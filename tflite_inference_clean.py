"""KokoroTFLiteTTS — clean TFLite inference pipeline for Kokoro TTS.

Pipeline (7 stages):
    BERT → Duration Predictor → Duration Expansion (Python)
    → Acoustic Expand → F0N Predictor → Vocos Conditioner → Vocos Streaming

Usage::

    tts = KokoroTFLiteTTS()
    input_ids, style = tts.text_to_inputs("Hello world.")
    for chunk in tts.forward(input_ids, style):
        play(chunk)   # float32 PCM at SAMPLE_RATE Hz

    # Or all-at-once:
    audio = tts.generate("Hello world.")
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ai_edge_litert.interpreter import Interpreter

from kokoro import KModel, KPipeline
from kokoro.model import KModelForONNX

# ── Pipeline constants ────────────────────────────────────────────────────────
MAX_INPUT_LEN = 510          # max phoneme tokens
T_ACOUSTIC    = 543          # static time dim for acoustic_expand / f0n
T_F0          = 1086         # 2 × T_ACOUSTIC; static time dim for conditioner
N_FFT         = 1200         # Vocos STFT / IRFFT size
HOP           = 300          # Vocos hop length
WIN_LEN       = 1200         # ISTFT window length (= N_FFT)
OLA_TAIL      = WIN_LEN - HOP  # overlap-add carry buffer = 900
CHUNK_FRAMES  = 16           # Vocos streaming chunk size (time frames)
COND_DIM      = 192          # Vocos conditioner output channels
SAMPLE_RATE   = 24_000


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_name(raw: str) -> str:
    """Strip onnx2tf 'serving_default_' prefix and ':N' suffix."""
    s = raw
    if s.startswith("serving_default_"):
        s = s[len("serving_default_"):]
    colon = s.rfind(":")
    if colon != -1:
        s = s[:colon]
    return s


def _overlap_add(
    frames: np.ndarray,  # [1, F, WIN_LEN]
    tail: np.ndarray,    # [1, OLA_TAIL]
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlap-add one chunk. Returns (audio [1, F*HOP], new_tail [1, OLA_TAIL])."""
    B, F, _ = frames.shape
    buf = np.zeros((B, F * HOP + OLA_TAIL), np.float32)
    for f in range(F):
        buf[:, f * HOP : f * HOP + WIN_LEN] += frames[:, f]
    buf[:, :OLA_TAIL] += tail
    return buf[:, :F * HOP], buf[:, F * HOP:]


def _expand_durations(pred_dur: np.ndarray) -> Tuple[np.ndarray, int]:
    """Rounded phoneme durations → per-frame phoneme indices.

    Returns (indices [T_acoustic], T_acoustic).
    """
    dur = torch.from_numpy(pred_dur.astype(np.float32))
    boundaries = torch.cumsum(dur, 0)
    T = int(boundaries[-1].item())
    if T == 0:
        return np.zeros(0, np.int64), 0
    vals = torch.arange(T, dtype=torch.int32)
    idx = torch.sum(boundaries.unsqueeze(1) <= vals.unsqueeze(0), dim=0)
    return idx.numpy().astype(np.int64), T


def _build_conditioner_features(
    asr: np.ndarray,    # [1, 512, T_a]
    f0: np.ndarray,     # [1, T_f0]
    n: np.ndarray,      # [1, T_f0]
    style: np.ndarray,  # [1, 256]
) -> np.ndarray:
    """Assemble [1, 642, T_f0] NCT feature array for the Vocos conditioner."""
    T = f0.shape[-1]
    asr_t = torch.from_numpy(asr)
    if asr_t.shape[-1] != T:
        asr_t = F.interpolate(asr_t, size=T, mode="linear", align_corners=False)
    f0_t  = torch.from_numpy(f0).unsqueeze(1)
    n_t   = torch.from_numpy(n).unsqueeze(1)
    s_t   = torch.from_numpy(style[:, :128]).unsqueeze(-1).expand(-1, -1, T)
    return torch.cat([asr_t, f0_t, n_t, s_t], dim=1).numpy()   # [1, 642, T]


# ── Main class ────────────────────────────────────────────────────────────────

class KokoroTFLiteTTS:
    """Kokoro TTS inference using onnx2tf-converted TFLite models."""

    SAMPLE_RATE: int = SAMPLE_RATE

    def __init__(
        self,
        saved_model_dir: str = "onnx2tf_conversion/saved_model",
        config_path: str = "checkpoints/config.json",
        kokoro_checkpoint: str = "checkpoints/kokoro-v1_0.pth",
        voice_path: str = "checkpoints/voices/af_bella.pt",
        num_threads: int = 4,
    ) -> None:
        self._voice_path = voice_path
        self._hann = torch.hann_window(WIN_LEN).numpy()

        # G2P + vocabulary
        kmodel = KModel(config=config_path, model=kokoro_checkpoint, disable_complex=True).cpu()
        self._kmodel_onnx = KModelForONNX(kmodel).eval()
        self._kpipeline = KPipeline(lang_code="a", model=kmodel, device="cpu")

        # TFLite interpreters
        d = Path(saved_model_dir)
        self._m: Dict[str, Interpreter] = {
            name: self._load_interp(d / f"{name}_float32.tflite", num_threads)
            for name in [
                "bert",
                "duration_predictor",
                "acoustic_expand",
                "f0n_predictor",
                "vocoder_conditioner",
                "vocoder_stream_chunk",
            ]
        }

        # Detect vocoder tensor layouts (onnx2tf may swap NCT ↔ NTC)
        K = N_FFT // 2 + 1
        in0  = self._m["vocoder_stream_chunk"].get_input_details()[0]["shape"]
        out0 = self._m["vocoder_stream_chunk"].get_output_details()[0]["shape"]
        self._vocos_in_ntc  = len(in0)  == 3 and in0[-1]  == COND_DIM
        self._vocos_out_ntk = len(out0) == 3 and out0[-1] == K

    @staticmethod
    def _load_interp(path: Path, num_threads: int) -> Interpreter:
        if not path.exists():
            raise FileNotFoundError(f"TFLite model not found: {path}")
        interp = Interpreter(model_path=str(path), num_threads=num_threads)
        interp.allocate_tensors()
        return interp

    def _ins(self, name: str) -> Dict[str, int]:
        """Return {normalized_tensor_name: index} for a model's input tensors."""
        return {_norm_name(d["name"]): d["index"]
                for d in self._m[name].get_input_details()}

    # ── Text / voice ──────────────────────────────────────────────────────────

    def text_to_inputs(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Convert text to (input_ids [1, T] int32, style [1, 256] float32).

        input_ids is NOT padded; pass directly to ``forward()``.
        """
        pipe = self._kpipeline
        if pipe.lang_code in "ab":
            _, tokens = pipe.g2p(text)
            ps: List[str] = []
            for _, p, _ in pipe.en_tokenize(tokens):
                ps.extend(p)
        else:
            ps, _ = pipe.g2p(text)
        ps = ps[:MAX_INPUT_LEN]

        vocab = pipe.model.vocab
        ids = [i for i in (vocab.get(p) for p in ps) if i is not None]
        input_ids = np.array([[0, *ids, 0]], dtype=np.int32)

        pack  = pipe.load_voice(self._voice_path).cpu()
        style = pack[len(ps) - 1].detach().numpy()
        if style.ndim == 1:
            style = style[np.newaxis]   # [256] → [1, 256]
        return input_ids, style.astype(np.float32)

    # ── Pipeline stages ───────────────────────────────────────────────────────

    def _run_bert(self, input_ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """input_ids [1,510], mask [1,510] → d_en [1, 510, 512] NTC"""
        ins = self._ins("bert")
        m = self._m["bert"]
        m.set_tensor(ins["input_ids"], input_ids)
        m.set_tensor(ins["text_mask"], mask)
        m.invoke()
        return m.get_tensor(m.get_output_details()[0]["index"])

    def _run_duration(
        self,
        d_en: np.ndarray,       # [1, 510, 512] NTC
        style: np.ndarray,      # [1, 256]
        mask: np.ndarray,       # [1, 510]
        speed: np.ndarray,      # [1] float32
        input_ids: np.ndarray,  # [1, 510]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """→ (pred_dur [510], d_enc [1,640,510] NCT, t_en [1,512,510] NCT)"""
        ins = self._ins("duration_predictor")
        dtypes = {_norm_name(d["name"]): d["dtype"]
                  for d in self._m["duration_predictor"].get_input_details()}
        m = self._m["duration_predictor"]
        m.set_tensor(ins["d_en"],       d_en)
        m.set_tensor(ins["style"],      style)
        m.set_tensor(ins["text_mask"],  mask)
        m.set_tensor(ins["speed"],      speed.astype(dtypes.get("speed", np.float32)))
        m.set_tensor(ins["input_ids"],  input_ids.astype(dtypes.get("input_ids", np.int64)))
        m.invoke()
        # Match outputs by shape: (510,)=pred_dur, (1,640,510)=d_enc, (1,512,510)=t_en
        outs = {tuple(d["shape"]): m.get_tensor(d["index"])
                for d in m.get_output_details()}
        return outs[(510,)], outs[(1, 640, 510)], outs[(1, 512, 510)]

    def _run_acoustic_expand(self, d_enc_ntc: np.ndarray) -> np.ndarray:
        """d_enc_expanded [1, T_ACOUSTIC, 640] NTC → en [1, T_ACOUSTIC, 512] NTC"""
        m = self._m["acoustic_expand"]
        m.set_tensor(m.get_input_details()[0]["index"], d_enc_ntc)
        m.invoke()
        return m.get_tensor(m.get_output_details()[0]["index"])

    def _run_f0n(
        self, en_nct: np.ndarray, style: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """en [1,512,T_ACOUSTIC] NCT, style [1,256] → F0 [1,T_F0], N [1,T_F0]"""
        ins = self._ins("f0n_predictor")
        m = self._m["f0n_predictor"]
        m.set_tensor(ins["en"],    en_nct)
        m.set_tensor(ins["style"], style)
        m.invoke()
        outs = {_norm_name(d["name"]): m.get_tensor(d["index"])
                for d in m.get_output_details()}
        return outs["F0_pred"], outs["N_pred"]

    def _run_conditioner(self, features_ntc: np.ndarray) -> np.ndarray:
        """features [1, T_F0, 642] NTC → conditioned [1, 192, T_F0] NCT"""
        m = self._m["vocoder_conditioner"]
        m.set_tensor(m.get_input_details()[0]["index"], features_ntc)
        m.invoke()
        return m.get_tensor(m.get_output_details()[0]["index"])

    def _stream_vocos(self, conditioned: np.ndarray) -> Iterator[np.ndarray]:
        """Yield audio chunks from conditioned [1, 192, T] NCT via streaming TFLite."""
        m    = self._m["vocoder_stream_chunk"]
        in_d = m.get_input_details()
        out_d = m.get_output_details()

        state    = [np.zeros(d["shape"], np.float32) for d in in_d[1:]]
        ola_tail = np.zeros((1, OLA_TAIL), np.float32)
        T = conditioned.shape[-1]

        for pos in range(0, T, CHUNK_FRAMES):
            valid = min(CHUNK_FRAMES, T - pos)
            chunk = conditioned[..., pos : pos + valid]
            if valid < CHUNK_FRAMES:
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, CHUNK_FRAMES - valid)))
            if self._vocos_in_ntc:
                chunk = chunk.transpose(0, 2, 1)    # NCT → NTC

            m.set_tensor(in_d[0]["index"], chunk.astype(np.float32))
            for i, s in enumerate(state):
                m.set_tensor(in_d[1 + i]["index"], s)
            m.invoke()

            # x_real / x_imag outputs: [1, F, K] NTK or [1, K, F] NKT
            xr = m.get_tensor(out_d[0]["index"])
            xi = m.get_tensor(out_d[1]["index"])
            if not self._vocos_out_ntk:
                xr, xi = xr.transpose(0, 2, 1), xi.transpose(0, 2, 1)

            # IRFFT + Hann window + overlap-add
            frames = np.fft.irfft(xr + 1j * xi, n=N_FFT, axis=-1)[..., :WIN_LEN].astype(np.float32)
            frames *= self._hann
            audio, ola_tail = _overlap_add(frames, ola_tail)
            yield audio[0, : valid * HOP]

            # Update recurrent state; transpose NCT → NTC where shape mismatches
            new_state = []
            for i, s_in in enumerate(in_d[1:]):
                s_out = m.get_tensor(out_d[2 + i]["index"])
                if s_out.ndim == 3 and tuple(s_out.shape) != tuple(s_in["shape"]):
                    s_out = s_out.transpose(0, 2, 1)
                new_state.append(s_out)
            state = new_state

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: np.ndarray,  # [1, T] int32, unpadded
        style: np.ndarray,      # [1, 256] float32
        speed: float = 1.0,
    ) -> Iterator[np.ndarray]:
        """Run the full TTS pipeline and yield float32 audio chunks at 24 kHz.

        Parameters
        ----------
        input_ids : np.ndarray, shape [1, T], dtype int32
            Phoneme token IDs (unpadded) from ``text_to_inputs()``.
        style : np.ndarray, shape [1, 256], dtype float32
            Voice embedding from ``text_to_inputs()``.
        speed : float
            Speaking rate multiplier (default 1.0).

        Yields
        ------
        chunk : np.ndarray, shape (N,), dtype float32
        """
        # Pad input_ids → [1, MAX_INPUT_LEN] and derive text_mask
        T_orig = input_ids.shape[1]
        mask = np.zeros((1, MAX_INPUT_LEN), np.float32)
        mask[0, :T_orig] = 1.0
        pad_len = MAX_INPUT_LEN - T_orig
        ids = np.concatenate(
            [input_ids, np.zeros((1, pad_len), np.int32)], axis=1
        ) if pad_len > 0 else input_ids[:, :MAX_INPUT_LEN]
        speed_arr = np.array([speed], np.float32)

        # Stage 1: BERT encoder
        d_en = self._run_bert(ids, mask)                # [1, 510, 512] NTC

        # Stage 2: Duration predictor
        pred_dur, d_enc, t_en = self._run_duration(     # [510], [1,640,510], [1,512,510]
            d_en, style, mask, speed_arr, ids
        )

        # Stage 3: Expand durations (Python)
        idx, T_a = _expand_durations(pred_dur)
        T_a  = min(T_a, T_ACOUSTIC)
        idx  = idx[:T_a]
        T_f0 = 2 * T_a

        d_enc_exp = np.take(d_enc, idx, axis=2)         # [1, 640, T_a]
        asr       = np.take(t_en,  idx, axis=2)         # [1, 512, T_a]

        # Stage 4: Acoustic expand (static T_ACOUSTIC)
        d_enc_ntc = d_enc_exp.transpose(0, 2, 1)        # NTC [1, T_a, 640]
        if T_a < T_ACOUSTIC:
            d_enc_ntc = np.pad(d_enc_ntc, ((0, 0), (0, T_ACOUSTIC - T_a), (0, 0)))
        en_nct = self._run_acoustic_expand(d_enc_ntc).transpose(0, 2, 1)  # [1, 512, T_ACOUSTIC]

        # Stage 5: F0 / noise predictor
        f0_full, n_full = self._run_f0n(en_nct, style)  # [1, T_F0] each
        f0 = f0_full[:, :T_f0]
        n  = n_full[:,  :T_f0]

        # Stage 6: Vocos conditioner (static T_F0)
        features_ntc = _build_conditioner_features(asr, f0, n, style).transpose(0, 2, 1)
        if T_f0 < T_F0:
            features_ntc = np.pad(features_ntc, ((0, 0), (0, T_F0 - T_f0), (0, 0)))
        conditioned = self._run_conditioner(features_ntc)[:, :, :T_f0]  # [1, 192, T_f0]

        # Stage 7: Vocos streaming decoder
        yield from self._stream_vocos(conditioned)

    def generate(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Synthesize text and return a mono float32 waveform at 24 kHz."""
        input_ids, style = self.text_to_inputs(text)
        return np.concatenate(list(self.forward(input_ids, style, speed)))
