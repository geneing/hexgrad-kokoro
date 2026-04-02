"""tflite_inference.py

TFLite inference pipeline for Kokoro TTS using onnx2tf-converted models.

Pipeline stages
---------------
1. ``bert_float32.tflite``              — BERT-style phoneme encoder
2. ``duration_predictor_float32.tflite``— Duration + text features predictor
3. ``acoustic_expand_float32.tflite``   — Shared BiLSTM on expanded features
4. ``f0n_predictor_float32.tflite``     — F0 / noise predictor
5. ``vocoder_conditioner_float32.tflite``— Vocos conditioner (full sequence)
6. ``vocoder_stream_chunk_float32.tflite``— Streaming Vocos backbone + iSTFT head

Tensor layout notes (onnx2tf NCHW → NHWC conversion)
-----------------------------------------------------
onnx2tf converts some I/O tensors between ONNX ``(N, C, T)`` and TFLite
``(N, T, C)`` layouts.  The exact conversion per module is:

  Module                 | Input layout  | Output layout
  -----------------------|---------------|---------------
  bert                   | N×T           | N×T×C (NTC)
  duration_predictor     | d_en: NTC     | d_enc: NCT, t_en_static: NCT
  acoustic_expand        | d_enc: NTC    | en: NTC
  f0n_predictor          | en: NCT       | F0/N: N×T
  vocoder_conditioner    | features: NTC | conditioned: NCT
  vocoder_stream_chunk   | chunk: NCT or NTC (auto-detected) + state → audio + state

This module handles all required transposes so that callers see the same
NumPy / Torch tensor conventions as the original PyTorch pipeline.

Usage
-----
>>> tflite = KokoroTFLiteTTS()
>>> wav = tflite.generate("Hello world.")

>>> tflite = KokoroTFLiteTTS(
...     saved_model_dir="onnx2tf_conversion/saved_model",
...     config_path="checkpoints/config.json",
...     kokoro_checkpoint="checkpoints/kokoro-v1_0.pth",
...     voice_path="checkpoints/voices/af_bella.pt",
... )
>>> wav = tflite.generate("Hello world, from Kokoro.")

>>> pt = KokoroPTTTS(
...     config_path="checkpoints/config.json",
...     kokoro_checkpoint="checkpoints/kokoro-v1_0.pth",
...     vocos_checkpoint="vocos_fp16.pt",
...     voice_path="checkpoints/voices/af_bella.pt",
... )
>>> wav_ref = pt.generate("Hello world.")
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import ai_edge_litert.interpreter as litert

from kokoro import KModel, KPipeline
from kokoro.model import KModelForONNX

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_INPUT_LENGTH = 510
_SAMPLE_RATE = 24_000
# Static TFLite model dimensions (set when the ONNX models were exported then
# converted with onnx2tf — values come from the actual tflite tensor shapes).
_T_ACOUSTIC = 543    # acoustic_expand / f0n_predictor time dimension
_T_F0 = 1086         # 2 × _T_ACOUSTIC; conditioner time dimension
_CHUNK_FRAMES = 16   # vocos backbone chunk size
_HOP = 300           # vocos hop length
_WIN_LEN = 1200      # ISTFT window length (= n_fft for vocos)
_VOCOS_N_FFT = 1200  # same as WIN_LEN for this model
_ISTFT_TAIL = _WIN_LEN - _HOP  # overlap tail between chunks (900)
_VOCOS_EMBED_IN = 192
_VOCOS_BLOCK_DIM = 384
_VOCOS_KERNEL_M1 = 6    # k−1 context frames per causal conv


# ---------------------------------------------------------------------------
# Shared helpers (mirror the notebook)
# ---------------------------------------------------------------------------

def _overlap_add(
    time_frames: np.ndarray,   # [1, F, win_len]  (NTC)
    istft_prev:  np.ndarray,   # [1, tail]
    hop: int,
    tail: int,
) -> "Tuple[np.ndarray, np.ndarray]":
    """Overlap-add one chunk of windowed time frames.

    Returns ``(audio [1, F*hop], new_istft_prev [1, tail])``.
    """
    B, F, win_len = time_frames.shape
    # Buffer extends to the end of the last frame: (F-1)*hop + win_len = F*hop + tail
    buf = np.zeros((B, F * hop + tail), dtype=np.float32)
    for f in range(F):
        buf[:, f * hop : f * hop + win_len] += time_frames[:, f, :]
    buf[:, :tail] += istft_prev
    audio          = buf[:, :F * hop]
    new_istft_prev = buf[:, F * hop:]
    return audio, new_istft_prev


def _expand_durations(pred_dur: np.ndarray) -> Tuple[np.ndarray, int]:
    """Compute per-frame phoneme indices from rounded per-phoneme durations.

    Parameters
    ----------
    pred_dur:
        float32 array of shape ``(510,)`` — rounded durations, zero for
        padding phonemes.

    Returns
    -------
    expanded_indices : np.ndarray, shape (T_acoustic,), dtype int64
        Index into the phoneme dimension for each acoustic frame.
    T_acoustic : int
        Actual number of acoustic frames (sum of positive durations).
    """
    dur = torch.from_numpy(pred_dur.astype(np.float32))
    boundaries = torch.cumsum(dur, dim=0)
    T_acoustic = int(boundaries[-1].item())
    values = torch.arange(T_acoustic, dtype=torch.int32)
    expanded_indices = torch.sum(
        boundaries.unsqueeze(1) <= values.unsqueeze(0), dim=0
    )
    return expanded_indices.numpy().astype(np.int64), T_acoustic


def _build_vocos_features(
    asr: np.ndarray,      # [1, 512, T_asr] NCT
    F0_pred: np.ndarray,  # [1, T_f0]
    N_pred: np.ndarray,   # [1, T_f0]
    style: np.ndarray,    # [1, 256]
) -> np.ndarray:
    """Assemble ``[1, 642, T_f0]`` feature array (NCT) for the Vocos conditioner.

    If ``T_asr != T_f0`` the ASR features are linearly interpolated to match.
    """
    asr_t = torch.from_numpy(asr.astype(np.float32))     # [1, 512, T_asr]
    T_f0 = F0_pred.shape[-1]
    if asr_t.shape[-1] != T_f0:
        asr_t = F.interpolate(asr_t, size=T_f0, mode="linear", align_corners=False)
    f0 = torch.from_numpy(F0_pred.astype(np.float32)).unsqueeze(1)   # [1, 1, T_f0]
    n  = torch.from_numpy(N_pred.astype(np.float32)).unsqueeze(1)    # [1, 1, T_f0]
    s  = torch.from_numpy(style.astype(np.float32))[:, :128].unsqueeze(-1).expand(-1, -1, T_f0)
    features = torch.cat([asr_t, f0, n, s], dim=1)  # [1, 642, T_f0]
    return features.numpy()


def _to_numpy(t: Tensor) -> np.ndarray:
    t = t.detach().cpu()
    return t.numpy() if t.is_floating_point() else t.numpy()


# ---------------------------------------------------------------------------
# KokoroTFLiteTTS
# ---------------------------------------------------------------------------

class KokoroTFLiteTTS:
    """Kokoro TTS inference using onnx2tf-converted TFLite models (full pipeline).

    All six pipeline stages run on TFLite: ``bert``, ``duration_predictor``,
    ``acoustic_expand``, ``f0n_predictor``, ``vocoder_conditioner``, and
    ``vocoder_stream_chunk``.

    The ``vocoder_stream_chunk_float32.tflite`` model is converted from
    ``VocosStreamChunkNoOLA``, which replaces ``torch.fft.irfft`` with explicit
    real-arithmetic IDFT (cos/sin matrix multiplications) and removes the
    ``ConvTranspose1d`` overlap-add layer.  The overlap-add is performed in
    Python/NumPy instead, avoiding the ~26× amplitude error that
    ``onnx2tf flatbuffer_direct`` produces for that ConvTranspose.

    Parameters
    ----------
    saved_model_dir:
        Directory containing ``*_float32.tflite`` files produced by onnx2tf.
        Defaults to ``"onnx2tf_conversion/saved_model"``.
    config_path:
        Kokoro ``config.json`` path.
        Defaults to ``"checkpoints/config.json"``.
    kokoro_checkpoint:
        Kokoro ``.pth`` weights path (only used for text/phoneme processing).
        Defaults to ``"checkpoints/kokoro-v1_0.pth"``.
    voice_path:
        Path to a ``voices/*.pt`` style vector file.
        Defaults to ``"checkpoints/voices/af_bella.pt"``.
    num_threads:
        Number of intra-op threads for the TFLite XNNPACK delegate.
        Defaults to ``4``.
    """

    # Static TFLite dimensions (read from model files; stored here for clarity)
    T_ACOUSTIC: int = _T_ACOUSTIC
    T_F0: int = _T_F0
    CHUNK_FRAMES: int = _CHUNK_FRAMES
    HOP: int = _HOP
    SAMPLE_RATE: int = _SAMPLE_RATE

    def __init__(
        self,
        saved_model_dir: str = "onnx2tf_conversion/saved_model",
        config_path: str = "checkpoints/config.json",
        kokoro_checkpoint: str = "checkpoints/kokoro-v1_0.pth",
        voice_path: str = "checkpoints/voices/af_bella.pt",
        num_threads: int = 4,
    ) -> None:
        self._model_dir = Path(saved_model_dir)
        self._voice_path = voice_path
        self._num_threads = num_threads

        # ── Text-processing pipeline (Kokoro KModel + KPipeline) ────────────
        kmodel = KModel(config=config_path, model=kokoro_checkpoint, disable_complex=True).to("cpu")
        self._kmodel_onnx = KModelForONNX(kmodel).eval()
        self._kpipeline = KPipeline(lang_code="a", model=kmodel, device="cpu")

        # ── TFLite models ────────────────────────────────────────────────────
        self._bert_interp  = self._load_interpreter("bert_float32.tflite")
        self._dur_interp   = self._load_interpreter("duration_predictor_float32.tflite")
        self._acexp_interp = self._load_interpreter("acoustic_expand_float32.tflite")
        self._f0n_interp   = self._load_interpreter("f0n_predictor_float32.tflite")
        self._cond_interp  = self._load_interpreter("vocoder_conditioner_float32.tflite")
        self._vocos_interp = self._load_interpreter("vocoder_stream_chunk_float32.tflite")

        # Verify static dimensions match expected constants
        self._verify_model_dimensions()

        # ── Pre-compute Hann window for IRFFT windowing ──────────────────────
        import torch as _torch
        self._vocos_window = _torch.hann_window(_WIN_LEN).numpy()   # [win_len]

        # ── Auto-detect vocoder chunk input/output layouts ────────────────
        # onnx2tf may transpose NCT tensors to NTC (last-dim = channels).
        # conditioned_chunk input: [1, 192, 16] NCT → [1, 16, 192] NTC
        # x_real/x_imag outputs: [1, F, K=601] NTK or [1, K, F] NKT
        _vsc_in0_shape  = self._vocos_interp.get_input_details()[0]["shape"]
        _vsc_out0_shape = self._vocos_interp.get_output_details()[0]["shape"]
        _K = _VOCOS_N_FFT // 2 + 1
        self._vocos_chunk_ntc: bool = (
            len(_vsc_in0_shape) == 3 and _vsc_in0_shape[-1] == _VOCOS_EMBED_IN
        )
        # x_real is NTK when last dim equals K (601)
        self._vocos_xreal_ntk: bool = (
            len(_vsc_out0_shape) == 3 and _vsc_out0_shape[-1] == _K
        )

    # ── Model loading helpers ────────────────────────────────────────────────

    def _load_interpreter(self, name: str) -> litert.Interpreter:
        path = self._model_dir / name
        if not path.exists():
            raise FileNotFoundError(f"TFLite model not found: {path}")
        interp = litert.Interpreter(
            model_path=str(path),
            num_threads=self._num_threads,
        )
        interp.allocate_tensors()
        return interp

    def _verify_model_dimensions(self) -> None:
        acexp_in = self._acexp_interp.get_input_details()[0]["shape"]
        f0n_in = self._f0n_interp.get_input_details()[0]["shape"]
        cond_in = self._cond_interp.get_input_details()[0]["shape"]
        # acexp_in: [1, T_ACOUSTIC, 640]
        # f0n_in:   [1, 512, T_ACOUSTIC]
        # cond_in:  [1, T_F0, 642]
        assert tuple(acexp_in) == (1, self.T_ACOUSTIC, 640), f"Unexpected acoustic_expand shape {acexp_in}"
        assert tuple(f0n_in) == (1, 512, self.T_ACOUSTIC), f"Unexpected f0n shape {f0n_in}"
        assert tuple(cond_in) == (1, self.T_F0, 642), f"Unexpected conditioner shape {cond_in}"

    # ── Stage helpers ────────────────────────────────────────────────────────

    def _run_bert(
        self, input_ids: np.ndarray, text_mask: np.ndarray
    ) -> np.ndarray:
        """BERT encoder: (input_ids [1,510], text_mask [1,510]) → d_en [1, 510, 512] NTC."""
        interp = self._bert_interp
        ins = {d["name"]: d["index"] for d in interp.get_input_details()}
        interp.set_tensor(ins["input_ids"], input_ids.astype(np.int32))
        interp.set_tensor(ins["text_mask"], text_mask.astype(np.float32))
        interp.invoke()
        return interp.get_tensor(interp.get_output_details()[0]["index"])  # [1, 510, 512]

    def _run_duration(
        self,
        d_en: np.ndarray,    # [1, 510, 512] NTC
        style: np.ndarray,   # [1, 256]
        text_mask: np.ndarray,  # [1, 510]
        speed: np.ndarray,   # [1] int32
        input_ids: np.ndarray,  # [1, 510] int32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Duration predictor → (pred_dur [510], d_enc [1,640,510] NCT, t_en_static [1,512,510] NCT)."""
        interp = self._dur_interp
        ins = {d["name"]: d["index"] for d in interp.get_input_details()}
        interp.set_tensor(ins["d_en"], d_en.astype(np.float32))
        interp.set_tensor(ins["style"], style.astype(np.float32))
        interp.set_tensor(ins["text_mask"], text_mask.astype(np.float32))
        interp.set_tensor(ins["speed"], speed.astype(np.int32))
        interp.set_tensor(ins["input_ids"], input_ids.astype(np.int32))
        interp.invoke()
        outs = {d["name"]: interp.get_tensor(d["index"]) for d in interp.get_output_details()}
        return outs["pred_dur"], outs["d_enc"], outs["t_en_static"]

    def _run_acoustic_expand(self, d_enc_exp_ntc: np.ndarray) -> np.ndarray:
        """Acoustic expand: d_enc_expanded [1, T_ACOUSTIC, 640] NTC → en [1, T_ACOUSTIC, 512] NTC."""
        assert d_enc_exp_ntc.shape == (1, self.T_ACOUSTIC, 640), (
            f"Expected [1, {self.T_ACOUSTIC}, 640], got {d_enc_exp_ntc.shape}"
        )
        interp = self._acexp_interp
        interp.set_tensor(interp.get_input_details()[0]["index"], d_enc_exp_ntc.astype(np.float32))
        interp.invoke()
        return interp.get_tensor(interp.get_output_details()[0]["index"])  # [1, T_ACOUSTIC, 512]

    def _run_f0n(
        self, en_nct: np.ndarray, style: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """F0N predictor: en [1, 512, T_ACOUSTIC], style [1,256] → F0 [1, T_F0], N [1, T_F0]."""
        assert en_nct.shape == (1, 512, self.T_ACOUSTIC), (
            f"Expected [1, 512, {self.T_ACOUSTIC}], got {en_nct.shape}"
        )
        interp = self._f0n_interp
        ins = {d["name"]: d["index"] for d in interp.get_input_details()}
        interp.set_tensor(ins["en"], en_nct.astype(np.float32))
        interp.set_tensor(ins["style"], style.astype(np.float32))
        interp.invoke()
        outs = {d["name"]: interp.get_tensor(d["index"]) for d in interp.get_output_details()}
        return outs["F0_pred"], outs["N_pred"]

    def _run_conditioner(self, features_ntc: np.ndarray) -> np.ndarray:
        """Conditioner: features [1, T_F0, 642] NTC → conditioned [1, 192, T_F0] NCT."""
        assert features_ntc.shape == (1, self.T_F0, 642), (
            f"Expected [1, {self.T_F0}, 642], got {features_ntc.shape}"
        )
        interp = self._cond_interp
        interp.set_tensor(interp.get_input_details()[0]["index"], features_ntc.astype(np.float32))
        interp.invoke()
        return interp.get_tensor(interp.get_output_details()[0]["index"])  # [1, 192, T_F0]

    def _run_vocos_stream_chunk(self, conditioned: np.ndarray) -> np.ndarray:
        """Run the streaming Vocos backbone via TFLite, apply numpy IRFFT + OLA.

        The TFLite model (VocosPreIRFFT) outputs x_real/x_imag [1,F,K] — the
        complex spectrum just before irfft.  numpy.fft.irfft (= tf.signal.irfft)
        is applied here in Python, followed by Hann-window and overlap-add.

        Parameters
        ----------
        conditioned:
            ``[1, 192, T_f0_actual]`` float32 NCT array.

        Returns
        -------
        audio: np.ndarray, shape ``(T_f0_actual * _HOP,)``
        """
        interp = self._vocos_interp
        in_d   = interp.get_input_details()
        out_d  = interp.get_output_details()

        total = conditioned.shape[-1]   # conditioned is NCT: [1, 192, T_f0]
        _K    = _VOCOS_N_FFT // 2 + 1

        state      = [np.zeros(d["shape"], dtype=np.float32) for d in in_d[1:]]
        istft_prev = np.zeros((1, _ISTFT_TAIL), dtype=np.float32)

        chunks: List[np.ndarray] = []
        pos = 0
        while pos < total:
            end   = min(total, pos + _CHUNK_FRAMES)
            valid = end - pos
            chunk = conditioned[..., pos:end]        # [1, 192, valid] NCT
            if valid < _CHUNK_FRAMES:
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, _CHUNK_FRAMES - valid)))

            if self._vocos_chunk_ntc:
                chunk = np.transpose(chunk, (0, 2, 1))   # NCT → NTC

            interp.set_tensor(in_d[0]["index"], chunk.astype(np.float32))
            for i, s in enumerate(state):
                interp.set_tensor(in_d[1 + i]["index"], s)
            interp.invoke()

            # x_real / x_imag: [1, F, K] NTK or [1, K, F] NKT
            xr_raw = interp.get_tensor(out_d[0]["index"])
            xi_raw = interp.get_tensor(out_d[1]["index"])
            if self._vocos_xreal_ntk:
                x_real, x_imag = xr_raw, xi_raw          # already [1, F, K]
            else:
                x_real = np.transpose(xr_raw, (0, 2, 1)) # NKT → NTK [1, F, K]
                x_imag = np.transpose(xi_raw, (0, 2, 1))

            # numpy IRFFT (= tf.signal.irfft) + Hann window
            spec = x_real + 1j * x_imag
            tf_chunk = np.fft.irfft(spec, n=_VOCOS_N_FFT, axis=-1)[..., : _WIN_LEN].astype(np.float32)
            tf_chunk = tf_chunk * self._vocos_window                # [1, F, win_len]

            audio_chunk, istft_prev = _overlap_add(tf_chunk, istft_prev, _HOP, _ISTFT_TAIL)
            chunks.append(audio_chunk[0, : valid * _HOP])

            # Update backbone state; transpose NCT → NTC where needed
            new_state: List[np.ndarray] = []
            for i, s_in_d in enumerate(in_d[1:]):
                s_out = interp.get_tensor(out_d[2 + i]["index"])   # outputs 0,1 = x_real,x_imag
                if s_out.ndim == 3 and tuple(s_out.shape) != tuple(s_in_d["shape"]):
                    s_out = np.transpose(s_out, (0, 2, 1))
                new_state.append(s_out)
            state = new_state
            pos = end

        return np.concatenate(chunks, axis=-1)

    # ── Text-processing helpers (mirror the notebook) ───────────────────────

    def _text_to_input_ids(self, text: str) -> Tuple[List[str], Tensor]:
        """Return (phonemes, input_ids [1, ≤510])."""
        if self._kpipeline.lang_code in "ab":
            _, tokens = self._kpipeline.g2p(text)
            ps: List[str] = []
            for _, p, _ in self._kpipeline.en_tokenize(tokens):
                ps.extend(p)
        else:
            ps, _ = self._kpipeline.g2p(text)
        ps = ps[:_MAX_INPUT_LENGTH]

        vocab = self._kpipeline.model.vocab
        ids = [i for i in (vocab.get(p) for p in ps) if i is not None]
        input_ids = torch.IntTensor([[0, *ids, 0]])
        return ps, input_ids

    def _load_voice(self, phonemes: List[str]) -> Tensor:
        pack = self._kpipeline.load_voice(self._voice_path).to("cpu")
        return pack[len(phonemes) - 1]

    def _build_padded_inputs(
        self, input_ids: Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pad input_ids to MAX_INPUT_LENGTH and build the text_mask."""
        text_mask = np.zeros((1, _MAX_INPUT_LENGTH), dtype=np.float32)
        text_mask[0, : input_ids.shape[1]] = 1.0
        pad_len = _MAX_INPUT_LENGTH - input_ids.shape[1]
        if pad_len > 0:
            input_ids = F.pad(input_ids, (0, pad_len))
        return input_ids.numpy().astype(np.int32), text_mask

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(self, text: str, speed: int = 1) -> np.ndarray:
        """Synthesise ``text`` and return a mono float32 waveform at 24 kHz.

        Parameters
        ----------
        text:
            Input text (English).
        speed:
            Speed multiplier (``1`` = normal).

        Returns
        -------
        audio : np.ndarray, shape ``(N,)``, dtype float32
        """
        # ── Text processing ───────────────────────────────────────────────
        phonemes, raw_input_ids = self._text_to_input_ids(text)
        style = _to_numpy(self._load_voice(phonemes))   # [1, 256]
        input_ids_np, text_mask_np = self._build_padded_inputs(raw_input_ids)
        speed_np = np.array([speed], dtype=np.int32)

        # ── Stage 1: BERT encoder ─────────────────────────────────────────
        # IN:  input_ids [1,510] int32, text_mask [1,510] float32
        # OUT: d_en      [1,510,512] NTC float32
        d_en_np = self._run_bert(input_ids_np, text_mask_np)

        # ── Stage 2: Duration predictor ───────────────────────────────────
        # IN:  d_en [1,510,512] NTC, style [1,256], text_mask [1,510],
        #      speed [1] int32, input_ids [1,510] int32
        # OUT: pred_dur [510], d_enc [1,640,510] NCT, t_en_static [1,512,510] NCT
        pred_dur_np, d_enc_np, t_en_static_np = self._run_duration(
            d_en_np, style, text_mask_np, speed_np, input_ids_np
        )

        # ── Stage 3: Duration expansion (Python — not exported) ───────────
        # d_enc and t_en_static are [1, C, T_text] NCT.
        # We index_select on the text dimension to expand to T_acoustic.
        expanded_indices, T_acoustic = _expand_durations(pred_dur_np)
        if T_acoustic > self.T_ACOUSTIC:
            warnings.warn(
                f"T_acoustic={T_acoustic} exceeds the TFLite static dimension "
                f"T_ACOUSTIC={self.T_ACOUSTIC}.  Input will be truncated.",
                RuntimeWarning,
                stacklevel=2,
            )
            T_acoustic = self.T_ACOUSTIC
            expanded_indices = expanded_indices[:T_acoustic]
        T_f0_actual = 2 * T_acoustic

        # Expand: [1, C, T_text] → [1, C, T_acoustic] (NCT)
        # Use np.take to avoid numpy advanced-indexing axis reordering
        d_enc_exp = np.take(d_enc_np, expanded_indices, axis=2)        # [1, 640, T_acoustic]
        asr = np.take(t_en_static_np, expanded_indices, axis=2)        # [1, 512, T_acoustic]

        # ── Stage 4: Acoustic expand (TFLite static T=T_ACOUSTIC) ─────────
        # TFLite expects d_enc_expanded [1, T_ACOUSTIC, 640] NTC.
        # Our d_enc_exp is [1, 640, T_acoustic] NCT → transpose + pad.
        d_enc_exp_ntc = np.transpose(d_enc_exp, (0, 2, 1))   # [1, T_acoustic, 640]
        if T_acoustic < self.T_ACOUSTIC:
            pad = self.T_ACOUSTIC - T_acoustic
            d_enc_exp_ntc = np.pad(d_enc_exp_ntc, ((0, 0), (0, pad), (0, 0)))

        en_ntc = self._run_acoustic_expand(d_enc_exp_ntc)     # [1, T_ACOUSTIC, 512]

        # Trim + transpose to NCT for f0n
        en_nct = np.transpose(en_ntc[:, :T_acoustic, :], (0, 2, 1))  # [1, 512, T_acoustic]
        if T_acoustic < self.T_ACOUSTIC:
            pad = self.T_ACOUSTIC - T_acoustic
            en_nct = np.pad(en_nct, ((0, 0), (0, 0), (0, pad)))      # [1, 512, T_ACOUSTIC]

        # ── Stage 5: F0 / noise predictor ────────────────────────────────
        # IN:  en [1, 512, T_ACOUSTIC] NCT, style [1, 256]
        # OUT: F0_pred [1, T_F0], N_pred [1, T_F0]
        F0_pred_full, N_pred_full = self._run_f0n(en_nct, style)
        F0_pred = F0_pred_full[:, :T_f0_actual]   # [1, T_f0_actual]
        N_pred = N_pred_full[:, :T_f0_actual]      # [1, T_f0_actual]

        # ── Stage 6: Vocos conditioner ───────────────────────────────────
        # Assemble features [1, 642, T_f0_actual] NCT, then transpose to NTC.
        features_nct = _build_vocos_features(asr, F0_pred, N_pred, style)
        # Transpose to NTC [1, T_f0_actual, 642] and pad to [1, T_F0, 642]
        features_ntc = np.transpose(features_nct, (0, 2, 1))  # [1, T_f0_actual, 642]
        if T_f0_actual < self.T_F0:
            pad = self.T_F0 - T_f0_actual
            features_ntc = np.pad(features_ntc, ((0, 0), (0, pad), (0, 0)))

        # IN:  features [1, T_F0, 642] NTC
        # OUT: conditioned [1, 192, T_F0] NCT
        conditioned_full = self._run_conditioner(features_ntc)     # [1, 192, T_F0]
        conditioned = conditioned_full[:, :, :T_f0_actual]         # [1, 192, T_f0_actual]

        # ── Stage 7: Vocos streaming chunks (PyTorch backbone + head) ─────
        # Input:  conditioned [1, 192, T_f0_actual] NCT
        # Output: audio (T_f0_actual * HOP,) float32
        audio = self._run_vocos_stream_chunk(conditioned)
        return audio


# ---------------------------------------------------------------------------
# KokoroPTTTS  (pure-PyTorch reference pipeline)
# ---------------------------------------------------------------------------

class KokoroPTTTS:
    """Pure-PyTorch Kokoro TTS reference pipeline (Vocos streaming decoder)."""

    SAMPLE_RATE: int = _SAMPLE_RATE

    def __init__(
        self,
        config_path: str,
        kokoro_checkpoint: str,
        vocos_checkpoint: str,
        voice_path: str,
        device: str = "cpu",
        chunk_frames: int = _CHUNK_FRAMES,
    ) -> None:
        self._voice_path = voice_path
        self._device = device
        self._chunk_frames = chunk_frames

        kmodel = KModel(config=config_path, model=kokoro_checkpoint, disable_complex=True).to(device)
        self._kmodel_onnx = KModelForONNX(kmodel).eval()
        self._kpipeline = KPipeline(lang_code="a", model=kmodel, device=device)
        from streaming_vocos import StreamingVocos
        self._vocos = StreamingVocos.from_checkpoint(
            vocos_checkpoint, chunk_frames=chunk_frames, device=device, use_fp16=False
        )
        self._hop = self._vocos.config.hop_length

        # Instantiate export modules (same weights, no copies)
        k = kmodel
        import torch.nn as nn

        class _BertModule(nn.Module):
            def __init__(self, k):
                super().__init__()
                self.bert = k.bert
                self.bert_encoder = k.bert_encoder
            def forward(self, ids, mask):
                return self.bert_encoder(self.bert(ids, attention_mask=mask)).transpose(-1, -2)

        class _DurModule(nn.Module):
            def __init__(self, k):
                super().__init__()
                self.predictor = k.predictor
                self.text_encoder = k.text_encoder
            def forward(self, d_en, style, text_mask, speed, input_ids):
                d = self.predictor.text_encoder(d_en, style[:, 128:], text_mask)
                duration = self.predictor.duration_proj(self.predictor.lstm(d)[0])
                duration = torch.sigmoid(duration).sum(dim=-1) / speed
                # Mask padding positions to zero — matches ONNX/TFLite model behaviour
                pred_dur = torch.round(duration).squeeze() * text_mask.squeeze()
                d_enc = d.transpose(-1, -2)
                t_en = self.text_encoder(input_ids, text_mask)
                return pred_dur, d_enc, t_en

        class _AcousticExpand(nn.Module):
            def __init__(self, k):
                super().__init__()
                self.shared = k.predictor.shared
            def forward(self, x):
                en, _ = self.shared(x.transpose(-1, -2))
                return en

        class _F0NModule(nn.Module):
            def __init__(self, k):
                super().__init__()
                self.predictor = k.predictor
            def forward(self, en, style):
                return self.predictor.F0Ntrain(en, style[:, 128:256])

        class _CondModule(nn.Module):
            def __init__(self, vocos):
                super().__init__()
                self.conditioner = vocos.model.conditioner
            def forward(self, x):
                return self.conditioner(x)

        self._bert_m = _BertModule(k).eval()
        self._dur_m = _DurModule(k).eval()
        self._acexp_m = _AcousticExpand(k).eval()
        self._f0n_m = _F0NModule(k).eval()
        self._cond_m = _CondModule(self._vocos).eval()

    def _text_to_input_ids(self, text: str) -> Tuple[List[str], Tensor]:
        if self._kpipeline.lang_code in "ab":
            _, tokens = self._kpipeline.g2p(text)
            ps: List[str] = []
            for _, p, _ in self._kpipeline.en_tokenize(tokens):
                ps.extend(p)
        else:
            ps, _ = self._kpipeline.g2p(text)
        ps = ps[:_MAX_INPUT_LENGTH]
        vocab = self._kpipeline.model.vocab
        ids = [i for i in (vocab.get(p) for p in ps) if i is not None]
        return ps, torch.IntTensor([[0, *ids, 0]])

    def _load_voice(self, phonemes: List[str]) -> Tensor:
        pack = self._kpipeline.load_voice(self._voice_path).to(self._device)
        return pack[len(phonemes) - 1]

    def _build_padded(self, input_ids: Tensor) -> Tuple[Tensor, Tensor]:
        mask = torch.zeros(1, _MAX_INPUT_LENGTH)
        mask[0, : input_ids.shape[1]] = 1
        pad = _MAX_INPUT_LENGTH - input_ids.shape[1]
        if pad > 0:
            input_ids = F.pad(input_ids, (0, pad))
        return input_ids, mask

    def generate(self, text: str, speed: int = 1) -> np.ndarray:
        """Synthesise ``text`` and return a mono float32 waveform at 24 kHz."""
        phonemes, raw_ids = self._text_to_input_ids(text)
        style = self._load_voice(phonemes)
        speed_t = torch.IntTensor([speed])
        input_ids, text_mask = self._build_padded(raw_ids)

        with torch.no_grad():
            # Stage 1: BERT
            d_en = self._bert_m(input_ids, text_mask)

            # Stage 2: Duration predictor
            pred_dur, d_enc, t_en_static = self._dur_m(d_en, style, text_mask, speed_t, input_ids)

            # Stage 3: Expand durations
            expanded_indices_np, T_acoustic = _expand_durations(pred_dur.cpu().numpy())
            expanded_indices = torch.from_numpy(expanded_indices_np).long()

            d_enc_exp = torch.index_select(d_enc, 2, expanded_indices)       # [1, h, T_acoustic]
            asr_full = torch.index_select(t_en_static, 2, expanded_indices)  # [1, 512, T_acoustic]

            # Stage 4: Acoustic expand
            en = self._acexp_m(d_enc_exp)   # [1, T_acoustic, h']

            # Stage 5: F0/N predictor
            F0_full, N_full = self._f0n_m(en, style)
            T_f0 = 2 * T_acoustic
            F0 = F0_full[:, :T_f0]
            N = N_full[:, :T_f0]
            asr = asr_full[:, :, :T_acoustic]

            # Stage 6: Vocos conditioner
            features = torch.from_numpy(
                _build_vocos_features(
                    _to_numpy(asr), _to_numpy(F0), _to_numpy(N), _to_numpy(style)
                )
            )
            conditioned = self._cond_m(features.float())   # [1, 192, T_f0]

            # Stage 7: Streaming Vocos (backbone + head)
            state = self._vocos.init_state(batch_size=1)
            chunks: List[Tensor] = []
            total = conditioned.shape[-1]
            pos = 0
            while pos < total:
                end = min(total, pos + self._chunk_frames)
                valid = end - pos
                chunk = conditioned[..., pos:end]
                if valid < self._chunk_frames:
                    chunk = F.pad(chunk, (0, self._chunk_frames - valid))
                self._vocos.model._start_streaming(1)
                try:
                    self._vocos.model.load_streaming_state_tensors(state)
                    x = self._vocos.model.backbone(chunk)
                    y = self._vocos.model.head(x)
                    state = {k: v.detach().clone()
                             for k, v in self._vocos.model.export_streaming_state_tensors().items()}
                finally:
                    self._vocos.model._stop_streaming()
                if y.ndim == 3 and y.shape[1] == 1:
                    y = y[:, 0, :]
                chunks.append(y[0, : valid * self._hop])
                pos = end

        audio = torch.cat(chunks).cpu().numpy()
        return audio


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_outputs(
    text: str,
    tflite_tts: KokoroTFLiteTTS,
    pt_tts: KokoroPTTTS,
    output_dir: str = ".",
    sample_rate: int = _SAMPLE_RATE,
) -> dict:
    """Run both pipelines and return a comparison summary dict.

    Also writes ``tflite_output.wav`` and ``pt_output.wav`` to *output_dir*.
    """
    import scipy.io.wavfile as wavfile

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"TFLite inference …")
    t0 = time.perf_counter()
    wav_tflite = tflite_tts.generate(text)
    t_tflite = time.perf_counter() - t0

    print(f"PyTorch inference …")
    t0 = time.perf_counter()
    wav_pt = pt_tts.generate(text)
    t_pt = time.perf_counter() - t0

    # Save
    for name, wav in [("tflite_output.wav", wav_tflite), ("pt_output.wav", wav_pt)]:
        path = out_dir / name
        wavfile.write(str(path), sample_rate, (wav * 32767).astype(np.int16))
        print(f"  Saved: {path}")

    # Numerical comparison
    L = min(len(wav_tflite), len(wav_pt))
    diff = np.abs(wav_tflite[:L] - wav_pt[:L])
    corr = float(np.corrcoef(wav_tflite[:L], wav_pt[:L])[0, 1])
    summary = {
        "text": text,
        "tflite_duration_s": round(t_tflite, 3),
        "pt_duration_s": round(t_pt, 3),
        "tflite_samples": len(wav_tflite),
        "pt_samples": len(wav_pt),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
        "pt_rms": float(np.sqrt(np.mean(wav_pt**2))),
        "tflite_rms": float(np.sqrt(np.mean(wav_tflite**2))),
        "correlation": corr,
    }
    print(
        f"\n{'─'*60}\n"
        f"  TFLite duration  : {t_tflite:.2f} s\n"
        f"  PT duration      : {t_pt:.2f} s\n"
        f"  TFLite audio len : {len(wav_tflite)} samples ({len(wav_tflite)/sample_rate:.2f} s)\n"
        f"  PT audio len     : {len(wav_pt)} samples ({len(wav_pt)/sample_rate:.2f} s)\n"
        f"  Max |TFLite−PT|  : {summary['max_abs_diff']:.6f}\n"
        f"  Mean |TFLite−PT| : {summary['mean_abs_diff']:.6f}\n"
        f"  RMS diff         : {summary['rms_diff']:.6f}\n"
        f"  Correlation      : {corr:.6f}\n"
        f"{'─'*60}"
    )

    # ── Waveform comparison plot ─────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t_axis = np.arange(L) / sample_rate
        zoom_end = min(L, sample_rate * 2)   # first 2 s for the zoom panel
        t_zoom = np.arange(zoom_end) / sample_rate

        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        fig.suptitle(
            f'Waveform comparison — TFLite vs PyTorch\n"{text[:80]}{"…" if len(text) > 80 else ""}"\n'
            f'Corr={corr:.4f}  RMS_diff={summary["rms_diff"]:.4f}  '
            f'Max_diff={summary["max_abs_diff"]:.4f}',
            fontsize=11,
        )

        # Full-length overlay
        ax = axes[0]
        ax.plot(t_axis, wav_pt[:L],     color="#2196F3", lw=0.6, alpha=0.85, label="PyTorch")
        ax.plot(t_axis, wav_tflite[:L], color="#FF5722", lw=0.6, alpha=0.75, label="TFLite")
        ax.set_title("Full waveform (overlay)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim(0, L / sample_rate)

        # First 2 s zoom overlay
        ax = axes[1]
        ax.plot(t_zoom, wav_pt[:zoom_end],     color="#2196F3", lw=0.8, alpha=0.85, label="PyTorch")
        ax.plot(t_zoom, wav_tflite[:zoom_end], color="#FF5722", lw=0.8, alpha=0.75, label="TFLite")
        ax.set_title("First 2 s (zoom overlay)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=9)

        # Difference signal
        ax = axes[2]
        ax.plot(t_axis, wav_tflite[:L] - wav_pt[:L], color="#9C27B0", lw=0.5, alpha=0.9)
        ax.axhline(0, color="black", lw=0.4, ls="--")
        ax.set_title("Difference (TFLite − PyTorch)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, L / sample_rate)

        # Spectrogram of difference
        ax = axes[3]
        ax.specgram(
            wav_tflite[:L] - wav_pt[:L],
            Fs=sample_rate,
            NFFT=512,
            noverlap=256,
            cmap="viridis",
        )
        ax.set_title("Spectrogram of difference")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        fig.tight_layout()
        png_path = out_dir / "waveform_comparison.png"
        fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {png_path}")
        summary["waveform_png"] = str(png_path)
    except Exception as _plot_err:
        print(f"  [waveform plot skipped: {_plot_err}]")

    return summary


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Kokoro TFLite TTS — inference and PT comparison")
    parser.add_argument("text", nargs="?",
                        default=(
                            "I had returned to civil practice and had finally abandoned Holmes in his "
                            "Baker Street rooms, although I continually visited him."
                        ),
                        help="Text to synthesise")
    parser.add_argument("--saved-model-dir", default="onnx2tf_conversion/saved_model")
    parser.add_argument("--vocos-ckpt", default="vocos_fp16.pt")
    parser.add_argument("--config", default="checkpoints/config.json")
    parser.add_argument("--kokoro-ckpt", default="checkpoints/kokoro-v1_0.pth")
    parser.add_argument("--voice", default="checkpoints/voices/af_bella.pt")
    parser.add_argument("--output-dir", default="tflite_outputs")
    parser.add_argument("--speed", type=int, default=1)
    args = parser.parse_args()

    print("Loading TFLite TTS …")
    tflite_tts = KokoroTFLiteTTS(
        saved_model_dir=args.saved_model_dir,
        config_path=args.config,
        kokoro_checkpoint=args.kokoro_ckpt,
        voice_path=args.voice,
    )

    print("Loading PyTorch TTS …")
    pt_tts = KokoroPTTTS(
        config_path=args.config,
        kokoro_checkpoint=args.kokoro_ckpt,
        vocos_checkpoint=args.vocos_ckpt,
        voice_path=args.voice,
    )

    compare_outputs(args.text, tflite_tts, pt_tts, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
