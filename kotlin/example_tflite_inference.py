#!/usr/bin/env python3
"""example_tflite_inference.py — Kokoro TTS streaming inference using LiteRT (TFLite)

Demonstrates full text-to-speech synthesis using the 6-stage Kokoro pipeline
with LiteRT (ai_edge_litert), producing audio via streaming chunked vocoder inference.

Pipeline
--------
  Text → [G2P] → (input_ids, style)
    → [S1 BERT]                 input_ids, text_mask → d_en [1,510,512] NTC
    → [S2 Duration Predictor]   d_en, style, text_mask, speed, input_ids
                                 → pred_dur, d_enc [1,640,510] NCT, t_en_static [1,512,510] NCT
    → [S3 Expand (Python)]      index_select → d_enc_exp, asr
    → [S4 Acoustic Expand]      d_enc_expanded [1,543,640] NTC → en [1,543,512] NTC
    → [S5 F0/N Predictor]       en [1,512,543] NCT, style → F0_pred, N_pred [1,1086]
    → [S6 Conditioner]          features [1,1086,642] NTC → conditioned [1,192,1086] NCT
    → [S7 Vocos stream loop]    16-frame chunks → audio [1, 4800] (IDFT + OLA inside model)

Tensor layout note: onnx2tf converts ONNX NCT tensors to TFLite NTC where it decides
to do so. The conditioner input is NTC in TFLite (unlike ONNX which is NCT). The vocos
chunk state outputs Identity_3..10 are NCT [1,384,6] and must be transposed to NTC
[1,6,384] before feeding back as inputs.

Models loaded from: export_models/tflite/
Precision is selected by the --precision flag.

Usage
-----
    uv run python kotlin/example_tflite_inference.py
    uv run python kotlin/example_tflite_inference.py --text "Hello world" --out hello.wav
    uv run python kotlin/example_tflite_inference.py --precision float16
    uv run python kotlin/example_tflite_inference.py --precision int8
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import scipy.io.wavfile as wavfile

import ai_edge_litert.interpreter as litert

from kokoro import KModel, KPipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = Path("export_models/tflite")
VOICE_PATH = "checkpoints/voices/af_bella.pt"
CONFIG_FILE = "checkpoints/config.json"
CHECKPOINT_PATH = "checkpoints/kokoro-v1_0.pth"

SAMPLE_RATE = 24_000
MAX_INPUT_LENGTH = 510
T_ACOUSTIC = 543
T_F0 = 1086
CHUNK_FRAMES = 16
HOP = 300
TAIL = 900              # overlap-add tail (win_len - hop = 1200 - 300)
KERN = 6
N_BLOCKS = 8

# Model filename patterns per precision
PRECISION_SUFFIX: Dict[str, Dict[str, str]] = {
    # precision → {model_name: filename_suffix}
    "float32": {m: "_float32" for m in [
        "bert", "duration_predictor", "acoustic_expand",
        "f0n_predictor", "vocoder_conditioner", "vocoder_stream_chunk",
    ]},
    "float16": {m: "_float16" for m in [
        "bert", "duration_predictor", "acoustic_expand",
        "f0n_predictor", "vocoder_conditioner", "vocoder_stream_chunk",
    ]},
    # Mixed: INT8 where available, float32 for bert and f0n_predictor.
    # Note: on Android with GPU delegate, use _float16 for bert and f0n_predictor instead.
    # float16 TFLite models require GPU/DSP delegate; they fail on the CPU backend.
    "int8": {
        "bert":                  "_float32",
        "duration_predictor":    "_full_int8",
        "acoustic_expand":       "_full_int8",
        "f0n_predictor":         "_float32",
        "vocoder_conditioner":   "_full_int8",
        "vocoder_stream_chunk":  "_full_int8",
    },
}


# ---------------------------------------------------------------------------
# LiteRT interpreter factory
# ---------------------------------------------------------------------------

def _make_interpreter(path: Path, num_threads: int = 4) -> litert.Interpreter:
    interp = litert.Interpreter(model_path=str(path), num_threads=num_threads)
    interp.allocate_tensors()
    return interp


def _norm_name(raw: str) -> str:
    """Strip onnx2tf tensor name decorations: 'serving_default_X:0' → 'X'."""
    n = raw
    if n.startswith("serving_default_"):
        n = n[len("serving_default_"):]
    colon = n.rfind(":")
    if colon != -1:
        n = n[:colon]
    return n


# ---------------------------------------------------------------------------
# Text → tokens
# ---------------------------------------------------------------------------

def _text_to_inputs(
    kpipeline: KPipeline,
    text: str,
    voice_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert text to (input_ids [1,510] int32, text_mask [1,510] float32, style [1,256])."""
    if kpipeline.lang_code in "ab":
        _, tokens = kpipeline.g2p(text)
        ps: List[str] = []
        for _, p, _ in kpipeline.en_tokenize(tokens):
            ps.extend(p)
    else:
        ps, _ = kpipeline.g2p(text)
    ps = ps[: MAX_INPUT_LENGTH - 2]

    vocab = kpipeline.model.vocab
    ids = [i for i in (vocab.get(p) for p in ps) if i is not None]
    raw_ids = torch.IntTensor([[0, *ids, 0]])

    pack = kpipeline.load_voice(voice_path).cpu()
    style = pack[len(ps) - 1].detach().numpy().astype(np.float32)  # [1, 256]

    seqlen = raw_ids.shape[1]
    text_mask = np.zeros((1, MAX_INPUT_LENGTH), dtype=np.float32)
    text_mask[0, :seqlen] = 1.0
    pad = MAX_INPUT_LENGTH - seqlen
    if pad > 0:
        raw_ids = F.pad(raw_ids, (0, pad))
    return raw_ids.numpy().astype(np.int32), text_mask, style


# ---------------------------------------------------------------------------
# Duration expansion
# ---------------------------------------------------------------------------

def _expand_durations(pred_dur: np.ndarray) -> Tuple[np.ndarray, int]:
    dur = torch.from_numpy(pred_dur.flatten().astype(np.float32))
    boundaries = torch.cumsum(dur, dim=0)
    T_acoustic = min(int(boundaries[-1].item()), T_ACOUSTIC)
    if T_acoustic == 0:
        return np.zeros(0, dtype=np.int64), 0
    values = torch.arange(T_acoustic, dtype=torch.int32)
    idx = torch.sum(boundaries.unsqueeze(1) <= values.unsqueeze(0), dim=0)
    return idx.numpy().astype(np.int64), T_acoustic


# ---------------------------------------------------------------------------
# Vocos feature assembly
# ---------------------------------------------------------------------------

def _build_vocos_features(
    asr: np.ndarray,      # [1, 512, T_acoustic] NCT
    f0_pred: np.ndarray,  # [1, T_f0_actual]
    n_pred: np.ndarray,   # [1, T_f0_actual]
    style: np.ndarray,    # [1, 256]
) -> np.ndarray:
    """Assemble NTC [1, T_f0_actual, 642] conditioner input for TFLite (onnx2tf layout)."""
    T_f0 = f0_pred.shape[-1]  # derive from actual prediction length
    asr_t = torch.from_numpy(asr.astype(np.float32))
    if asr_t.shape[-1] != T_f0:
        asr_t = F.interpolate(asr_t, size=T_f0, mode="linear", align_corners=False)
    f0 = torch.from_numpy(f0_pred.astype(np.float32)).unsqueeze(1)
    n  = torch.from_numpy(n_pred.astype(np.float32)).unsqueeze(1)
    s  = torch.from_numpy(style.astype(np.float32))[:, :128].unsqueeze(-1).expand(-1, -1, T_f0)
    nct = torch.cat([asr_t, f0, n, s], dim=1).numpy()   # [1, 642, T_f0] NCT
    return np.transpose(nct, (0, 2, 1))                  # → [1, T_f0, 642] NTC


# ---------------------------------------------------------------------------
# Stage inference functions
# ---------------------------------------------------------------------------

def _run_bert(
    interp: litert.Interpreter,
    input_ids: np.ndarray,
    text_mask: np.ndarray,
) -> np.ndarray:
    """BERT: input_ids [1,510] int32 + text_mask [1,510] → d_en [1,510,512] NTC."""
    ins = {_norm_name(d["name"]): d["index"] for d in interp.get_input_details()}
    interp.set_tensor(ins["input_ids"], input_ids.astype(np.int32))
    interp.set_tensor(ins["text_mask"], text_mask.astype(np.float32))
    interp.invoke()
    return interp.get_tensor(interp.get_output_details()[0]["index"])


def _run_duration(
    interp: litert.Interpreter,
    d_en: np.ndarray,
    style: np.ndarray,
    text_mask: np.ndarray,
    speed: float,
    input_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duration predictor → (pred_dur [510], d_enc [1,640,510] NCT, t_en [1,512,510] NCT)."""
    ins_detail = interp.get_input_details()
    ins = {_norm_name(d["name"]): d for d in ins_detail}
    dtype_map = {_norm_name(d["name"]): d["dtype"] for d in ins_detail}

    interp.set_tensor(ins["d_en"]["index"],       d_en.astype(np.float32))
    interp.set_tensor(ins["style"]["index"],      style.astype(np.float32))
    interp.set_tensor(ins["text_mask"]["index"],  text_mask.astype(np.float32))
    # speed is float32 in the re-exported duration predictor
    interp.set_tensor(ins["speed"]["index"], np.array([speed], dtype=dtype_map.get("speed", np.float32)))
    # input_ids is int64 in the re-exported duration predictor
    interp.set_tensor(ins["input_ids"]["index"], input_ids.astype(dtype_map.get("input_ids", np.int64)))
    interp.invoke()

    outs = interp.get_output_details()
    # Outputs may be named or indexed; map by shape
    by_shape: Dict[tuple, np.ndarray] = {
        tuple(d["shape"]): interp.get_tensor(d["index"]) for d in outs
    }
    pred_dur    = by_shape.get((510,), by_shape.get((1, 510)))
    if pred_dur is None:
        pred_dur = next(
            interp.get_tensor(d["index"]).flatten()
            for d in outs if interp.get_tensor(d["index"]).size == 510
        )
    pred_dur = pred_dur.flatten()
    d_enc = by_shape[(1, 640, 510)]
    t_en  = by_shape[(1, 512, 510)]
    return pred_dur, d_enc, t_en


def _run_acoustic_expand(
    interp: litert.Interpreter,
    d_enc_exp_ntc: np.ndarray,   # [1, T_ACOUSTIC, 640] NTC
) -> np.ndarray:
    """Acoustic expand: d_enc_expanded NTC → en [1, T_ACOUSTIC, 512] NTC."""
    interp.set_tensor(interp.get_input_details()[0]["index"], d_enc_exp_ntc.astype(np.float32))
    interp.invoke()
    return interp.get_tensor(interp.get_output_details()[0]["index"])


def _run_f0n(
    interp: litert.Interpreter,
    en_nct: np.ndarray,   # [1, 512, T_ACOUSTIC] NCT
    style: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """F0/N predictor: en NCT, style → F0_pred [1,T_F0], N_pred [1,T_F0]."""
    ins = {d["name"]: d["index"] for d in interp.get_input_details()}
    interp.set_tensor(ins["en"],    en_nct.astype(np.float32))
    interp.set_tensor(ins["style"], style.astype(np.float32))
    interp.invoke()
    outs = {d["name"]: interp.get_tensor(d["index"]) for d in interp.get_output_details()}
    return outs["F0_pred"], outs["N_pred"]


def _run_conditioner(
    interp: litert.Interpreter,
    features_ntc: np.ndarray,   # [1, T_F0, 642] NTC
) -> np.ndarray:
    """Conditioner: features NTC → conditioned [1, 192, T_F0] NCT (normalised)."""
    interp.set_tensor(interp.get_input_details()[0]["index"], features_ntc.astype(np.float32))
    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]["index"])
    # onnx2tf may output NCT [1, 192, T_F0] (float32) or NTC [1, T_F0, 192] (int8).
    # Normalise to NCT so callers always receive [1, 192, T_F0].
    if out.ndim == 3 and out.shape[1] != 192:
        out = np.transpose(out, (0, 2, 1))
    return out


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def generate(
    text: str,
    interps: Dict[str, litert.Interpreter],
    kpipeline: KPipeline,
    voice_path: str = VOICE_PATH,
    speed: float = 1.0,
) -> np.ndarray:
    """Synthesise text to a mono float32 waveform at 24 kHz."""
    # ── Text → tokens ──────────────────────────────────────────────────────
    input_ids, text_mask, style = _text_to_inputs(kpipeline, text, voice_path)

    # ── S1: BERT ──────────────────────────────────────────────────────────
    d_en = _run_bert(interps["bert"], input_ids, text_mask)  # [1,510,512] NTC

    # ── S2: Duration predictor ────────────────────────────────────────────
    pred_dur, d_enc, t_en_static = _run_duration(
        interps["duration_predictor"], d_en, style, text_mask, speed, input_ids
    )

    # ── S3: Duration expansion ────────────────────────────────────────────
    expanded_idx, T_acoustic = _expand_durations(pred_dur)
    if T_acoustic == 0:
        return np.zeros(0, dtype=np.float32)
    if T_acoustic > T_ACOUSTIC:
        warnings.warn(f"T_acoustic={T_acoustic} > {T_ACOUSTIC}, truncating.")
        T_acoustic = T_ACOUSTIC
        expanded_idx = expanded_idx[:T_acoustic]
    T_f0_actual = 2 * T_acoustic

    d_enc_exp = np.take(d_enc, expanded_idx, axis=2)       # [1, 640, T_acoustic]
    asr = np.take(t_en_static, expanded_idx, axis=2)       # [1, 512, T_acoustic]

    # ── S4: Acoustic expand ────────────────────────────────────────────────
    d_enc_exp_ntc = np.transpose(d_enc_exp, (0, 2, 1))     # [1, T_acoustic, 640]
    if T_acoustic < T_ACOUSTIC:
        d_enc_exp_ntc = np.pad(
            d_enc_exp_ntc, ((0, 0), (0, T_ACOUSTIC - T_acoustic), (0, 0))
        )
    en_ntc = _run_acoustic_expand(interps["acoustic_expand"], d_enc_exp_ntc)

    # Transpose to NCT — do NOT trim to T_acoustic
    en_nct = np.transpose(en_ntc, (0, 2, 1))               # [1, 512, T_ACOUSTIC]

    # ── S5: F0/N predictor ────────────────────────────────────────────────
    f0_full, n_full = _run_f0n(interps["f0n_predictor"], en_nct, style)
    f0_pred = f0_full[:, :T_f0_actual]
    n_pred  = n_full[:, :T_f0_actual]

    # ── S6: Conditioner ────────────────────────────────────────────────────
    # TFLite conditioner expects NTC [1, T_F0, 642] (onnx2tf transposed)
    features_ntc = _build_vocos_features(asr, f0_pred, n_pred, style)  # [1, T_f0, 642]
    if T_f0_actual < T_F0:
        features_ntc = np.pad(features_ntc, ((0, 0), (0, T_F0 - T_f0_actual), (0, 0)))

    conditioned_full = _run_conditioner(interps["vocoder_conditioner"], features_ntc)
    conditioned = conditioned_full[:, :, :T_f0_actual]      # [1, 192, T_f0_actual] NCT

    # ── S7: Vocos streaming chunk loop ────────────────────────────────────
    # TFLite model: VocosStreamChunkReal — real-matmul IDFT + pad-sum OLA fully inside.
    # IN  [0]: conditioned_chunk  [1, 16, 192]  NTC
    # IN  [1+]: state tensors (embed, 8 blocks, istft_prev) — positional
    # OUT [0]: audio              [1, 4800]
    # OUT [1+]: updated state tensors — positional, some may need NCT→NTC transpose
    vocos = interps["vocoder_stream_chunk"]
    in_d  = vocos.get_input_details()
    out_d = vocos.get_output_details()

    # Auto-detect chunk input layout (NCT vs NTC) from first input tensor shape
    in0_shape = in_d[0]["shape"]
    chunk_is_ntc = (len(in0_shape) == 3 and in0_shape[-1] == 192)

    # Initialise state: all inputs after conditioned_chunk (embed + 8 blocks + istft_prev)
    state = [np.zeros(d["shape"], dtype=np.float32) for d in in_d[1:]]

    audio_chunks: List[np.ndarray] = []
    pos = 0

    while pos < T_f0_actual:
        end   = min(T_f0_actual, pos + CHUNK_FRAMES)
        valid = end - pos

        chunk = conditioned[:, :, pos:end]   # [1, 192, valid] NCT
        if valid < CHUNK_FRAMES:
            chunk = np.pad(chunk, ((0, 0), (0, 0), (0, CHUNK_FRAMES - valid)))
        if chunk_is_ntc:
            chunk = np.transpose(chunk, (0, 2, 1))   # NCT → NTC

        vocos.set_tensor(in_d[0]["index"], chunk.astype(np.float32))
        for i, s in enumerate(state):
            vocos.set_tensor(in_d[1 + i]["index"], s)
        vocos.invoke()

        # Output 0 is audio [1, CHUNK_FRAMES * HOP]
        audio_chunk = vocos.get_tensor(out_d[0]["index"])
        audio_chunks.append(audio_chunk.flatten()[: valid * HOP])

        # Update state: outputs 1+ map positionally to inputs 1+
        new_state: List[np.ndarray] = []
        for i, s_in in enumerate(in_d[1:]):
            s_out = vocos.get_tensor(out_d[1 + i]["index"])
            if s_out.ndim == 3 and tuple(s_out.shape) != tuple(s_in["shape"]):
                s_out = np.transpose(s_out, (0, 2, 1))
            new_state.append(s_out)
        state = new_state
        pos = end

    return np.concatenate(audio_chunks).astype(np.float32)


# ---------------------------------------------------------------------------
# Interpreter loader
# ---------------------------------------------------------------------------

def load_interpreters(
    model_dir: Path,
    precision: str,
    num_threads: int = 4,
) -> Dict[str, litert.Interpreter]:
    """Load all 6 LiteRT interpreters for the given precision."""
    suffix_map = PRECISION_SUFFIX[precision]
    interps: Dict[str, litert.Interpreter] = {}
    for name, suffix in suffix_map.items():
        path = model_dir / f"{name}{suffix}.tflite"
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                f"Run 'python export_tflite.py' to generate export_models/tflite/ first."
            )
        print(f"  loading {path.name:55s}  {path.stat().st_size/1e6:.1f} MB")
        interps[name] = _make_interpreter(path, num_threads)
    return interps


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kokoro TTS LiteRT inference example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--text",
        default=(
            "Hello world. This is Kokoro TTS running on LiteRT "
            "with streaming chunked vocoder inference."
        ),
        help="Text to synthesise.",
    )
    parser.add_argument(
        "--out",
        default="output_tflite.wav",
        help="Output WAV file path.",
    )
    parser.add_argument(
        "--precision",
        default="int8",
        choices=["float32", "float16", "int8"],
        help=(
            "Model precision: float32, float16, or int8 "
            "(int8 = mixed: float16 for bert/f0n, full_int8 for others)."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        dest="model_dir",
        help="Directory with exported TFLite models.",
    )
    parser.add_argument(
        "--voice",
        default=VOICE_PATH,
        help="Path to voice .pt file.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier (1.0 = normal).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of TFLite interpreter threads.",
    )
    args = parser.parse_args()

    print("Loading Kokoro pipeline (G2P)...")
    kmodel = KModel(
        config=CONFIG_FILE,
        model=CHECKPOINT_PATH,
        disable_complex=True,
    ).cpu().eval()
    kpipeline = KPipeline(lang_code="a", model=kmodel, device="cpu")

    prec_label = args.precision
    if args.precision == "int8":
        prec_label = "mixed INT8 (float16 for bert/f0n, full_int8 for others)"
    print(f"Loading {prec_label} TFLite models from {args.model_dir}/")
    interps = load_interpreters(args.model_dir, args.precision, args.threads)

    print(f"\nSynthesising: \"{args.text[:70]}\"")
    t0 = time.perf_counter()
    audio = generate(
        text=args.text,
        interps=interps,
        kpipeline=kpipeline,
        voice_path=args.voice,
        speed=args.speed,
    )
    elapsed = time.perf_counter() - t0

    duration = len(audio) / SAMPLE_RATE
    rtf = elapsed / duration if duration > 0 else float("inf")
    print(f"Generated {duration:.2f}s audio in {elapsed:.2f}s  (RTF={rtf:.2f})")

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    wavfile.write(args.out, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
