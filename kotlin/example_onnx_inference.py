#!/usr/bin/env python3
"""example_onnx_inference.py — Kokoro TTS streaming inference using ONNX Runtime

Demonstrates full text-to-speech synthesis using the 6-stage Kokoro pipeline
with ONNX Runtime, producing audio via streaming chunked vocoder inference.

Pipeline
--------
  Text → [G2P] → (input_ids, style)
    → [S1 BERT]                 input_ids, text_mask → d_en [1,510,512] NTC
    → [S2 Duration Predictor]   d_en, style, text_mask, speed, input_ids
                                 → pred_dur, d_enc [1,640,510] NCT, t_en_static [1,512,510] NCT
    → [S3 Expand (Python)]      index_select → d_enc_exp, asr
    → [S4 Acoustic Expand]      d_enc_expanded [1,543,640] NTC → en [1,543,512] NTC
    → [S5 F0/N Predictor]       en [1,512,543] NCT, style → F0_pred, N_pred [1,1086]
    → [S6 Conditioner]          features [1,642,1086] NCT → conditioned [1,192,1086] NCT
    → [S7 Vocos stream loop]    16-frame chunks → audio [1, 4800] (IDFT + OLA inside model)

Models loaded from: export_models/onnx/
All models use float32 inputs; load *_fp16.onnx or *_int8_static.onnx variants
by changing MODEL_PRECISION below.

Usage
-----
    uv run python kotlin/example_onnx_inference.py
    uv run python kotlin/example_onnx_inference.py --text "Hello world" --out hello.wav
    uv run python kotlin/example_onnx_inference.py --precision fp16
    uv run python kotlin/example_onnx_inference.py --precision int8
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

import onnxruntime as ort

from kokoro import KModel, KPipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = Path("export_models/onnx")
VOICE_PATH = "checkpoints/voices/af_bella.pt"
CONFIG_FILE = "checkpoints/config.json"
CHECKPOINT_PATH = "checkpoints/kokoro-v1_0.pth"

SAMPLE_RATE = 24_000
MAX_INPUT_LENGTH = 510
T_ACOUSTIC = 543        # static BiLSTM time dimension
T_F0 = 1086             # 2 × T_ACOUSTIC — conditioner time dimension
CHUNK_FRAMES = 16       # vocoder streaming chunk size
HOP = 300               # samples per vocoder frame
TAIL = 900              # overlap-add tail (win_len - hop = 1200 - 300)
KERN = 6                # causal conv context depth (kernel_size - 1)
N_BLOCKS = 8            # Vocos ConvNeXt blocks

# Model name → suffix map per precision.
# For 'int8' the map is per-model to keep fragile models in fp32:
#   bert: ALBERT attention MatMuls degrade too much even with Softmax/LN excluded
#   vocoder_conditioner: tiny (1.1 MB), catastrophic INT8 quality
#   vocoder_stream_chunk: streaming state accumulates INT8 errors over chunks → use fp16
#                         (fp16 ONNX ≡ fp32 quality on CPU; 2× smaller on-disk)
PRECISION_SUFFIX: Dict[str, Dict[str, str]] = {
    "fp32": {m: "_fp32"         for m in [
        "bert", "duration_predictor", "acoustic_expand",
        "f0n_predictor", "vocoder_conditioner", "vocoder_stream_chunk",
    ]},
    "fp16": {m: "_fp16"         for m in [
        "bert", "duration_predictor", "acoustic_expand",
        "f0n_predictor", "vocoder_conditioner", "vocoder_stream_chunk",
    ]},
    # int8: all models use static INT8 (for compatibility / testing all INT8 models)
    "int8": {m: "_int8_static"  for m in [
        "bert", "duration_predictor", "acoustic_expand",
        "f0n_predictor", "vocoder_conditioner", "vocoder_stream_chunk",
    ]},
    # int8_mixed: quality-preserving config (~114 MB vs 151 MB fp32, 25% reduction)
    # Keeps models with poor INT8 quality in fp32/fp16.
    "int8_mixed": {
        "bert":                 "_fp32",           # poor INT8 quality (0.17 cosine_sim)
        "duration_predictor":   "_int8_static",    # improved: LSTM excluded; 0.9992 cosine_sim
        "acoustic_expand":      "_int8_static",    # lossless INT8 (1.000 cosine_sim)
        "f0n_predictor":        "_int8_static",    # good quality (0.76 pearson)
        "vocoder_conditioner":  "_fp32",           # tiny (1.1 MB); catastrophic INT8
        "vocoder_stream_chunk": "_fp16",           # fp16 = fp32 quality on CPU; avoids state accumulation
    },
}


# ---------------------------------------------------------------------------
# ORT session factory
# ---------------------------------------------------------------------------

def _make_session(path: Path, use_nnapi: bool = False) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    opts.log_severity_level = 3  # errors only
    providers = ["CPUExecutionProvider"]
    # On Android, replace CPUExecutionProvider with NNAPIExecutionProvider:
    #   providers = ["NNAPIExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


_ORT_FLOAT_TYPES = {"tensor(float16)": np.float16, "tensor(float)": np.float32}


def _run(session: ort.InferenceSession, feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """Run session, auto-casting float inputs to the session's declared dtypes."""
    casted = {}
    for inp in session.get_inputs():
        if inp.name in feed:
            arr = feed[inp.name]
            target = _ORT_FLOAT_TYPES.get(inp.type)
            if target is not None and arr.dtype != target:
                arr = arr.astype(target)
            casted[inp.name] = arr
    return session.run(None, casted)


# ---------------------------------------------------------------------------
# Text → tokens
# ---------------------------------------------------------------------------

def _text_to_inputs(
    kpipeline: KPipeline,
    text: str,
    voice_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert text to padded (input_ids, text_mask, style).

    Returns
    -------
    input_ids : int32 [1, MAX_INPUT_LENGTH]
    text_mask : float32 [1, MAX_INPUT_LENGTH]
    style     : float32 [1, 256]
    """
    # G2P
    if kpipeline.lang_code in "ab":
        _, tokens = kpipeline.g2p(text)
        ps: List[str] = []
        for _, p, _ in kpipeline.en_tokenize(tokens):
            ps.extend(p)
    else:
        ps, _ = kpipeline.g2p(text)
    ps = ps[: MAX_INPUT_LENGTH - 2]  # leave room for BOS/EOS

    vocab = kpipeline.model.vocab
    ids = [i for i in (vocab.get(p) for p in ps) if i is not None]
    raw_ids = torch.IntTensor([[0, *ids, 0]])  # BOS + phoneme IDs + EOS

    # Style vector from voice pack (row = num_phonemes - 1)
    pack = kpipeline.load_voice(voice_path).cpu()
    style = pack[len(ps) - 1].detach().numpy().astype(np.float32)  # [1, 256]

    # Pad to MAX_INPUT_LENGTH
    seqlen = raw_ids.shape[1]
    text_mask = np.zeros((1, MAX_INPUT_LENGTH), dtype=np.float32)
    text_mask[0, :seqlen] = 1.0
    pad = MAX_INPUT_LENGTH - seqlen
    if pad > 0:
        raw_ids = F.pad(raw_ids, (0, pad))
    input_ids = raw_ids.numpy().astype(np.int32)

    return input_ids, text_mask, style


# ---------------------------------------------------------------------------
# Duration expansion
# ---------------------------------------------------------------------------

def _expand_durations(pred_dur: np.ndarray) -> Tuple[np.ndarray, int]:
    """Compute acoustic-frame phoneme indices from per-phoneme durations.

    Returns (expanded_indices [T_acoustic], T_acoustic).
    """
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
    asr: np.ndarray,       # [1, 512, T_acoustic] NCT
    f0_pred: np.ndarray,   # [1, T_f0_actual]
    n_pred: np.ndarray,    # [1, T_f0_actual]
    style: np.ndarray,     # [1, 256]
) -> np.ndarray:
    """Assemble conditioner input [1, 642, T_f0_actual] NCT."""
    T_f0 = f0_pred.shape[-1]  # derive from actual prediction length
    asr_t = torch.from_numpy(asr.astype(np.float32))    # [1, 512, T_acoustic]
    if asr_t.shape[-1] != T_f0:
        asr_t = F.interpolate(asr_t, size=T_f0, mode="linear", align_corners=False)
    f0 = torch.from_numpy(f0_pred.astype(np.float32)).unsqueeze(1)   # [1,1,T_f0]
    n  = torch.from_numpy(n_pred.astype(np.float32)).unsqueeze(1)    # [1,1,T_f0]
    s  = torch.from_numpy(style.astype(np.float32))[:, :128].unsqueeze(-1).expand(-1, -1, T_f0)
    return torch.cat([asr_t, f0, n, s], dim=1).numpy()  # [1, 642, T_f0]


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def generate(
    text: str,
    sessions: Dict[str, ort.InferenceSession],
    kpipeline: KPipeline,
    voice_path: str = VOICE_PATH,
    speed: int = 1,
) -> np.ndarray:
    """Synthesise text to a mono float32 waveform at 24 kHz.

    Parameters
    ----------
    text:
        Input English text.
    sessions:
        Dict of model_name → OrtInferenceSession.
    kpipeline:
        KPipeline for G2P / tokenisation.
    voice_path:
        Path to a voices/*.pt style vector file.
    speed:
        Speed multiplier (1 = normal).

    Returns
    -------
    audio : np.ndarray, shape (N,), dtype float32
    """
    # ── Text → tokens ──────────────────────────────────────────────────────
    input_ids, text_mask, style = _text_to_inputs(kpipeline, text, voice_path)

    # ── S1: BERT encoder ──────────────────────────────────────────────────
    # IN:  input_ids [1,510] int32,  text_mask [1,510] float32
    # OUT: d_en      [1,512,510] NCT float32
    d_en = _run(sessions["bert"], {
        "input_ids": input_ids,
        "text_mask": text_mask,
    })[0]   # [1, 512, 510] NCT

    # ── S2: Duration predictor ────────────────────────────────────────────
    # IN:  d_en [1,512,510] NCT, style [1,256], text_mask [1,510], speed [1], input_ids [1,510]
    # OUT: pred_dur [510], d_enc [1,640,510] NCT, t_en_static [1,512,510] NCT
    dur_outs = _run(sessions["duration_predictor"], {
        "d_en":       d_en,
        "style":      style,
        "text_mask":  text_mask,
        "speed":      np.array([speed], dtype=np.int32),
        "input_ids":  input_ids,
    })

    # Map outputs by shape
    by_shape: Dict[tuple, np.ndarray] = {tuple(a.shape): a for a in dur_outs}
    pred_dur = by_shape.get((510,))
    if pred_dur is None:
        pred_dur = next(a.flatten() for a in dur_outs if a.size == 510)
    else:
        pred_dur = pred_dur.flatten()
    d_enc = by_shape[(1, 640, 510)]         # [1, 640, 510] NCT
    t_en_static = by_shape[(1, 512, 510)]   # [1, 512, 510] NCT

    # ── S3: Duration expansion (Python — no model) ────────────────────────
    expanded_idx, T_acoustic = _expand_durations(pred_dur)
    if T_acoustic == 0:
        return np.zeros(0, dtype=np.float32)
    if T_acoustic > T_ACOUSTIC:
        warnings.warn(f"T_acoustic={T_acoustic} > {T_ACOUSTIC}, truncating.")
        T_acoustic = T_ACOUSTIC
        expanded_idx = expanded_idx[:T_acoustic]
    T_f0_actual = 2 * T_acoustic

    # Index-select text dim (axis 2 of NCT) to expand phonemes → acoustic frames
    d_enc_exp = np.take(d_enc, expanded_idx, axis=2)     # [1, 640, T_acoustic]
    asr = np.take(t_en_static, expanded_idx, axis=2)     # [1, 512, T_acoustic]

    # ── S4: Acoustic expand ────────────────────────────────────────────────
    # IN:  d_enc_expanded [1, 640, T_ACOUSTIC] NCT  (zero-pad in axis=2)
    # OUT: en             [1, T_ACOUSTIC, 512] NTC
    if T_acoustic < T_ACOUSTIC:
        d_enc_exp_padded = np.pad(
            d_enc_exp, ((0, 0), (0, 0), (0, T_ACOUSTIC - T_acoustic))
        )
    else:
        d_enc_exp_padded = d_enc_exp

    en_ntc = _run(sessions["acoustic_expand"], {
        "d_enc_expanded": d_enc_exp_padded.astype(np.float32),
    })[0]   # [1, T_ACOUSTIC, 512] NTC

    # ── S5: F0/N predictor ────────────────────────────────────────────────
    # IN:  en [1, T_ACOUSTIC, 512] NTC,  style [1, 256]
    # OUT: F0_pred [1, T_F0],  N_pred [1, T_F0]
    # Note: feed en_ntc directly — f0n expects NTC (do NOT transpose)
    f0n_outs = _run(sessions["f0n_predictor"], {
        "en":    en_ntc.astype(np.float32),
        "style": style,
    })
    f0_full = f0n_outs[0]   # [1, T_F0]
    n_full  = f0n_outs[1]   # [1, T_F0]
    f0_pred = f0_full[:, :T_f0_actual]
    n_pred  = n_full[:, :T_f0_actual]

    # ── S6: Vocos conditioner ─────────────────────────────────────────────
    # IN:  features [1, 642, T_F0] NCT
    # OUT: conditioned [1, 192, T_F0] NCT
    features_nct = _build_vocos_features(asr, f0_pred, n_pred, style)  # [1, 642, T_f0_actual]
    # Pad T dimension to T_F0 static size
    if T_f0_actual < T_F0:
        features_nct = np.pad(features_nct, ((0, 0), (0, 0), (0, T_F0 - T_f0_actual)))

    conditioned_full = _run(sessions["vocoder_conditioner"], {
        "features": features_nct.astype(np.float32),
    })[0]   # [1, 192, T_F0] NCT

    # Trim to actual length
    conditioned = conditioned_full[:, :, :T_f0_actual]   # [1, 192, T_f0_actual]

    # ── S7: Vocos streaming chunk loop ────────────────────────────────────
    # Each chunk: conditioned_chunk [1, 192, 16] NCT + 10 state tensors
    # Outputs: audio [1, CHUNK_FRAMES*HOP], 10 updated states (embed + 8 blocks + istft)
    embed_prev  = np.zeros((1, 192, KERN), dtype=np.float32)
    block_prevs = [np.zeros((1, 384, KERN), dtype=np.float32) for _ in range(N_BLOCKS)]
    istft_prev  = np.zeros((1, TAIL), dtype=np.float32)

    audio_chunks: List[np.ndarray] = []
    pos = 0

    while pos < T_f0_actual:
        end   = min(T_f0_actual, pos + CHUNK_FRAMES)
        valid = end - pos

        # Extract chunk; zero-pad last chunk if shorter than CHUNK_FRAMES
        chunk = conditioned[:, :, pos:end]   # [1, 192, valid] NCT
        if valid < CHUNK_FRAMES:
            chunk = np.pad(chunk, ((0, 0), (0, 0), (0, CHUNK_FRAMES - valid)))

        feed: Dict[str, np.ndarray] = {
            "conditioned_chunk": chunk.astype(np.float32),
            "embed_prev":        embed_prev,
        }
        for b in range(N_BLOCKS):
            feed[f"block_{b}_prev"] = block_prevs[b]
        feed["istft_prev"] = istft_prev

        vocos_outs = _run(sessions["vocoder_stream_chunk"], feed)
        # OUT order: audio, embed_prev_new, block_0_prev_new .. block_7_prev_new, istft_prev_new
        audio_chunk = vocos_outs[0]   # [1, CHUNK_FRAMES * HOP]
        embed_prev  = vocos_outs[1]
        for b in range(N_BLOCKS):
            block_prevs[b] = vocos_outs[2 + b]
        istft_prev = vocos_outs[2 + N_BLOCKS]   # [1, TAIL]

        audio_chunks.append(audio_chunk[0, : valid * HOP])
        pos = end

    return np.concatenate(audio_chunks).astype(np.float32)


# ---------------------------------------------------------------------------
# Session loader
# ---------------------------------------------------------------------------

def load_sessions(model_dir: Path, precision: str) -> Dict[str, ort.InferenceSession]:
    """Load all 6 ORT sessions from model_dir.

    Parameters
    ----------
    model_dir:
        Directory containing the exported ONNX models.
    precision:
        One of 'fp32', 'fp16', 'int8', 'int8_mixed'.
    """
    suffix_map = PRECISION_SUFFIX[precision]
    sessions: Dict[str, ort.InferenceSession] = {}
    for name, suffix in suffix_map.items():
        path = model_dir / f"{name}{suffix}.onnx"
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                f"Run 'python export_onnx.py' to generate export_models/onnx/ first."
            )
        print(f"  loading {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
        sessions[name] = _make_session(path)
    return sessions


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kokoro TTS ONNX inference example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--text",
        default=(
            "Hello world. This is Kokoro TTS running on ONNX Runtime "
            "with streaming chunked vocoder inference."
        ),
        help="Text to synthesise.",
    )
    parser.add_argument(
        "--out",
        default="output_onnx.wav",
        help="Output WAV file path.",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp32", "fp16", "int8", "int8_mixed"],
        help="Model precision to use.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        dest="model_dir",
        help="Directory with exported ONNX models.",
    )
    parser.add_argument(
        "--voice",
        default=VOICE_PATH,
        help="Path to voice .pt file.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        help="Speed multiplier (integer, 1 = normal).",
    )
    args = parser.parse_args()

    print(f"Loading Kokoro pipeline (G2P)...")
    kmodel = KModel(
        config=CONFIG_FILE,
        model=CHECKPOINT_PATH,
        disable_complex=True,
    ).cpu().eval()
    kpipeline = KPipeline(lang_code="a", model=kmodel, device="cpu")

    print(f"Loading {args.precision.upper()} ONNX models from {args.model_dir}/")
    sessions = load_sessions(args.model_dir, args.precision)

    print(f"\nSynthesising: \"{args.text[:70]}\"")
    t0 = time.perf_counter()
    audio = generate(
        text=args.text,
        sessions=sessions,
        kpipeline=kpipeline,
        voice_path=args.voice,
        speed=args.speed,
    )
    elapsed = time.perf_counter() - t0

    duration = len(audio) / SAMPLE_RATE
    rtf = elapsed / duration if duration > 0 else float("inf")
    print(f"Generated {duration:.2f}s audio in {elapsed:.2f}s  (RTF={rtf:.2f})")

    # Normalise to int16 range and save
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(args.out, SAMPLE_RATE, audio_int16)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
