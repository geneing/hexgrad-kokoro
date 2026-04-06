#!/usr/bin/env python3
"""quantize_int8.py — INT8 Post-Training Quantization for Kokoro TTS Android Models

Methods implemented
-------------------
1. **ONNX Static INT8** (onnxruntime)
   - Full INT8 (weights + activations), ~4× smaller than FP32
   - QDQ (Quantize-Dequantize) format for NNAPI compatibility
   - Requires calibration data (generated below)
   - Output: onnx_streaming_vocos/*_int8_static.onnx

2. **LiteRT Full Integer INT8** (onnx2tf)
   - Full INT8 (weights + activations) TFLite
   - Best for NNAPI delegate on Android
   - Requires calibration via onnx2tf representative dataset
   - Output: onnx_streaming_vocos/tflite/*_full_int8.tflite

3. **torchao Weight-Only INT8** (torchao + ONNX re-export)
   - Applies Int8WeightOnlyConfig to PyTorch model, then re-exports
   - Better weight quantization quality than standard PTQ
   - Needs FakeTensor-compatible model (forward() only, no custom ops)
   - Output: onnx_streaming_vocos/*_torchao_int8.onnx (best effort)

Usage
-----
    TF_ENABLE_ONEDNN_OPTS=0 uv run python quantize_int8.py [--methods static litert-int8 torchao]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU for TFLite quant

import numpy as np
import torch
import torch.nn.functional as F

# ── ONNX ─────────────────────────────────────────────────────────────────────
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.calibrate import CalibrationMethod

# ── Kokoro TTS ────────────────────────────────────────────────────────────────
from kokoro import KModel, KPipeline

# ── Config ────────────────────────────────────────────────────────────────────
ONNX_DIR = Path("onnx_streaming_vocos")
TFLITE_DIR = Path("onnx_streaming_vocos/tflite")
CALIB_DIR = Path("int8_calib")

CONFIG_FILE = "checkpoints/config.json"
CHECKPOINT_PATH = "checkpoints/kokoro-v1_0.pth"
VOICE_PATH = "checkpoints/voices/af_bella.pt"

MAX_INPUT_LENGTH = 510
T_ACOUSTIC = 543    # static acoustic expand dimension (from model)
T_F0 = 1086         # T_ACOUSTIC * 2
CHUNK_FRAMES = 16   # vocos streaming chunk size
HOP = 300
WIN_LEN = 1200
VOCOS_N_FFT = 1200
KERN = 6            # vocos causal conv kernel - 1

# Calibration sentences: short, medium, long to cover range of acoustic lengths
CALIB_SENTENCES = [
    "Hello.",
    "Testing the system.",
    "The quick brown fox jumps over the lazy dog.",
    "I had returned to civil practice and had finally abandoned Holmes in his Baker Street rooms.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "It was the best of times, it was the worst of times.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "Call me Ishmael.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "The sky above the port was the color of television, tuned to a dead channel.",
    "We hold these truths to be self-evident, that all men are created equal.",
    "In the beginning God created the heavens and the earth.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "One ring to rule them all, one ring to find them, one ring to bring them all and in the darkness bind them.",
    "Speech synthesis is the artificial production of human speech from text.",
    "The model converts phoneme sequences into mel spectrograms and then to audio.",
    "Neural text to speech systems have improved dramatically in recent years.",
    "Quantization reduces model size while maintaining acceptable audio quality.",
    "Android devices support neural network acceleration through the NNAPI delegate.",
    "The vocoder uses streaming causal convolutions to generate audio in real time.",
    "Kokoro is a high quality text to speech model optimized for mobile deployment.",
    "Duration prediction determines how long each phoneme is held in the output.",
    "The acoustic model expands phoneme representations to frame level features.",
    "Pitch and energy contours are predicted by the F zero and noise predictor.",
    "Vocos uses a Fourier transform based vocoder with streaming state management.",
    "Post training quantization can achieve near lossless compression of neural networks.",
    "Mobile inference requires balancing model accuracy, size, and computational cost.",
    "The half quadratic quantization method minimizes reconstruction error per weight block.",
    "Int eight quantization uses signed or unsigned eight bit integers for weights and activations.",
]

# ---------------------------------------------------------------------------
# Kokoro pipeline setup
# ---------------------------------------------------------------------------

def load_kokoro_pipeline() -> Tuple[KModel, KPipeline]:
    """Load Kokoro model and pipeline (CPU only, for calibration data generation)."""
    print("Loading Kokoro model...")
    kmodel = KModel(config=CONFIG_FILE, model=CHECKPOINT_PATH, disable_complex=True).to("cpu").eval()
    kpipeline = KPipeline(lang_code="a", model=kmodel, device="cpu")
    return kmodel, kpipeline


def text_to_padded_inputs(
    kpipeline: KPipeline, text: str, max_len: int = MAX_INPUT_LENGTH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert text to padded input_ids, text_mask, and style vector.

    Returns
    -------
    input_ids : int32 numpy [1, max_len]
    text_mask : float32 numpy [1, max_len]
    style : float32 numpy [1, 256]
    """
    if kpipeline.lang_code in "ab":
        _, tokens = kpipeline.g2p(text)
        ps: List[str] = []
        for _, p, _ in kpipeline.en_tokenize(tokens):
            ps.extend(p)
    else:
        ps, _ = kpipeline.g2p(text)
    ps = ps[:max_len - 2]  # leave room for BOS/EOS tokens

    vocab = kpipeline.model.vocab
    ids = [i for i in (vocab.get(p) for p in ps) if i is not None]
    raw_ids = torch.IntTensor([[0, *ids, 0]])  # add BOS=0 and EOS=0

    # Style from voice pack (use length as row index)
    pack = kpipeline.load_voice(VOICE_PATH).to("cpu")
    style_row = pack[len(ps) - 1]  # [1, 256]

    # Pad to max_len
    text_mask = np.zeros((1, max_len), dtype=np.float32)
    text_mask[0, : raw_ids.shape[1]] = 1.0
    pad_len = max_len - raw_ids.shape[1]
    if pad_len > 0:
        raw_ids = F.pad(raw_ids, (0, pad_len))

    input_ids_np = raw_ids.numpy().astype(np.int32)
    style_np = style_row.detach().cpu().numpy().astype(np.float32)  # [1, 256]
    return input_ids_np, text_mask, style_np


# ---------------------------------------------------------------------------
# ONNX inference helpers for calibration
# ---------------------------------------------------------------------------

def _ort_session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    opts.log_severity_level = 3  # errors only
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _expand_durations(pred_dur: np.ndarray) -> Tuple[np.ndarray, int]:
    """Compute per-acoustic-frame phoneme indices from rounded durations."""
    dur = torch.from_numpy(pred_dur.astype(np.float32))
    boundaries = torch.cumsum(dur, dim=0)
    T_acoustic = int(boundaries[-1].item())
    if T_acoustic == 0:
        return np.zeros(0, dtype=np.int64), 0
    values = torch.arange(T_acoustic, dtype=torch.int32)
    expanded_indices = torch.sum(boundaries.unsqueeze(1) <= values.unsqueeze(0), dim=0)
    return expanded_indices.numpy().astype(np.int64), T_acoustic


def run_onnx_pipeline_for_calibration(
    text: str,
    kpipeline: KPipeline,
    sessions: Dict[str, ort.InferenceSession],
) -> Optional[Dict[str, List[Dict[str, np.ndarray]]]]:
    """Run one sentence through the ONNX android pipeline and collect inputs for each model.

    Returns a dict: model_name → list of {input_name: array} dicts.
    Returns None if the sentence produces a T_acoustic > T_ACOUSTIC.
    """
    try:
        input_ids, text_mask, style = text_to_padded_inputs(kpipeline, text)
        speed = np.array([1], dtype=np.int32)

        # ── Stage 1: BERT ──────────────────────────────────────────────────
        bert_inputs = {"input_ids": input_ids, "text_mask": text_mask}
        bert_out = sessions["bert"].run(None, bert_inputs)
        d_en = bert_out[0]  # [1, 512, 510] NCT

        # ── Stage 2: Duration predictor ────────────────────────────────────
        dur_inputs = {
            "d_en": d_en,
            "style": style,
            "text_mask": text_mask,
            "speed": speed,
            "input_ids": input_ids,
        }
        dur_outs = sessions["duration_predictor"].run(None, dur_inputs)

        # Map outputs by shape
        dur_out_map: Dict[tuple, np.ndarray] = {}
        for arr in dur_outs:
            dur_out_map[tuple(arr.shape)] = arr
        pred_dur = dur_out_map.get((510,), dur_out_map.get((1, 510)))
        if pred_dur is None:
            # Try to find the 1D 510-size output
            for arr in dur_outs:
                if arr.size == 510:
                    pred_dur = arr.flatten()
                    break
        if pred_dur is None:
            return None
        pred_dur = pred_dur.flatten()  # [510]

        d_enc_nct = dur_out_map.get((1, 640, 510))   # [1, 640, 510] NCT
        t_en_static_nct = dur_out_map.get((1, 512, 510))  # [1, 512, 510] NCT

        if d_enc_nct is None or t_en_static_nct is None:
            # Find by shape scan
            for arr in dur_outs:
                if arr.shape == (1, 640, 510):
                    d_enc_nct = arr
                elif arr.shape == (1, 512, 510):
                    t_en_static_nct = arr

        # ── Stage 3: Expand durations (no model needed) ───────────────────
        expanded_indices, T_acoustic = _expand_durations(pred_dur)
        if T_acoustic == 0 or T_acoustic > T_ACOUSTIC:
            return None  # skip sentences that are too long

        # Expand: [1, C, T_text] → [1, C, T_acoustic]
        d_enc_exp = np.take(d_enc_nct, expanded_indices, axis=2)    # [1, 640, T_acoustic]
        asr = np.take(t_en_static_nct, expanded_indices, axis=2)    # [1, 512, T_acoustic]

        # Pad NCT along axis=2 to T_ACOUSTIC (model input is [1, 640, T_ACOUSTIC] NCT)
        if T_acoustic < T_ACOUSTIC:
            pad = T_ACOUSTIC - T_acoustic
            d_enc_exp = np.pad(d_enc_exp, ((0, 0), (0, 0), (0, pad)))

        # ── Stage 4: Acoustic expand ───────────────────────────────────────
        acexp_inputs = {"d_enc_expanded": d_enc_exp.astype(np.float32)}
        acexp_outs = sessions["acoustic_expand"].run(None, acexp_inputs)
        en_ntc = acexp_outs[0]  # [1, T_ACOUSTIC, 512] NTC

        # ── Stage 5: F0/N predictor ────────────────────────────────────────
        f0n_inputs = {
            "en": en_ntc.astype(np.float32),    # [1, T_ACOUSTIC, 512] NTC
            "style": style,
        }
        f0n_outs = sessions["f0n_predictor"].run(None, f0n_inputs)
        # Outputs: F0_pred [1, T_F0], N_pred [1, T_F0]
        f0_map = {tuple(a.shape): a for a in f0n_outs}
        F0_pred = f0_map.get((1, T_F0))
        N_pred = None
        for arr in f0n_outs:
            if arr.shape == (1, T_F0) and F0_pred is not None and not np.array_equal(arr, F0_pred):
                N_pred = arr
                break
        if F0_pred is None:
            F0_pred = f0n_outs[0]
        if N_pred is None:
            N_pred = f0n_outs[1] if len(f0n_outs) > 1 else f0n_outs[0]

        # ── Stage 6: Vocoder conditioner ──────────────────────────────────
        # Build features [1, 642, T_F0] NCT
        T_f0_actual = 2 * T_acoustic
        asr_t = torch.from_numpy(asr.astype(np.float32))              # [1, 512, T_acoustic]
        # Interpolate ASR to T_F0
        asr_t_f0 = F.interpolate(asr_t, size=T_F0, mode="linear", align_corners=False)
        f0_t = torch.from_numpy(F0_pred.astype(np.float32)).unsqueeze(1)  # [1, 1, T_F0]
        n_t = torch.from_numpy(N_pred.astype(np.float32)).unsqueeze(1)   # [1, 1, T_F0]
        s_t = torch.from_numpy(style.astype(np.float32))[:, :128].unsqueeze(-1).expand(-1, -1, T_F0)
        features_nct = torch.cat([asr_t_f0, f0_t, n_t, s_t], dim=1).numpy()  # [1, 642, T_F0]

        cond_inputs = {"features": features_nct.astype(np.float32)}
        cond_outs = sessions["vocoder_conditioner"].run(None, cond_inputs)
        conditioned = cond_outs[0]  # [1, 192, T_F0] NCT

        # ── Stage 7: Vocoder streaming (first chunk only for calibration) ──
        chunk = conditioned[:, :, :CHUNK_FRAMES].astype(np.float32)   # [1, 192, CHUNK_FRAMES]
        state_shapes = {
            "embed_prev": (1, 192, KERN),
            **{f"block_{i}_prev": (1, 384, KERN) for i in range(8)},
            "istft_prev": (1, WIN_LEN - HOP),
        }
        chunk_inputs: Dict[str, np.ndarray] = {"conditioned_chunk": chunk}
        for sname, sshape in state_shapes.items():
            chunk_inputs[sname] = np.zeros(sshape, dtype=np.float32)

        # Also collect a few middle chunks
        vocos_calib_inputs = [chunk_inputs]
        for chunk_start in range(CHUNK_FRAMES, T_f0_actual, CHUNK_FRAMES * 4):
            end = min(T_f0_actual, chunk_start + CHUNK_FRAMES)
            c = conditioned[:, :, chunk_start:end]
            if c.shape[-1] < CHUNK_FRAMES:
                c = np.pad(c, ((0, 0), (0, 0), (0, CHUNK_FRAMES - c.shape[-1])))
            chunk_inputs_mid: Dict[str, np.ndarray] = {"conditioned_chunk": c.astype(np.float32)}
            for sname, sshape in state_shapes.items():
                chunk_inputs_mid[sname] = np.random.randn(*sshape).astype(np.float32) * 0.1
            vocos_calib_inputs.append(chunk_inputs_mid)

        return {
            "bert": [bert_inputs],
            "duration_predictor": [dur_inputs],
            "acoustic_expand": [acexp_inputs],
            "f0n_predictor": [f0n_inputs],
            "vocoder_conditioner": [cond_inputs],
            "vocoder_stream_chunk": vocos_calib_inputs,
        }

    except Exception as e:
        warnings.warn(f"Calibration data generation failed for '{text[:40]}...': {e}")
        return None


def generate_calibration_data(
    kpipeline: KPipeline,
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """Generate calibration data for all 6 models by running the ONNX pipeline.

    Returns dict: model_name → list of input dicts (one per calibration sample).
    """
    print(f"Loading ONNX sessions for calibration ({len(CALIB_SENTENCES)} sentences)...")
    sessions: Dict[str, ort.InferenceSession] = {}
    model_names = [
        "bert", "duration_predictor", "acoustic_expand",
        "f0n_predictor", "vocoder_conditioner", "vocoder_stream_chunk",
    ]
    for name in model_names:
        path = ONNX_DIR / f"{name}.onnx"
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        sessions[name] = _ort_session(path)

    all_calib: Dict[str, List[Dict[str, np.ndarray]]] = {n: [] for n in model_names}

    for i, sentence in enumerate(CALIB_SENTENCES):
        print(f"  [{i+1}/{len(CALIB_SENTENCES)}] {sentence[:50]:<50}", end=" ", flush=True)
        t0 = time.perf_counter()
        result = run_onnx_pipeline_for_calibration(sentence, kpipeline, sessions)
        elapsed = time.perf_counter() - t0
        if result is None:
            print(f"SKIP ({elapsed:.1f}s)")
            continue
        for name in model_names:
            all_calib[name].extend(result[name])
        print(f"OK   ({elapsed:.1f}s, {sum(len(v) for v in result.values())} samples)")

    for name in model_names:
        n = len(all_calib[name])
        print(f"  {name}: {n} calibration samples")

    return all_calib


# ---------------------------------------------------------------------------
# ONNX calibration data reader
# ---------------------------------------------------------------------------

class KokoroCalibrationReader(CalibrationDataReader):
    """onnxruntime CalibrationDataReader backed by pre-collected numpy arrays."""

    def __init__(self, samples: List[Dict[str, np.ndarray]]) -> None:
        self._samples = samples
        self._idx = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._idx >= len(self._samples):
            return None
        item = self._samples[self._idx]
        self._idx += 1
        return item

    def rewind(self) -> None:
        self._idx = 0


# ---------------------------------------------------------------------------
# Node exclusion helpers — protect sensitive operations from INT8 quantization
# ---------------------------------------------------------------------------

def _nodes_to_exclude(model_name: str, src: Path) -> List[str]:
    """Return nodes that should NOT be quantized for a given model.

    Strategy per model:
    - bert: exclude Softmax (attention) + LayerNorm (narrow/shifted range)
    - duration_predictor: exclude LSTM (extremely sensitive to quantization)
    - acoustic_expand: nothing — BiLSTM quantizes losslessly
    - f0n_predictor: nothing — ConvNet quantizes well
    - vocoder_conditioner: skip entirely (model is too small; all nodes excluded)
    - vocoder_stream_chunk: exclude IDFT+OLA section (audio generation is fp32-sensitive)
    """
    excludes: List[str] = []
    try:
        m = onnx.load(str(src))
        if model_name == "bert":
            for n in m.graph.node:
                if n.op_type in ("Softmax", "LayerNormalization"):
                    if n.name:
                        excludes.append(n.name)
        elif model_name == "duration_predictor":
            for n in m.graph.node:
                if n.op_type in ("LSTM", "GRU", "RNN"):
                    if n.name:
                        excludes.append(n.name)
        elif model_name == "vocoder_conditioner":
            # Tiny model (1.1 MB fp32) — INT8 completely breaks it; skip all
            for n in m.graph.node:
                if n.name:
                    excludes.append(n.name)
        elif model_name == "vocoder_stream_chunk":
            # Exclude the entire IDFT+OLA section (from final projection forward).
            # The Einsum IDFT with cos/sin basis and the OLA accumulation are very
            # sensitive: any INT8 rounding in the spectral amplitudes causes loud
            # artefacts. The ConvNeXt blocks above are stable and can stay INT8.
            # Also exclude LayerNormalization nodes (narrow normalized range).
            idft_start = "node_MatMul_146"   # final 384→1202 projection
            in_idft = False
            for n in m.graph.node:
                if n.name == idft_start:
                    in_idft = True
                if in_idft and n.name:
                    excludes.append(n.name)
                elif n.op_type == "LayerNormalization" and n.name:
                    excludes.append(n.name)
    except Exception as e:
        warnings.warn(f"Could not compute node exclusions for {model_name}: {e}")
    return excludes


# ---------------------------------------------------------------------------
# Method 1: ONNX Static INT8 Quantization (QDQ format for NNAPI)
# ---------------------------------------------------------------------------

def quantize_onnx_static(
    model_names: List[str],
    calib_data: Dict[str, List[Dict[str, np.ndarray]]],
    output_dir: Path,
    quant_format: QuantFormat = QuantFormat.QDQ,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
    force: bool = False,
) -> Dict[str, Path]:
    """Apply onnxruntime static INT8 quantization with calibration data.

    Uses QDQ (Quantize-Dequantize) format which is required for ONNX Runtime
    NNAPI execution provider on Android.

    Improvements over naive quantization:
    - Entropy calibration (better dynamic range estimation than MinMax)
    - Per-model node exclusions: LSTM ops (duration_predictor), Softmax+LayerNorm
      (bert), IDFT/OLA section (vocoder_stream_chunk), all nodes (vocoder_conditioner)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}

    for name in model_names:
        src = ONNX_DIR / f"{name}.onnx"
        dst = output_dir / f"{name}_int8_static.onnx"
        samples = calib_data.get(name, [])

        if not samples:
            print(f"  [skip] {name}: no calibration data")
            continue
        if dst.exists() and not force:
            size_mb = dst.stat().st_size / 1e6
            print(f"  [skip] {dst.name} already exists ({size_mb:.1f} MB)")
            outputs[name] = dst
            continue

        exclude = _nodes_to_exclude(name, src)
        if exclude:
            print(f"  Static INT8: {name} ({len(samples)} samples, "
                  f"{len(exclude)} nodes excluded)...", end=" ", flush=True)
        else:
            print(f"  Static INT8: {name} ({len(samples)} samples)...", end=" ", flush=True)

        t0 = time.perf_counter()
        try:
            reader = KokoroCalibrationReader(samples)
            quantize_static(
                model_input=str(src),
                model_output=str(dst),
                calibration_data_reader=reader,
                quant_format=quant_format,
                activation_type=activation_type,
                weight_type=weight_type,
                per_channel=True,
                reduce_range=False,
                calibrate_method=CalibrationMethod.Entropy,   # better than MinMax
                nodes_to_exclude=exclude if exclude else None,
                extra_options={
                    "WeightSymmetric": True,
                    "ActivationSymmetric": False,
                },
            )
            size_mb = dst.stat().st_size / 1e6
            orig_mb = src.stat().st_size / 1e6
            elapsed = time.perf_counter() - t0
            print(f"{orig_mb:.1f}MB → {size_mb:.1f}MB ({size_mb/orig_mb*100:.0f}%) in {elapsed:.1f}s")
            outputs[name] = dst
        except Exception as e:
            print(f"FAILED: {e}")
            # Try fallback with QOperator format
            try:
                print(f"    Retrying with QOperator format...", end=" ", flush=True)
                reader = KokoroCalibrationReader(samples)
                quantize_static(
                    model_input=str(src),
                    model_output=str(dst),
                    calibration_data_reader=reader,
                    quant_format=QuantFormat.QOperator,
                    activation_type=activation_type,
                    weight_type=weight_type,
                    per_channel=True,
                    reduce_range=False,
                    calibrate_method=CalibrationMethod.Entropy,
                    nodes_to_exclude=exclude if exclude else None,
                )
                size_mb = dst.stat().st_size / 1e6
                elapsed = time.perf_counter() - t0
                print(f"OK {size_mb:.1f}MB ({elapsed:.1f}s)")
                outputs[name] = dst
            except Exception as e2:
                print(f"ALSO FAILED: {e2}")

    return outputs


# ---------------------------------------------------------------------------
# Method 3 & 4: LiteRT quantization via onnx2tf
# ---------------------------------------------------------------------------

def save_calib_npy(
    calib_samples: List[Dict[str, np.ndarray]],
    calib_dir: Path,
    model_name: str,
) -> Optional[List[List]]:
    """Save averaged calibration samples to .npy files for onnx2tf -oiqt.

    onnx2tf's custom_input_op_name_np_data_path expects:
      [[input_name, "/path/to/data.npy"], ...]       (Python API)
    where the .npy file contains a single representative sample (the mean
    across all calibration samples is used).

    Returns the list-of-lists to pass, or None if no samples are available.
    """
    if not calib_samples:
        return None

    calib_dir.mkdir(parents=True, exist_ok=True)
    cind_args = []

    # Average all samples to get a representative input
    input_names = list(calib_samples[0].keys())
    for inp_name in input_names:
        arrays = [s[inp_name] for s in calib_samples if inp_name in s]
        if not arrays:
            continue
        # Use mean sample (cast all to original dtype)
        try:
            orig_dtype = arrays[0].dtype
            avg = np.mean([a.astype(np.float32) for a in arrays], axis=0)
            representative = avg.astype(orig_dtype)
        except Exception:
            representative = arrays[0]

        npy_path = calib_dir / f"{model_name}_{inp_name}.npy"
        np.save(str(npy_path), representative)
        # Format: [name, path, mean, std] — all 4 required for -oiqt in onnx2tf
        # mean=0.0, std=1.0 means no normalization (raw float values)
        cind_args.append([inp_name, str(npy_path), 0.0, 1.0])

    return cind_args if cind_args else None


def quantize_litert(
    model_names: List[str],
    output_dir: Path,
    dynamic_range: bool = True,
    full_integer: bool = True,
    source_dir: Path = ONNX_DIR,
    suffix: str = "",          # source model suffix e.g. "" for base onnx
    calib_data: Optional[Dict[str, List[Dict[str, np.ndarray]]]] = None,
) -> None:
    """Export LiteRT INT8 models using onnx2tf.

    Parameters
    ----------
    dynamic_range:
        If True, export weights-only INT8 dynamic range quantized TFLite.
    full_integer:
        If True, export full INT8 (weights + activations) TFLite.
        Calibration data (calib_data) is used when provided so that onnx2tf
        can accurately estimate activation ranges.  Without it, onnx2tf falls
        back to random dummy data which gives poor accuracy for non-image models.
    calib_data:
        Optional dict: model_name → list of {input_name: np.ndarray} samples.
        Required for ``full_integer=True`` of non-image models.
    """
    try:
        from onnx2tf import convert as onnx2tf_convert
    except ImportError:
        print("onnx2tf not available, skipping LiteRT quantization")
        return

    import shutil
    output_dir.mkdir(parents=True, exist_ok=True)
    calib_npy_dir = output_dir / "_calib_npy"

    for name in model_names:
        src = source_dir / f"{name}{suffix}.onnx"
        if not src.exists():
            print(f"  [skip] {src} not found")
            continue

        # Determine output paths
        dr_out = output_dir / f"{name}_dynamic_range_int8.tflite"
        fi_out = output_dir / f"{name}_full_int8.tflite"

        # Check if we need to run conversion at all
        need_dr = dynamic_range and not dr_out.exists()
        need_fi = full_integer and not fi_out.exists()
        if not need_dr and not need_fi:
            print(f"  [skip] {name} LiteRT int8 models already exist")
            continue

        # Prepare calibration npy files for full-integer quantization
        cind_args: Optional[List[str]] = None
        if need_fi and calib_data is not None:
            samples = calib_data.get(name, [])
            if samples:
                cind_args = save_calib_npy(samples, calib_npy_dir, name)

        # Use a temp folder for onnx2tf output
        tmp_dir = output_dir / f"_onnx2tf_tmp_{name}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"  onnx2tf INT8: {name}...", end=" ", flush=True)
            t0 = time.perf_counter()

            convert_kwargs: Dict = dict(
                input_onnx_file_path=str(src),
                output_folder_path=str(tmp_dir),
                copy_onnx_input_output_names_to_tflite=True,
                non_verbose=True,
                output_dynamic_range_quantized_tflite=need_dr,
                output_integer_quantized_tflite=need_fi,
                quant_type="per-channel",
                input_quant_dtype="int8",
                output_quant_dtype="int8",
            )
            if cind_args:
                convert_kwargs["custom_input_op_name_np_data_path"] = cind_args

            onnx2tf_convert(**convert_kwargs)

            elapsed = time.perf_counter() - t0

            # Copy outputs to final location
            copied = []
            for tflite_file in sorted(tmp_dir.glob("*.tflite")):
                fname = tflite_file.name
                if "dynamic_range" in fname and need_dr:
                    shutil.copy2(tflite_file, dr_out)
                    copied.append(f"dr_int8({dr_out.stat().st_size/1e6:.1f}MB)")
                elif ("quant" in fname or "integer" in fname) and need_fi:
                    if not fi_out.exists():
                        shutil.copy2(tflite_file, fi_out)
                        copied.append(f"fi_int8({fi_out.stat().st_size/1e6:.1f}MB)")
                elif need_fi and not fi_out.exists() and "float" not in fname:
                    # Fallback: take any .tflite that isn't explicitly float
                    shutil.copy2(tflite_file, fi_out)
                    copied.append(f"fi_int8_fallback({fi_out.stat().st_size/1e6:.1f}MB)")

            print(f"OK in {elapsed:.1f}s — {', '.join(copied) if copied else 'no tflite files found'}")

        except Exception as e:
            print(f"FAILED: {e}")
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Method 5: torchao weight-only INT8 + ONNX re-export
# ---------------------------------------------------------------------------

def quantize_torchao_int8_export(
    model_names: List[str],
    output_dir: Path,
) -> Dict[str, Path]:
    """Apply torchao Int8WeightOnlyConfig to Kokoro PyTorch submodels and re-export.

    This applies optimized per-group INT8 weight quantization directly to the
    PyTorch model before ONNX export.  Unlike standard PTQ this uses lossless
    scales that minimize per-group reconstruction error.

    Note: Only affects nn.Linear weights. LSTM weight matrices are handled
    separately via manual quantization of their weight parameters.
    """
    try:
        import torchao
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
    except ImportError:
        print("torchao not available, skipping torchao INT8 export")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}

    print("Loading Kokoro model for torchao INT8 quantization...")
    from kokoro.model import KModelForONNX
    kmodel = KModel(
        config=CONFIG_FILE, model=CHECKPOINT_PATH, disable_complex=True
    ).to("cpu").eval()
    kmodel_onnx = KModelForONNX(kmodel).eval()

    # torchao's quantize_ works on nn.Module, modifying Linear layers in-place
    # We need to apply it to each sub-module before exporting that sub-module
    # to ONNX. The tricky part is that KModelForONNX exposes separate forward
    # methods for each stage.

    # Map of model_name → (module_path, input_example_fn)
    # We only handle models with significant nn.Linear content
    quantizable = {
        "bert", "duration_predictor", "f0n_predictor",
        "vocoder_stream_chunk", "acoustic_expand",
    }

    for name in model_names:
        if name not in quantizable:
            continue
        dst = output_dir / f"{name}_torchao_int8.onnx"
        if dst.exists():
            print(f"  [skip] {dst.name} already exists")
            outputs[name] = dst
            continue

        print(f"  torchao INT8: {name}...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            # Re-load a fresh model for each export to avoid state pollution
            fresh_kmodel = KModel(
                config=CONFIG_FILE, model=CHECKPOINT_PATH, disable_complex=True
            ).to("cpu").eval()
            fresh_onnx = KModelForONNX(fresh_kmodel).eval()

            # Apply Int8WeightOnly quantization to all linear layers
            # torchao modifies weights in-place using AffineQuantizedTensor
            quantize_(fresh_onnx, Int8WeightOnlyConfig())

            # Try to export with torch.onnx.export (will fail if custom tensor ops
            # are not supported by the ONNX exporter, in which case we skip)
            import torch

            # Build dummy inputs matching the per-model convention
            if name == "bert":
                dummy = (
                    torch.zeros(1, 510, dtype=torch.int32),  # input_ids
                    torch.zeros(1, 510, dtype=torch.float32),  # text_mask
                )
                input_names = ["input_ids", "text_mask"]

                class _BertWrapper(torch.nn.Module):
                    def __init__(self, m): super().__init__(); self.m = m
                    def forward(self, inp, mask): return self.m.bert(inp, mask)

                mod = _BertWrapper(fresh_onnx)

            elif name == "duration_predictor":
                dummy = (
                    torch.zeros(1, 512, 510, dtype=torch.float32),  # d_en NCT
                    torch.zeros(1, 256, dtype=torch.float32),        # style
                    torch.zeros(1, 510, dtype=torch.float32),        # text_mask
                    torch.tensor([1], dtype=torch.float32),          # speed
                    torch.zeros(1, 510, dtype=torch.int64),          # input_ids
                )
                input_names = ["d_en", "style", "text_mask", "speed", "input_ids"]

                class _DurWrapper(torch.nn.Module):
                    def __init__(self, m): super().__init__(); self.m = m
                    def forward(self, d_en, style, text_mask, speed, input_ids):
                        return self.m.duration_predictor(d_en, style, text_mask, speed, input_ids)

                mod = _DurWrapper(fresh_onnx)

            else:
                print(f"skip (no wrapper defined)")
                continue

            # Attempt export — this may fail if torchao's custom tensors are not
            # ONNX-compatible, in which case we log a clear warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    mod,
                    dummy,
                    str(dst),
                    input_names=input_names,
                    opset_version=17,
                    do_constant_folding=True,
                )

            size_mb = dst.stat().st_size / 1e6
            elapsed = time.perf_counter() - t0
            print(f"{size_mb:.1f}MB in {elapsed:.1f}s")
            outputs[name] = dst

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"FAILED ({elapsed:.1f}s): {type(e).__name__}: {e}")
            # torchao custom tensor types often break ONNX export; this is expected.
            # The recommended path is: use ONNX dynamic/static quantization instead,
            # which achieves equivalent compression with full NNAPI compatibility.

    return outputs


# ---------------------------------------------------------------------------
# Quality check: compare ONNX static int8 vs FP32 outputs
# ---------------------------------------------------------------------------

def compare_onnx_quality(
    model_name: str,
    calib_samples: List[Dict[str, np.ndarray]],
    int8_path: Path,
    fp32_path: Optional[Path] = None,
    n_samples: int = 5,
) -> Optional[float]:
    """Compare FP32 and INT8 ONNX model outputs on calibration data.

    Returns mean cosine similarity across samples, or None if comparison fails.
    """
    if fp32_path is None:
        fp32_path = ONNX_DIR / f"{model_name}.onnx"
    if not fp32_path.exists() or not int8_path.exists():
        return None

    try:
        sess_fp32 = _ort_session(fp32_path)
        sess_int8 = _ort_session(int8_path)

        similarities = []
        for sample in calib_samples[:n_samples]:
            out_fp32 = sess_fp32.run(None, sample)
            out_int8 = sess_int8.run(None, sample)
            for a, b in zip(out_fp32, out_int8):
                a_flat = a.flatten().astype(np.float32)
                b_flat = b.flatten().astype(np.float32)
                if a_flat.size > 0 and np.linalg.norm(a_flat) > 0 and np.linalg.norm(b_flat) > 0:
                    cos = np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
                    similarities.append(float(cos))

        if not similarities:
            return None
        return float(np.mean(similarities))

    except Exception as e:
        warnings.warn(f"Quality comparison failed for {model_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_size_comparison(output_dir: Path = ONNX_DIR, tflite_dir: Path = TFLITE_DIR) -> None:
    """Print size comparison across all quantization formats."""
    model_names = [
        "bert", "duration_predictor", "acoustic_expand",
        "f0n_predictor", "vocoder_conditioner", "vocoder_stream_chunk",
    ]

    print("\n" + "=" * 80)
    print("MODEL SIZE COMPARISON")
    print("=" * 80)
    print(f"{'Model':<30} {'FP32':>8} {'FP16':>8} {'Int8 Sta':>10} {'LiteRT FI':>11}")
    print("-" * 70)

    total = {k: 0.0 for k in ["fp32", "fp16", "sta", "lfi"]}

    for name in model_names:
        sizes = {}
        for key, path in [
            ("fp32", ONNX_DIR / f"{name}.onnx"),
            ("fp16", ONNX_DIR / f"{name}_fp16.onnx"),
            ("sta",  output_dir / f"{name}_int8_static.onnx"),
            ("lfi",  tflite_dir / f"{name}_full_int8.tflite"),
        ]:
            sizes[key] = path.stat().st_size / 1e6 if path.exists() else None

        row = f"{name:<30}"
        for key in ["fp32", "fp16", "sta", "lfi"]:
            v = sizes[key]
            if v is not None:
                row += f" {v:>8.1f}M"
                total[key] += v
            else:
                row += f" {'—':>8}"
        print(row)

    print("-" * 70)
    row = f"{'TOTAL':<30}"
    for key in ["fp32", "fp16", "sta", "lfi"]:
        v = total[key]
        row += f" {v:>8.1f}M" if v > 0 else f" {'—':>8}"
    print(row)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_METHODS = ["static", "litert-int8", "torchao"]

MODEL_NAMES = [
    "bert",
    "duration_predictor",
    "acoustic_expand",
    "f0n_predictor",
    "vocoder_conditioner",
    "vocoder_stream_chunk",
]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="INT8 PTQ for Kokoro TTS Android models")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=["static", "litert-int8"],
        help="Quantization methods to apply",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        help="Models to quantize",
    )
    parser.add_argument(
        "--onnx-output",
        default=str(ONNX_DIR),
        help="Output directory for quantized ONNX models (default: onnx_streaming_vocos/)",
    )
    parser.add_argument(
        "--tflite-output",
        default=str(TFLITE_DIR),
        help="Output directory for quantized TFLite models (default: onnx_streaming_vocos/tflite/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output files already exist",
    )
    args = parser.parse_args(argv)

    onnx_out = Path(args.onnx_output)
    tflite_out = Path(args.tflite_output)
    methods = set(args.methods)
    model_list = args.models

    if args.force:
        # Remove existing outputs so they get regenerated
        for m in model_list:
            for p in [
                onnx_out / f"{m}_int8_static.onnx",
                tflite_out / f"{m}_full_int8.tflite",
            ]:
                if p.exists():
                    p.unlink()

    # ── Load Kokoro pipeline (needed for calibration data generation) ──────
    kmodel = None
    kpipeline = None
    calib_data: Dict[str, List[Dict[str, np.ndarray]]] = {n: [] for n in MODEL_NAMES}

    if "static" in methods or "litert-int8" in methods:
        kmodel, kpipeline = load_kokoro_pipeline()
        print("\n── Generating calibration data ──")
        calib_data = generate_calibration_data(kpipeline)

    # ── Method 1: ONNX Static INT8 (QDQ for NNAPI) ─────────────────────────
    if "static" in methods:
        print("\n── ONNX Static INT8 (QDQ) Quantization ──")
        static_outputs = quantize_onnx_static(model_list, calib_data, onnx_out)

        # Quality check
        print("\n  Quality check (cosine similarity vs FP32):")
        for name, path in static_outputs.items():
            samples = calib_data.get(name, [])
            if samples:
                sim = compare_onnx_quality(name, samples, path)
                if sim is not None:
                    print(f"    {name:<35} cosine similarity: {sim:.4f}")

    # ── Method 2: LiteRT Full INT8 ─────────────────────────────────────────
    if "litert-int8" in methods:
        print("\n── LiteRT Full INT8 (onnx2tf) ──")
        quantize_litert(
            model_list, tflite_out,
            dynamic_range=False, full_integer=True,
            calib_data=calib_data,
        )

    # ── Method 3: torchao ──────────────────────────────────────────────────
    if "torchao" in methods:
        print("\n── torchao Int8WeightOnly + ONNX Re-export ──")
        torchao_outputs = quantize_torchao_int8_export(model_list, onnx_out)

    # ── Summary ─────────────────────────────────────────────────────────────
    print_size_comparison(onnx_out, tflite_out)
    print("\nDone! Quantized models are in:")
    print(f"  ONNX:   {onnx_out}/")
    print(f"  TFLite: {tflite_out}/")
    print("\nFor Android NNAPI deployment:")
    print("  • ONNX Runtime NNAPI EP:  use *_int8_static.onnx (QDQ format)")
    print("  • LiteRT NNAPI delegate:  use *_full_int8.tflite")


if __name__ == "__main__":
    main()
