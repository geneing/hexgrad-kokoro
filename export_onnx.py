#!/usr/bin/env python3
"""export_onnx.py — Export Kokoro TTS ONNX models to export_models/onnx/

Produces three precision variants for each of the 6 pipeline stages:

  bert_fp32.onnx                      float32  (24 MB)
  bert_fp16.onnx                      float16  (12 MB)
  bert_int8_static.onnx               INT8 QDQ (6.9 MB)

  duration_predictor_fp32.onnx        float32  (52 MB)
  duration_predictor_fp16.onnx        float16  (26 MB)
  duration_predictor_int8_static.onnx INT8 QDQ (39 MB)

  acoustic_expand_fp32.onnx           float32  (7.1 MB)
  acoustic_expand_fp16.onnx           float16  (3.6 MB)
  acoustic_expand_int8_static.onnx    INT8 QDQ (7.1 MB)

  f0n_predictor_fp32.onnx             float32  (26 MB)
  f0n_predictor_fp16.onnx             float16  (13 MB)
  f0n_predictor_int8_static.onnx      INT8 QDQ (23 MB)

  vocoder_conditioner_fp32.onnx       float32  (1.1 MB)
  vocoder_conditioner_fp16.onnx       float16  (0.5 MB)
  vocoder_conditioner_int8_static.onnx INT8 QDQ (0.3 MB)

  vocoder_stream_chunk_fp32.onnx      float32  (31 MB)
  vocoder_stream_chunk_fp16.onnx      float16  (16 MB)
  vocoder_stream_chunk_int8_static.onnx INT8 QDQ (8.1 MB)

INT8 models use QDQ (Quantize-Dequantize) format, calibrated with real TTS audio
features. They are compatible with the ONNX Runtime NNAPI execution provider on Android.

Source for FP32/FP16: onnx_streaming_vocos/  (exports from streaming_vocos_export.ipynb
with FastGELU activation and VocosStreamChunkReal — real-matmul IDFT, no external IRFFT).

Usage
-----
    TF_ENABLE_ONEDNN_OPTS=0 uv run python export_onnx.py
    TF_ENABLE_ONEDNN_OPTS=0 uv run python export_onnx.py --precision fp32 fp16
    TF_ENABLE_ONEDNN_OPTS=0 uv run python export_onnx.py --models bert --force
    TF_ENABLE_ONEDNN_OPTS=0 uv run python export_onnx.py --precision int8
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Source: ONNX exports from streaming_vocos_export.ipynb (VocosStreamChunkReal — internal DFT)
ONNX_SRC_DIR = Path("onnx_streaming_vocos")

# Destination
OUTPUT_DIR = Path("export_models/onnx")

ALL_MODELS = [
    "bert",
    "duration_predictor",
    "acoustic_expand",
    "f0n_predictor",
    "vocoder_conditioner",
    "vocoder_stream_chunk",
]

ALL_PRECISIONS = ["fp32", "fp16", "int8"]

# Suffix mapping in onnx_streaming_vocos/ (source filenames)
SRC_SUFFIX: Dict[str, str] = {
    "fp32": "",          # e.g. bert.onnx
    "fp16": "_fp16",     # e.g. bert_fp16.onnx
    "int8": "_int8_static",  # e.g. bert_int8_static.onnx
}

# Destination filename suffix
DST_SUFFIX: Dict[str, str] = {
    "fp32": "_fp32",
    "fp16": "_fp16",
    "int8": "_int8_static",
}


# ---------------------------------------------------------------------------
# FP32 / FP16  →  copy from onnx_streaming_vocos/
# ---------------------------------------------------------------------------

def _copy_model(
    model_name: str,
    precision: str,
    output_dir: Path,
    force: bool,
    src_dir: Path = None,
) -> Optional[Path]:
    """Copy a pre-exported model from onnx_streaming_vocos/ to output_dir."""
    if src_dir is None:
        src_dir = ONNX_SRC_DIR
    src = src_dir / f"{model_name}{SRC_SUFFIX[precision]}.onnx"
    dst = output_dir / f"{model_name}{DST_SUFFIX[precision]}.onnx"

    if not src.exists():
        print(f"  [missing] {src} — skipping")
        return None

    if dst.exists() and not force:
        size = dst.stat().st_size / 1e6
        print(f"  [exists]  {dst.name}  ({size:.1f} MB)")
        return dst

    shutil.copy2(src, dst)
    # Also copy the external-data sidecar file if present (e.g. *.onnx.data)
    src_data = src_dir / f"{model_name}{SRC_SUFFIX[precision]}.onnx.data"
    if src_data.exists():
        dst_data = output_dir / f"{model_name}{DST_SUFFIX[precision]}.onnx.data"
        shutil.copy2(src_data, dst_data)
    size = dst.stat().st_size / 1e6
    print(f"  [copied]  {dst.name}  ({size:.1f} MB)")
    return dst


# ---------------------------------------------------------------------------
# INT8 static  —  generate via quantize_int8.py helpers
# ---------------------------------------------------------------------------

def _build_int8_models(
    model_names: List[str],
    output_dir: Path,
    force: bool,
    src_dir: Path = None,
) -> None:
    """Generate INT8 static QDQ models using onnxruntime calibration."""
    if src_dir is None:
        src_dir = ONNX_SRC_DIR
    # We reuse the generation logic from quantize_int8.py so all calibration
    # sentence handling, pipeline setup, and ONNX session management is shared.
    import quantize_int8 as q8

    # Check which models actually need INT8 generation
    needed = []
    for name in model_names:
        dst = output_dir / f"{name}_int8_static.onnx"
        if dst.exists() and not force:
            size = dst.stat().st_size / 1e6
            print(f"  [exists]  {dst.name}  ({size:.1f} MB)")
        else:
            # Also verify the FP32 source exists in onnx_streaming_vocos/ (needed for calibration)
            src = src_dir / f"{name}.onnx"
            if not src.exists():
                print(f"  [missing] {q8.ONNX_DIR / name}.onnx source — skipping INT8 for {name}")
            else:
                needed.append(name)

    if not needed:
        return

    print(f"\n  Generating INT8 static models for: {', '.join(needed)}")

    # Temporarily point quantize_int8.py at the correct source directory
    original_onnx_dir = q8.ONNX_DIR
    q8.ONNX_DIR = src_dir

    try:
        _, kpipeline = q8.load_kokoro_pipeline()
        calib_data = q8.generate_calibration_data(kpipeline)
        # quantize_onnx_static writes to output_dir directly
        q8.quantize_onnx_static(needed, calib_data, output_dir)
    finally:
        q8.ONNX_DIR = original_onnx_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Kokoro TTS ONNX models to export_models/onnx/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
        metavar="MODEL",
        help="Models to export.",
    )
    parser.add_argument(
        "--precision",
        nargs="+",
        default=ALL_PRECISIONS,
        choices=ALL_PRECISIONS,
        metavar="PREC",
        help="Precisions to export: fp32, fp16, int8.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ONNX_SRC_DIR,
        help="Source directory containing pre-exported onnx_streaming_vocos/ models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export even if output files already exist.",
    )
    args = parser.parse_args()

    src_dir = args.source
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print(
            f"ERROR: Source directory '{src_dir}' not found.\n"
            "Run the ONNX export notebook (streaming_vocos_export.ipynb) first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Exporting to {output_dir}/")
    print(f"Models:     {', '.join(args.models)}")
    print(f"Precisions: {', '.join(args.precision)}")
    print()

    t_total = time.perf_counter()

    # ── FP32 and FP16 (copy from onnx_streaming_vocos/) ───────────────────────
    for prec in ("fp32", "fp16"):
        if prec not in args.precision:
            continue
        label = "Float32" if prec == "fp32" else "Float16"
        print(f"── {label} (copy from {src_dir}/) ──")
        for name in args.models:
            _copy_model(name, prec, output_dir, args.force, src_dir)
        print()

    # ── INT8 static (calibrated QDQ) ─────────────────────────────────────────
    if "int8" in args.precision:
        print("── INT8 Static QDQ (onnxruntime calibration) ──")
        _build_int8_models(args.models, output_dir, args.force, src_dir)
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total
    files = sorted(output_dir.glob("*.onnx"))
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"── Summary ──")
    print(f"  {len(files)} models in {output_dir}/  ({total_mb:.0f} MB total)  [{elapsed:.0f}s]")
    for f in files:
        print(f"  {f.stat().st_size/1e6:6.1f} MB  {f.name}")


if __name__ == "__main__":
    main()
