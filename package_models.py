#!/usr/bin/env python3
"""
Package the ONNX and TFLite inference models into zip files for download.

Usage:
    python package_models.py [--out-fp32 kokoro_streaming_vocos_fp32.zip]
                             [--out-fp16 kokoro_streaming_vocos_fp16.zip]

Each zip has the layout:
    onnx/
        bert.onnx
        duration_predictor.onnx
        acoustic_expand.onnx
        f0n_predictor.onnx
        vocoder_conditioner.onnx
        vocoder_stream_chunk.onnx
    tflite/
        bert.tflite
        duration_predictor.tflite
        acoustic_expand.tflite
        f0n_predictor.tflite
        vocoder_conditioner.tflite
        vocoder_stream_chunk.tflite
"""

import argparse
import zipfile
from pathlib import Path

# Pipeline stages in order
STEMS = [
    "bert",
    "duration_predictor",
    "acoustic_expand",
    "f0n_predictor",
    "vocoder_conditioner",
    "vocoder_stream_chunk",
]

ONNX_DIR   = Path("onnx_streaming_vocos")
TFLITE_DIR = Path("onnx_streaming_vocos/tflite")

# Vocoder TFLite was converted via onnx2tf -osd; files live in this subdir
VOCODER_OSD_DIR = TFLITE_DIR / "vocoder_stream_chunk_osd"


def onnx_path(stem: str, fp16: bool) -> Path:
    suffix = "_fp16" if fp16 else ""
    return ONNX_DIR / f"{stem}{suffix}.onnx"


def tflite_path(stem: str, fp16: bool) -> Path:
    if stem == "vocoder_stream_chunk":
        # onnx2tf -osd names output after the ONNX input stem (without _fp16 suffix)
        fname = "vocoder_stream_chunk_float16.tflite" if fp16 else "vocoder_stream_chunk_float32.tflite"
        return VOCODER_OSD_DIR / fname
    if fp16:
        # Use the float16 TFLite emitted by onnx2tf alongside the fp32 conversion
        # (much smaller than converting the _fp16.onnx which inflates with cast ops)
        return TFLITE_DIR / f"{stem}_onnx2tf" / f"{stem}_float16.tflite"
    return TFLITE_DIR / f"{stem}.tflite"


def build_zip(out: Path, fp16: bool) -> None:
    label = "fp16" if fp16 else "fp32"

    missing = []
    for stem in STEMS:
        for p in (onnx_path(stem, fp16), tflite_path(stem, fp16)):
            if not p.exists():
                missing.append(str(p))

    if missing:
        print(f"ERROR: missing {label} files (run the export notebook first):")
        for m in missing:
            print(f"  {m}")
        raise SystemExit(1)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for stem in STEMS:
            zf.write(onnx_path(stem, fp16),   arcname=f"onnx/{stem}.onnx")
            zf.write(tflite_path(stem, fp16), arcname=f"tflite/{stem}.tflite")

        print(f"\n{label.upper()} → {out}")
        print(f"  {'File':<53}  {'Size':>8}")
        print("  " + "-" * 64)
        for info in zf.infolist():
            print(f"  {info.filename:<53}  {info.file_size / 1024 / 1024:>6.1f} MB")

    total_mb = out.stat().st_size / 1024 / 1024
    print(f"  Compressed: {total_mb:.1f} MB  →  {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-fp32", default="kokoro_streaming_vocos_fp32.zip",
                        help="fp32 output zip (default: kokoro_streaming_vocos_fp32.zip)")
    parser.add_argument("--out-fp16", default="kokoro_streaming_vocos_fp16.zip",
                        help="fp16 output zip (default: kokoro_streaming_vocos_fp16.zip)")
    args = parser.parse_args()

    build_zip(Path(args.out_fp32), fp16=False)
    build_zip(Path(args.out_fp16), fp16=True)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
