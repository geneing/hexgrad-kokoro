"""tflite_export_clean.py — Convert Kokoro ONNX models to TFLite.

Takes the Android-optimised ONNX models from onnx_android/ (or the legacy
models from onnx2tf_conversion/) and converts each one to a float32 TFLite
flatbuffer via onnx2tf.  The resulting *_float32.tflite files are compatible
with the KokoroTFLiteTTS class in tflite_inference_clean.py and with
LiteRT (TFLite) on Android.

Pipeline stages exported (6 models):
    bert_float32.tflite               S1  BERT phoneme encoder
    duration_predictor_float32.tflite S2  Duration + text feature predictor
    acoustic_expand_float32.tflite    S4  Shared BiLSTM on expanded features
    f0n_predictor_float32.tflite      S5  F0 / noise predictor
    vocoder_conditioner_float32.tflite S6  Vocos conditioner (FastGELU)
    vocoder_stream_chunk_float32.tflite S7 Streaming Vocos backbone (no DFT)

Per-model ONNX source overrides are supported via MODEL_ONNX_OVERRIDES below.
This is useful when the android model fails onnx2tf conversion (e.g. due to
ConvTranspose issues at opset 20) and the legacy opset-19 model should be used
instead.

Run:
    TF_ENABLE_ONEDNN_OPTS=0 uv run python tflite_export_clean.py
    TF_ENABLE_ONEDNN_OPTS=0 uv run python tflite_export_clean.py \\
        --onnx-dir onnx_android --fallback-dir onnx2tf_conversion \\
        --out-dir tflite_android --validate
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_ONNX_DIR     = "onnx_android"
DEFAULT_FALLBACK_DIR = "onnx2tf_conversion"
DEFAULT_OUT_DIR      = "tflite_android"

# ── Model names (stem only — .onnx / _float32.tflite added automatically) ─────
MODEL_NAMES = [
    "bert",
    "duration_predictor",
    "acoustic_expand",
    "f0n_predictor",
    "vocoder_conditioner",
    "vocoder_stream_chunk",
]

# ── Per-model ONNX source override (relative to fallback dir, not primary dir).
# Edit this to force specific models to use the legacy onnx2tf_conversion/ ONNX
# when the android model fails to convert (e.g. ConvTranspose at opset 20).
# Maps model stem → onnx stem in fallback dir (usually the same name).
# Set to {} to always use the primary onnx_dir for all models.
MODEL_FALLBACK_OVERRIDES: dict[str, str] = {
    # f0n_predictor: opset-20 android model has ConvTranspose that onnx2tf
    # can't convert (groups=512 doesn't divide spatial dim=543).  The opset-19
    # model from onnx2tf_conversion/ converts cleanly.
    "f0n_predictor": "f0n_predictor",
}


# ── Conversion ────────────────────────────────────────────────────────────────

def convert_model(
    onnx_path: Path,
    out_dir: Path,
    *,
    verbosity: int = 0,
) -> Path:
    """Convert a single ONNX model to TFLite via onnx2tf.

    Parameters
    ----------
    onnx_path : Path
        Source ``.onnx`` file.
    out_dir : Path
        Directory to write the ``*_float32.tflite`` file.
    verbosity : int
        0 = quiet, 1 = show onnx2tf stdout, 2 = show stdout + stderr.

    Returns
    -------
    tflite_path : Path
        Path to the written ``{stem}_float32.tflite`` file.

    Raises
    ------
    RuntimeError
        If onnx2tf exits with a non-zero status.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = out_dir / f"{onnx_path.stem}_float32.tflite"

    cmd = [
        "uv", "run", "onnx2tf",
        "-i", str(onnx_path.resolve()),
        "-osd",                         # output SavedModel + TFLite flatbuffer
        "-tb", "flatbuffer_direct",     # skip full SavedModel compilation (~seconds, not minutes)
        "-o", str(out_dir.resolve()),
    ]

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=(verbosity == 0),
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        stderr = result.stderr or ""
        raise RuntimeError(
            f"onnx2tf failed for {onnx_path.name} "
            f"(exit {result.returncode}):\n{stderr[-2000:]}"
        )

    if verbosity >= 1 and result.stdout:
        print(result.stdout)
    if verbosity >= 2 and result.stderr:
        print(result.stderr, file=sys.stderr)

    if not tflite_path.exists():
        raise FileNotFoundError(
            f"onnx2tf ran but {tflite_path} was not created. "
            f"Check onnx2tf output."
        )

    size_mb = tflite_path.stat().st_size / 1e6
    print(f"  {tflite_path.name:50s}  {size_mb:6.1f} MB  ({elapsed:.1f}s)")
    return tflite_path


# ── Validation ────────────────────────────────────────────────────────────────

def validate_tflite(tflite_path: Path) -> None:
    """Load the TFLite model and do one forward pass with zero inputs.

    Raises RuntimeError if any input/output has an unexpected dtype, or if
    inference fails.
    """
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(tflite_path), num_threads=1)
    interp.allocate_tensors()

    in_details  = interp.get_input_details()
    out_details = interp.get_output_details()

    for d in in_details:
        data = np.zeros(d["shape"], dtype=d["dtype"])
        # Integer inputs should be non-zero to avoid degenerate behaviour
        if np.issubdtype(d["dtype"], np.integer):
            data = np.ones(d["shape"], dtype=d["dtype"])
        interp.set_tensor(d["index"], data)

    interp.invoke()

    for d in out_details:
        out = interp.get_tensor(d["index"])
        if not np.all(np.isfinite(out)):
            raise RuntimeError(
                f"{tflite_path.name}: output '{d['name']}' contains inf/NaN"
            )

    print(f"  {tflite_path.name:50s}  "
          f"inputs={len(in_details)}  outputs={len(out_details)}  OK")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    onnx_dir: str = DEFAULT_ONNX_DIR,
    out_dir: str  = DEFAULT_OUT_DIR,
    validate: bool = False,
    models: Optional[list[str]] = None,
    verbosity: int = 0,
) -> None:
    src = Path(onnx_dir)
    dst = Path(out_dir)

    model_list = models or MODEL_NAMES

    # ── Check sources ─────────────────────────────────────────────────────────
    missing = [m for m in model_list if not (src / f"{m}.onnx").exists()]
    if missing:
        print("ERROR: the following ONNX models are missing from", src)
        for m in missing:
            print(f"  {m}.onnx")
        print(
            "\nRun  TF_ENABLE_ONEDNN_OPTS=0 uv run python onnx_export_android.py  first."
        )
        sys.exit(1)

    # ── Convert ───────────────────────────────────────────────────────────────
    print(f"Converting {len(model_list)} ONNX model(s): {src!s} → {dst!s}\n")
    tflite_paths: list[Path] = []
    for name in model_list:
        onnx_path = src / f"{name}.onnx"
        print(f"[{name}]")
        tp = convert_model(onnx_path, dst, verbosity=verbosity)
        tflite_paths.append(tp)
    print()

    # ── Validate ─────────────────────────────────────────────────────────────
    if validate:
        print("Validating TFLite models …\n")
        all_ok = True
        for tp in tflite_paths:
            try:
                validate_tflite(tp)
            except Exception as exc:
                print(f"  FAIL  {tp.name}: {exc}")
                all_ok = False
        print()
        if not all_ok:
            sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_mb = sum(p.stat().st_size for p in tflite_paths) / 1e6
    print(f"Output: {dst.resolve()}")
    for p in tflite_paths:
        print(f"  {p.name}")
    print(f"\nTotal: {total_mb:.1f} MB across {len(tflite_paths)} models")
    print(
        "\nTo run TTS inference:\n"
        "  from tflite_inference_clean import KokoroTFLiteTTS\n"
        f"  tts = KokoroTFLiteTTS(saved_model_dir='{dst}')\n"
        "  audio = tts.generate('Hello world.')"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Kokoro Android ONNX models to TFLite (onnx2tf)."
    )
    parser.add_argument(
        "--onnx-dir", default=DEFAULT_ONNX_DIR,
        help=f"Directory containing source .onnx files (default: {DEFAULT_ONNX_DIR})",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help=f"Output directory for *_float32.tflite files (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="After conversion, load each TFLite model and run a zero-input forward pass.",
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=None,
        help="Convert a subset of models only (default: all).",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase onnx2tf output verbosity (-v shows stdout, -vv shows stderr too).",
    )
    args = parser.parse_args()
    main(
        onnx_dir=args.onnx_dir,
        out_dir=args.out_dir,
        validate=args.validate,
        models=args.models,
        verbosity=args.verbose,
    )
