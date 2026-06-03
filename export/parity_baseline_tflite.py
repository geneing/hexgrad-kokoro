"""
Compare exported Kokoro TFLite submodules against promoted PyTorch baseline tensors.

Uses:
  test_output/baseline/tensors/line_XX/chunk_YY/*.npz

Writes:
  test_output/<git_hash>/baseline_parity/<module>/line_XX_chunk_YY_*.npy
  test_output/<git_hash>/baseline_parity/summary.tsv

Run with:
  uv run python export/parity_baseline_tflite.py
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ai_edge_litert import interpreter


@dataclass(frozen=True)
class Bucket:
    name: str
    size: int


BERT_BUCKETS = [
    Bucket("bert_short", 32),
    Bucket("bert_medium", 128),
    Bucket("bert_long", 256),
    Bucket("bert_max", 510),
]
TEXT_BUCKETS = [
    Bucket("text_encoder_short", 32),
    Bucket("text_encoder_medium", 128),
    Bucket("text_encoder_long", 256),
]
PRED_DUR_BUCKETS = [
    Bucket("predictor_dur_short", 32),
    Bucket("predictor_dur_medium", 128),
    Bucket("predictor_dur_long", 256),
]
F0N_BUCKETS = [
    Bucket("predictor_f0n", 200),
    Bucket("predictor_f0n_long", 800),
]
DECODER_BUCKETS = [
    Bucket("decoder_short", 200),
    Bucket("decoder_medium", 800),
    Bucket("decoder_long", 2000),
]


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


def pick_bucket(length: int, buckets: list[Bucket]) -> Bucket | None:
    for bucket in buckets:
        if length <= bucket.size:
            return bucket
    return None


def pad_axis(array: np.ndarray, axis: int, target: int, value=0) -> np.ndarray:
    if array.shape[axis] > target:
        raise ValueError(f"Cannot pad axis {axis} from {array.shape[axis]} down to {target}")
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, target - array.shape[axis])
    return np.pad(array, pad_width, mode="constant", constant_values=value)


def max_abs_diff(reference: np.ndarray, actual: np.ndarray) -> float:
    if reference.size == 0:
        return 0.0
    return float(np.max(np.abs(reference.astype(np.float32) - actual.astype(np.float32))))


def rms(array: np.ndarray) -> float:
    array = array.astype(np.float32)
    return float(np.sqrt(np.mean(array * array))) if array.size else 0.0


class TFLiteModule:
    def __init__(self, path: Path):
        self.path = path
        self.interpreter = interpreter.Interpreter(model_path=str(path))
        self.signatures = self.interpreter.get_signature_list()

    def run(self, signature_name: str, *args: np.ndarray):
        runner = self.interpreter.get_signature_runner(signature_name)
        kwargs = {f"args_{index}": arg for index, arg in enumerate(args)}
        outputs = runner(**kwargs)
        names = sorted(outputs)
        values = [outputs[name] for name in names]
        return values[0] if len(values) == 1 else tuple(values)


def module_path(outputs_root: Path, current_hash: str, preferred_name: str, fallback: str) -> Path:
    preferred = outputs_root / current_hash / preferred_name
    if preferred.exists():
        return preferred
    return Path(fallback)


def buckets_from_signatures(module: TFLiteModule, prefix: str, fallback: list[Bucket]) -> list[Bucket]:
    buckets = []
    for name in module.signatures:
        marker = f"{prefix}_T"
        if name.startswith(marker):
            try:
                buckets.append(Bucket(name, int(name.removeprefix(marker))))
            except ValueError:
                pass
    return sorted(buckets, key=lambda bucket: bucket.size) if buckets else fallback


def save_array(out_dir: Path, name: str, array: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", array)


def add_result(
    rows: list[dict[str, str]],
    module: str,
    case: str,
    signature: str,
    tensor: str,
    diff: float | None,
    status: str,
    note: str = "",
) -> None:
    rows.append(
        {
            "module": module,
            "case": case,
            "signature": signature,
            "tensor": tensor,
            "max_abs_diff": "" if diff is None else f"{diff:.8g}",
            "status": status,
            "note": note,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline tensor parity against exported TFLite modules.")
    parser.add_argument("--baseline", type=Path, default=Path("test_output/baseline/tensors"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    current_hash = git_hash()
    out_dir = args.out_dir or Path("test_output") / current_hash / "baseline_parity"
    out_dir.mkdir(parents=True, exist_ok=True)

    modules = {
        "bert": TFLiteModule(args.outputs_root / "e7de808" / "kokoro_bert_multisig_fp32.tflite"),
        "text_encoder": TFLiteModule(
            module_path(
                args.outputs_root,
                current_hash,
                "kokoro_text_encoder_baseline_fp32.tflite",
                "outputs/5c9f727/kokoro_text_encoder_multisig_fp32.tflite",
            )
        ),
        "predictor_dur": TFLiteModule(
            module_path(
                args.outputs_root,
                current_hash,
                "kokoro_predictor_dur_baseline_fp32.tflite",
                "outputs/c46f2e1/kokoro_predictor_dur_multisig_fp32.tflite",
            )
        ),
        "predictor_f0n": TFLiteModule(
            module_path(
                args.outputs_root,
                current_hash,
                "kokoro_predictor_f0n_baseline_fp32.tflite",
                "outputs/c46f2e1/kokoro_predictor_f0n_multisig_fp32.tflite",
            )
        ),
        "decoder": TFLiteModule(args.outputs_root / "c46f2e1" / "kokoro_decoder_multisig_fp32.tflite"),
    }
    text_buckets = buckets_from_signatures(modules["text_encoder"], "text_encoder", TEXT_BUCKETS)
    pred_dur_buckets = buckets_from_signatures(modules["predictor_dur"], "predictor_dur", PRED_DUR_BUCKETS)
    f0n_buckets = buckets_from_signatures(modules["predictor_f0n"], "predictor_f0n", F0N_BUCKETS)

    rows: list[dict[str, str]] = []
    chunk_dirs = sorted(args.baseline.glob("line_*/chunk_*"))
    if not chunk_dirs:
        raise FileNotFoundError(f"No baseline chunks found under {args.baseline}")

    for chunk_dir in chunk_dirs:
        case = f"{chunk_dir.parent.name}_{chunk_dir.name}"
        inputs = np.load(chunk_dir / "inputs.npz")
        bert_ref = np.load(chunk_dir / "bert.npz")
        text_ref = np.load(chunk_dir / "text_encoder.npz")
        dur_ref = np.load(chunk_dir / "predictor_dur.npz")
        f0n_ref = np.load(chunk_dir / "predictor_f0n.npz")
        dec_ref = np.load(chunk_dir / "decoder.npz")

        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        text_mask = inputs["text_mask"].astype(bool)
        style_predictor = inputs["style_predictor"].astype(np.float32)
        style_decoder = inputs["style_decoder"].astype(np.float32)

        token_len = input_ids.shape[1]
        aligned_len = dec_ref["asr"].shape[-1]

        # BERT
        bucket = pick_bucket(token_len, BERT_BUCKETS)
        if bucket is None:
            add_result(rows, "bert", case, "", "output", None, "SKIP", f"T={token_len} exceeds max bucket")
        else:
            ids = pad_axis(input_ids, 1, bucket.size).astype(np.int64)
            mask = pad_axis(attention_mask, 1, bucket.size).astype(np.int64)
            output = modules["bert"].run(bucket.name, ids, mask)[:, :, :token_len]
            save_array(out_dir / "bert", f"{case}_{bucket.name}_output_tflite", output)
            diff = max_abs_diff(bert_ref["output"], output)
            add_result(rows, "bert", case, bucket.name, "output", diff, "PASS" if diff < 2e-3 else "FAIL")

        # TextEncoder
        bucket = pick_bucket(token_len, text_buckets)
        if bucket is None:
            add_result(rows, "text_encoder", case, "", "output", None, "SKIP", f"T={token_len} exceeds max bucket")
        else:
            ids = pad_axis(input_ids, 1, bucket.size).astype(np.int64)
            mask = pad_axis(text_mask, 1, bucket.size, value=True).astype(bool)
            output = modules["text_encoder"].run(bucket.name, ids, mask)[:, :, :token_len]
            save_array(out_dir / "text_encoder", f"{case}_{bucket.name}_output_tflite", output)
            diff = max_abs_diff(text_ref["output"], output)
            add_result(rows, "text_encoder", case, bucket.name, "output", diff, "PASS" if diff < 2e-3 else "FAIL")

        # Predictor duration
        bucket = pick_bucket(token_len, pred_dur_buckets)
        if bucket is None:
            add_result(rows, "predictor_dur", case, "", "logits", None, "SKIP", f"T={token_len} exceeds max bucket")
            add_result(rows, "predictor_dur", case, "", "duration_encoded", None, "SKIP", f"T={token_len} exceeds max bucket")
        else:
            d_en = pad_axis(bert_ref["output"], 2, bucket.size).astype(np.float32)
            logits, duration_encoded = modules["predictor_dur"].run(bucket.name, d_en, style_predictor)
            logits = logits[:, :token_len, :]
            duration_encoded = duration_encoded[:, :token_len, :]
            save_array(out_dir / "predictor_dur", f"{case}_{bucket.name}_logits_tflite", logits)
            save_array(out_dir / "predictor_dur", f"{case}_{bucket.name}_duration_encoded_tflite", duration_encoded)
            diff = max_abs_diff(dur_ref["logits"], logits)
            add_result(rows, "predictor_dur", case, bucket.name, "logits", diff, "PASS" if diff < 2e-3 else "FAIL")
            diff = max_abs_diff(dur_ref["duration_encoded"], duration_encoded)
            add_result(rows, "predictor_dur", case, bucket.name, "duration_encoded", diff, "PASS" if diff < 2e-3 else "FAIL")

        # F0/N
        bucket = pick_bucket(aligned_len, f0n_buckets)
        if bucket is None:
            add_result(rows, "predictor_f0n", case, "", "f0", None, "SKIP", f"T_aligned={aligned_len} exceeds max bucket")
            add_result(rows, "predictor_f0n", case, "", "n", None, "SKIP", f"T_aligned={aligned_len} exceeds max bucket")
        else:
            f0n_input = pad_axis(f0n_ref["input"], 2, bucket.size).astype(np.float32)
            f0, n = modules["predictor_f0n"].run(bucket.name, f0n_input, style_predictor)
            f0 = f0[:, : aligned_len * 2]
            n = n[:, : aligned_len * 2]
            save_array(out_dir / "predictor_f0n", f"{case}_{bucket.name}_f0_tflite", f0)
            save_array(out_dir / "predictor_f0n", f"{case}_{bucket.name}_n_tflite", n)
            diff = max_abs_diff(f0n_ref["f0"], f0)
            add_result(rows, "predictor_f0n", case, bucket.name, "f0", diff, "PASS" if diff < 2e-3 else "FAIL")
            diff = max_abs_diff(f0n_ref["n"], n)
            add_result(rows, "predictor_f0n", case, bucket.name, "n", diff, "PASS" if diff < 2e-3 else "FAIL")

        # Decoder
        bucket = pick_bucket(aligned_len, DECODER_BUCKETS)
        if bucket is None:
            add_result(rows, "decoder", case, "", "audio", None, "SKIP", f"T_aligned={aligned_len} exceeds max bucket")
        else:
            asr = pad_axis(dec_ref["asr"], 2, bucket.size).astype(np.float32)
            f0 = pad_axis(f0n_ref["f0"], 1, bucket.size * 2).astype(np.float32)
            n = pad_axis(f0n_ref["n"], 1, bucket.size * 2).astype(np.float32)
            audio = modules["decoder"].run(bucket.name, asr, f0, n, style_decoder)
            audio = audio.reshape(1, -1)[:, : dec_ref["audio"].shape[0]]
            save_array(out_dir / "decoder", f"{case}_{bucket.name}_audio_tflite", audio)
            ref_audio = dec_ref["audio"].reshape(1, -1)
            diff = max_abs_diff(ref_audio, audio)
            ratio = max(rms(ref_audio), rms(audio)) / (min(rms(ref_audio), rms(audio)) + 1e-9)
            status = "PASS" if np.isfinite(audio).all() and ratio < 10.0 else "FAIL"
            add_result(rows, "decoder", case, bucket.name, "audio", diff, status, f"rms_ratio={ratio:.4g}")

    summary_path = out_dir / "summary.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["module", "case", "signature", "tensor", "max_abs_diff", "status", "note"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {summary_path}")
    for row in rows:
        print(
            f"{row['status']:>4} {row['module']:<15} {row['case']:<16} "
            f"{row['signature']:<24} {row['tensor']:<18} {row['max_abs_diff']:<12} {row['note']}"
        )

    failed = [row for row in rows if row["status"] == "FAIL"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
