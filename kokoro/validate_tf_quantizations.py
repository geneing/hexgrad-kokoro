from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf
import torch
from loguru import logger

from .tf_checkpoint_utils import (
    build_feature_from_pair_payload,
    build_tf_generator,
    infer_tf_generator_config,
    load_pytorch_generator_state,
    load_pytorch_state_into_tf_generator,
    save_wav_16bit,
)


@dataclass
class ValItem:
    index: int
    wav_path: Path
    pair_path: Path


class _ServingWrapper(tf.Module):
    def __init__(self, model: tf.keras.Model, in_channels: int):
        super().__init__()
        self.model = model
        self.in_channels = int(in_channels)

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, None], dtype=tf.float32, name="features_bct")])
    def serve(self, features_bct: tf.Tensor) -> tf.Tensor:
        # Enforce channel count at runtime while allowing variable frame lengths.
        with tf.control_dependencies(
            [tf.debugging.assert_equal(tf.shape(features_bct)[1], self.in_channels, message="unexpected channel count")]
        ):
            return self.model(features_bct, training=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Quantize TensorFlow Vocos (fp16/int8) and validate on val set samples")
    parser.add_argument("--pytorch-checkpoint", type=Path, default=Path("output/checkpoints/last.pt"))
    parser.add_argument("--val-filelist", type=Path, default=Path("inputs/filelists/vocos.val.txt"))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--hop-length", type=int, default=300)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--padding", type=str, default="same", choices=["same", "center"])
    parser.add_argument("--quant-output-dir", type=Path, default=Path("output/tf_quantized"))
    parser.add_argument("--validation-output-dir", type=Path, default=Path("output/tf_quant_validation"))
    parser.add_argument("--skip-fp32", action="store_true", help="Skip fp32 baseline generation/metrics")
    return parser.parse_args()


def _read_wav_mono(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sr = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        raw = wav_file.readframes(n_frames)

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483647.0
    else:
        raise ValueError(f"Unsupported sample width {sampwidth}: {path}")
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio, sr


def _derive_pair_path(wav_path: Path) -> Path:
    s = str(wav_path)
    if "/audio/vocoder/audio/" in s:
        return Path(s.replace("/audio/vocoder/audio/", "/audio/vocoder/pairs/")).with_suffix(".pt")
    if "/audio/" in s:
        return Path(s.replace("/audio/", "/pairs/")).with_suffix(".pt")
    raise ValueError(f"Cannot derive pair path from wav path: {wav_path}")


def _load_val_items(val_filelist: Path, num_samples: int) -> List[ValItem]:
    wavs = [Path(x.strip()) for x in val_filelist.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not wavs:
        raise RuntimeError(f"No entries in val filelist: {val_filelist}")
    n = min(len(wavs), max(1, num_samples))
    items: List[ValItem] = []
    for i, wav in enumerate(wavs[:n], start=1):
        pair = _derive_pair_path(wav)
        if not pair.exists():
            raise FileNotFoundError(f"Pair not found for val sample: {pair}")
        if not wav.exists():
            raise FileNotFoundError(f"Val wav missing: {wav}")
        items.append(ValItem(index=i, wav_path=wav, pair_path=pair))
    return items


def _build_tf_model_from_checkpoint(
    pytorch_checkpoint: Path, hop_length: int, padding: str
) -> Tuple[tf.keras.Model, Dict[str, object], Dict[str, object]]:
    state, metadata = load_pytorch_generator_state(pytorch_checkpoint)
    cfg = infer_tf_generator_config(state, hop_length=hop_length, padding=padding)
    model = build_tf_generator(cfg)
    report = load_pytorch_state_into_tf_generator(model, state)
    return model, metadata, {"config": cfg, "report": report}


def _converter_with_select_tf_ops(model: tf.keras.Model, in_channels: int) -> tf.lite.TFLiteConverter:
    wrapper = _ServingWrapper(model=model, in_channels=in_channels)
    concrete = wrapper.serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapper)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    return converter


def _convert_fp32_tflite(model: tf.keras.Model, in_channels: int) -> bytes:
    converter = _converter_with_select_tf_ops(model, in_channels=in_channels)
    return converter.convert()


def _convert_fp16_tflite(model: tf.keras.Model, in_channels: int) -> bytes:
    converter = _converter_with_select_tf_ops(model, in_channels=in_channels)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def _convert_int8_tflite(
    model: tf.keras.Model, in_channels: int, representative_features: Iterable[np.ndarray]
) -> tuple[bytes, str]:
    rep_list = [np.asarray(x, dtype=np.float32) for x in representative_features]
    if not rep_list:
        raise RuntimeError("Representative dataset is empty for int8 conversion")

    def rep_dataset():
        for feat in rep_list:
            yield [feat]

    attempts = []

    def _attempt_full_int8() -> bytes:
        converter = _converter_with_select_tf_ops(model, in_channels=in_channels)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        return converter.convert()

    def _attempt_int8_weights_float_io() -> bytes:
        converter = _converter_with_select_tf_ops(model, in_channels=in_channels)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        return converter.convert()

    def _attempt_dynamic_range() -> bytes:
        converter = _converter_with_select_tf_ops(model, in_channels=in_channels)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter.convert()

    for name, fn in (
        ("full_int8_io", _attempt_full_int8),
        ("int8_weights_float_io", _attempt_int8_weights_float_io),
        ("dynamic_range_fallback", _attempt_dynamic_range),
    ):
        try:
            return fn(), name
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{name}: {type(exc).__name__}: {exc}")
            continue

    raise RuntimeError("All int8 conversion attempts failed:\n" + "\n".join(attempts))


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _prepare_tflite_interpreter(model_path: Path) -> tf.lite.Interpreter:
    # Use default delegate stack so Flex ops required by ISTFT/complex path are available.
    return tf.lite.Interpreter(model_path=str(model_path))


def _run_tflite_infer(interpreter: tf.lite.Interpreter, features_bct: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_idx = input_details["index"]
    output_idx = output_details["index"]

    interpreter.resize_tensor_input(input_idx, list(features_bct.shape), strict=False)
    interpreter.allocate_tensors()

    in_dtype = input_details["dtype"]
    in_scale, in_zero = input_details.get("quantization", (0.0, 0))
    if in_dtype in (np.int8, np.uint8) and in_scale and in_scale > 0:
        qmin, qmax = (np.iinfo(in_dtype).min, np.iinfo(in_dtype).max)
        x = np.round(features_bct / in_scale + in_zero).clip(qmin, qmax).astype(in_dtype)
    else:
        x = features_bct.astype(in_dtype, copy=False)
    interpreter.set_tensor(input_idx, x)
    interpreter.invoke()

    y = interpreter.get_tensor(output_idx)
    out_dtype = output_details["dtype"]
    out_scale, out_zero = output_details.get("quantization", (0.0, 0))
    if out_dtype in (np.int8, np.uint8) and out_scale and out_scale > 0:
        y = (y.astype(np.float32) - out_zero) * out_scale
    else:
        y = y.astype(np.float32, copy=False)
    return np.asarray(y[0], dtype=np.float32)


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    n = min(len(pred), len(target))
    if n <= 0:
        return {"samples": 0.0, "mae": math.nan, "rmse": math.nan, "corr": math.nan}
    p = pred[:n]
    t = target[:n]
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    if float(np.std(p)) > 1e-8 and float(np.std(t)) > 1e-8:
        corr = float(np.corrcoef(p, t)[0, 1])
    else:
        corr = math.nan
    return {"samples": float(n), "mae": mae, "rmse": rmse, "corr": corr}


def _clone_model_from_base(base: tf.keras.Model, in_channels: int, hop_length: int, padding: str) -> tf.keras.Model:
    # Rebuild a fresh graph and copy variables to avoid in-place corruption.
    cfg = {
        "in_channels": in_channels,
        "model_input_channels": int(base.conditioner.layers[0].filters),
        "backbone_dim": int(base.backbone.embed.filters),
        "backbone_intermediate_dim": int(base.backbone.blocks[0].pwconv1.units),
        "backbone_layers": int(len(base.backbone.blocks)),
        "n_fft": int(base.head.n_fft),
        "hop_length": int(hop_length),
        "padding": padding,
    }
    from .tf_vocos import PairedVocosGeneratorTF

    model = PairedVocosGeneratorTF(**cfg)
    _ = model(tf.zeros([1, in_channels, 16], dtype=tf.float32), training=False)
    for src, dst in zip(base.variables, model.variables):
        dst.assign(src)
    return model


def _apply_fp16_weight_quantization(model: tf.keras.Model) -> Dict[str, float]:
    total = 0
    changed = 0
    for v in model.trainable_variables:
        x = v.numpy().astype(np.float32, copy=False)
        q = np.asarray(x.astype(np.float16), dtype=np.float32)
        v.assign(q.astype(tf.as_dtype(v.dtype).as_numpy_dtype, copy=False))
        total += x.size
        changed += int(np.count_nonzero(x != q))
    ratio = float(changed / max(1, total))
    return {"changed_values": float(changed), "total_values": float(total), "changed_ratio": ratio}


def _apply_int8_weight_quantization(model: tf.keras.Model) -> Dict[str, float]:
    total = 0
    changed = 0
    zero_scale = 0
    for v in model.trainable_variables:
        x = v.numpy().astype(np.float32, copy=False)
        max_abs = float(np.max(np.abs(x))) if x.size > 0 else 0.0
        if max_abs < 1e-12:
            qd = x
            zero_scale += 1
        else:
            scale = max_abs / 127.0
            q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
            qd = q.astype(np.float32) * scale
        v.assign(qd.astype(tf.as_dtype(v.dtype).as_numpy_dtype, copy=False))
        total += x.size
        changed += int(np.count_nonzero(x != qd))
    ratio = float(changed / max(1, total))
    return {
        "changed_values": float(changed),
        "total_values": float(total),
        "changed_ratio": ratio,
        "zero_scale_tensors": float(zero_scale),
    }


def main() -> None:
    args = parse_args()
    if not args.pytorch_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.pytorch_checkpoint}")
    if not args.val_filelist.exists():
        raise FileNotFoundError(f"Val filelist not found: {args.val_filelist}")

    tf.get_logger().setLevel("ERROR")
    items = _load_val_items(args.val_filelist, args.num_samples)
    logger.info(f"Loaded {len(items)} validation items from {args.val_filelist}")

    model, metadata, load_info = _build_tf_model_from_checkpoint(
        pytorch_checkpoint=args.pytorch_checkpoint,
        hop_length=args.hop_length,
        padding=args.padding,
    )
    logger.info(
        "Loaded TF generator from PT checkpoint "
        f"(loaded_keys={load_info['report']['num_loaded_keys']}, "
        f"ignored={load_info['report']['ignored_keys']}, meta={metadata})"
    )
    in_channels = int(load_info["config"].in_channels)

    rep_features: List[np.ndarray] = []
    sample_cache: List[tuple[ValItem, np.ndarray]] = []
    for item in items:
        pair = torch.load(item.pair_path, map_location="cpu", weights_only=False)
        feat = build_feature_from_pair_payload(pair).astype(np.float32)
        sample_cache.append((item, feat))
        rep_features.append(feat)

    quant_dir = args.quant_output_dir.resolve()
    quant_dir.mkdir(parents=True, exist_ok=True)
    quant_manifest_path = quant_dir / "quantization_manifest.json"

    model_fp16 = _clone_model_from_base(model, in_channels=in_channels, hop_length=args.hop_length, padding=args.padding)
    model_int8 = _clone_model_from_base(model, in_channels=in_channels, hop_length=args.hop_length, padding=args.padding)
    fp16_stats = _apply_fp16_weight_quantization(model_fp16)
    int8_stats = _apply_int8_weight_quantization(model_int8)
    variant_models = {"fp16": model_fp16, "int8": model_int8}

    quant_manifest = {
        "checkpoint": str(args.pytorch_checkpoint.resolve()),
        "num_calibration_samples": len(rep_features),  # kept for compatibility with previous output schema
        "runtime": "tensorflow_weight_quantized",
        "variants": {
            "fp16": {"mode": "weights_cast_fp16_then_dequant_to_fp32", "stats": fp16_stats},
            "int8": {"mode": "weights_symmetric_int8_quant_dequant", "stats": int8_stats},
        },
        "load_report": {
            "loaded_keys": load_info["report"]["num_loaded_keys"],
            "ignored_keys": load_info["report"]["ignored_keys"],
            "checkpoint_metadata": metadata,
        },
    }
    quant_manifest_path.write_text(json.dumps(quant_manifest, indent=2), encoding="utf-8")
    logger.info(f"Prepared quantized model variants (fp16/int8). Manifest: {quant_manifest_path}")

    out_root = args.validation_output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    variants: List[str] = ["fp16", "int8"]
    if not args.skip_fp32:
        variants = ["fp32"] + variants

    rows: List[Dict[str, object]] = []
    for item, feat in sample_cache:
        target, sr = _read_wav_mono(item.wav_path)
        if sr != args.sample_rate:
            raise RuntimeError(
                f"Unexpected sample rate for target {item.wav_path}: {sr} (expected {args.sample_rate})"
            )
        expected = int(feat.shape[-1] * args.hop_length)

        for variant in variants:
            if variant == "fp32":
                pred = model(tf.convert_to_tensor(feat, dtype=tf.float32), training=False).numpy()[0].astype(np.float32)
            else:
                pred = variant_models[variant](tf.convert_to_tensor(feat, dtype=tf.float32), training=False).numpy()[0]
                pred = pred.astype(np.float32, copy=False)
            pred = pred[:expected]

            variant_dir = out_root / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            out_wav = variant_dir / f"{item.index:02d}_{item.wav_path.stem}_{variant}.wav"
            save_wav_16bit(out_wav, pred, sample_rate=args.sample_rate)

            m = _compute_metrics(pred, target)
            row = {
                "variant": variant,
                "index": item.index,
                "wav_path": str(item.wav_path),
                "pair_path": str(item.pair_path),
                "generated_wav": str(out_wav),
                "samples": int(m["samples"]),
                "seconds": float(m["samples"] / float(args.sample_rate)) if m["samples"] > 0 else 0.0,
                "mae": float(m["mae"]),
                "rmse": float(m["rmse"]),
                "corr": float(m["corr"]),
            }
            rows.append(row)
            logger.info(
                f"{variant} idx={item.index:02d} mae={row['mae']:.6f} "
                f"rmse={row['rmse']:.6f} corr={row['corr']:.6f}"
            )

    csv_path = out_root / "metrics_per_sample.csv"
    fieldnames = ["variant", "index", "wav_path", "pair_path", "generated_wav", "samples", "seconds", "mae", "rmse", "corr"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary: Dict[str, Dict[str, float]] = {}
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        summary[variant] = {
            "count": float(len(subset)),
            "mae_mean": float(np.mean([r["mae"] for r in subset])) if subset else math.nan,
            "rmse_mean": float(np.mean([r["rmse"] for r in subset])) if subset else math.nan,
            "corr_mean": float(np.mean([r["corr"] for r in subset])) if subset else math.nan,
        }
    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "num_samples": len(items),
                "variants": variants,
                "quant_manifest": str(quant_manifest_path),
                "summary": summary,
                "metrics_csv": str(csv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Wrote validation metrics: {csv_path}")
    logger.info(f"Wrote validation summary: {summary_path}")


if __name__ == "__main__":
    main()
