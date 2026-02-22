"""TensorFlow Vocos LiteRT export + QAT tuning + validation pipeline.

Progress log (2026-02-22):
- [done] Add mixed export path: int8 core ops + float-sensitive path from fp16 base model.
- [done] Keep auto-selection quality-first while preferring higher int8 arithmetic coverage.
- [done] Extend validation reports with fp16/int8 vs fp32 closeness metrics.
- [done] Add optional intermediate tensor drift report across fp32/fp16/int8 TFLite models.
- [done] Make auto mode robust to candidate-export failures (skip failed candidate, keep pipeline running).
- [done] Add scoped int8 candidates to reduce remaining float ops (pre conv + optional post FC).
- [done] Add narrow post-FC scoped candidate to test extra int8 coverage with lower quality risk.
- [done] In auto mode, prefer higher int8 coverage among quality-equivalent candidates.

Examples:
1) End-to-end run (QAT + fp32/fp16/int8 export + 20-sample validation)
   uv run python -m kokoro.tf_litert_pipeline \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --data-root inputs \
     --train-filelist inputs/filelists/vocos.train.txt \
     --val-filelist inputs/filelists/vocos.val.txt \
     --output-dir output/tf_litert

2) Export only (skip QAT)
   uv run python -m kokoro.tf_litert_pipeline \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --data-root inputs \
     --train-filelist inputs/filelists/vocos.train.txt \
     --val-filelist inputs/filelists/vocos.val.txt \
     --qat-steps 0 \
     --output-dir output/tf_litert_no_qat

3) QAT-only update and export with custom tuning schedule
   uv run python -m kokoro.tf_litert_pipeline \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --data-root inputs \
     --train-filelist inputs/filelists/vocos.train.txt \
     --val-filelist inputs/filelists/vocos.val.txt \
     --qat-steps 400 --qat-lr 5e-6 --batch-size 2 \
     --output-dir output/tf_litert_qat400

4) Validation generation only using already-exported LiteRT files
   uv run python -m kokoro.tf_litert_pipeline \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --data-root inputs \
     --train-filelist inputs/filelists/vocos.train.txt \
     --val-filelist inputs/filelists/vocos.val.txt \
     --skip-export --skip-qat \
     --output-dir output/tf_litert

5) Force weight-only int8 export (quality-first)
   uv run python -m kokoro.tf_litert_pipeline \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --data-root inputs \
     --train-filelist inputs/filelists/vocos.train.txt \
     --val-filelist inputs/filelists/vocos.val.txt \
     --int8-recipe weight_only \
     --output-dir output/tf_litert_weight_only

6) Force static-core int8 export (NPU-friendly)
   uv run python -m kokoro.tf_litert_pipeline \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --data-root inputs \
     --train-filelist inputs/filelists/vocos.train.txt \
     --val-filelist inputs/filelists/vocos.val.txt \
     --int8-recipe static \
     --output-dir output/tf_litert_static

7) Force mixed-core int8 export (preferred for quality/perf balance)
   uv run python -m kokoro.tf_litert_pipeline \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --data-root inputs \
     --train-filelist inputs/filelists/vocos.train.txt \
     --val-filelist inputs/filelists/vocos.val.txt \
     --int8-recipe mixed \
     --output-dir output/tf_litert_mixed

Notes:
- Fixed-frame export is enforced at 520 frames (input shape [1, 642, 520]).
- Quantization split mirrors PyTorch flow:
  - Float pre blocks: conditioner + embed + input norm
  - Quantized core: ConvNeXt stack
  - Float post blocks: final layer norm + export-safe ISTFT head
- int8 export mode is configurable: `auto`, `mixed`, `static`, `static_plus_postfc`, `static_plus_pre`, `static_plus_pre_postfc`, `static_full`, or `weight_only`.
- `mixed` quantizes core ConvNeXt-heavy ops from an fp16 base model (sensitive blocks stay float path).
- `static` quantizes ConvNeXt core ops only (keeps sensitive blocks float).
- `static_plus_postfc` quantizes core plus post-head FC projections (higher coverage, moderate risk).
- `static_plus_pre` quantizes core plus pre-block convs (extra int8 coverage, quality-sensitive).
- `static_plus_pre_postfc` quantizes core + pre convs + post FC projections (higher int8 coverage).
- `static_full` fully static-quantizes int8 activations (diagnostic mode).
- `auto` exports candidates and chooses by fp32 probe match with int8 arithmetic coverage preference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import tensorflow as tf
import torch
from ai_edge_quantizer import quantizer as aeq_quantizer
from ai_edge_quantizer import recipe as aeq_recipe
from ai_edge_quantizer.utils import tfl_interpreter_utils
from loguru import logger

from .tf_checkpoint_utils import (
    build_tf_export_generator,
    infer_tf_generator_config,
    load_pytorch_generator_state,
    load_pytorch_state_into_tf_generator,
    save_wav_16bit,
)
from .tf_vocos import (
    DEFAULT_FIXED_FRAMES,
    FloatPostBlocksTF,
    FloatPreBlocksTF,
    MultiResolutionGroupDelayLossTF,
    MultiResolutionSTFTLossTF,
    QuantizableVocosCoreTF,
    QuantizedVocosInferenceTF,
)


@dataclass
class ArithmeticOpStats:
    float_ops: int
    int_ops: int

    @property
    def total(self) -> int:
        return int(self.float_ops + self.int_ops)


@dataclass
class ValItem:
    index: int
    wav_path: Path
    pair_path: Path


@dataclass
class TrainItem:
    wav_path: Path
    pair_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export TensorFlow Vocos LiteRT models (fp32/fp16/int8) with QAT tuning")
    parser.add_argument("--pytorch-checkpoint", type=Path, default=Path("output/checkpoints/last.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("inputs"))
    parser.add_argument("--train-filelist", type=Path, default=None)
    parser.add_argument("--val-filelist", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/tf_litert"))

    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=300)
    parser.add_argument("--n-fft", type=int, default=1200)
    parser.add_argument("--fixed-frames", type=int, default=DEFAULT_FIXED_FRAMES)

    parser.add_argument("--qat-steps", type=int, default=200)
    parser.add_argument("--qat-lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-items", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--int8-calib-samples", type=int, default=64)
    parser.add_argument("--num-val-samples", type=int, default=20)
    parser.add_argument(
        "--int8-recipe",
        choices=(
            "auto",
            "mixed",
            "static",
            "static_plus_postfc",
            "static_plus_pre",
            "static_plus_pre_postfc",
            "static_full",
            "weight_only",
        ),
        default="auto",
    )
    parser.add_argument("--dump-intermediates", action="store_true")
    parser.add_argument("--intermediate-max-tensors", type=int, default=192)

    parser.add_argument("--skip-qat", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
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
        raise ValueError(f"Unsupported WAV sample width {sampwidth} for {path}")
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio, sr


def _repeat_pad_1d(x: np.ndarray, target_length: int) -> np.ndarray:
    if len(x) >= target_length:
        return x[:target_length]
    if len(x) == 0:
        return np.zeros(target_length, dtype=np.float32)
    rep = 1 + target_length // len(x)
    return np.tile(x, rep)[:target_length]


def _repeat_pad_2d(x: np.ndarray, target_length: int) -> np.ndarray:
    if x.shape[1] >= target_length:
        return x[:, :target_length]
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], target_length), dtype=np.float32)
    rep = 1 + target_length // x.shape[1]
    return np.tile(x, (1, rep))[:, :target_length]


def _interp_asr_to_frames(asr: np.ndarray, total_frames: int) -> np.ndarray:
    if asr.shape[-1] == total_frames:
        return asr
    src_x = np.linspace(0.0, 1.0, asr.shape[-1], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, total_frames, dtype=np.float32)
    return np.stack([np.interp(dst_x, src_x, ch).astype(np.float32) for ch in asr], axis=0)


def _derive_pair_path(wav_path: Path) -> Path:
    s = str(wav_path)
    if "/audio/vocoder/audio/" in s:
        return Path(s.replace("/audio/vocoder/audio/", "/audio/vocoder/pairs/")).with_suffix(".pt")
    if "/audio/" in s:
        return Path(s.replace("/audio/", "/pairs/")).with_suffix(".pt")
    raise ValueError(f"Cannot derive pair path from wav path: {wav_path}")


def _load_filelist(path: Path) -> list[Path]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [Path(x).resolve() for x in lines]


def _build_train_items(train_filelist: Path, max_items: int, seed: int) -> list[TrainItem]:
    wavs = _load_filelist(train_filelist)
    items: list[TrainItem] = []
    for wav in wavs:
        pair = _derive_pair_path(wav)
        if pair.exists() and wav.exists():
            items.append(TrainItem(wav_path=wav, pair_path=pair))
    if not items:
        raise RuntimeError(f"No train items found from {train_filelist}")
    if max_items > 0 and len(items) > max_items:
        rng = random.Random(seed)
        rng.shuffle(items)
        items = items[:max_items]
    return items


def _build_val_items(val_filelist: Path, num_samples: int) -> list[ValItem]:
    wavs = _load_filelist(val_filelist)
    n = min(len(wavs), max(1, num_samples))
    out: list[ValItem] = []
    for i, wav in enumerate(wavs[:n], start=1):
        pair = _derive_pair_path(wav)
        if not pair.exists():
            raise FileNotFoundError(f"Missing pair file for val sample: {pair}")
        out.append(ValItem(index=i, wav_path=wav, pair_path=pair))
    return out


def _load_fixed_feature_and_target_audio(
    pair_path: Path,
    wav_path: Path,
    fixed_frames: int,
    hop_length: int,
    sample_rate: int,
    start_frame: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    pair = torch.load(pair_path, map_location="cpu", weights_only=False)
    asr = pair["asr"].float().cpu().numpy().astype(np.float32)
    f0 = pair["f0"].float().cpu().numpy().astype(np.float32)
    noise = pair["noise"].float().cpu().numpy().astype(np.float32)
    style = pair["style"].float().cpu().numpy().astype(np.float32)

    total_frames = int(f0.shape[-1])
    if total_frames <= 0:
        raise RuntimeError(f"Empty f0 frames in {pair_path}")
    start = int(max(0, min(start_frame, max(0, total_frames - 1))))
    end = min(total_frames, start + fixed_frames)

    asr = _interp_asr_to_frames(asr, total_frames)
    asr_s = _repeat_pad_2d(asr[:, start:end], fixed_frames)
    f0_s = _repeat_pad_1d(f0[start:end], fixed_frames)[None, :]
    noise_s = _repeat_pad_1d(noise[start:end], fixed_frames)[None, :]
    style_s = np.repeat(style[:, None], fixed_frames, axis=1)
    feat = np.concatenate([asr_s, f0_s, noise_s, style_s], axis=0).astype(np.float32)

    wav, sr = _read_wav_mono(wav_path)
    if sr != sample_rate:
        raise RuntimeError(f"Unexpected sample rate {sr} for {wav_path}; expected {sample_rate}")
    target_len = fixed_frames * hop_length
    s = start * hop_length
    t = _repeat_pad_1d(wav[s : s + target_len], target_len).astype(np.float32)
    return feat, t


class FixedFrameTrainLoader:
    def __init__(
        self,
        items: Sequence[TrainItem],
        fixed_frames: int,
        hop_length: int,
        sample_rate: int,
        batch_size: int,
        seed: int,
    ):
        self.items = list(items)
        self.fixed_frames = int(fixed_frames)
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        self.batch_size = int(batch_size)
        self.rng = random.Random(seed)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        feats: list[np.ndarray] = []
        audio: list[np.ndarray] = []
        for _ in range(self.batch_size):
            item = self.rng.choice(self.items)
            pair = torch.load(item.pair_path, map_location="cpu", weights_only=False)
            total_frames = int(pair["f0"].shape[-1])
            start = 0
            if total_frames > self.fixed_frames:
                start = self.rng.randint(0, total_frames - self.fixed_frames)
            feat, target = _load_fixed_feature_and_target_audio(
                pair_path=item.pair_path,
                wav_path=item.wav_path,
                fixed_frames=self.fixed_frames,
                hop_length=self.hop_length,
                sample_rate=self.sample_rate,
                start_frame=start,
            )
            feats.append(feat)
            audio.append(target)
        return np.stack(feats, axis=0).astype(np.float32), np.stack(audio, axis=0).astype(np.float32)


def _quant_dequant_weight_int8(x: tf.Tensor) -> tf.Tensor:
    x = tf.cast(x, tf.float32)
    max_abs = tf.reduce_max(tf.abs(x))
    scale = tf.where(max_abs > 1e-12, max_abs / 127.0, tf.constant(1.0, dtype=tf.float32))
    q = tf.clip_by_value(tf.round(x / scale), -127.0, 127.0)
    qd = q * scale
    return qd


def _project_core_weights_to_int8(core: tf.keras.Model) -> None:
    for v in core.trainable_variables:
        qd = _quant_dequant_weight_int8(v)
        v.assign(tf.cast(qd, v.dtype))


def run_qat_tuning(
    inference_model: QuantizedVocosInferenceTF,
    core: QuantizableVocosCoreTF,
    train_loader: FixedFrameTrainLoader,
    qat_steps: int,
    qat_lr: float,
    log_every: int,
) -> dict[str, float]:
    if qat_steps <= 0:
        return {"steps": 0.0}

    mrstft_loss = MultiResolutionSTFTLossTF(sample_rate=24000)
    gd_loss = MultiResolutionGroupDelayLossTF()
    opt = tf.keras.optimizers.Adam(learning_rate=qat_lr, beta_1=0.8, beta_2=0.9)

    last = {"loss": 0.0, "mrstft": 0.0, "group_delay": 0.0, "l1": 0.0}
    for step in range(1, qat_steps + 1):
        feats_np, target_np = train_loader.next_batch()
        feats = tf.convert_to_tensor(feats_np, dtype=tf.float32)
        target = tf.convert_to_tensor(target_np, dtype=tf.float32)

        # Quantization-aware projection: keep core weights on int8 manifold during tuning.
        _project_core_weights_to_int8(core)

        with tf.GradientTape() as tape:
            pred = inference_model(feats, training=True)
            n = tf.minimum(tf.shape(pred)[1], tf.shape(target)[1])
            pred = pred[:, :n]
            target_s = target[:, :n]

            l_mrstft = mrstft_loss(target_s, pred)
            l_gd = gd_loss(target_s, pred)
            l_l1 = tf.reduce_mean(tf.abs(pred - target_s))
            loss = 45.0 * l_mrstft + 2.0 * l_gd + 1.0 * l_l1
        grads = tape.gradient(loss, core.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, core.trainable_variables) if g is not None]
        if grads_and_vars:
            opt.apply_gradients(grads_and_vars)

        last = {
            "loss": float(loss.numpy()),
            "mrstft": float(l_mrstft.numpy()),
            "group_delay": float(l_gd.numpy()),
            "l1": float(l_l1.numpy()),
        }
        if step == 1 or step == qat_steps or step % max(1, log_every) == 0:
            logger.info(
                f"QAT step={step}/{qat_steps} total={last['loss']:.4f} "
                f"mrstft={last['mrstft']:.4f} gd={last['group_delay']:.4f} l1={last['l1']:.4f}"
            )

    _project_core_weights_to_int8(core)
    return {"steps": float(qat_steps), **last}


class _FixedServingModule(tf.Module):
    def __init__(self, model: tf.keras.Model, in_channels: int, fixed_frames: int):
        super().__init__()
        self.model = model
        self.in_channels = int(in_channels)
        self.fixed_frames = int(fixed_frames)

        spec = tf.TensorSpec(
            shape=[1, self.in_channels, self.fixed_frames],
            dtype=tf.float32,
            name="features_bct",
        )
        self.serve = tf.function(self._serve_impl, input_signature=[spec])

    def _serve_impl(self, features_bct: tf.Tensor) -> dict[str, tf.Tensor]:
        return {"audio": self.model(features_bct, training=False)}


def _export_fp32_tflite(model: tf.keras.Model, in_channels: int, fixed_frames: int, out_path: Path) -> Path:
    wrapper = _FixedServingModule(model, in_channels=in_channels, fixed_frames=fixed_frames)
    concrete = wrapper.serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapper)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_bytes = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_bytes)
    return out_path


def _export_fp16_tflite(model: tf.keras.Model, in_channels: int, fixed_frames: int, out_path: Path) -> Path:
    wrapper = _FixedServingModule(model, in_channels=in_channels, fixed_frames=fixed_frames)
    concrete = wrapper.serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapper)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_bytes)
    return out_path


def _export_int8_static_full_tflite(
    model: tf.keras.Model,
    in_channels: int,
    fixed_frames: int,
    out_path: Path,
    rep_feats: Sequence[np.ndarray],
) -> tuple[Path, str]:
    wrapper = _FixedServingModule(model, in_channels=in_channels, fixed_frames=fixed_frames)
    concrete = wrapper.serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapper)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    rep = list(rep_feats)
    if not rep:
        raise RuntimeError("Representative dataset is empty for int8 export")

    def representative_dataset() -> Iterable[list[np.ndarray]]:
        for feat in rep:
            yield [feat[None, ...].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    int8_model = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(int8_model)
    return out_path, "tflite_representative_int8_hybrid_static_full"


def _export_int8_scoped_from_tflite(
    source_tflite_path: Path,
    out_path: Path,
    rep_feats: Sequence[np.ndarray],
    source_tag: str,
    static_configs: Sequence[tuple[str, str]],
    recipe_tag: str,
) -> tuple[Path, str]:
    source_model = source_tflite_path.read_bytes()
    qt = aeq_quantizer.Quantizer(source_model)
    for op_regex, op_type in static_configs:
        qt.add_static_config(op_regex, op_type, 8, 8)

    rep = list(rep_feats)
    if not rep:
        raise RuntimeError("Representative dataset is empty for static-core int8 export")

    # AI Edge calibrator requires signatures; these exports currently have none.
    # Patch the signature helpers to use subgraph-0 input/output for single-input models.
    orig_get_sig = tfl_interpreter_utils.get_signature_main_subgraph_index
    orig_invoke_sig = tfl_interpreter_utils.invoke_interpreter_signature

    def _patched_get_signature_main_subgraph_index(interpreter, signature_key=None):
        signatures = interpreter.get_signature_list()
        if not signatures:
            return 0
        return orig_get_sig(interpreter, signature_key)

    def _patched_invoke_interpreter_signature(interpreter, signature_input_data, signature_key=None, quantize_input=True):
        signatures = interpreter.get_signature_list()
        if signatures:
            return orig_invoke_sig(interpreter, signature_input_data, signature_key, quantize_input)
        in_meta = interpreter.get_input_details()[0]
        key = next(iter(signature_input_data.keys()))
        x = np.asarray(signature_input_data[key], dtype=np.float32)
        if in_meta["dtype"] != np.float32:
            x = x.astype(in_meta["dtype"], copy=False)
        interpreter.set_tensor(in_meta["index"], x)
        interpreter.invoke()
        out_meta = interpreter.get_output_details()[0]
        return {out_meta["name"]: interpreter.get_tensor(out_meta["index"])}

    tfl_interpreter_utils.get_signature_main_subgraph_index = _patched_get_signature_main_subgraph_index
    tfl_interpreter_utils.invoke_interpreter_signature = _patched_invoke_interpreter_signature
    try:
        calibration_rows = [{"features_bct": feat[None, ...].astype(np.float32)} for feat in rep]
        calibration_result = qt.calibrate({"default": calibration_rows})
        quant_result = qt.quantize(calibration_result)
    finally:
        tfl_interpreter_utils.get_signature_main_subgraph_index = orig_get_sig
        tfl_interpreter_utils.invoke_interpreter_signature = orig_invoke_sig

    if quant_result.quantized_model is None:
        raise RuntimeError("AI Edge Quantizer did not produce static-core int8 model")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(quant_result.quantized_model))
    return out_path, f"ai_edge_{recipe_tag}_wi8_ai8_from_{source_tag}"


def _export_int8_weight_only_from_fp32(fp32_tflite_path: Path, out_path: Path) -> tuple[Path, str]:
    fp32_model = fp32_tflite_path.read_bytes()
    qt = aeq_quantizer.Quantizer(fp32_model)
    qt.load_quantization_recipe(aeq_recipe.weight_only_wi8_afp32())
    quant_result = qt.quantize()
    if quant_result.quantized_model is None:
        raise RuntimeError("AI Edge Quantizer did not produce weight-only int8 model")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(quant_result.quantized_model))
    return out_path, "ai_edge_weight_only_wi8_afp32"


def _probe_int8_quality(
    fp32_path: Path,
    int8_path: Path,
    probe_feature: np.ndarray,
) -> dict[str, float]:
    y_fp32 = _run_tflite(fp32_path, probe_feature).astype(np.float32)
    y_int8 = _run_tflite(int8_path, probe_feature).astype(np.float32)
    n = min(len(y_fp32), len(y_int8))
    if n <= 0:
        return {"mae": math.inf, "rmse": math.inf, "corr": -1.0}
    y_fp32 = y_fp32[:n]
    y_int8 = y_int8[:n]
    mae = float(np.mean(np.abs(y_fp32 - y_int8)))
    rmse = float(np.sqrt(np.mean((y_fp32 - y_int8) ** 2)))
    corr = float(np.corrcoef(y_fp32, y_int8)[0, 1]) if (np.std(y_fp32) > 1e-8 and np.std(y_int8) > 1e-8) else -1.0
    return {"mae": mae, "rmse": rmse, "corr": corr}


def _score_int8_candidates(
    fp32_path: Path,
    candidates: Mapping[str, Path],
    probe_feature: np.ndarray,
    fixed_frames: int,
) -> dict[str, dict[str, float]]:
    scored: dict[str, dict[str, float]] = {}
    for name, path in candidates.items():
        q = _probe_int8_quality(fp32_path, path, probe_feature)
        ops = estimate_tflite_arithmetic_ops(path, fixed_frames=fixed_frames)
        total = max(1, ops.total)
        int_ratio = float(ops.int_ops) / float(total)
        scored[name] = {
            "mae": float(q["mae"]),
            "rmse": float(q["rmse"]),
            "corr": float(q["corr"]),
            "float_ops": float(ops.float_ops),
            "int_ops": float(ops.int_ops),
            "total_ops": float(ops.total),
            "int_ops_ratio": float(int_ratio),
        }
    return scored


def _select_int8_candidate(scored: Mapping[str, Mapping[str, float]]) -> tuple[str, dict[str, object]]:
    if not scored:
        raise RuntimeError("No scored int8 candidates to select from")
    best_corr = max(float(v["corr"]) for v in scored.values())
    corr_floor = best_corr - 0.005
    shortlist = {k: v for k, v in scored.items() if float(v["corr"]) >= corr_floor}
    if not shortlist:
        shortlist = dict(scored)
    # Candidates in shortlist are already quality-bounded by corr_floor.
    # Within that set, prefer higher int8 coverage.
    ranked = sorted(
        shortlist.items(),
        key=lambda kv: (
            -float(kv[1]["int_ops_ratio"]),
            -float(kv[1]["corr"]),
            float(kv[1]["mae"]),
            float(kv[1]["rmse"]),
        ),
    )
    selected = ranked[0][0]
    decision = {
        "best_corr": float(best_corr),
        "corr_floor": float(corr_floor),
        "shortlist": list(shortlist.keys()),
        "ranking": [
            {
                "name": k,
                "corr": float(v["corr"]),
                "int_ops_ratio": float(v["int_ops_ratio"]),
                "mae": float(v["mae"]),
                "rmse": float(v["rmse"]),
            }
            for k, v in ranked
        ],
    }
    return selected, decision


def _static_scope_configs(scope: str) -> list[tuple[str, str]]:
    core_regex = r".*quantizable_vocos_core_tf.*"
    pre_regex = r".*float_pre_blocks_tf.*"
    post_head_regex = r".*float_post_blocks_tf.*export_safe_istft_head_tf.*"

    cfg: list[tuple[str, str]] = [
        (core_regex, "CONV_2D"),
        (core_regex, "DEPTHWISE_CONV_2D"),
        (core_regex, "FULLY_CONNECTED"),
    ]
    if scope in {"plus_pre", "plus_pre_postfc"}:
        cfg.extend(
            [
                (pre_regex, "CONV_2D"),
                (pre_regex, "FULLY_CONNECTED"),
            ]
        )
    if scope in {"plus_postfc", "plus_pre_postfc"}:
        cfg.extend(
            [
                (post_head_regex, "FULLY_CONNECTED"),
            ]
        )
    return cfg


def _dtype_is_integer(dtype_obj: object) -> bool:
    try:
        return np.dtype(dtype_obj).kind in ("i", "u", "b")
    except Exception:
        return False


def _dtype_is_float(dtype_obj: object) -> bool:
    try:
        return np.dtype(dtype_obj).kind == "f"
    except Exception:
        return False


def _shape_numel(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        if int(d) <= 0:
            return 0
        n *= int(d)
    return int(n)


def _get_tensor_shape(interpreter: object, tensor_idx: int) -> tuple[int, ...]:
    if int(tensor_idx) < 0:
        return ()
    details = interpreter._get_tensor_details(int(tensor_idx), subgraph_index=0)
    return tuple(int(d) for d in details["shape"])


def _estimate_op_arithmetic_count(op: dict[str, object], interpreter: object) -> int:
    op_name = str(op.get("op_name", ""))
    inputs = list(op.get("inputs", []))
    outputs = list(op.get("outputs", []))
    out_shape = _get_tensor_shape(interpreter, outputs[0]) if outputs else ()
    out_numel = _shape_numel(out_shape)

    if op_name in {"RESHAPE", "TRANSPOSE", "SLICE", "PAD", "DEQUANTIZE", "QUANTIZE"}:
        return 0
    if op_name in {"ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM"}:
        return out_numel
    if op_name in {"RSQRT", "EXP", "SIN", "COS"}:
        return out_numel
    if op_name == "MEAN":
        in_shape = _get_tensor_shape(interpreter, inputs[0]) if inputs else ()
        return _shape_numel(in_shape)
    if op_name == "CONV_2D":
        if len(inputs) < 2:
            return 0
        w_shape = _get_tensor_shape(interpreter, inputs[1])  # [O, KH, KW, I]
        if len(w_shape) != 4 or len(out_shape) != 4:
            return 0
        _, kh, kw, cin = w_shape
        _, oh, ow, oc = out_shape
        return int(2 * oh * ow * oc * kh * kw * cin)
    if op_name == "DEPTHWISE_CONV_2D":
        if len(inputs) < 2:
            return 0
        w_shape = _get_tensor_shape(interpreter, inputs[1])
        if len(w_shape) != 4 or len(out_shape) != 4:
            return 0
        _, kh, kw, _ = w_shape
        _, oh, ow, oc = out_shape
        return int(2 * oh * ow * oc * kh * kw)
    if op_name == "TRANSPOSE_CONV":
        if len(inputs) < 3:
            return 0
        in_shape = _get_tensor_shape(interpreter, inputs[2])
        w_shape = _get_tensor_shape(interpreter, inputs[1])  # [KH, KW, OC, IC]
        if len(in_shape) != 4 or len(w_shape) != 4:
            return 0
        n, ih, iw, cin = in_shape
        kh, kw, oc, wcin = w_shape
        cin = min(cin, wcin) if (wcin > 0 and cin > 0) else cin
        return int(2 * n * ih * iw * cin * kh * kw * oc)
    if op_name in {"FULLY_CONNECTED", "BATCH_MATMUL", "MATMUL"}:
        if len(inputs) < 2:
            return 0
        lhs_shape = _get_tensor_shape(interpreter, inputs[0])
        k = int(lhs_shape[-1]) if len(lhs_shape) >= 2 else 0
        return int(2 * out_numel * k) if k > 0 else 0
    return 0


def estimate_tflite_arithmetic_ops(model_path: Path, fixed_frames: int) -> ArithmeticOpStats:
    interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        str(model_path),
        allocate_tensors=False,
        use_xnnpack=False,
        preserve_all_tensors=True,
    )
    for in_detail in interpreter.get_input_details():
        shape = [int(d) for d in in_detail["shape_signature"]]
        if len(shape) >= 1 and shape[0] <= 0:
            shape[0] = 1
        if len(shape) >= 2 and shape[1] <= 0:
            shape[1] = 642
        if len(shape) >= 3 and shape[2] <= 0:
            shape[2] = int(fixed_frames)
        shape = [1 if d <= 0 else int(d) for d in shape]
        interpreter.resize_tensor_input(in_detail["index"], shape, strict=False)
    interpreter.allocate_tensors()

    f_ops = 0
    i_ops = 0
    for op in interpreter._get_ops_details():
        n = _estimate_op_arithmetic_count(op, interpreter)
        if n <= 0:
            continue
        result_types = list(op.get("result_types", []))
        operand_types = list(op.get("operand_types", []))
        dtype = result_types[0] if result_types else (operand_types[0] if operand_types else None)
        if _dtype_is_integer(dtype):
            i_ops += int(n)
        elif _dtype_is_float(dtype):
            f_ops += int(n)
        else:
            f_ops += int(n)
    return ArithmeticOpStats(float_ops=int(f_ops), int_ops=int(i_ops))


def _fmt_int(x: int) -> str:
    return f"{int(x):,}"


def print_ops_table(rows: Mapping[str, ArithmeticOpStats]) -> None:
    print()
    print("| model | float_ops | int_ops | total_arith_ops |")
    print("|---|---:|---:|---:|")
    for k in ("fp32", "fp16", "int8"):
        s = rows[k]
        print(f"| {k} | {_fmt_int(s.float_ops)} | {_fmt_int(s.int_ops)} | {_fmt_int(s.total)} |")
    print()


def _run_tflite_interpreter(model: object, features_bct: np.ndarray) -> np.ndarray:
    sigs = list(model.get_signature_list().keys())
    x = np.asarray(features_bct[None, ...], dtype=np.float32)
    if sigs:
        sig = sigs[0]
        runner = model.get_signature_runner(sig)
        in_details = runner.get_input_details()
        if len(in_details) != 1:
            raise RuntimeError("Expected single input in signature runner")
        in_name = next(iter(in_details.keys()))
        in_meta = in_details[in_name]
        in_dtype = np.dtype(in_meta["dtype"])
        if _dtype_is_integer(in_dtype):
            scale, zp = in_meta.get("quantization", (0.0, 0))
            if float(scale) <= 0.0:
                raise RuntimeError("Invalid input quantization scale")
            q = np.round(x / float(scale) + int(zp))
            q = np.clip(q, np.iinfo(in_dtype).min, np.iinfo(in_dtype).max).astype(in_dtype)
            x = q
        else:
            x = x.astype(in_dtype, copy=False)
        out_map = runner(**{in_name: x})
        out_name = next(iter(out_map.keys()))
        y = np.asarray(out_map[out_name], dtype=np.float32)
    else:
        in_details = model.get_input_details()
        if len(in_details) != 1:
            raise RuntimeError("Expected single input in model input details")
        in_meta = in_details[0]
        in_dtype = np.dtype(in_meta["dtype"])
        if _dtype_is_integer(in_dtype):
            scale, zp = in_meta.get("quantization", (0.0, 0))
            if float(scale) <= 0.0:
                raise RuntimeError("Invalid input quantization scale")
            q = np.round(x / float(scale) + int(zp))
            x = np.clip(q, np.iinfo(in_dtype).min, np.iinfo(in_dtype).max).astype(in_dtype)
        else:
            x = x.astype(in_dtype, copy=False)
        model.set_tensor(in_meta["index"], x)
        model.invoke()
        out_meta = model.get_output_details()[0]
        y = np.asarray(model.get_tensor(out_meta["index"]), dtype=np.float32)
        out_dtype = np.dtype(out_meta["dtype"])
        if _dtype_is_integer(out_dtype):
            scale, zp = out_meta.get("quantization", (0.0, 0))
            if float(scale) > 0.0:
                y = (y - float(zp)) * float(scale)
    return y[0]


def _run_tflite(model_path: Path, features_bct: np.ndarray) -> np.ndarray:
    model = tfl_interpreter_utils.create_tfl_interpreter(str(model_path), allocate_tensors=True, use_xnnpack=False)
    return _run_tflite_interpreter(model, features_bct)


def _collect_tflite_intermediates(
    model_path: Path,
    features_bct: np.ndarray,
    max_tensors: int,
) -> dict[tuple[str, tuple[int, ...]], dict[str, object]]:
    interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        str(model_path),
        allocate_tensors=True,
        use_xnnpack=False,
        preserve_all_tensors=True,
    )
    _ = _run_tflite_interpreter(interpreter, features_bct)

    op_output_indexes: set[int] = set()
    for op in interpreter._get_ops_details():
        for out_idx in list(op.get("outputs", [])):
            if int(out_idx) >= 0:
                op_output_indexes.add(int(out_idx))

    entries: list[dict[str, object]] = []
    for detail in interpreter.get_tensor_details():
        idx = int(detail.get("index", -1))
        if idx < 0 or idx not in op_output_indexes:
            continue
        name = str(detail.get("name", ""))
        shape = tuple(int(d) for d in detail.get("shape", []))
        if _shape_numel(shape) <= 0:
            continue
        try:
            arr = np.asarray(interpreter.get_tensor(idx))
        except Exception:
            continue
        if arr.size <= 0 or not np.issubdtype(arr.dtype, np.number):
            continue
        arrf = arr.astype(np.float32, copy=False).reshape(-1)
        entries.append(
            {
                "key": (name, shape),
                "name": name,
                "shape": shape,
                "dtype": str(arr.dtype),
                "numel": int(arr.size),
                "values": arrf,
            }
        )

    entries.sort(key=lambda e: int(e["numel"]), reverse=True)
    capped = entries[: max(1, int(max_tensors))]
    out: dict[tuple[str, tuple[int, ...]], dict[str, object]] = {}
    for e in capped:
        out[e["key"]] = e
    return out


def _compare_tflite_intermediates(
    ref_map: Mapping[tuple[str, tuple[int, ...]], Mapping[str, object]],
    cand_map: Mapping[tuple[str, tuple[int, ...]], Mapping[str, object]],
    candidate_name: str,
    top_k: int = 40,
) -> dict[str, object]:
    keys = sorted(set(ref_map.keys()) & set(cand_map.keys()))
    per_tensor: list[dict[str, object]] = []
    for key in keys:
        r = np.asarray(ref_map[key]["values"], dtype=np.float32)
        c = np.asarray(cand_map[key]["values"], dtype=np.float32)
        n = min(len(r), len(c))
        if n <= 0:
            continue
        r = r[:n]
        c = c[:n]
        diff = c - r
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        corr = float(np.corrcoef(r, c)[0, 1]) if (np.std(r) > 1e-8 and np.std(c) > 1e-8) else math.nan
        per_tensor.append(
            {
                "tensor_name": str(ref_map[key]["name"]),
                "shape": [int(d) for d in ref_map[key]["shape"]],
                "numel": int(n),
                "reference_dtype": str(ref_map[key]["dtype"]),
                "candidate_dtype": str(cand_map[key]["dtype"]),
                "mae": mae,
                "rmse": rmse,
                "corr": corr,
            }
        )
    per_tensor.sort(key=lambda r: float(r["rmse"]), reverse=True)
    return {
        "candidate": candidate_name,
        "common_tensors": int(len(keys)),
        "compared_tensors": int(len(per_tensor)),
        "top_drift_tensors": per_tensor[: max(1, int(top_k))],
    }


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    n = min(len(pred), len(target))
    if n <= 0:
        return {"samples": 0.0, "mae": math.nan, "rmse": math.nan, "corr": math.nan}
    p = pred[:n]
    t = target[:n]
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    corr = float(np.corrcoef(p, t)[0, 1]) if (np.std(p) > 1e-8 and np.std(t) > 1e-8) else math.nan
    return {"samples": float(n), "mae": mae, "rmse": rmse, "corr": corr}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if int(args.fixed_frames) != DEFAULT_FIXED_FRAMES:
        raise ValueError(
            f"This pipeline is fixed to {DEFAULT_FIXED_FRAMES} frames for LiteRT export; "
            f"received --fixed-frames={args.fixed_frames}"
        )

    if not args.pytorch_checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.pytorch_checkpoint}")

    data_root = args.data_root.resolve()
    train_filelist = args.train_filelist or (data_root / "filelists" / "vocos.train.txt")
    val_filelist = args.val_filelist or (data_root / "filelists" / "vocos.val.txt")
    if not train_filelist.exists():
        raise FileNotFoundError(f"Missing train filelist: {train_filelist}")
    if not val_filelist.exists():
        raise FileNotFoundError(f"Missing val filelist: {val_filelist}")

    state, ckpt_meta = load_pytorch_generator_state(args.pytorch_checkpoint)
    cfg = infer_tf_generator_config(state, hop_length=args.hop_length, padding="same")
    export_gen = build_tf_export_generator(cfg, fixed_frames=args.fixed_frames)
    load_report = load_pytorch_state_into_tf_generator(export_gen, state)
    logger.info(
        "Loaded PT->TF export generator: "
        f"keys={load_report['num_loaded_keys']} ignored={load_report['ignored_keys']} meta={ckpt_meta}"
    )

    pre = FloatPreBlocksTF(export_gen)
    core = QuantizableVocosCoreTF(export_gen)
    post = FloatPostBlocksTF(export_gen)
    inference_model = QuantizedVocosInferenceTF(pre=pre, core=core, post=post)
    _ = inference_model(tf.zeros([1, cfg.in_channels, args.fixed_frames], dtype=tf.float32), training=False)

    # Keep pre/post float (quality-critical), tune and quantize core only.
    pre.trainable = False
    post.trainable = False
    core.trainable = True

    out_dir = args.output_dir.resolve()
    tflite_dir = out_dir / "tflite"
    val_out_dir = out_dir / "validation"
    tflite_dir.mkdir(parents=True, exist_ok=True)
    val_out_dir.mkdir(parents=True, exist_ok=True)

    train_items = _build_train_items(train_filelist, max_items=args.max_train_items, seed=args.seed)
    train_loader = FixedFrameTrainLoader(
        items=train_items,
        fixed_frames=args.fixed_frames,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    qat_stats = {"steps": 0.0}
    if not args.skip_qat and args.qat_steps > 0:
        logger.info("Starting quantization-aware tuning of core blocks")
        qat_stats = run_qat_tuning(
            inference_model=inference_model,
            core=core,
            train_loader=train_loader,
            qat_steps=args.qat_steps,
            qat_lr=args.qat_lr,
            log_every=args.log_every,
        )
        logger.info(f"QAT complete: {qat_stats}")

    fp32_path = tflite_dir / "vocos_fp32.tflite"
    fp16_path = tflite_dir / "vocos_fp16.tflite"
    int8_path = tflite_dir / "vocos_int8.tflite"
    int8_mixed_path = tflite_dir / "vocos_int8_mixed.tflite"
    int8_static_path = tflite_dir / "vocos_int8_static.tflite"
    int8_static_plus_postfc_path = tflite_dir / "vocos_int8_static_plus_postfc.tflite"
    int8_static_plus_pre_path = tflite_dir / "vocos_int8_static_plus_pre.tflite"
    int8_static_plus_pre_postfc_path = tflite_dir / "vocos_int8_static_plus_pre_postfc.tflite"
    int8_static_full_path = tflite_dir / "vocos_int8_static_full.tflite"
    int8_weight_only_path = tflite_dir / "vocos_int8_weight_only.tflite"
    int8_mode = "none"
    int8_probe: dict[str, object] = {}
    if not args.skip_export:
        _export_fp32_tflite(inference_model, in_channels=cfg.in_channels, fixed_frames=args.fixed_frames, out_path=fp32_path)
        _export_fp16_tflite(inference_model, in_channels=cfg.in_channels, fixed_frames=args.fixed_frames, out_path=fp16_path)

        probe_item = _build_val_items(val_filelist, 1)[0]
        probe_feat, _ = _load_fixed_feature_and_target_audio(
            pair_path=probe_item.pair_path,
            wav_path=probe_item.wav_path,
            fixed_frames=args.fixed_frames,
            hop_length=args.hop_length,
            sample_rate=args.sample_rate,
            start_frame=0,
        )

        int8_candidates: dict[str, Path] = {}
        int8_errors: dict[str, str] = {}
        rep_feats: list[np.ndarray] = []
        if args.int8_recipe in {
            "auto",
            "mixed",
            "static",
            "static_plus_postfc",
            "static_plus_pre",
            "static_plus_pre_postfc",
            "static_full",
        }:
            for _ in range(max(1, args.int8_calib_samples)):
                f, _ = train_loader.next_batch()
                rep_feats.append(f[0])

        def _record_candidate(name: str, export_fn, out_path: Path) -> None:
            try:
                export_fn()
                int8_candidates[name] = out_path
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                int8_errors[name] = msg
                if args.int8_recipe == name:
                    raise RuntimeError(f"int8 export failed for recipe={name}: {msg}") from exc
                logger.warning(f"int8 candidate '{name}' failed, skipping: {msg}")

        if args.int8_recipe in {"auto", "mixed"}:
            _record_candidate(
                "mixed",
                lambda: _export_int8_scoped_from_tflite(
                    source_tflite_path=fp16_path,
                    out_path=int8_mixed_path,
                    rep_feats=rep_feats,
                    source_tag="fp16",
                    static_configs=_static_scope_configs("core_only"),
                    recipe_tag="mixed_core",
                ),
                int8_mixed_path,
            )

        if args.int8_recipe in {"auto", "static"}:
            _record_candidate(
                "static",
                lambda: _export_int8_scoped_from_tflite(
                    source_tflite_path=fp32_path,
                    out_path=int8_static_path,
                    rep_feats=rep_feats,
                    source_tag="fp32",
                    static_configs=_static_scope_configs("core_only"),
                    recipe_tag="static_core",
                ),
                int8_static_path,
            )

        if args.int8_recipe in {"auto", "static_plus_postfc"}:
            _record_candidate(
                "static_plus_postfc",
                lambda: _export_int8_scoped_from_tflite(
                    source_tflite_path=fp32_path,
                    out_path=int8_static_plus_postfc_path,
                    rep_feats=rep_feats,
                    source_tag="fp32",
                    static_configs=_static_scope_configs("plus_postfc"),
                    recipe_tag="static_plus_postfc",
                ),
                int8_static_plus_postfc_path,
            )

        if args.int8_recipe in {"auto", "static_plus_pre"}:
            _record_candidate(
                "static_plus_pre",
                lambda: _export_int8_scoped_from_tflite(
                    source_tflite_path=fp32_path,
                    out_path=int8_static_plus_pre_path,
                    rep_feats=rep_feats,
                    source_tag="fp32",
                    static_configs=_static_scope_configs("plus_pre"),
                    recipe_tag="static_plus_pre",
                ),
                int8_static_plus_pre_path,
            )

        if args.int8_recipe in {"auto", "static_plus_pre_postfc"}:
            _record_candidate(
                "static_plus_pre_postfc",
                lambda: _export_int8_scoped_from_tflite(
                    source_tflite_path=fp32_path,
                    out_path=int8_static_plus_pre_postfc_path,
                    rep_feats=rep_feats,
                    source_tag="fp32",
                    static_configs=_static_scope_configs("plus_pre_postfc"),
                    recipe_tag="static_plus_pre_postfc",
                ),
                int8_static_plus_pre_postfc_path,
            )

        if args.int8_recipe in {"auto", "static_full"}:
            _record_candidate(
                "static_full",
                lambda: _export_int8_static_full_tflite(
                    inference_model,
                    in_channels=cfg.in_channels,
                    fixed_frames=args.fixed_frames,
                    out_path=int8_static_full_path,
                    rep_feats=rep_feats,
                ),
                int8_static_full_path,
            )

        if args.int8_recipe in {"auto", "weight_only"}:
            _record_candidate(
                "weight_only",
                lambda: _export_int8_weight_only_from_fp32(fp32_path, int8_weight_only_path),
                int8_weight_only_path,
            )

        if not int8_candidates:
            raise RuntimeError(f"No int8 export candidate for recipe={args.int8_recipe}. errors={int8_errors}")

        if args.int8_recipe == "auto" and len(int8_candidates) > 1:
            scored = _score_int8_candidates(
                fp32_path=fp32_path,
                candidates=int8_candidates,
                probe_feature=probe_feat,
                fixed_frames=args.fixed_frames,
            )
            selected, decision = _select_int8_candidate(scored)
            int8_probe["candidates"] = scored
            int8_probe["selection"] = decision
            int8_path_selected = int8_candidates[selected]
            int8_mode = f"auto:{selected}"
            logger.info(f"int8 auto selection: {decision} selected={selected}")
        else:
            selected = next(iter(int8_candidates.keys()))
            int8_path_selected = int8_candidates[selected]
            int8_mode = selected
            int8_probe["selected_only"] = {
                "name": selected,
                "metrics": _score_int8_candidates(
                    fp32_path=fp32_path,
                    candidates={selected: int8_path_selected},
                    probe_feature=probe_feat,
                    fixed_frames=args.fixed_frames,
                )[selected],
            }

        if int8_path_selected != int8_path:
            shutil.copy2(int8_path_selected, int8_path)
        int8_probe["selected"] = {"mode": int8_mode}
        if int8_errors:
            int8_probe["errors"] = int8_errors
        logger.info(f"Exported LiteRT models to {tflite_dir} (int8_mode={int8_mode})")

    op_rows: dict[str, ArithmeticOpStats] = {}
    for name, path in (("fp32", fp32_path), ("fp16", fp16_path), ("int8", int8_path)):
        if path.exists():
            op_rows[name] = estimate_tflite_arithmetic_ops(path, fixed_frames=args.fixed_frames)
        else:
            op_rows[name] = ArithmeticOpStats(float_ops=0, int_ops=0)
    print_ops_table(op_rows)

    summary_manifest = {
        "checkpoint": str(args.pytorch_checkpoint.resolve()),
        "fixed_frames": int(args.fixed_frames),
        "hop_length": int(args.hop_length),
        "sample_rate": int(args.sample_rate),
        "tflite": {
            "fp32": str(fp32_path),
            "fp16": str(fp16_path),
            "int8": str(int8_path),
            "int8_mixed": str(int8_mixed_path),
            "int8_static": str(int8_static_path),
            "int8_static_plus_postfc": str(int8_static_plus_postfc_path),
            "int8_static_plus_pre": str(int8_static_plus_pre_path),
            "int8_static_plus_pre_postfc": str(int8_static_plus_pre_postfc_path),
            "int8_static_full": str(int8_static_full_path),
            "int8_weight_only": str(int8_weight_only_path),
            "int8_mode": int8_mode,
        },
        "ops": {k: {"float_ops": v.float_ops, "int_ops": v.int_ops, "total": v.total} for k, v in op_rows.items()},
        "qat": qat_stats,
        "int8_probe": int8_probe,
        "load_report": {
            "loaded_keys": load_report["num_loaded_keys"],
            "ignored_keys": load_report["ignored_keys"],
            "checkpoint_meta": ckpt_meta,
        },
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary_manifest, indent=2), encoding="utf-8")

    if args.skip_validation:
        return

    val_items = _build_val_items(val_filelist, args.num_val_samples)
    rows: list[dict[str, object]] = []
    for item in val_items:
        feat, target = _load_fixed_feature_and_target_audio(
            pair_path=item.pair_path,
            wav_path=item.wav_path,
            fixed_frames=args.fixed_frames,
            hop_length=args.hop_length,
            sample_rate=args.sample_rate,
            start_frame=0,
        )
        preds: dict[str, np.ndarray] = {}
        for variant, model_path in (("fp32", fp32_path), ("fp16", fp16_path), ("int8", int8_path)):
            pred = _run_tflite(model_path, feat).astype(np.float32)
            pred = pred[: args.fixed_frames * args.hop_length]
            out_wav = val_out_dir / variant / f"{item.index:02d}_{item.wav_path.stem}_{variant}.wav"
            save_wav_16bit(out_wav, pred, sample_rate=args.sample_rate)
            preds[variant] = pred
            m = _metrics(pred, target)
            if variant == "fp32":
                m_vs_fp32 = {"mae": 0.0, "rmse": 0.0, "corr": 1.0}
            else:
                m_vs_fp32 = _metrics(pred, preds["fp32"])
            rows.append(
                {
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
                    "mae_vs_fp32": float(m_vs_fp32["mae"]),
                    "rmse_vs_fp32": float(m_vs_fp32["rmse"]),
                    "corr_vs_fp32": float(m_vs_fp32["corr"]),
                }
            )
        logger.info(f"Validated sample {item.index}/{len(val_items)}: {item.wav_path.name}")

    csv_path = val_out_dir / "metrics_per_sample.csv"
    fields = [
        "variant",
        "index",
        "wav_path",
        "pair_path",
        "generated_wav",
        "samples",
        "seconds",
        "mae",
        "rmse",
        "corr",
        "mae_vs_fp32",
        "rmse_vs_fp32",
        "corr_vs_fp32",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary: dict[str, dict[str, float]] = {}
    for variant in ("fp32", "fp16", "int8"):
        subset = [r for r in rows if r["variant"] == variant]
        summary[variant] = {
            "count": float(len(subset)),
            "mae_mean": float(np.mean([r["mae"] for r in subset])) if subset else math.nan,
            "rmse_mean": float(np.mean([r["rmse"] for r in subset])) if subset else math.nan,
            "corr_mean": float(np.mean([r["corr"] for r in subset])) if subset else math.nan,
            "mae_vs_fp32_mean": float(np.mean([r["mae_vs_fp32"] for r in subset])) if subset else math.nan,
            "rmse_vs_fp32_mean": float(np.mean([r["rmse_vs_fp32"] for r in subset])) if subset else math.nan,
            "corr_vs_fp32_mean": float(np.mean([r["corr_vs_fp32"] for r in subset])) if subset else math.nan,
        }
    summary_json = {
        "num_samples": len(val_items),
        "variants": ["fp32", "fp16", "int8"],
        "metrics_csv": str(csv_path),
        "summary": summary,
    }
    (val_out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    if args.dump_intermediates and val_items:
        probe_feat, _ = _load_fixed_feature_and_target_audio(
            pair_path=val_items[0].pair_path,
            wav_path=val_items[0].wav_path,
            fixed_frames=args.fixed_frames,
            hop_length=args.hop_length,
            sample_rate=args.sample_rate,
            start_frame=0,
        )
        fp32_map = _collect_tflite_intermediates(fp32_path, probe_feat, max_tensors=args.intermediate_max_tensors)
        fp16_map = _collect_tflite_intermediates(fp16_path, probe_feat, max_tensors=args.intermediate_max_tensors)
        int8_map = _collect_tflite_intermediates(int8_path, probe_feat, max_tensors=args.intermediate_max_tensors)
        intermediate_report = {
            "sample_index": int(val_items[0].index),
            "sample_wav": str(val_items[0].wav_path),
            "max_tensors": int(args.intermediate_max_tensors),
            "fp32_tensor_count": int(len(fp32_map)),
            "fp16_tensor_count": int(len(fp16_map)),
            "int8_tensor_count": int(len(int8_map)),
            "comparisons": {
                "fp16_vs_fp32": _compare_tflite_intermediates(fp32_map, fp16_map, candidate_name="fp16"),
                "int8_vs_fp32": _compare_tflite_intermediates(fp32_map, int8_map, candidate_name="int8"),
            },
        }
        (val_out_dir / "intermediate_drift.json").write_text(json.dumps(intermediate_report, indent=2), encoding="utf-8")

    logger.info(f"Wrote validation outputs to {val_out_dir}")


if __name__ == "__main__":
    main()
