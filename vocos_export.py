"""Export Kokoro Vocos inference weights to LiteRT and validate outputs.

This script consumes a `third_party/vocos/train_kokoro_decoder.py` checkpoint
or prepared generator weights (typically produced by `prepare_weights.py`) and
exports LiteRT variants:
- fp32
- fp16
- int8 (AI Edge Quantizer; full-integer static calibration preferred)

It then runs quick validation inference on sample feature tensors and writes
WAV artifacts for qualitative sanity checking.

Inputs:
- `--checkpoint models/vocos/last.pt` by default, or
- `vocos.pt` and `vocos_fp16.pt` in `--weights-dir`

Outputs in `--output-dir`:
- `vocos_fp32_litert.tflite`
- `vocos_fp16_litert.tflite`
- `vocos_int8_full_integer_litert.tflite` (if quantization succeeds)
- `vocos_int8_litert.tflite` (selected int8 artifact used for validation)
- `sample_audio/*.wav` generated validation clips

Examples:

1) Export all variants from the trained checkpoint
   uv run python vocos_export.py \
     --checkpoint models/vocos/last.pt \
     --output-dir output/litert_vocos

2) Export all variants from prepared weights
   uv run python vocos_export.py \
     --weights-dir output/saved_infer_weights \
     --output-dir output/litert_vocos

3) Export with 520-frame fixed input and extra calibration samples
   uv run python vocos_export.py \
     --checkpoint models/vocos/last.pt \
     --output-dir output/litert_vocos_520f \
     --num-frames 520 \
     --int8-calib-samples 64

4) Lightweight conversion mode
   uv run python vocos_export.py \
     --checkpoint models/vocos/last.pt \
     --output-dir output/litert_vocos_light \
     --lightweight-conversion

5) Export and run Pixel/Android ARM GPU delegate benchmark
   uv run python vocos_export.py \
     --checkpoint models/vocos/last.pt \
     --output-dir output/litert_vocos_pixel10 \
     --android-gpu-test \
     --android-benchmark-model-apk path/to/android_aarch64_benchmark_model.apk \
     --android-model-variant fp16
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shlex
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence
import glob

import numpy as np
import torch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
logging.getLogger("torchao").setLevel(logging.ERROR)

import litert_torch
from ai_edge_quantizer import quantizer as aeq_quantizer
from ai_edge_quantizer import recipe as aeq_recipe
from ai_edge_quantizer.utils import tfl_interpreter_utils
from loguru import logger
from torch import nn

from prepare_weights import (
    GeneratorConfig,
    InferenceSample,
    build_generator,
    build_inference_samples,
    build_train_loader,
    infer_generator_config,
    load_checkpoint,
)
from vocos.spectral_ops import ISTFT


@dataclass
class ExportedVariant:
    name: str
    litert_path: Path


@dataclass
class ArithmeticOpStats:
    float_ops: int
    int_ops: int

    @property
    def total(self) -> int:
        return int(self.float_ops + self.int_ops)


@dataclass
class AndroidGpuBenchmarkResult:
    variant: str
    model_path: Path
    device_info_path: Path
    gpu_compile_log_path: Path
    gpu_benchmark_log_path: Path
    cpu_benchmark_log_path: Path | None


class ExportSafeISTFT(nn.Module):
    """ISTFT forward variant without data-dependent Python asserts (export-safe)."""

    def __init__(self, src: ISTFT):
        super().__init__()
        self.padding = src.padding
        self.n_fft = src.n_fft
        self.hop_length = src.hop_length
        self.win_length = src.win_length
        self.register_buffer("window", src.window.detach().clone())

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.padding == "center":
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        if self.padding != "same":
            raise ValueError("Padding must be 'center' or 'same'.")

        pad = (self.win_length - self.hop_length) // 2
        _, _, t_frames = spec.shape

        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        output_size = (t_frames - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        window_sq = self.window.square().expand(1, t_frames, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        return y / window_envelope.clamp_min(1e-11)


class ExportSafeISTFTHead(nn.Module):
    """ISTFTHead variant that avoids complex dtype ops for LiteRT conversion."""

    def __init__(self, src_head: nn.Module):
        super().__init__()
        self.out = src_head.out
        assert isinstance(src_head.istft, ISTFT)
        self.istft = ExportSafeISTFT(src_head.istft)

        n_fft = int(self.istft.n_fft)
        num_bins = n_fft // 2 + 1
        k = torch.arange(1, num_bins - 1, dtype=torch.float32)
        n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(1)
        angle = 2.0 * torch.pi * n * k.unsqueeze(0) / float(n_fft)
        self.register_buffer("_cos_basis", torch.cos(angle))
        self.register_buffer("_sin_basis", torch.sin(angle))
        self.register_buffer("_nyquist_sign", torch.pow(torch.tensor(-1.0), torch.arange(n_fft, dtype=torch.float32)))
        self.register_buffer("_ola_kernel", torch.eye(self.istft.win_length, dtype=torch.float32).unsqueeze(1))

    def _irfft_real(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        # real/imag: [B, F, T], with F = n_fft//2 + 1
        n_fft = int(self.istft.n_fft)
        dc = real[:, 0:1, :]
        nyquist = real[:, -1:, :] * self._nyquist_sign.view(1, n_fft, 1)
        real_mid = real[:, 1:-1, :]
        imag_mid = imag[:, 1:-1, :]
        inner = torch.einsum("bkt,nk->bnt", real_mid, self._cos_basis)
        inner = inner - torch.einsum("bkt,nk->bnt", imag_mid, self._sin_basis)
        return (dc + nyquist + (2.0 * inner)) / float(n_fft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        mag, phase = x.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)

        ifft = self._irfft_real(real, imag)
        window = self.istft.window
        ifft = ifft * window[None, :, None]

        t_frames = ifft.shape[-1]
        output_size = (t_frames - 1) * self.istft.hop_length + self.istft.win_length
        pad = (self.istft.win_length - self.istft.hop_length) // 2 if self.istft.padding == "same" else 0
        ola = torch.nn.functional.conv_transpose1d(
            ifft,
            self._ola_kernel,
            stride=self.istft.hop_length,
            groups=1,
        )
        y = ola[:, 0, pad : (output_size - pad)]

        window_sq = window.square().expand(1, t_frames, -1).transpose(1, 2)
        env_ola = torch.nn.functional.conv_transpose1d(
            window_sq,
            self._ola_kernel,
            stride=self.istft.hop_length,
            groups=1,
        )
        envelope = env_ola[0, 0, pad : (output_size - pad)]
        return y / envelope.clamp_min(1e-11)


def _patch_istft_for_export(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "istft") and isinstance(module.istft, ISTFT):
            module.istft = ExportSafeISTFT(module.istft)
        if hasattr(module, "head") and hasattr(module.head, "istft") and isinstance(module.head.istft, ISTFT):
            module.head = ExportSafeISTFTHead(module.head)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export pre-saved Vocos inference weights to LiteRT and validate by generating WAVs")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/vocos/last.pt"),
        help="Raw checkpoint produced by third_party/vocos/train_kokoro_decoder.py",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Optional directory containing prepared vocos.pt and vocos_fp16.pt; overrides --checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/litert"),
        help="Output directory for LiteRT files and validation WAVs",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=42,
        help="Fixed input frame count for exported LiteRT Vocos model",
    )
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=300)
    parser.add_argument("--padding", type=str, default="same")

    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--sample-max-frames", type=int, default=480)
    parser.add_argument(
        "--input-feature-glob",
        action="append",
        default=["data/af_alloy_0000*_00.pt"],
        help="Glob for Kokoro feature .pt files to use as model inputs before dataset/random fallbacks.",
    )

    parser.add_argument("--data-root", type=Path, default=Path("data/outputs"))
    parser.add_argument("--train-filelist", type=Path, default=None)
    parser.add_argument("--val-filelist", type=Path, default=None)
    parser.add_argument("--max-train-items", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--frame-cap", type=int, default=520)
    parser.add_argument(
        "--int8-recipe",
        type=str,
        choices=("static",),
        default="static",
        help="Int8 conversion mode. Uses full-integer quantization with representative dataset calibration.",
    )
    parser.add_argument(
        "--int8-calib-samples",
        type=int,
        default=32,
        help="Number of representative dataset samples (from prepare_weights pipeline) used for int8 calibration.",
    )

    parser.add_argument(
        "--lightweight-conversion",
        action="store_true",
        help="Use lightweight LiteRT conversion path",
    )
    parser.add_argument(
        "--android-gpu-test",
        action="store_true",
        help="After export, push a selected LiteRT model to an Android phone and test it with the GPU delegate.",
    )
    parser.add_argument(
        "--android-benchmark-model-bin",
        type=Path,
        default=None,
        help="Local android_arm64 TensorFlow Lite benchmark_model binary to push to the phone.",
    )
    parser.add_argument(
        "--android-benchmark-model-apk",
        type=Path,
        default=None,
        help="Local android_aarch64 TensorFlow Lite benchmark_model APK to install and run on the phone.",
    )
    parser.add_argument(
        "--adb",
        type=str,
        default="/mnt/c/Users/genei/Downloads/platform-tools/adb.exe",
        help="adb executable to use for Android GPU testing.",
    )
    parser.add_argument("--adb-serial", type=str, default=None, help="Optional adb device serial for Pixel/Android testing.")
    parser.add_argument(
        "--android-work-dir",
        type=str,
        default="/data/local/tmp/kokoro_vocos_litert",
        help="Writable directory on the Android device for benchmark binary, model, logs, and GPU cache.",
    )
    parser.add_argument(
        "--android-model-variant",
        type=str,
        choices=("fp32", "fp16", "int8"),
        default="fp16",
        help="Exported LiteRT model variant to test with the Android GPU delegate.",
    )
    parser.add_argument("--android-gpu-warmup-runs", type=int, default=1)
    parser.add_argument("--android-gpu-runs", type=int, default=20)
    parser.add_argument("--android-cpu-runs", type=int, default=10)
    parser.add_argument("--android-skip-cpu-baseline", action="store_true")
    parser.add_argument(
        "--android-gpu-extra-flag",
        action="append",
        default=[],
        help="Extra flag to pass to benchmark_model GPU runs, e.g. --android-gpu-extra-flag=--gpu_backend=cl.",
    )
    return parser.parse_args()


def save_wav_16bit(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm16 = (waveform * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def _load_state(path: Path) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected mapping in {path}, got {type(state)}")
    return dict(state)


def _trim_or_pad_features(features: torch.Tensor, target_frames: int) -> torch.Tensor:
    # Feature layout: [channels, frames]
    frames = int(features.shape[-1])
    if frames == target_frames:
        return features
    if frames > target_frames:
        return features[:, :target_frames]
    pad = torch.zeros(features.shape[0], target_frames - frames, dtype=features.dtype)
    return torch.cat([features, pad], dim=-1)


def _feature_paths_from_globs(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(pattern))
    return sorted(dict.fromkeys(p.resolve() for p in paths))


def _compose_features_from_pt(path: Path) -> torch.Tensor:
    row = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(row, Mapping):
        raise TypeError(f"Expected mapping in {path}, got {type(row)}")
    asr = row["asr"].float()
    f0 = row["f0"].float()
    noise = row["noise"].float()
    style = row["style"].float()
    if not all(torch.is_tensor(x) for x in (asr, f0, noise, style)):
        raise TypeError(f"Expected tensor asr/f0/noise/style in {path}")

    total_frames = int(f0.shape[-1])
    if asr.shape[-1] != total_frames:
        asr = torch.nn.functional.interpolate(
            asr.unsqueeze(0),
            size=total_frames,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
    return torch.cat(
        [
            asr[:, :total_frames],
            f0[:total_frames].unsqueeze(0),
            noise[:total_frames].unsqueeze(0),
            style.unsqueeze(-1).expand(style.shape[0], total_frames),
        ],
        dim=0,
    )


def _collect_feature_pt_samples(args: argparse.Namespace, input_channels: int, count: int) -> list[InferenceSample]:
    paths = _feature_paths_from_globs(args.input_feature_glob or [])
    samples: list[InferenceSample] = []
    for path in paths[: max(1, int(count))]:
        try:
            features = _compose_features_from_pt(path)
        except Exception as exc:
            logger.warning(f"Skipping feature input {path}: {exc}")
            continue
        if int(features.shape[0]) != int(input_channels):
            logger.warning(f"Skipping feature input {path}: expected {input_channels} channels, got {features.shape[0]}")
            continue
        samples.append(InferenceSample(tag=path.stem, features=features))
    if samples:
        logger.info(f"Using {len(samples)} feature .pt sample(s) from {args.input_feature_glob}")
    return samples


def _collect_samples(args: argparse.Namespace, input_channels: int) -> list[InferenceSample]:
    feature_samples = _collect_feature_pt_samples(args, input_channels=input_channels, count=args.sample_count)
    if feature_samples:
        return feature_samples

    try:
        loader = build_train_loader(args)
        return build_inference_samples(
            dataset=loader.dataset,
            count=args.sample_count,
            seed=args.seed,
            max_frames=args.sample_max_frames,
        )
    except Exception as exc:
        logger.warning(f"Could not build dataset-backed samples ({exc}); using random fallback samples")

    rng = torch.Generator().manual_seed(args.seed)
    samples: list[InferenceSample] = []
    for i in range(max(1, args.sample_count)):
        features = torch.randn(input_channels, args.num_frames, generator=rng)
        samples.append(InferenceSample(tag=f"{i+1:02d}_random", features=features))
    return samples


def _collect_representative_samples(args: argparse.Namespace, input_channels: int) -> list[InferenceSample]:
    feature_samples = _collect_feature_pt_samples(args, input_channels=input_channels, count=args.int8_calib_samples)
    if feature_samples:
        return feature_samples

    try:
        loader = build_train_loader(args)
        reps = build_inference_samples(
            dataset=loader.dataset,
            count=max(1, args.int8_calib_samples),
            seed=args.seed,
            max_frames=max(args.sample_max_frames, args.num_frames),
        )
    except Exception as exc:
        logger.warning(f"Could not build representative dataset samples ({exc}); using random calibration samples")
        rng = torch.Generator().manual_seed(args.seed)
        reps = [
            InferenceSample(
                tag=f"{i+1:02d}_random_calib",
                features=torch.randn(input_channels, args.num_frames, generator=rng),
            )
            for i in range(max(1, args.int8_calib_samples))
        ]
    if not reps:
        raise RuntimeError(
            "Representative dataset sampling returned no items; cannot run full-integer int8 calibration."
        )
    return reps


def _make_static_calibration_data(
    model_content: bytes,
    samples: list[InferenceSample],
    args: argparse.Namespace,
    fixed_frames: int,
) -> dict[str, list[dict[str, np.ndarray]]]:
    if not samples:
        raise RuntimeError(
            "No representative samples available for full-integer int8 calibration."
        )

    interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        model_content,
        allocate_tensors=False,
        use_xnnpack=False,
    )
    signatures = list(interpreter.get_signature_list().keys())
    if not signatures:
        raise RuntimeError("No TFLite signatures found for static calibration.")
    signature_key = signatures[0]
    input_details = interpreter.get_signature_runner(signature_key).get_input_details()
    if len(input_details) != 1:
        raise RuntimeError(
            "Model has non-single-input signature; representative dataset calibration expects single input."
        )

    input_name = next(iter(input_details.keys()))
    calibration_inputs: list[dict[str, np.ndarray]] = []
    limit = max(1, args.int8_calib_samples)
    for i in range(limit):
        sample = samples[i % len(samples)]
        fixed = _trim_or_pad_features(sample.features.float(), fixed_frames)
        calibration_inputs.append({input_name: fixed.unsqueeze(0).numpy().astype(np.float32)})
    return {signature_key: calibration_inputs}


def _quantize_fp32_tflite_to_int8(
    fp32_tflite_path: Path,
    int8_tflite_path: Path,
    samples: list[InferenceSample],
    args: argparse.Namespace,
    recipe_name: str = "static",
) -> Path:
    fp32_model = fp32_tflite_path.read_bytes()
    qt = aeq_quantizer.Quantizer(fp32_model)
    if recipe_name == "static":
        qt.load_quantization_recipe(aeq_recipe.static_wi8_ai8())
    elif recipe_name == "weight_only":
        qt.load_quantization_recipe(aeq_recipe.weight_only_wi8_afp32())
    else:
        raise ValueError(f"Unsupported int8 quantization recipe: {recipe_name}")

    if qt.need_calibration:
        calibration_data = _make_static_calibration_data(
            model_content=fp32_model,
            samples=samples,
            args=args,
            fixed_frames=args.num_frames,
        )
        calibration_result = qt.calibrate(calibration_data)
        quant_result = qt.quantize(calibration_result)
    else:
        quant_result = qt.quantize()

    if quant_result.quantized_model is None:
        raise RuntimeError("AI Edge Quantizer did not produce an int8 model.")
    int8_tflite_path.write_bytes(bytes(quant_result.quantized_model))
    return int8_tflite_path


def _waveform_rms(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def _load_models(args: argparse.Namespace) -> tuple[GeneratorConfig, nn.Module, nn.Module]:
    if args.weights_dir is not None:
        weights_dir = args.weights_dir.resolve()
        fp32_path = weights_dir / "vocos.pt"
        fp16_path = weights_dir / "vocos_fp16.pt"

        for path in (fp32_path, fp16_path):
            if not path.exists():
                raise FileNotFoundError(path)

        fp32_state = _load_state(fp32_path)
        fp16_state = _load_state(fp16_path)
        config = infer_generator_config(fp32_state, args=args)
        source = str(weights_dir)
    else:
        checkpoint_path = args.checkpoint.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        ckpt, fp32_state = load_checkpoint(checkpoint_path)
        config = infer_generator_config(fp32_state, args=args, ckpt=ckpt)
        fp16_state = {k: v.half() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in fp32_state.items()}
        source = str(checkpoint_path)

    logger.info(
        "Using inferred export config: "
        f"source={source}, backend=third_party/vocos, in_channels={config.in_channels}, "
        f"backbone_dim={config.backbone_dim}, layers={config.backbone_layers}, "
        f"n_fft={config.n_fft}, hop={config.hop_length}"
    )

    fp32_model = build_generator(config, fp32_state).eval()
    _patch_istft_for_export(fp32_model)

    fp16_as_fp32 = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in fp16_state.items()}
    fp16_model = build_generator(config, fp32_state).eval()
    fp16_model.load_state_dict(fp16_as_fp32, strict=True)
    _patch_istft_for_export(fp16_model)

    return config, fp32_model, fp16_model


def _export_litert(
    model: nn.Module,
    sample_arg: torch.Tensor,
    out_path: Path,
    lightweight_conversion: bool,
) -> Path:
    model.eval()
    edge_model = litert_torch.convert(
        model,
        sample_args=(sample_arg,),
        strict_export=False,
        lightweight_conversion=lightweight_conversion,
    )
    edge_model.export(str(out_path))
    return out_path


def _run_litert_inference(model_path: Path, features: np.ndarray) -> np.ndarray:
    model = litert_torch.load(str(model_path))
    interpreter = model._get_interpreter()
    signatures = list(interpreter.get_signature_list().keys())
    if not signatures:
        raise RuntimeError(f"No TFLite signatures found in {model_path}")
    signature_key = signatures[0]
    runner = interpreter.get_signature_runner(signature_key)

    input_details = runner.get_input_details()
    if len(input_details) != 1:
        raise RuntimeError(f"Expected single-input signature in {model_path}, got {len(input_details)} inputs")
    input_name = next(iter(input_details.keys()))
    in_meta = input_details[input_name]
    in_dtype = np.dtype(in_meta["dtype"])

    model_input = np.asarray(features, dtype=np.float32)
    if _dtype_is_integer(in_dtype):
        in_scale, in_zero_point = in_meta.get("quantization", (0.0, 0))
        if float(in_scale) <= 0.0:
            raise RuntimeError(f"Invalid quantization scale for input {input_name} in {model_path}")
        q = np.round(model_input / float(in_scale) + int(in_zero_point))
        q = np.clip(q, np.iinfo(in_dtype).min, np.iinfo(in_dtype).max).astype(in_dtype)
        model_input = q
    else:
        model_input = model_input.astype(in_dtype, copy=False)

    pred_map = runner(**{input_name: model_input})
    if not isinstance(pred_map, Mapping) or not pred_map:
        raise RuntimeError(f"Unexpected LiteRT output type from {model_path}: {type(pred_map)}")
    output_name = next(iter(pred_map.keys()))
    pred = np.asarray(pred_map[output_name])

    output_details = runner.get_output_details()
    if output_name in output_details:
        out_meta = output_details[output_name]
        out_dtype = np.dtype(out_meta["dtype"])
        if _dtype_is_integer(out_dtype):
            out_scale, out_zero_point = out_meta.get("quantization", (0.0, 0))
            if float(out_scale) > 0.0:
                pred = (pred.astype(np.float32) - float(out_zero_point)) * float(out_scale)
            else:
                pred = pred.astype(np.float32)
        else:
            pred = pred.astype(np.float32)
    else:
        pred = pred.astype(np.float32)

    return pred


def _shape_numel(shape: np.ndarray | list[int] | tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        if int(d) <= 0:
            return 0
        n *= int(d)
    return int(n)


def _dtype_is_integer(dtype_obj: object) -> bool:
    try:
        kind = np.dtype(dtype_obj).kind
    except Exception:
        return False
    return kind in ("i", "u", "b")


def _dtype_is_float(dtype_obj: object) -> bool:
    try:
        kind = np.dtype(dtype_obj).kind
    except Exception:
        return False
    return kind == "f"


def _get_tensor_shape_from_idx(interpreter: object, tensor_idx: int) -> tuple[int, ...]:
    if int(tensor_idx) < 0:
        return ()
    try:
        details = interpreter._get_tensor_details(int(tensor_idx), subgraph_index=0)
        return tuple(int(d) for d in details["shape"])
    except Exception:
        return ()


def _estimate_op_arithmetic_count(op: dict[str, object], interpreter: object) -> int:
    op_name = str(op.get("op_name", ""))
    inputs = list(op.get("inputs", []))
    outputs = list(op.get("outputs", []))
    out_shape = _get_tensor_shape_from_idx(interpreter, outputs[0]) if outputs else ()
    out_numel = _shape_numel(out_shape)

    if op_name in {"RESHAPE", "TRANSPOSE", "BROADCAST_TO", "SLICE", "DEQUANTIZE", "QUANTIZE"}:
        return 0
    if op_name in {"ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM"}:
        return out_numel
    if op_name in {"RSQRT", "GELU", "EXP", "SIN", "COS"}:
        return out_numel
    if op_name == "SQUARED_DIFFERENCE":
        return out_numel * 2
    if op_name == "MEAN":
        in_shape = _get_tensor_shape_from_idx(interpreter, inputs[0]) if inputs else ()
        return _shape_numel(in_shape)
    if op_name == "CONV_2D":
        if len(inputs) < 2:
            return 0
        w_shape = _get_tensor_shape_from_idx(interpreter, inputs[1])  # [O, KH, KW, I]
        if len(w_shape) != 4 or len(out_shape) != 4:
            return 0
        _, kh, kw, cin = w_shape
        _, oh, ow, oc = out_shape
        macs = oh * ow * oc * kh * kw * cin
        return int(2 * macs)
    if op_name == "DEPTHWISE_CONV_2D":
        if len(inputs) < 2:
            return 0
        w_shape = _get_tensor_shape_from_idx(interpreter, inputs[1])  # [1, KH, KW, OC]
        if len(w_shape) != 4 or len(out_shape) != 4:
            return 0
        _, kh, kw, _ = w_shape
        _, oh, ow, oc = out_shape
        macs = oh * ow * oc * kh * kw
        return int(2 * macs)
    if op_name == "TRANSPOSE_CONV":
        if len(inputs) < 3:
            return 0
        # TFLite TRANSPOSE_CONV inputs:
        #   0: output_shape, 1: filter [KH, KW, OC, IC], 2: input [N, H, W, IC]
        in_shape = _get_tensor_shape_from_idx(interpreter, inputs[2])
        w_shape = _get_tensor_shape_from_idx(interpreter, inputs[1])
        if len(in_shape) != 4 or len(w_shape) != 4:
            return 0
        n, ih, iw, cin = in_shape
        kh, kw, oc, wcin = w_shape
        if wcin > 0 and cin > 0 and wcin != cin:
            cin = min(cin, wcin)
        macs = n * ih * iw * cin * kh * kw * oc
        return int(2 * macs)
    if op_name == "FULLY_CONNECTED":
        if len(inputs) < 2:
            return 0
        w_shape = _get_tensor_shape_from_idx(interpreter, inputs[1])  # [N, K]
        k = int(w_shape[-1]) if len(w_shape) >= 2 else 0
        return int(2 * out_numel * k) if k > 0 else 0
    if op_name in {"BATCH_MATMUL", "MATMUL"}:
        if len(inputs) < 2:
            return 0
        lhs_shape = _get_tensor_shape_from_idx(interpreter, inputs[0])
        k = int(lhs_shape[-1]) if len(lhs_shape) >= 2 else 0
        return int(2 * out_numel * k) if k > 0 else 0
    return 0


def _estimate_tflite_arithmetic_ops(model_path: Path, num_frames: int) -> ArithmeticOpStats:
    interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        str(model_path),
        allocate_tensors=False,
        use_xnnpack=False,
        preserve_all_tensors=True,
    )
    for input_detail in interpreter.get_input_details():
        shape = [int(d) for d in input_detail["shape_signature"]]
        if len(shape) >= 1 and shape[0] <= 0:
            shape[0] = 1
        if len(shape) >= 3 and shape[-1] <= 0:
            shape[-1] = int(num_frames)
        for i, dim in enumerate(shape):
            if dim <= 0:
                shape[i] = 1
        interpreter.resize_tensor_input(input_detail["index"], shape, strict=False)
    interpreter.allocate_tensors()

    float_ops = 0
    int_ops = 0
    for op in interpreter._get_ops_details():
        op_arith = _estimate_op_arithmetic_count(op, interpreter)
        if op_arith <= 0:
            continue
        result_types = list(op.get("result_types", []))
        operand_types = list(op.get("operand_types", []))
        dtype = result_types[0] if result_types else (operand_types[0] if operand_types else None)
        if _dtype_is_integer(dtype):
            int_ops += int(op_arith)
        elif _dtype_is_float(dtype):
            float_ops += int(op_arith)
        else:
            float_ops += int(op_arith)
    return ArithmeticOpStats(float_ops=int(float_ops), int_ops=int(int_ops))


def _format_int(n: int) -> str:
    return f"{int(n):,}"


def _print_arithmetic_ops_table(rows: Mapping[str, ArithmeticOpStats]) -> None:
    print(flush=True)
    print("Estimated arithmetic ops per inference:", flush=True)
    print("| model | float_ops | int_ops | total_arith_ops |", flush=True)
    print("|---|---:|---:|---:|", flush=True)
    for model_name in ("fp32", "fp16", "int8"):
        stats = rows[model_name]
        print(
            f"| {model_name} | {_format_int(stats.float_ops)} | "
            f"{_format_int(stats.int_ops)} | {_format_int(stats.total)} |",
            flush=True,
        )
    print(flush=True)


def _save_validation_wavs(
    exported: Iterable[ExportedVariant],
    samples: list[InferenceSample],
    output_dir: Path,
    sample_rate: int,
    fixed_frames: int,
) -> None:
    wav_dir = output_dir / "sample_audio"
    wav_dir.mkdir(parents=True, exist_ok=True)
    for old_wav in wav_dir.glob("*.wav"):
        old_wav.unlink()

    for sample in samples:
        fixed = _trim_or_pad_features(sample.features.float(), fixed_frames)
        input_np = fixed.unsqueeze(0).numpy().astype(np.float32)
        for variant in exported:
            pred = _run_litert_inference(variant.litert_path, input_np)
            save_wav_16bit(
                wav_dir / f"{sample.tag}_{variant.name}_litert.wav",
                pred,
                sample_rate,
            )


def _adb_base_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [str(args.adb)]
    if args.adb_serial:
        cmd.extend(["-s", str(args.adb_serial)])
    return cmd


def _run_logged_command(cmd: Sequence[str], log_path: Path | None = None, check: bool = True) -> str:
    proc = subprocess.run(
        list(cmd),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = proc.stdout or ""
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("$ " + " ".join(str(part) for part in cmd) + "\n\n" + output, encoding="utf-8")
    if check and proc.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code {code}: {cmd}\nLog: {log}".format(
                code=proc.returncode,
                cmd=" ".join(str(part) for part in cmd),
                log=log_path if log_path is not None else output[-2000:],
            )
        )
    return output


def _adb_shell(args: argparse.Namespace, shell_cmd: str, log_path: Path | None = None, check: bool = True) -> str:
    return _run_logged_command([*_adb_base_cmd(args), "shell", shell_cmd], log_path=log_path, check=check)


def _adb_push(args: argparse.Namespace, local_path: Path, remote_path: str) -> None:
    _run_logged_command([*_adb_base_cmd(args), "push", str(local_path), remote_path], check=True)


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _collect_android_device_info(args: argparse.Namespace, log_path: Path) -> None:
    info_cmd = (
        "printf 'model='; getprop ro.product.model; "
        "printf 'device='; getprop ro.product.device; "
        "printf 'hardware='; getprop ro.hardware; "
        "printf 'board_platform='; getprop ro.board.platform; "
        "printf 'abi='; getprop ro.product.cpu.abi; "
        "printf 'sdk='; getprop ro.build.version.sdk; "
        "printf 'egl='; dumpsys SurfaceFlinger 2>/dev/null | grep -m 1 GLES || true"
    )
    _adb_shell(args, info_cmd, log_path=log_path, check=True)


def _select_android_model(
    exported_variants: Sequence[ExportedVariant],
    variant_name: str,
) -> ExportedVariant:
    for variant in exported_variants:
        if variant.name == variant_name:
            return variant
    available = ", ".join(v.name for v in exported_variants)
    raise RuntimeError(f"Android GPU requested model variant={variant_name!r}, available variants: {available}")


def _benchmark_flags(
    graph_path: str,
    warmup_runs: int,
    num_runs: int,
    use_gpu: bool,
    extra_gpu_flags: Sequence[str],
    input_value_file: str | None = None,
    input_shape: str | None = None,
) -> list[str]:
    flags = [
        f"--graph={graph_path}",
        f"--warmup_runs={max(0, int(warmup_runs))}",
        f"--num_runs={max(1, int(num_runs))}",
    ]
    if input_value_file:
        flags.append("--input_layer=serving_default_args_0")
        if input_shape:
            flags.append(f"--input_layer_shape={input_shape}")
        flags.append(f"--input_layer_value_files=serving_default_args_0:{input_value_file}")
    if use_gpu:
        flags.extend(
            [
                "--use_gpu=true",
                "--gpu_precision_loss_allowed=true",
            ]
        )
        flags.extend(str(flag) for flag in extra_gpu_flags)
    else:
        flags.extend(["--use_xnnpack=true", "--num_threads=4"])
    return flags


def _run_android_benchmark_apk_once(
    args: argparse.Namespace,
    bench_flags: Sequence[str],
    log_path: Path,
) -> str:
    activity = "org.tensorflow.lite.benchmark/.BenchmarkModelActivity"
    bench_args = " ".join(str(flag) for flag in bench_flags)
    _run_logged_command([*_adb_base_cmd(args), "logcat", "-c"], check=True)
    _adb_shell(
        args,
        _shell_join(["am", "start", "-S", "-n", activity, "--es", "args", bench_args]),
        check=True,
    )
    wait_seconds = 45
    _adb_shell(args, f"sleep {wait_seconds}", check=True)
    output = _run_logged_command([*_adb_base_cmd(args), "logcat", "-d"], check=True)
    filtered = "\n".join(
        line
        for line in output.splitlines()
        if any(token in line.lower() for token in ("tflite", "benchmark", "inference timings", "gpu", "delegate", "error"))
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ adb logcat -c\n"
        "$ adb shell am start ...\n"
        f"args={bench_args}\n\n"
        + (filtered if filtered.strip() else output),
        encoding="utf-8",
    )
    if "inference timings" not in output.lower():
        logger.warning(f"Android benchmark APK log did not include inference timings; inspect {log_path}")
    return output


def _run_android_gpu_test_with_apk(
    args: argparse.Namespace,
    selected: ExportedVariant,
    remote_model: str,
    report_dir: Path,
    device_info_path: Path,
    remote_input_value_file: str | None,
) -> AndroidGpuBenchmarkResult:
    apk_path = args.android_benchmark_model_apk.resolve()
    if not apk_path.exists():
        raise FileNotFoundError(apk_path)

    package_state = _adb_shell(args, "cmd package path org.tensorflow.lite.benchmark", check=False)
    if "package:" in package_state:
        (report_dir / "apk_install.txt").write_text(
            "Skipped install; org.tensorflow.lite.benchmark is already installed.\n" + package_state,
            encoding="utf-8",
        )
    else:
        _run_logged_command(
            [*_adb_base_cmd(args), "install", "-r", "-d", "-g", str(apk_path)],
            log_path=report_dir / "apk_install.txt",
            check=True,
        )

    compile_flags = _benchmark_flags(
        graph_path=remote_model,
        warmup_runs=1,
        num_runs=1,
        use_gpu=True,
        extra_gpu_flags=args.android_gpu_extra_flag,
        input_value_file=remote_input_value_file,
        input_shape=getattr(args, "android_input_shape", None),
    )
    gpu_compile_log_path = report_dir / f"{selected.name}_gpu_compile_apk.txt"
    _run_android_benchmark_apk_once(args, compile_flags, gpu_compile_log_path)

    gpu_flags = _benchmark_flags(
        graph_path=remote_model,
        warmup_runs=args.android_gpu_warmup_runs,
        num_runs=args.android_gpu_runs,
        use_gpu=True,
        extra_gpu_flags=args.android_gpu_extra_flag,
        input_value_file=remote_input_value_file,
        input_shape=getattr(args, "android_input_shape", None),
    )
    gpu_benchmark_log_path = report_dir / f"{selected.name}_gpu_benchmark_apk.txt"
    _run_android_benchmark_apk_once(args, gpu_flags, gpu_benchmark_log_path)

    cpu_benchmark_log_path: Path | None = None
    if not args.android_skip_cpu_baseline:
        cpu_flags = _benchmark_flags(
            graph_path=remote_model,
            warmup_runs=1,
            num_runs=args.android_cpu_runs,
            use_gpu=False,
            extra_gpu_flags=[],
            input_value_file=remote_input_value_file,
            input_shape=getattr(args, "android_input_shape", None),
        )
        cpu_benchmark_log_path = report_dir / f"{selected.name}_cpu_benchmark_apk.txt"
        _run_android_benchmark_apk_once(args, cpu_flags, cpu_benchmark_log_path)

    return AndroidGpuBenchmarkResult(
        variant=selected.name,
        model_path=selected.litert_path,
        device_info_path=device_info_path,
        gpu_compile_log_path=gpu_compile_log_path,
        gpu_benchmark_log_path=gpu_benchmark_log_path,
        cpu_benchmark_log_path=cpu_benchmark_log_path,
    )


def _run_android_gpu_test(
    args: argparse.Namespace,
    exported_variants: Sequence[ExportedVariant],
) -> AndroidGpuBenchmarkResult:
    if args.android_benchmark_model_bin is None and args.android_benchmark_model_apk is None:
        raise ValueError("--android-benchmark-model-bin or --android-benchmark-model-apk is required with --android-gpu-test")

    selected = _select_android_model(exported_variants, args.android_model_variant)
    if not selected.litert_path.exists():
        raise FileNotFoundError(selected.litert_path)

    report_dir = args.output_dir / "android_gpu"
    report_dir.mkdir(parents=True, exist_ok=True)

    remote_dir = str(args.android_work_dir).rstrip("/")
    remote_cache_dir = f"{remote_dir}/gpu_cache"
    remote_bin = f"{remote_dir}/benchmark_model"
    remote_model = f"{remote_dir}/{selected.litert_path.name}"
    remote_input_value_file = None
    local_input_value_file = getattr(args, "android_input_value_file", None)
    if local_input_value_file is not None:
        remote_input_value_file = f"{remote_dir}/{Path(local_input_value_file).name}"

    _run_logged_command([*_adb_base_cmd(args), "get-state"], log_path=report_dir / "adb_get_state.txt", check=True)
    _adb_shell(args, _shell_join(["mkdir", "-p", remote_dir, remote_cache_dir]), check=True)
    _adb_push(args, selected.litert_path, remote_model)
    if local_input_value_file is not None and remote_input_value_file is not None:
        _adb_push(args, Path(local_input_value_file), remote_input_value_file)

    device_info_path = report_dir / "device_info.txt"
    _collect_android_device_info(args, device_info_path)

    if args.android_benchmark_model_apk is not None:
        result = _run_android_gpu_test_with_apk(
            args=args,
            selected=selected,
            remote_model=remote_model,
            report_dir=report_dir,
            device_info_path=device_info_path,
            remote_input_value_file=remote_input_value_file,
        )
        logger.info(
            "Completed Android ARM GPU APK benchmark for "
            f"{selected.name}: compile_log={result.gpu_compile_log_path}, gpu_log={result.gpu_benchmark_log_path}"
        )
        return result

    benchmark_bin = args.android_benchmark_model_bin.resolve()
    if not benchmark_bin.exists():
        raise FileNotFoundError(benchmark_bin)

    _adb_push(args, benchmark_bin, remote_bin)
    _adb_shell(args, _shell_join(["chmod", "755", remote_bin]), check=True)

    compile_flags = _benchmark_flags(
        graph_path=remote_model,
        warmup_runs=1,
        num_runs=1,
        use_gpu=True,
        extra_gpu_flags=args.android_gpu_extra_flag,
        input_value_file=remote_input_value_file,
        input_shape=getattr(args, "android_input_shape", None),
    )
    gpu_compile_log_path = report_dir / f"{selected.name}_gpu_compile.txt"
    compile_output = _adb_shell(args, _shell_join([remote_bin, *compile_flags]), log_path=gpu_compile_log_path, check=True)
    if "gpu" not in compile_output.lower():
        logger.warning(
            "Android GPU compile log does not mention GPU. Inspect log for delegate support: "
            f"{gpu_compile_log_path}"
        )

    gpu_flags = _benchmark_flags(
        graph_path=remote_model,
        warmup_runs=args.android_gpu_warmup_runs,
        num_runs=args.android_gpu_runs,
        use_gpu=True,
        extra_gpu_flags=args.android_gpu_extra_flag,
        input_value_file=remote_input_value_file,
        input_shape=getattr(args, "android_input_shape", None),
    )
    gpu_benchmark_log_path = report_dir / f"{selected.name}_gpu_benchmark.txt"
    _adb_shell(args, _shell_join([remote_bin, *gpu_flags]), log_path=gpu_benchmark_log_path, check=True)

    cpu_benchmark_log_path: Path | None = None
    if not args.android_skip_cpu_baseline:
        cpu_flags = _benchmark_flags(
            graph_path=remote_model,
            warmup_runs=1,
            num_runs=args.android_cpu_runs,
            use_gpu=False,
            extra_gpu_flags=[],
            input_value_file=remote_input_value_file,
            input_shape=getattr(args, "android_input_shape", None),
        )
        cpu_benchmark_log_path = report_dir / f"{selected.name}_cpu_benchmark.txt"
        _adb_shell(args, _shell_join([remote_bin, *cpu_flags]), log_path=cpu_benchmark_log_path, check=True)

    result = AndroidGpuBenchmarkResult(
        variant=selected.name,
        model_path=selected.litert_path,
        device_info_path=device_info_path,
        gpu_compile_log_path=gpu_compile_log_path,
        gpu_benchmark_log_path=gpu_benchmark_log_path,
        cpu_benchmark_log_path=cpu_benchmark_log_path,
    )
    logger.info(
        "Completed Android ARM GPU benchmark for "
        f"{selected.name}: compile_log={gpu_compile_log_path}, gpu_log={gpu_benchmark_log_path}"
    )
    return result


def _write_android_input_value_file(
    sample: InferenceSample,
    output_dir: Path,
    fixed_frames: int,
) -> Path:
    input_dir = output_dir / "android_gpu" / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    fixed = _trim_or_pad_features(sample.features.float(), fixed_frames)
    value_path = input_dir / f"{sample.tag}_{fixed_frames}f_float32.bin"
    fixed.unsqueeze(0).contiguous().numpy().astype(np.float32).tofile(value_path)
    logger.info(f"Saved Android benchmark input value file: {value_path}")
    return value_path


def main() -> None:
    logger.enable("vocos_export")
    args = parse_args()

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.weights_dir is not None:
        args.weights_dir = args.weights_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config, fp32_model, fp16_model = _load_models(args)

    sample_features_fp32 = torch.randn(1, config.in_channels, args.num_frames, dtype=torch.float32)

    export_targets = {
        "fp32": args.output_dir / "vocos_fp32_litert.tflite",
        "fp16": args.output_dir / "vocos_fp16_litert.tflite",
        "int8": args.output_dir / "vocos_int8_litert.tflite",
        "int8_full_integer": args.output_dir / "vocos_int8_full_integer_litert.tflite",
    }
    exported_variants: list[ExportedVariant] = []

    _export_litert(
        model=fp32_model,
        sample_arg=sample_features_fp32,
        out_path=export_targets["fp32"],
        lightweight_conversion=args.lightweight_conversion,
    )
    exported_variants.append(ExportedVariant("fp32", export_targets["fp32"]))
    logger.info(f"Exported LiteRT fp32 model: {export_targets['fp32']}")

    _export_litert(
        model=fp16_model,
        sample_arg=sample_features_fp32,
        out_path=export_targets["fp16"],
        lightweight_conversion=args.lightweight_conversion,
    )
    exported_variants.append(ExportedVariant("fp16", export_targets["fp16"]))
    logger.info(f"Exported LiteRT fp16 model: {export_targets['fp16']}")

    samples = _collect_samples(args, input_channels=config.in_channels)
    if args.android_gpu_test and samples:
        args.android_input_shape = f"1,{config.in_channels},{args.num_frames}"
        args.android_input_value_file = _write_android_input_value_file(
            sample=samples[0],
            output_dir=args.output_dir,
            fixed_frames=args.num_frames,
        )
    try:
        representative_samples = _collect_representative_samples(args, input_channels=config.in_channels)
    except Exception as exc:
        raise RuntimeError(
            "Failed to build representative dataset samples for full-integer int8 calibration. "
            "Check --data-root/--train-filelist/--val-filelist settings."
        ) from exc
    logger.info(
        "Using representative dataset calibration samples from prepare_weights pipeline: "
        f"{len(representative_samples)}"
    )
    try:
        _quantize_fp32_tflite_to_int8(
            fp32_tflite_path=export_targets["fp32"],
            int8_tflite_path=export_targets["int8_full_integer"],
            samples=representative_samples,
            args=args,
            recipe_name="static",
        )
        int8_selected_path = export_targets["int8_full_integer"]
        if samples:
            probe_fixed = _trim_or_pad_features(samples[0].features.float(), args.num_frames)
            probe_audio = _run_litert_inference(int8_selected_path, probe_fixed.unsqueeze(0).numpy().astype(np.float32))
            probe_rms = _waveform_rms(probe_audio)
            if probe_rms < 1e-5:
                logger.warning(
                    "Full-integer int8 model produced near-silent output on probe sample "
                    f"(rms={probe_rms:.3e}). Falling back to weight-only int8 for audible validation WAVs. "
                    f"Full-integer model kept at: {export_targets['int8_full_integer']}"
                )
                _quantize_fp32_tflite_to_int8(
                    fp32_tflite_path=export_targets["fp32"],
                    int8_tflite_path=export_targets["int8"],
                    samples=representative_samples,
                    args=args,
                    recipe_name="weight_only",
                )
                int8_selected_path = export_targets["int8"]
            else:
                shutil.copy2(export_targets["int8_full_integer"], export_targets["int8"])
                int8_selected_path = export_targets["int8"]
        else:
            shutil.copy2(export_targets["int8_full_integer"], export_targets["int8"])
            int8_selected_path = export_targets["int8"]

        exported_variants.append(ExportedVariant("int8", int8_selected_path))
        logger.info(
            "Exported LiteRT int8 model via AI Edge Quantizer "
            f"(requested recipe={args.int8_recipe}, selected={int8_selected_path.name}): {int8_selected_path}"
        )
    except Exception as exc:
        raise RuntimeError(f"LiteRT int8 export failed: {exc}") from exc

    stats_rows = {
        "fp32": _estimate_tflite_arithmetic_ops(export_targets["fp32"], num_frames=args.num_frames),
        "fp16": _estimate_tflite_arithmetic_ops(export_targets["fp16"], num_frames=args.num_frames),
        "int8": _estimate_tflite_arithmetic_ops(exported_variants[-1].litert_path, num_frames=args.num_frames),
    }
    _print_arithmetic_ops_table(stats_rows)

    _save_validation_wavs(
        exported=exported_variants,
        samples=samples,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        fixed_frames=args.num_frames,
    )
    logger.info(f"Saved LiteRT validation WAVs to: {args.output_dir / 'sample_audio'}")

    if args.android_gpu_test:
        result = _run_android_gpu_test(args=args, exported_variants=exported_variants)
        logger.info(
            "Android GPU reports saved: "
            f"device={result.device_info_path}, gpu_compile={result.gpu_compile_log_path}, "
            f"gpu_benchmark={result.gpu_benchmark_log_path}, cpu_benchmark={result.cpu_benchmark_log_path}"
        )


if __name__ == "__main__":
    main()
