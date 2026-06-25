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
import contextlib
import copy
import io
import logging
import os
import random
import re
import shlex
import shutil
import subprocess
import wave
from collections import Counter
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
from ai_edge_quantizer import algorithm_manager as aeq_algorithm_manager
from ai_edge_quantizer import qtyping as aeq_qtyping
from ai_edge_quantizer import recipe_manager as aeq_recipe_manager
from ai_edge_quantizer.utils import tfl_interpreter_utils
from litert_torch.generative.quantize import quant_recipes
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


@dataclass
class AotCompileResult:
    model_path: Path
    compiled_dir: Path
    report_path: Path
    raw_log_path: Path
    copied_error_logs: list[Path]
    success: bool


@dataclass
class LocalQualityResult:
    sample_tag: str
    staging_path: Path
    fp16_path: Path
    output_shape: tuple[int, ...]
    finite: bool
    rms: float
    peak: float
    max_abs_error: float
    mean_abs_error: float
    error_rms: float
    rms_delta: float


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
        self.out = _linear_to_conv1d(src_head.out)
        assert isinstance(src_head.istft, ISTFT)
        self.istft = ExportSafeISTFT(src_head.istft)

        n_fft = int(self.istft.n_fft)
        num_bins = n_fft // 2 + 1
        k = torch.arange(1, num_bins - 1, dtype=torch.float32)
        n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(1)
        angle = 2.0 * torch.pi * n * k.unsqueeze(0) / float(n_fft)
        self.register_buffer("_cos_basis", torch.cos(angle))
        self.register_buffer("_sin_basis", torch.sin(angle))
        # 1x1 projections avoid einsum->BATCH_MATMUL lowering on Android GPU delegate.
        self.register_buffer("_cos_proj", torch.cos(angle).unsqueeze(-1))
        self.register_buffer("_sin_proj", torch.sin(angle).unsqueeze(-1))
        self.register_buffer("_nyquist_sign", torch.pow(torch.tensor(-1.0), torch.arange(n_fft, dtype=torch.float32)))
        self.register_buffer("_ola_kernel", torch.eye(self.istft.win_length, dtype=torch.float32).unsqueeze(1))
        shift_kernel = torch.zeros(self.istft.win_length, 1, self.istft.win_length, dtype=torch.float32)
        for c in range(self.istft.win_length):
            shift_kernel[c, 0, self.istft.win_length - 1 - c] = 1.0
        self.register_buffer("_ola_shift_kernel", shift_kernel)
        hop_mask = torch.zeros(1, 1, self.istft.hop_length, dtype=torch.float32)
        hop_mask[..., 0] = 1.0
        self.register_buffer("_hop_mask", hop_mask)
        self.register_buffer("_fixed_window_sq", torch.empty(0, dtype=torch.float32))
        self.use_transpose_conv_overlap_add = False

    def set_fixed_frames(self, frames: int | None) -> None:
        if frames is None:
            self._fixed_window_sq = torch.empty(0, dtype=self.istft.window.dtype, device=self.istft.window.device)
            return
        fixed = self.istft.window.square().view(1, self.istft.win_length, 1)
        self._fixed_window_sq = fixed.repeat(1, 1, int(frames))

    def _irfft_real(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        # real/imag: [B, F, T], with F = n_fft//2 + 1
        n_fft = int(self.istft.n_fft)
        dc = real[:, 0:1, :]
        nyquist = real[:, -1:, :] * self._nyquist_sign.view(1, n_fft, 1)
        real_mid = real[:, 1:-1, :]
        imag_mid = imag[:, 1:-1, :]
        cos_w = self._cos_proj.to(dtype=real_mid.dtype)
        sin_w = self._sin_proj.to(dtype=imag_mid.dtype)
        inner = torch.nn.functional.conv1d(real_mid, cos_w)
        inner = inner - torch.nn.functional.conv1d(imag_mid, sin_w)
        return (dc + nyquist + (2.0 * inner)) / float(n_fft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x.transpose(1, 2))
        mag, phase = x.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)

        ifft = self._irfft_real(real, imag)
        window = self.istft.window
        ifft = ifft * window[None, :, None]

        pad = (self.istft.win_length - self.istft.hop_length) // 2 if self.istft.padding == "same" else 0
        if self.use_transpose_conv_overlap_add:
            y = self._overlap_add_transpose_conv(ifft)
        else:
            y = self._overlap_add_no_transpose_conv(ifft)
        y = y[:, pad:-pad] if pad > 0 else y

        if self._fixed_window_sq.numel() > 0:
            window_sq = self._fixed_window_sq.to(dtype=ifft.dtype)
        else:
            window_sq = window.square().view(1, self.istft.win_length, 1) * torch.ones_like(ifft[:, :1, :])
        if self.use_transpose_conv_overlap_add:
            envelope = self._overlap_add_transpose_conv(window_sq)[0]
        else:
            envelope = self._overlap_add_no_transpose_conv(window_sq)[0]
        envelope = envelope[pad:-pad] if pad > 0 else envelope
        return y / envelope.clamp_min(1e-11)

    def _overlap_add_transpose_conv(self, x: torch.Tensor) -> torch.Tensor:
        ola = torch.nn.functional.conv_transpose1d(
            x,
            self._ola_kernel.to(dtype=x.dtype, device=x.device),
            stride=self.istft.hop_length,
            groups=1,
        )
        return ola[:, 0, :]

    def _overlap_add_no_transpose_conv(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, win_length, frames]
        hop = int(self.istft.hop_length)
        if hop > 1:
            x_up = x.repeat_interleave(hop, dim=2)
            frames = x.shape[-1]
            mask = self._hop_mask.to(dtype=x.dtype, device=x.device).repeat(1, 1, frames)
            x_up = x_up * mask
        else:
            x_up = x
        shifted = torch.nn.functional.conv1d(
            x_up,
            self._ola_shift_kernel.to(dtype=x.dtype, device=x.device),
            bias=None,
            stride=1,
            padding=self.istft.win_length - 1,
            groups=self.istft.win_length,
        )
        y = shifted.sum(dim=1)
        out_size = (x.shape[-1] - 1) * hop + self.istft.win_length
        return y[:, :out_size]


class ExportSafeLayerNorm(nn.Module):
    """LayerNorm decomposition that keeps constants in the module dtype for fp16 LiteRT export."""

    def __init__(self, src: nn.LayerNorm):
        super().__init__()
        if isinstance(src.normalized_shape, int):
            normalized_shape = (src.normalized_shape,)
        else:
            normalized_shape = tuple(int(dim) for dim in src.normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError(f"ExportSafeLayerNorm only supports 1D normalized_shape, got {normalized_shape}")
        self.normalized_dim = int(normalized_shape[0])
        self.register_buffer("eps", torch.tensor(float(src.eps), dtype=torch.float32))
        if src.elementwise_affine:
            assert src.weight is not None and src.bias is not None
            self.weight = nn.Parameter(src.weight.detach().clone())
            self.bias = nn.Parameter(src.bias.detach().clone())
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var = centered.square().mean(dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + self.eps.to(dtype=x.dtype))
        if self.weight is not None:
            y = y * self.weight.view(*([1] * (x.ndim - 1)), self.normalized_dim)
        if self.bias is not None:
            y = y + self.bias.view(*([1] * (x.ndim - 1)), self.normalized_dim)
        return y


class ExportSafeChannelLayerNorm(nn.Module):
    """LayerNorm over channel dimension for channel-first export graphs."""

    def __init__(self, src: nn.LayerNorm):
        super().__init__()
        if isinstance(src.normalized_shape, int):
            normalized_shape = (src.normalized_shape,)
        else:
            normalized_shape = tuple(int(dim) for dim in src.normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError(f"ExportSafeChannelLayerNorm only supports 1D normalized_shape, got {normalized_shape}")
        self.normalized_dim = int(normalized_shape[0])
        self.register_buffer("eps", torch.tensor(float(src.eps), dtype=torch.float32))
        if src.elementwise_affine:
            assert src.weight is not None and src.bias is not None
            self.weight = nn.Parameter(src.weight.detach().clone())
            self.bias = nn.Parameter(src.bias.detach().clone())
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        var = centered.square().mean(dim=1, keepdim=True)
        y = centered * torch.rsqrt(var + self.eps.to(dtype=x.dtype))
        if self.weight is not None:
            y = y * self.weight.view(1, self.normalized_dim, 1)
        if self.bias is not None:
            y = y + self.bias.view(1, self.normalized_dim, 1)
        return y


def _linear_to_conv1d(src: nn.Linear) -> nn.Conv1d:
    conv = nn.Conv1d(
        in_channels=src.in_features,
        out_channels=src.out_features,
        kernel_size=1,
        bias=src.bias is not None,
    )
    with torch.no_grad():
        conv.weight.copy_(src.weight.detach().unsqueeze(-1))
        if src.bias is not None:
            conv.bias.copy_(src.bias.detach())
    return conv


def _as_1d_tuple(value: int | tuple[int, ...], name: str) -> tuple[int]:
    if isinstance(value, int):
        return (int(value),)
    if len(value) != 1:
        raise ValueError(f"Expected 1D {name}, got {value}")
    return (int(value[0]),)


class ExportSafeTemporalConv1d(nn.Module):
    """Conv1d equivalent expressed without torch conv lowering."""

    def __init__(self, src: nn.Conv1d):
        super().__init__()
        if src.padding_mode != "zeros":
            raise ValueError(f"ExportSafeTemporalConv1d only supports zero padding, got {src.padding_mode}")

        kernel_size = _as_1d_tuple(src.kernel_size, "kernel_size")[0]
        stride = _as_1d_tuple(src.stride, "stride")[0]
        padding = _as_1d_tuple(src.padding, "padding")[0]
        dilation = _as_1d_tuple(src.dilation, "dilation")[0]
        if stride != 1:
            raise ValueError(f"ExportSafeTemporalConv1d only supports stride=1, got {stride}")
        if dilation != 1:
            raise ValueError(f"ExportSafeTemporalConv1d only supports dilation=1, got {dilation}")
        if src.groups not in (1, src.in_channels) or (src.groups == src.in_channels and src.out_channels != src.in_channels):
            raise ValueError(
                "ExportSafeTemporalConv1d only supports groups=1 or depthwise groups=in_channels=out_channels, "
                f"got in={src.in_channels}, out={src.out_channels}, groups={src.groups}"
            )

        self.in_channels = int(src.in_channels)
        self.out_channels = int(src.out_channels)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.groups = int(src.groups)
        self.weight = nn.Parameter(src.weight.detach().clone())
        if src.bias is not None:
            self.bias = nn.Parameter(src.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = torch.nn.functional.pad(x, (self.padding, self.padding))

        acc: torch.Tensor | None = None
        for offset in range(self.kernel_size):
            if self.kernel_size == 1 or offset == self.kernel_size - 1:
                segment = x[:, :, offset:]
            else:
                segment = x[:, :, offset : offset - (self.kernel_size - 1)]

            if self.groups == self.in_channels:
                term = segment * self.weight[:, 0, offset].view(1, self.out_channels, 1)
            else:
                term = (segment.unsqueeze(1) * self.weight[:, :, offset].view(1, self.out_channels, self.in_channels, 1)).sum(dim=2)
            acc = term if acc is None else acc + term

        assert acc is not None
        if self.bias is not None:
            acc = acc + self.bias.view(1, self.out_channels, 1)
        return acc


def _replace_conv1ds_with_temporal_conv2d(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d):
            setattr(module, name, ExportSafeTemporalConv1d(child))
        else:
            _replace_conv1ds_with_temporal_conv2d(child)


class ExportSafeConvNeXtBlock(nn.Module):
    """ConvNeXt block kept channel-first to avoid symbolic Linear bias expands."""

    def __init__(self, src: nn.Module):
        super().__init__()
        if getattr(src, "adanorm", False):
            raise ValueError("ExportSafeConvNeXtBlock does not support AdaLayerNorm")
        if not isinstance(src.norm, nn.LayerNorm):
            raise TypeError(f"Expected ConvNeXtBlock.norm to be LayerNorm, got {type(src.norm)}")
        self.dwconv = src.dwconv
        self.norm = ExportSafeChannelLayerNorm(src.norm)
        self.pwconv1 = _linear_to_conv1d(src.pwconv1)
        self.act = src.act
        self.pwconv2 = _linear_to_conv1d(src.pwconv2)
        if src.gamma is not None:
            self.gamma = nn.Parameter(src.gamma.detach().clone(), requires_grad=src.gamma.requires_grad)
        else:
            self.register_parameter("gamma", None)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor | None = None) -> torch.Tensor:
        del cond_embedding_id
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma.view(1, -1, 1)
        return residual + x


class ExportSafeVocosBackbone(nn.Module):
    """Vocos backbone export wrapper with channel-first ConvNeXt blocks."""

    def __init__(self, src: nn.Module):
        super().__init__()
        if getattr(src, "adanorm", False):
            raise ValueError("ExportSafeVocosBackbone does not support AdaLayerNorm")
        if not isinstance(src.norm, nn.LayerNorm):
            raise TypeError(f"Expected VocosBackbone.norm to be LayerNorm, got {type(src.norm)}")
        if not isinstance(src.final_layer_norm, nn.LayerNorm):
            raise TypeError(
                f"Expected VocosBackbone.final_layer_norm to be LayerNorm, got {type(src.final_layer_norm)}"
            )
        self.input_channels = int(src.input_channels)
        self.embed = src.embed
        self.norm = ExportSafeChannelLayerNorm(src.norm)
        self.convnext = nn.ModuleList(ExportSafeConvNeXtBlock(block) for block in src.convnext)
        self.final_layer_norm = ExportSafeChannelLayerNorm(src.final_layer_norm)

    def forward(self, x: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        del kwargs
        x = self.embed(x)
        x = self.norm(x)
        for conv_block in self.convnext:
            x = conv_block(x)
        x = self.final_layer_norm(x)
        return x.transpose(1, 2)


def _widen_conv1d_input(src: nn.Conv1d, total_channels: int, start_channel: int) -> nn.Conv1d:
    widened = nn.Conv1d(
        in_channels=total_channels,
        out_channels=src.out_channels,
        kernel_size=src.kernel_size,
        stride=src.stride,
        padding=src.padding,
        dilation=src.dilation,
        groups=1,
        bias=src.bias is not None,
        padding_mode=src.padding_mode,
    )
    with torch.no_grad():
        widened.weight.zero_()
        in_channels = int(src.in_channels)
        widened.weight[:, start_channel : start_channel + in_channels, :] = src.weight.detach().clone()
        if src.bias is not None:
            widened.bias.copy_(src.bias.detach())
    return widened


def _widen_control_branch(branch: nn.Sequential, total_channels: int, start_channel: int) -> nn.Sequential:
    if not branch or not isinstance(branch[0], nn.Conv1d):
        raise TypeError("Expected control branch to start with Conv1d")
    layers = list(branch.children())
    layers[0] = _widen_conv1d_input(layers[0], total_channels=total_channels, start_channel=start_channel)
    return nn.Sequential(*layers)


class ExportSafeKokoroFeatureConditioner(nn.Module):
    """Kokoro conditioner with slice-free full-channel projection branches for fp16 export."""

    def __init__(self, src: nn.Module):
        super().__init__()
        self.asr_channels = int(src.asr_channels)
        self.style_channels = int(src.style_channels)
        self.total_channels = self.asr_channels + self.style_channels + 2
        asr_start = 0
        f0_start = self.asr_channels
        noise_start = self.asr_channels + 1
        style_start = self.asr_channels + 2

        self.asr_proj = _widen_conv1d_input(src.asr_proj, self.total_channels, asr_start)
        self.f0_proj = _widen_control_branch(src.f0_proj, self.total_channels, f0_start)
        self.noise_proj = _widen_control_branch(src.noise_proj, self.total_channels, noise_start)
        self.style_proj = _widen_conv1d_input(src.style_proj, self.total_channels, style_start)
        self.fuse = src.fuse

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fuse(
            torch.cat(
                [
                    self.asr_proj(features),
                    self.f0_proj(features),
                    self.noise_proj(features),
                    self.style_proj(features),
                ],
                dim=1,
            )
        )


def _replace_layer_norms_for_export(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, ExportSafeLayerNorm(child))
        else:
            _replace_layer_norms_for_export(child)


def _patch_istft_for_export(model: nn.Module) -> None:
    if hasattr(model, "conditioner") and all(
        hasattr(model.conditioner, attr)
        for attr in ("asr_channels", "style_channels", "asr_proj", "f0_proj", "noise_proj", "style_proj", "fuse")
    ):
        model.conditioner = ExportSafeKokoroFeatureConditioner(model.conditioner)
    if hasattr(model, "backbone") and all(
        hasattr(model.backbone, attr)
        for attr in ("input_channels", "embed", "norm", "convnext", "final_layer_norm")
    ):
        model.backbone = ExportSafeVocosBackbone(model.backbone)
    _replace_layer_norms_for_export(model)
    for module in model.modules():
        if hasattr(module, "istft") and isinstance(module.istft, ISTFT):
            module.istft = ExportSafeISTFT(module.istft)
        if hasattr(module, "head") and hasattr(module.head, "istft") and isinstance(module.head.istft, ISTFT):
            module.head = ExportSafeISTFTHead(module.head)
    if os.environ.get("KOKORO_EXPORT_CONV_FREE_FP16_PROBE") == "1":
        _replace_conv1ds_with_temporal_conv2d(model)


def _set_export_fixed_frames(model: nn.Module, frames: int | None) -> None:
    for module in model.modules():
        if isinstance(module, ExportSafeISTFTHead):
            module.set_fixed_frames(frames)


def _set_export_transpose_conv_overlap_add(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, ExportSafeISTFTHead):
            module.use_transpose_conv_overlap_add = bool(enabled)


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
        default=["data/af_alloy_0*_00.pt"],
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
    parser.add_argument(
        "--pixel10-fp16-aot",
        action="store_true",
        help=(
            "Compatibility alias for --pixel10-fp16-gpu. Exports Pixel 10 GPU-targeted FP16-weight LiteRT models "
            "and writes diagnostics under output-dir/diagnostics. Does not run Google Tensor AOT."
        ),
    )
    parser.add_argument(
        "--pixel10-fp16-gpu",
        action="store_true",
        help=(
            "Export Pixel 10 GPU-targeted FP16-weight LiteRT models from real feature inputs. "
            "FP16 is selected at conversion/export time; GPU acceleration is selected at Android runtime."
        ),
    )
    parser.add_argument(
        "--pixel10-multisignature-static",
        action="store_true",
        help=(
            "In --pixel10-fp16-aot mode, also export one static multi-signature model with a signature "
            "for each discovered real input frame length."
        ),
    )
    parser.add_argument(
        "--dynamic-frames",
        action="store_true",
        help="In --pixel10-fp16-aot mode, first attempt a dynamic-frame LiteRT export before fixed-length fallbacks.",
    )
    parser.add_argument("--dynamic-frame-min", type=int, default=16)
    parser.add_argument("--dynamic-frame-max", type=int, default=1200)
    parser.add_argument(
        "--google-tensor-compiler-lib",
        type=Path,
        default=Path("tools/google_tensor_ml_sdk"),
        help="Deprecated for Pixel GPU export; Google Tensor AOT is not used by --pixel10-fp16-gpu.",
    )
    parser.add_argument(
        "--google-tensor-soc-model",
        type=str,
        default="TENSOR_G5",
        choices=("TENSOR_G3", "TENSOR_G4", "TENSOR_G5", "TENSOR_G6"),
        help="Deprecated for Pixel GPU export; Google Tensor AOT is not used by --pixel10-fp16-gpu.",
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


def _fp16_weight_only_recipe() -> list[dict[str, object]]:
    rp_manager = aeq_recipe_manager.RecipeManager()
    rp_manager.add_weight_only_config(
        regex=".*",
        operation_name=aeq_qtyping.TFLOperationName.ALL_SUPPORTED,
        num_bits=16,
        algorithm_key=aeq_algorithm_manager.AlgorithmName.FLOAT_CASTING,
    )
    return rp_manager.get_quantization_recipe()


def _quantize_tflite_to_fp16_weights(fp32_tflite_path: Path, fp16_tflite_path: Path) -> Path:
    qt = aeq_quantizer.Quantizer(fp32_tflite_path.read_bytes())
    qt.load_quantization_recipe(_fp16_weight_only_recipe())
    quant_result = qt.quantize()
    if quant_result.quantized_model is None:
        raise RuntimeError("AI Edge Quantizer did not produce an fp16-weight model.")
    fp16_tflite_path.write_bytes(bytes(quant_result.quantized_model))
    return fp16_tflite_path


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
    dynamic_shapes: tuple[object, ...] | None = None,
    quant_config: object | None = None,
) -> Path:
    model.eval()
    edge_model = litert_torch.convert(
        model,
        sample_args=(sample_arg,),
        strict_export=False,
        quant_config=quant_config,
        dynamic_shapes=dynamic_shapes,
        lightweight_conversion=lightweight_conversion,
    )
    edge_model.export(str(out_path))
    return out_path


def _export_litert_multisignature_static(
    model: nn.Module,
    samples: Sequence[InferenceSample],
    out_path: Path,
    lightweight_conversion: bool,
    sample_dtype: torch.dtype,
    quant_config: object | None = None,
) -> Path:
    if not samples:
        raise RuntimeError("Cannot export a multi-signature model without samples.")

    first_sample = samples[0]
    first_frames = int(first_sample.features.shape[-1])
    first_model = copy.deepcopy(model).eval()
    _set_export_fixed_frames(first_model, first_frames)
    converter = litert_torch.signature(
        f"frames_{first_frames}",
        first_model,
        sample_args=(first_sample.features.to(dtype=sample_dtype).unsqueeze(0),),
    )

    seen_frames = {first_frames}
    for sample in samples[1:]:
        frames = int(sample.features.shape[-1])
        if frames in seen_frames:
            continue
        seen_frames.add(frames)
        signature_model = copy.deepcopy(model).eval()
        _set_export_fixed_frames(signature_model, frames)
        converter.add_signature(
            f"frames_{frames}",
            signature_model,
            sample_args=(sample.features.to(dtype=sample_dtype).unsqueeze(0),),
        )

    edge_model = converter.convert(
        strict_export=False,
        quant_config=quant_config,
        lightweight_conversion=lightweight_conversion,
    )
    edge_model.export(str(out_path))
    return out_path


def _schema_enum_names(enum_cls: object) -> dict[int, str]:
    names: dict[int, str] = {}
    for name in dir(enum_cls):
        if not name.isupper():
            continue
        value = getattr(enum_cls, name)
        if isinstance(value, int):
            names[int(value)] = name
    return names


def _write_tflite_diagnostics(model_path: Path, out_path: Path, num_frames: int) -> None:
    from ai_edge_litert import schema_py_generated as schema

    model_bytes = model_path.read_bytes()
    fb_model = schema.Model.GetRootAsModel(model_bytes, 0)
    op_code_names = _schema_enum_names(schema.BuiltinOperator)
    tensor_type_names = _schema_enum_names(schema.TensorType)

    opcode_indices: list[int] = []
    custom_codes: list[str] = []
    for i in range(fb_model.OperatorCodesLength()):
        code = fb_model.OperatorCodes(i)
        builtin = int(code.BuiltinCode())
        opcode_indices.append(builtin)
        custom = code.CustomCode()
        custom_codes.append(custom.decode("utf-8") if custom else "")

    op_hist: Counter[str] = Counter()
    tensor_hist: Counter[str] = Counter()
    lines: list[str] = [f"Model: {model_path}", ""]

    for subgraph_idx in range(fb_model.SubgraphsLength()):
        subgraph = fb_model.Subgraphs(subgraph_idx)
        lines.append(f"Subgraph {subgraph_idx}")
        lines.append(f"  tensors: {subgraph.TensorsLength()}")
        lines.append(f"  operators: {subgraph.OperatorsLength()}")

        tensor_meta: dict[int, tuple[str, list[int]]] = {}
        for tensor_idx in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(tensor_idx)
            tensor_type = tensor_type_names.get(int(tensor.Type()), str(int(tensor.Type())))
            tensor_shape = [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())]
            tensor_meta[tensor_idx] = (tensor_type, tensor_shape)
            tensor_hist[tensor_type] += 1

        lines.append("  operators:")
        for op_idx in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(op_idx)
            opcode_index = int(op.OpcodeIndex())
            builtin = opcode_indices[opcode_index]
            op_name = custom_codes[opcode_index] or op_code_names.get(builtin, str(builtin))
            op_hist[op_name] += 1
            inputs = [int(op.Inputs(i)) for i in range(op.InputsLength())]
            outputs = [int(op.Outputs(i)) for i in range(op.OutputsLength())]
            input_shapes = [tensor_meta.get(idx, ("", []))[1] for idx in inputs]
            output_shapes = [tensor_meta.get(idx, ("", []))[1] for idx in outputs]
            input_types = [tensor_meta.get(idx, ("", []))[0] for idx in inputs]
            output_types = [tensor_meta.get(idx, ("", []))[0] for idx in outputs]
            lines.append(
                f"    {op_idx:03d} {op_name} inputs={inputs} outputs={outputs} "
                f"input_shapes={input_shapes} output_shapes={output_shapes} "
                f"input_types={input_types} output_types={output_types}"
            )
        lines.append("")

    arith = _estimate_tflite_arithmetic_ops(model_path, num_frames=num_frames)
    lines.append("Operator histogram:")
    for op_name, count in op_hist.most_common():
        lines.append(f"  {op_name}: {count}")
    lines.append("")
    lines.append("Tensor type histogram:")
    for type_name, count in tensor_hist.most_common():
        lines.append(f"  {type_name}: {count}")
    lines.append("")
    lines.append(
        "Estimated arithmetic ops: "
        f"float={_format_int(arith.float_ops)}, int={_format_int(arith.int_ops)}, total={_format_int(arith.total)}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_aot_error_logs(report_text: str, diagnostics_dir: Path, model_stem: str) -> list[Path]:
    copied: list[Path] = []
    for match in re.finditer(r"See\s+(/tmp/\S+?\.error)", report_text):
        source = Path(match.group(1))
        if not source.exists():
            continue
        dest = diagnostics_dir / f"{model_stem}_{source.name}"
        shutil.copy2(source, dest)
        copied.append(dest)
    return copied


def _run_google_tensor_aot(
    model_path: Path,
    output_dir: Path,
    diagnostics_dir: Path,
    soc_model_name: str,
    compiler_lib: Path,
) -> AotCompileResult:
    from ai_edge_litert.aot.aot_compile import aot_compile
    from ai_edge_litert.aot.vendors.google_tensor.target import SocManufacturer, SocModel, Target

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    compiled_dir = output_dir / "compiled" / model_path.stem
    compiled_dir.mkdir(parents=True, exist_ok=True)

    compiler_lib = compiler_lib.resolve()
    if compiler_lib.exists():
        os.environ["GOOGLE_TENSOR_BACKEND_ENABLED"] = "1"
        os.environ["GOOGLE_TENSOR_COMPILER_LIB"] = str(compiler_lib)

    # Importing the backend module registers backend id "GOOGLE" with LiteRT AOT.
    import ai_edge_litert.aot.vendors.google_tensor.google_tensor_backend  # noqa: F401

    target = Target(getattr(SocModel, soc_model_name), SocManufacturer.GOOGLE)
    raw = io.StringIO()
    result = None
    exc: Exception | None = None
    with contextlib.redirect_stdout(raw), contextlib.redirect_stderr(raw):
        try:
            result = aot_compile(
                str(model_path),
                output_dir=compiled_dir,
                target=target,
                keep_going=True,
            )
        except Exception as err:  # Keep diagnostics even when the wrapper raises.
            exc = err

    report = result.compilation_report() if result is not None else ""
    success = bool(result is not None and result.models_with_backend and not result.failed_backends)
    if exc is not None:
        report += ("\n" if report else "") + f"AOT wrapper exception: {type(exc).__name__}: {exc}"

    report_path = diagnostics_dir / f"{model_path.stem}_aot_{repr(target).lower()}.txt"
    raw_log_path = diagnostics_dir / f"{model_path.stem}_aot_{repr(target).lower()}_raw.txt"
    report_path.write_text(
        f"GOOGLE_TENSOR_COMPILER_LIB={os.environ.get('GOOGLE_TENSOR_COMPILER_LIB', '')}\n\n"
        + (report if report.strip() else "No compilation report was produced.")
        + "\n",
        encoding="utf-8",
    )
    raw_log_path.write_text(raw.getvalue(), encoding="utf-8")
    copied_errors = _copy_aot_error_logs(report, diagnostics_dir, f"{model_path.stem}_aot_{repr(target).lower()}")

    logger.info(
        f"AOT compile {'succeeded' if success else 'failed'} for {model_path.name}; "
        f"report={report_path}, raw={raw_log_path}"
    )
    return AotCompileResult(
        model_path=model_path,
        compiled_dir=compiled_dir,
        report_path=report_path,
        raw_log_path=raw_log_path,
        copied_error_logs=copied_errors,
        success=success,
    )


def _attempt_dynamic_fp16_export(
    model: nn.Module,
    sample: InferenceSample,
    output_dir: Path,
    diagnostics_dir: Path,
    args: argparse.Namespace,
    sample_dtype: torch.dtype = torch.float32,
    quant_config: object | None = None,
) -> Path | None:
    out_path = output_dir / "vocos_fp16_dynamic_litert.tflite"
    try:
        _set_export_fixed_frames(model, None)
        _set_export_transpose_conv_overlap_add(model, True)
        dynamic_shapes = (
            {2: torch.export.Dim("frames", min=args.dynamic_frame_min, max=args.dynamic_frame_max)},
        )
        _export_litert(
            model=model,
            sample_arg=sample.features.to(dtype=sample_dtype).unsqueeze(0),
            out_path=out_path,
            lightweight_conversion=args.lightweight_conversion,
            dynamic_shapes=dynamic_shapes,
            quant_config=quant_config,
        )
        logger.info(f"Exported dynamic-frame fp16 LiteRT model: {out_path}")
        return out_path
    except Exception as exc:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        failure_path = diagnostics_dir / "dynamic_export_failure.txt"
        failure_path.write_text(
            "Dynamic-frame litert_torch export failed.\n\n"
            f"sample={sample.tag}\n"
            f"sample_shape={tuple(sample.features.shape)}\n"
            f"dynamic_frame_min={args.dynamic_frame_min}\n"
            f"dynamic_frame_max={args.dynamic_frame_max}\n\n"
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        logger.warning(f"Dynamic-frame fp16 LiteRT export failed; diagnostic={failure_path}")
        return None
    finally:
        _set_export_transpose_conv_overlap_add(model, False)


def _select_pixel10_export_model(
    fp16_model: nn.Module,
    sample: InferenceSample,
    output_dir: Path,
    diagnostics_dir: Path,
    args: argparse.Namespace,
) -> tuple[nn.Module, torch.dtype, str]:
    frames = int(sample.features.shape[-1])
    probe_path = output_dir / f"vocos_true_fp16_probe_{sample.tag}_{frames}f_litert.tflite"
    try:
        true_fp16_model = copy.deepcopy(fp16_model).half().eval()
        _set_export_fixed_frames(true_fp16_model, frames)
        _export_litert(
            model=true_fp16_model,
            sample_arg=sample.features.half().unsqueeze(0),
            out_path=probe_path,
            lightweight_conversion=args.lightweight_conversion,
        )
        _write_tflite_diagnostics(
            model_path=probe_path,
            out_path=diagnostics_dir / f"{probe_path.stem}_op_inventory.txt",
            num_frames=frames,
        )
        (diagnostics_dir / "true_fp16_export.txt").write_text(
            f"True fp16 LiteRT export succeeded for probe sample {sample.tag}: {probe_path}\n",
            encoding="utf-8",
        )
        logger.info(f"True fp16 LiteRT probe export succeeded: {probe_path}")
        return true_fp16_model, torch.float16, "true_fp16"
    except Exception as exc:
        failure_path = diagnostics_dir / "true_fp16_export_failure.txt"
        failure_path.write_text(
            "True all-fp16 LiteRT export failed; falling back to float32 LiteRT export plus fp16 weight casting.\n\n"
            f"sample={sample.tag}\n"
            f"sample_shape={tuple(sample.features.shape)}\n\n"
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        logger.warning(f"True fp16 LiteRT export failed; diagnostic={failure_path}")
        return fp16_model, torch.float32, "fp16_weight_quantized_tflite"


def _run_pixel10_fp16_aot(args: argparse.Namespace, config: GeneratorConfig, fp16_model: nn.Module) -> None:
    diagnostics_dir = args.output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    samples = _collect_feature_pt_samples(args, input_channels=config.in_channels, count=max(args.sample_count, 3))
    if not samples:
        raise RuntimeError(
            "--pixel10-fp16-aot requires feature .pt inputs. "
            f"No usable files matched {args.input_feature_glob}."
        )

    export_model = fp16_model
    sample_dtype = torch.float32
    export_precision = "litert_torch_full_fp16_recipe"
    fp16_quant_config = quant_recipes.full_fp16_recipe()

    exported: list[Path] = []
    exported_pairs: list[tuple[InferenceSample, Path, Path]] = []
    quality_results: list[LocalQualityResult] = []
    multisig_path: Path | None = None
    dynamic_path: Path | None = None
    if args.dynamic_frames:
        dynamic_path = _attempt_dynamic_fp16_export(
            model=export_model,
            sample=samples[0],
            output_dir=args.output_dir,
            diagnostics_dir=diagnostics_dir,
            args=args,
            sample_dtype=sample_dtype,
            quant_config=fp16_quant_config,
        )
        if dynamic_path is not None:
            exported.append(dynamic_path)
            _write_tflite_diagnostics(
                model_path=dynamic_path,
                out_path=diagnostics_dir / f"{dynamic_path.stem}_op_inventory.txt",
                num_frames=int(samples[0].features.shape[-1]),
            )

    for sample in samples:
        frames = int(sample.features.shape[-1])
        staging_path = args.output_dir / f"vocos_fp32_for_fp16_{sample.tag}_{frames}f_litert.tflite"
        out_path = args.output_dir / f"vocos_fp16_{sample.tag}_{frames}f_litert.tflite"
        _set_export_fixed_frames(export_model, frames)
        _export_litert(
            model=export_model,
            sample_arg=sample.features.to(dtype=sample_dtype).unsqueeze(0),
            out_path=staging_path,
            lightweight_conversion=args.lightweight_conversion,
        )
        _export_litert(
            model=export_model,
            sample_arg=sample.features.to(dtype=sample_dtype).unsqueeze(0),
            out_path=out_path,
            lightweight_conversion=args.lightweight_conversion,
            quant_config=fp16_quant_config,
        )
        exported.append(out_path)
        exported_pairs.append((sample, staging_path, out_path))
        _write_tflite_diagnostics(
            model_path=out_path,
            out_path=diagnostics_dir / f"{out_path.stem}_op_inventory.txt",
            num_frames=frames,
        )
        quality_results.append(
            _run_fixed_frame_quality_check(
                sample=sample,
                staging_path=staging_path,
                fp16_path=out_path,
                diagnostics_dir=diagnostics_dir,
            )
        )
        logger.info(f"Exported fixed-frame fp16 LiteRT model for {sample.tag}: {out_path}")

    if args.pixel10_multisignature_static:
        multisig_path = args.output_dir / "vocos_fp16_multisig_static_litert.tflite"
        _export_litert_multisignature_static(
            model=export_model,
            samples=samples,
            out_path=multisig_path,
            lightweight_conversion=args.lightweight_conversion,
            sample_dtype=sample_dtype,
            quant_config=fp16_quant_config,
        )
        _write_tflite_diagnostics(
            model_path=multisig_path,
            out_path=diagnostics_dir / f"{multisig_path.stem}_op_inventory.txt",
            num_frames=max(int(sample.features.shape[-1]) for sample in samples),
        )
        logger.info(f"Exported static multi-signature fp16 LiteRT model: {multisig_path}")

    _save_pixel10_sample_audio(
        exports=exported_pairs,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        multisig_path=multisig_path,
        dynamic_path=dynamic_path,
        dynamic_frame_max=args.dynamic_frame_max if args.dynamic_frames else None,
    )

    summary_lines = [
        "Pixel 10 GPU FP16 export diagnostics",
        "",
        "Input examples:",
    ]
    for sample in samples:
        summary_lines.append(f"  {sample.tag}: [1,{config.in_channels},{int(sample.features.shape[-1])}]")
    summary_lines.extend(
        [
            "",
            f"Export precision path: {export_precision}",
            "GPU runtime: Android LiteRT CompiledModel with Accelerator.GPU (or older TFLite GPU delegate with FP16/reduced precision enabled).",
            "AOT compilation: skipped; Google Tensor AOT is for NPU/TPU vendor targets, not Pixel GPU runtime selection.",
            "",
            "Dynamic-frame export:",
            "  attempted: " + str(bool(args.dynamic_frames)),
            "  result: " + ("exported " + str(dynamic_path) if dynamic_path else "not exported; see dynamic_export_failure.txt"),
            "",
            "Fixed-frame LiteRT exports:",
        ]
    )
    summary_lines.extend(f"  {path.name}" for path in exported if path != dynamic_path)
    if multisig_path is not None:
        summary_lines.extend(
            [
                "",
                "Static multi-signature export:",
                f"  {multisig_path.name}",
                "  signatures: " + ", ".join(f"frames_{int(sample.features.shape[-1])}" for sample in samples),
            ]
        )
    summary_lines.extend(["", "Local LiteRT quality checks:"])
    for result in quality_results:
        summary_lines.append(
            f"  {result.sample_tag}: finite={result.finite}, shape={list(result.output_shape)}, "
            f"rms={result.rms:.6g}, peak={result.peak:.6g}, "
            f"max_abs_error={result.max_abs_error:.6g}, mean_abs_error={result.mean_abs_error:.6g}, "
            f"error_rms={result.error_rms:.6g}, rms_delta={result.rms_delta:.6g}"
        )
    summary_lines.extend(
        [
            "",
            "Notes:",
            "  Tensor and op inventories are saved as *_op_inventory.txt.",
            "  Sample WAV outputs are saved under sample_audio/.",
            "  FP16 is a conversion/export choice; GPU acceleration is a runtime loading choice.",
        ]
    )
    (diagnostics_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    _write_pixel10_gpu_runtime_notes(diagnostics_dir)
    logger.info(f"Saved Pixel 10 GPU FP16 diagnostics to: {diagnostics_dir}")


def _run_litert_inference(model_path: Path, features: np.ndarray, signature_name: str | None = None) -> np.ndarray:
    try:
        model = litert_torch.load(str(model_path))
        interpreter = model._get_interpreter()
    except RuntimeError as exc:
        logger.warning(
            f"Default LiteRT interpreter failed for {model_path.name}; retrying without XNNPACK/default delegates: {exc}"
        )
        interpreter = tfl_interpreter_utils.create_tfl_interpreter(
            str(model_path),
            allocate_tensors=False,
            use_xnnpack=False,
            preserve_all_tensors=False,
        )
    signatures = list(interpreter.get_signature_list().keys())
    if not signatures:
        raise RuntimeError(f"No TFLite signatures found in {model_path}")
    signature_key = signature_name or signatures[0]
    if signature_key not in signatures:
        raise RuntimeError(f"Signature {signature_key!r} not found in {model_path}; available={signatures}")
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


def _run_fixed_frame_quality_check(
    sample: InferenceSample,
    staging_path: Path,
    fp16_path: Path,
    diagnostics_dir: Path,
) -> LocalQualityResult:
    input_np = sample.features.float().unsqueeze(0).numpy().astype(np.float32)
    staging_pred = _run_litert_inference(staging_path, input_np)
    fp16_pred = _run_litert_inference(fp16_path, input_np)
    diff = fp16_pred.astype(np.float32) - staging_pred.astype(np.float32)
    fp16_flat = fp16_pred.astype(np.float32).reshape(-1)
    diff_flat = diff.reshape(-1)

    result = LocalQualityResult(
        sample_tag=sample.tag,
        staging_path=staging_path,
        fp16_path=fp16_path,
        output_shape=tuple(int(dim) for dim in fp16_pred.shape),
        finite=bool(np.isfinite(fp16_pred).all()),
        rms=_waveform_rms(fp16_pred),
        peak=float(np.max(np.abs(fp16_flat))) if fp16_flat.size else 0.0,
        max_abs_error=float(np.max(np.abs(diff_flat))) if diff_flat.size else 0.0,
        mean_abs_error=float(np.mean(np.abs(diff_flat))) if diff_flat.size else 0.0,
        error_rms=_waveform_rms(diff),
        rms_delta=abs(_waveform_rms(fp16_pred) - _waveform_rms(staging_pred)),
    )

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    report_path = diagnostics_dir / f"{fp16_path.stem}_quality.txt"
    report_path.write_text(
        "\n".join(
            [
                f"sample={result.sample_tag}",
                f"staging_model={result.staging_path}",
                f"fp16_model={result.fp16_path}",
                f"output_shape={list(result.output_shape)}",
                f"finite={result.finite}",
                f"rms={result.rms:.9g}",
                f"peak={result.peak:.9g}",
                f"max_abs_error={result.max_abs_error:.9g}",
                f"mean_abs_error={result.mean_abs_error:.9g}",
                f"error_rms={result.error_rms:.9g}",
                f"rms_delta={result.rms_delta:.9g}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _write_pixel10_gpu_runtime_notes(diagnostics_dir: Path) -> None:
    notes = """Pixel 10 GPU FP16 runtime notes

FP16 is selected when exporting the LiteRT/TFLite flatbuffer:

```python
from litert_torch.generative.quantize import quant_recipes

edge_model = litert_torch.convert(
    model.eval(),
    sample_args,
    quant_config=quant_recipes.full_fp16_recipe(),
)
edge_model.export("model_fp16.tflite")
```

GPU acceleration is selected when loading the model on Android. Do not use Google Tensor AOT compilation for Pixel GPU runtime selection.

```kotlin
val env = Environment.create()

val model = CompiledModel.create(
    context.assets,
    "model_fp16.tflite",
    CompiledModel.Options(Accelerator.GPU),
    env,
)
```

If using the older TensorFlow Lite `Interpreter` GPU delegate instead of LiteRT `CompiledModel`, enable FP16/reduced precision in the GPU delegate options.
"""
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    (diagnostics_dir / "pixel10_gpu_runtime_notes.md").write_text(notes, encoding="utf-8")


def _save_pixel10_sample_audio(
    exports: Sequence[tuple[InferenceSample, Path, Path]],
    output_dir: Path,
    sample_rate: int,
    multisig_path: Path | None = None,
    dynamic_path: Path | None = None,
    dynamic_frame_max: int | None = None,
) -> None:
    wav_dir = output_dir / "sample_audio"
    wav_dir.mkdir(parents=True, exist_ok=True)

    for sample, staging_path, fp16_path in exports:
        input_np = sample.features.float().unsqueeze(0).numpy().astype(np.float32)
        staging_pred = _run_litert_inference(staging_path, input_np)
        save_wav_16bit(wav_dir / f"{sample.tag}_fp32_staging_litert.wav", staging_pred, sample_rate)

        fp16_pred = _run_litert_inference(fp16_path, input_np)
        save_wav_16bit(wav_dir / f"{sample.tag}_fp16_litert.wav", fp16_pred, sample_rate)

        if multisig_path is not None and multisig_path.exists():
            frames = int(sample.features.shape[-1])
            multisig_pred = _run_litert_inference(multisig_path, input_np, signature_name=f"frames_{frames}")
            save_wav_16bit(wav_dir / f"{sample.tag}_fp16_multisig_litert.wav", multisig_pred, sample_rate)

        if dynamic_path is not None and dynamic_path.exists():
            frames = int(sample.features.shape[-1])
            if dynamic_frame_max is None or frames <= int(dynamic_frame_max):
                try:
                    dynamic_pred = _run_litert_inference(dynamic_path, input_np)
                except Exception as exc:
                    logger.warning(f"Skipping dynamic sample audio for {sample.tag}: {exc}")
                else:
                    save_wav_16bit(wav_dir / f"{sample.tag}_fp16_dynamic_litert.wav", dynamic_pred, sample_rate)


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

    if args.pixel10_fp16_gpu or args.pixel10_fp16_aot:
        _run_pixel10_fp16_aot(args=args, config=config, fp16_model=fp16_model)
        return

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
