"""Export Kokoro Wavehax weights to LiteRT/TFLite and generate validation WAVs.

Defaults:
  uv run python wavehax_export.py

Inputs:
  models/wavehax/last.pt
  data/af_alloy*.pt

Outputs:
  runs/wavehax/wavehax_fp32_litert.tflite
  runs/wavehax/wavehax_fp16_litert.tflite
  runs/wavehax/sample_audio/*.wav
  runs/wavehax/diagnostics/*
  PROGRESS.md
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import random
import resource
import sys
import time
import wave
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from loguru import logger
from torch import Tensor, nn
import torch.nn.functional as F

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

ROOT = Path(__file__).resolve().parent
WAVEHAX_ROOT = ROOT / "third_party" / "wavehax"
for path in (ROOT, WAVEHAX_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import litert_torch
from ai_edge_quantizer.utils import tfl_interpreter_utils
from litert_torch.generative.quantize import quant_recipes

from wavehax.generators.wavehax import MultiScaleWavehaxGenerator
from wavehax.modules.decomposer import MultiStream1d
from wavehax.modules.stft import STFT


@dataclass
class FeatureSample:
    tag: str
    path: Path
    features: torch.Tensor
    sample_rate: int
    frame_hop: int


@dataclass
class LiteRTVariant:
    name: str
    path: Path


class KokoroFeatureConditioner(nn.Module):
    """Project Kokoro ASR/F0/noise/style tensors into vocoder conditioning."""

    def __init__(
        self,
        asr_channels: int = 512,
        style_channels: int = 128,
        out_channels: int = 192,
        control_channels: int = 32,
        control_layers: int = 2,
    ):
        super().__init__()
        self.asr_channels = int(asr_channels)
        self.style_channels = int(style_channels)
        self.asr_proj = nn.Conv1d(self.asr_channels, out_channels, 1)
        self.style_proj = nn.Conv1d(self.style_channels, out_channels, 1)
        self.f0_proj = self._control_branch(out_channels, control_channels, control_layers)
        self.noise_proj = self._control_branch(out_channels, control_channels, control_layers)
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels * 4, out_channels, 1),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 1),
        )

    @staticmethod
    def _control_branch(out_channels: int, control_channels: int, control_layers: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_ch = 1
        hidden = max(1, int(control_channels))
        for _ in range(max(1, int(control_layers))):
            layers.extend([nn.Conv1d(in_ch, hidden, 5, padding=2), nn.GELU()])
            in_ch = hidden
        layers.append(nn.Conv1d(hidden, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        asr_end = self.asr_channels
        f0_end = asr_end + 1
        noise_end = f0_end + 1
        return self.fuse(
            torch.cat(
                [
                    self.asr_proj(features[:, :asr_end]),
                    self.f0_proj(features[:, asr_end:f0_end]),
                    self.noise_proj(features[:, f0_end:noise_end]),
                    self.style_proj(features[:, noise_end:]),
                ],
                dim=1,
            )
        )


class KokoroMultiScaleWavehaxGenerator(nn.Module):
    def __init__(self, config: Mapping[str, object], sample_rate: int):
        super().__init__()
        self.conditioner = KokoroFeatureConditioner(
            out_channels=int(config["model_input_channels"]),
            control_channels=int(config["control_channels"]),
            control_layers=int(config["control_layers"]),
        )
        self.generator = MultiScaleWavehaxGenerator(
            in_channels=int(config["model_input_channels"]),
            channels=int(config["channels"]),
            mult_channels=int(config["mult_channels"]),
            kernel_size=int(config["kernel_size"]),
            num_blocks=int(config["num_blocks"]),
            decomposer=str(config["decomposer"]),
            num_splits=int(config["num_splits"]),
            n_fft=int(config["n_fft"]),
            hop_length=int(config["hop_length"]),
            sample_rate=int(sample_rate),
            prior_type=str(config["prior_type"]),
            drop_prob=float(config["drop_prob"]),
            framewise_norm=bool(config["framewise_norm"]),
            use_gradient_checkpointing=False,
        )

    def forward(self, features: Tensor) -> Tensor:
        cond = self.conditioner(features)
        f0 = None if isinstance(self.generator, ExportOptimizedMultiScaleWavehaxGenerator) else features[:, 512:513, :]
        audio, _prior = self.generator(cond, f0)
        return audio[:, 0, :] if audio.ndim == 3 and audio.shape[1] == 1 else audio


class ExportSafeKokoroFeatureConditioner(nn.Module):
    """Slice-free conditioner for dynamic-shape LiteRT export."""

    def __init__(self, src: KokoroFeatureConditioner):
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

    def forward(self, features: Tensor) -> Tensor:
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


class ExportSafeSTFT(nn.Module):
    """STFT variant using real-valued DFT projections instead of torch.fft."""

    def __init__(self, src: STFT):
        super().__init__()
        self.n_fft = int(src.n_fft)
        self.n_bins = int(src.n_bins)
        self.hop_length = int(src.hop_length)
        self.register_buffer("window", src.window.detach().clone())
        self.register_buffer("window_envelope", src.window_envelope.detach().clone())
        self.register_buffer("enframe_kernel", src.enframe_kernel.detach().clone())

        n = torch.arange(self.n_fft, dtype=torch.float32).unsqueeze(0)
        k = torch.arange(self.n_bins, dtype=torch.float32).unsqueeze(1)
        angle = 2.0 * torch.pi * k * n / float(self.n_fft)
        self.register_buffer("dft_cos", torch.cos(angle).unsqueeze(-1))
        self.register_buffer("dft_sin", torch.sin(angle).unsqueeze(-1))

        mid_k = torch.arange(1, self.n_bins - 1, dtype=torch.float32)
        n_col = torch.arange(self.n_fft, dtype=torch.float32).unsqueeze(1)
        mid_angle = 2.0 * torch.pi * n_col * mid_k.unsqueeze(0) / float(self.n_fft)
        self.register_buffer("idft_cos_mid", torch.cos(mid_angle).unsqueeze(-1))
        self.register_buffer("idft_sin_mid", torch.sin(mid_angle).unsqueeze(-1))
        self.register_buffer(
            "nyquist_sign",
            torch.pow(torch.tensor(-1.0, dtype=torch.float32), torch.arange(self.n_fft, dtype=torch.float32)),
        )
        shift_kernel = torch.zeros(self.n_fft, 1, self.n_fft, dtype=torch.float32)
        for c in range(self.n_fft):
            shift_kernel[c, 0, self.n_fft - 1 - c] = 1.0
        self.register_buffer("ola_shift_kernel", shift_kernel)
        hop_mask = torch.zeros(1, 1, self.hop_length, dtype=torch.float32)
        hop_mask[..., 0] = 1.0
        self.register_buffer("hop_mask", hop_mask)

    def forward(self, x: Tensor, norm: str | None = None) -> tuple[Tensor, Tensor]:
        if norm is not None:
            raise ValueError("ExportSafeSTFT only supports norm=None")
        pad = self.n_fft - self.hop_length
        pad_left = pad // 2
        x = F.pad(x, (pad_left, pad - pad_left))
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = F.conv1d(x, self.enframe_kernel.to(dtype=x.dtype), stride=self.hop_length)
        x = x * self.window.to(dtype=x.dtype)
        real = F.conv1d(x, self.dft_cos.to(dtype=x.dtype))
        imag = -F.conv1d(x, self.dft_sin.to(dtype=x.dtype))
        return real, imag

    def inverse(self, real: Tensor, imag: Tensor, norm: str | None = None) -> Tensor:
        if norm is not None:
            raise ValueError("ExportSafeSTFT only supports norm=None")
        frames = real.shape[2]
        samples = frames * self.hop_length

        dc = real[:, 0:1, :]
        nyquist = real[:, -1:, :] * self.nyquist_sign.to(dtype=real.dtype).view(1, self.n_fft, 1)
        real_mid = real[:, 1:-1, :]
        imag_mid = imag[:, 1:-1, :]
        inner = F.conv1d(real_mid, self.idft_cos_mid.to(dtype=real.dtype))
        inner = inner - F.conv1d(imag_mid, self.idft_sin_mid.to(dtype=imag.dtype))
        x = (dc + nyquist + 2.0 * inner) / float(self.n_fft)

        x = x * self.window.to(dtype=x.dtype)
        x = self._overlap_add(x)
        window_envelope = self._overlap_add(self.window_envelope.to(dtype=x.dtype).repeat(1, 1, frames))

        pad = (self.n_fft - self.hop_length) // 2
        x = x[..., pad : samples + pad]
        window_envelope = window_envelope[..., pad : samples + pad]
        return x / window_envelope.clamp_min(1e-11)

    def _overlap_add(self, x: Tensor) -> Tensor:
        hop = int(self.hop_length)
        if hop > 1:
            frames = x.shape[-1]
            x_up = x.repeat_interleave(hop, dim=2)
            mask = self.hop_mask.to(dtype=x.dtype, device=x.device).repeat(1, 1, frames)
            x_up = x_up * mask
        else:
            x_up = x
        shifted = F.conv1d(
            x_up,
            self.ola_shift_kernel.to(dtype=x.dtype, device=x.device),
            stride=1,
            padding=self.n_fft - 1,
            groups=self.n_fft,
        )
        out_size = (x.shape[-1] - 1) * hop + self.n_fft
        return shifted.sum(dim=1, keepdim=True)[..., :out_size]


class ExportSafeMultiStream1d(nn.Module):
    """MultiStream synthesis with zero-insertion expressed without TRANSPOSE_CONV."""

    def __init__(self, src: MultiStream1d):
        super().__init__()
        self.num_split = int(src.num_split)
        self.conv_synthesis = _conv1d_zero_padding_clone(src.conv_synthesis)
        mask = torch.zeros(1, 1, self.num_split, dtype=torch.float32)
        mask[..., 0] = float(self.num_split)
        self.register_buffer("upsample_mask", mask)

    def synthesis(self, xs: list[Tensor]) -> Tensor:
        x = torch.cat(xs, dim=1)
        frames = x.shape[-1]
        x = x.repeat_interleave(self.num_split, dim=2)
        x = x * self.upsample_mask.to(dtype=x.dtype, device=x.device).repeat(1, 1, frames)
        return self.conv_synthesis(x)


class ExportOptimizedMultiScaleWavehaxGenerator(nn.Module):
    """Fast export wrapper: skips audio-rate harmonic prior/STFT analysis."""

    def __init__(self, src: MultiScaleWavehaxGenerator):
        super().__init__()
        self.in_channels = int(src.in_channels)
        self.n_fft = int(src.n_fft)
        self.n_bins = int(src.n_bins)
        self.hop_length = int(src.hop_length)
        self.sample_rate = int(src.sample_rate)
        self.num_splits = int(src.num_splits)
        self.stft = ExportSafeSTFT(src.stft)
        if isinstance(src.decomposer, MultiStream1d):
            self.decomposer = ExportSafeMultiStream1d(src.decomposer)
        else:
            self.decomposer = src.decomposer
        self.cond_proj = _conv1d_zero_padding_clone(src.cond_proj)
        self.input_proj = _conv2d_zero_padding_clone(src.input_proj)
        self.input_norm = src.input_norm
        self.blocks = src.blocks
        self.output_norm = src.output_norm
        self.output_proj = _conv2d_zero_padding_clone(src.output_proj)

    def forward(self, cond: Tensor, f0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        del f0
        cond = self.cond_proj(cond)
        b, _channels, frames = cond.shape
        cond = cond.view(b, self.num_splits, self.n_bins, frames)
        zero = cond * 0.0
        x = torch.cat([cond, zero, zero], dim=1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        x = self.output_proj(x)

        xs = list(x.chunk(2 * self.num_splits, dim=1))
        ys: list[Tensor] = []
        for real, imag in zip(xs[:-1:2], xs[1::2]):
            ys.append(self.stft.inverse(real.squeeze(1), imag.squeeze(1)))
        y = self.decomposer.synthesis(ys)
        prior = torch.zeros_like(y)
        return y, prior


def deterministic_pcph_closed_form(
    f0: Tensor,
    hop_length: int,
    sample_rate: int,
    power_factor: float = 0.1,
    max_frequency: float | None = None,
    epsilon: float = 1e-6,
) -> Tensor:
    """Deterministic Wavehax prior generator, with no runtime random ops."""

    f0_upsampled = F.interpolate(f0, scale_factor=hop_length, mode="linear", align_corners=False)
    phase_increment = f0_upsampled / float(sample_rate)
    phase = torch.cumsum(phase_increment, dim=2) * (2.0 * torch.pi)
    phase = torch.fmod(phase, 2.0 * torch.pi)

    limit_freq = float(max_frequency) if max_frequency is not None else float(sample_rate) / 2.0
    safe_f0 = torch.clamp(f0_upsampled, min=1e-5)
    n_harmonics = torch.floor(limit_freq / safe_f0)

    half_phase = phase / 2.0
    numerator = torch.cos(half_phase) - torch.cos((n_harmonics + 0.5) * phase)
    denominator = 2.0 * torch.sin(half_phase)
    harmonics = torch.where(
        torch.abs(denominator) > float(epsilon),
        numerator / denominator,
        torch.zeros_like(phase),
    )

    amp_scale = float(power_factor) * torch.sqrt(2.0 / torch.clamp(n_harmonics, min=1.0))
    vuv_mask = (f0_upsampled > 0.0).to(dtype=f0_upsampled.dtype)
    return harmonics * amp_scale * vuv_mask


def _conv1d_zero_padding_clone(src: nn.Conv1d) -> nn.Conv1d:
    conv = nn.Conv1d(
        in_channels=src.in_channels,
        out_channels=src.out_channels,
        kernel_size=src.kernel_size,
        stride=src.stride,
        padding=src.padding,
        dilation=src.dilation,
        groups=src.groups,
        bias=src.bias is not None,
        padding_mode="zeros",
    )
    with torch.no_grad():
        conv.weight.copy_(src.weight.detach())
        if src.bias is not None:
            conv.bias.copy_(src.bias.detach())
    return conv


def _conv2d_zero_padding_clone(src: nn.Conv2d) -> nn.Conv2d:
    conv = nn.Conv2d(
        in_channels=src.in_channels,
        out_channels=src.out_channels,
        kernel_size=src.kernel_size,
        stride=src.stride,
        padding=src.padding,
        dilation=src.dilation,
        groups=src.groups,
        bias=src.bias is not None,
        padding_mode="zeros",
    )
    with torch.no_grad():
        conv.weight.copy_(src.weight.detach())
        if src.bias is not None:
            conv.bias.copy_(src.bias.detach())
    return conv


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
        padding_mode="zeros",
    )
    with torch.no_grad():
        widened.weight.zero_()
        in_channels = int(src.in_channels)
        widened.weight[:, start_channel : start_channel + in_channels, :] = src.weight.detach()
        if src.bias is not None:
            widened.bias.copy_(src.bias.detach())
    return widened


def _widen_control_branch(branch: nn.Sequential, total_channels: int, start_channel: int) -> nn.Sequential:
    if not branch or not isinstance(branch[0], nn.Conv1d):
        raise TypeError("Expected control branch to start with Conv1d")
    layers = list(branch.children())
    layers[0] = _widen_conv1d_input(layers[0], total_channels, start_channel)
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Conv1d) and layer.padding_mode != "zeros":
            layers[i] = _conv1d_zero_padding_clone(layer)
    return nn.Sequential(*layers)


def _replace_reflect_convs_with_zero_padding(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv1d) and child.padding_mode != "zeros":
            setattr(module, name, _conv1d_zero_padding_clone(child))
        elif isinstance(child, nn.Conv2d) and child.padding_mode != "zeros":
            setattr(module, name, _conv2d_zero_padding_clone(child))
        else:
            _replace_reflect_convs_with_zero_padding(child)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export Kokoro Wavehax to LiteRT/TFLite")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/wavehax/last.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/wavehax"))
    parser.add_argument("--input-feature-glob", action="append", default=["data/af_alloy*.pt"])
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--num-frames", type=int, default=330)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--lightweight-conversion", action="store_true")
    parser.add_argument("--export-all-sample-lengths", action="store_true")
    parser.add_argument("--dynamic-frames", action="store_true")
    parser.add_argument("--dynamic-frame-min", type=int, default=16)
    parser.add_argument("--dynamic-frame-max", type=int, default=2400)
    parser.add_argument("--multisignature-static", action="store_true")
    parser.add_argument("--optimized-prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--zero-pad-reflect-convs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-litert-validation", action="store_true")
    parser.add_argument("--progress-file", type=Path, default=Path("PROGRESS.md"))
    return parser.parse_args()


def save_wav_16bit(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    waveform = np.nan_to_num(waveform)
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm16 = (waveform * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())


def _feature_paths_from_globs(patterns: Sequence[str], limit: int) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(pattern))
    return sorted(dict.fromkeys(p.resolve() for p in paths))[: max(1, int(limit))]


def _compose_features_from_pt(path: Path) -> FeatureSample:
    row = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(row, Mapping):
        raise TypeError(f"Expected mapping in {path}, got {type(row)}")
    asr = row["asr"].float()
    f0 = row["f0"].float()
    noise = row["noise"].float()
    style = row["style"].float()
    total_frames = int(f0.shape[-1])
    if asr.shape[-1] != total_frames:
        asr = F.interpolate(asr.unsqueeze(0), size=total_frames, mode="linear", align_corners=False).squeeze(0)
    features = torch.cat(
        [
            asr[:, :total_frames],
            f0[:total_frames].unsqueeze(0),
            noise[:total_frames].unsqueeze(0),
            style.unsqueeze(-1).expand(style.shape[0], total_frames),
        ],
        dim=0,
    )
    return FeatureSample(
        tag=path.stem,
        path=path,
        features=features.contiguous(),
        sample_rate=int(row.get("sample_rate", 24000)),
        frame_hop=int(row.get("frame_hop", 300)),
    )


def _trim_or_pad_features(features: torch.Tensor, target_frames: int) -> torch.Tensor:
    frames = int(features.shape[-1])
    if frames == target_frames:
        return features
    if frames > target_frames:
        return features[:, :target_frames]
    pad = torch.zeros(features.shape[0], target_frames - frames, dtype=features.dtype)
    return torch.cat([features, pad], dim=-1)


def _load_wavehax_model(
    checkpoint_path: Path,
    sample_rate: int,
    *,
    optimized_prior: bool,
    zero_pad_reflect_convs: bool,
) -> tuple[dict[str, object], nn.Module]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, Mapping):
        raise TypeError(f"Expected checkpoint mapping in {checkpoint_path}, got {type(ckpt)}")
    config = dict(ckpt["backend_config"])
    model = KokoroMultiScaleWavehaxGenerator(config, sample_rate=sample_rate).eval()
    model.load_state_dict(ckpt["generator"], strict=True)
    _patch_model_for_export(
        model,
        optimized_prior=optimized_prior,
        zero_pad_reflect_convs=zero_pad_reflect_convs,
    )
    return config, model


def _patch_model_for_export(
    model: nn.Module,
    *,
    optimized_prior: bool,
    zero_pad_reflect_convs: bool,
) -> None:
    if optimized_prior:
        if isinstance(model.conditioner, KokoroFeatureConditioner):
            model.conditioner = ExportSafeKokoroFeatureConditioner(model.conditioner)
        if not isinstance(model.generator, MultiScaleWavehaxGenerator):
            raise TypeError(f"Expected MultiScaleWavehaxGenerator, got {type(model.generator)}")
        model.generator = ExportOptimizedMultiScaleWavehaxGenerator(model.generator)
    else:
        if isinstance(model.conditioner, KokoroFeatureConditioner):
            model.conditioner = ExportSafeKokoroFeatureConditioner(model.conditioner)
        for module in model.modules():
            if hasattr(module, "stft") and isinstance(module.stft, STFT):
                module.stft = ExportSafeSTFT(module.stft)
            if hasattr(module, "prior_generator") and hasattr(module, "hop_length") and hasattr(module, "sample_rate"):
                module.prior_generator = partial(
                    deterministic_pcph_closed_form,
                    hop_length=int(module.hop_length),
                    sample_rate=int(module.sample_rate),
                )
    if zero_pad_reflect_convs:
        _replace_reflect_convs_with_zero_padding(model)


def _export_litert(
    model: nn.Module,
    sample_arg: torch.Tensor,
    out_path: Path,
    *,
    lightweight_conversion: bool,
    dynamic_shapes: tuple[object, ...] | None = None,
    quant_config: object | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    samples: Sequence[FeatureSample],
    out_path: Path,
    *,
    lightweight_conversion: bool,
    quant_config: object | None = None,
) -> Path:
    if not samples:
        raise RuntimeError("Cannot export a multi-signature model without samples.")

    first = samples[0]
    first_frames = int(first.features.shape[-1])
    converter = litert_torch.signature(
        f"frames_{first_frames}",
        copy.deepcopy(model).eval(),
        sample_args=(first.features.float().unsqueeze(0),),
    )

    seen = {first_frames}
    for sample in samples[1:]:
        frames = int(sample.features.shape[-1])
        if frames in seen:
            continue
        seen.add(frames)
        converter.add_signature(
            f"frames_{frames}",
            copy.deepcopy(model).eval(),
            sample_args=(sample.features.float().unsqueeze(0),),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    edge_model = converter.convert(
        strict_export=False,
        quant_config=quant_config,
        lightweight_conversion=lightweight_conversion,
    )
    edge_model.export(str(out_path))
    return out_path


def _run_litert_inference(model_path: Path, features: np.ndarray) -> np.ndarray:
    try:
        model = litert_torch.load(str(model_path))
        interpreter = model._get_interpreter()
    except RuntimeError:
        interpreter = tfl_interpreter_utils.create_tfl_interpreter(
            str(model_path),
            allocate_tensors=False,
            use_xnnpack=False,
            preserve_all_tensors=False,
        )
    signatures = list(interpreter.get_signature_list().keys())
    if not signatures:
        raise RuntimeError(f"No TFLite signatures found in {model_path}")
    runner = interpreter.get_signature_runner(signatures[0])
    input_details = runner.get_input_details()
    input_name = next(iter(input_details.keys()))
    in_dtype = np.dtype(input_details[input_name]["dtype"])
    pred_map = runner(**{input_name: np.asarray(features, dtype=in_dtype)})
    return np.asarray(next(iter(pred_map.values())), dtype=np.float32)


def _schema_enum_names(enum_cls: object) -> dict[int, str]:
    names: dict[int, str] = {}
    for name in dir(enum_cls):
        if name.isupper() and isinstance(getattr(enum_cls, name), int):
            names[int(getattr(enum_cls, name))] = name
    return names


def _write_tflite_diagnostics(model_path: Path, out_path: Path) -> None:
    from ai_edge_litert import schema_py_generated as schema

    fb_model = schema.Model.GetRootAsModel(model_path.read_bytes(), 0)
    op_code_names = _schema_enum_names(schema.BuiltinOperator)
    tensor_type_names = _schema_enum_names(schema.TensorType)
    opcode_builtin: list[int] = []
    opcode_custom: list[str] = []
    for i in range(fb_model.OperatorCodesLength()):
        code = fb_model.OperatorCodes(i)
        opcode_builtin.append(int(code.BuiltinCode()))
        custom = code.CustomCode()
        opcode_custom.append(custom.decode("utf-8") if custom else "")

    lines = [f"Model: {model_path}", ""]
    op_hist: dict[str, int] = {}
    type_hist: dict[str, int] = {}
    for subgraph_idx in range(fb_model.SubgraphsLength()):
        subgraph = fb_model.Subgraphs(subgraph_idx)
        lines.append(f"Subgraph {subgraph_idx}: tensors={subgraph.TensorsLength()} ops={subgraph.OperatorsLength()}")
        tensor_meta: dict[int, tuple[str, list[int]]] = {}
        for tensor_idx in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(tensor_idx)
            tensor_type = tensor_type_names.get(int(tensor.Type()), str(int(tensor.Type())))
            shape = [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())]
            tensor_meta[tensor_idx] = (tensor_type, shape)
            type_hist[tensor_type] = type_hist.get(tensor_type, 0) + 1
        for op_idx in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(op_idx)
            code_idx = int(op.OpcodeIndex())
            name = opcode_custom[code_idx] or op_code_names.get(opcode_builtin[code_idx], str(opcode_builtin[code_idx]))
            op_hist[name] = op_hist.get(name, 0) + 1
            inputs = [int(op.Inputs(i)) for i in range(op.InputsLength())]
            outputs = [int(op.Outputs(i)) for i in range(op.OutputsLength())]
            lines.append(
                f"  {op_idx:03d} {name} "
                f"input_shapes={[tensor_meta.get(i, ('', []))[1] for i in inputs]} "
                f"output_shapes={[tensor_meta.get(i, ('', []))[1] for i in outputs]}"
            )
        lines.append("")
    lines.append("Operator histogram:")
    lines.extend(f"  {name}: {count}" for name, count in sorted(op_hist.items(), key=lambda kv: (-kv[1], kv[0])))
    lines.append("")
    lines.append("Tensor type histogram:")
    lines.extend(f"  {name}: {count}" for name, count in sorted(type_hist.items(), key=lambda kv: (-kv[1], kv[0])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _memory_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _append_progress(progress_file: Path, lines: Sequence[str]) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    if not progress_file.exists():
        progress_file.write_text("# Progress\n\n", encoding="utf-8")
    with progress_file.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    logger.enable("wavehax_export")
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    start = time.time()
    args.output_dir = args.output_dir.resolve()
    diagnostics_dir = args.output_dir / "diagnostics"
    wav_dir = args.output_dir / "sample_audio"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    samples = [_compose_features_from_pt(path) for path in _feature_paths_from_globs(args.input_feature_glob, args.sample_count)]
    if not samples:
        raise RuntimeError(f"No feature samples matched: {args.input_feature_glob}")

    config, model = _load_wavehax_model(
        args.checkpoint.resolve(),
        sample_rate=args.sample_rate,
        optimized_prior=args.optimized_prior,
        zero_pad_reflect_convs=args.zero_pad_reflect_convs,
    )
    input_channels = 512 + 1 + 1 + 128
    fixed_frames = int(args.num_frames)
    fixed = _trim_or_pad_features(samples[0].features.float(), fixed_frames).unsqueeze(0)

    with torch.inference_mode():
        for sample in samples:
            audio = model(sample.features.float().unsqueeze(0)).detach().cpu().numpy()
            save_wav_16bit(wav_dir / f"{sample.tag}_pytorch.wav", audio, sample.sample_rate)
            logger.info(f"Wrote PyTorch WAV for {sample.tag}: shape={audio.shape}")

    variants: list[LiteRTVariant] = []
    fp32_path = args.output_dir / "wavehax_fp32_litert.tflite"
    fp16_path = args.output_dir / "wavehax_fp16_litert.tflite"

    _export_litert(
        model=model,
        sample_arg=fixed,
        out_path=fp32_path,
        lightweight_conversion=args.lightweight_conversion,
    )
    variants.append(LiteRTVariant("fp32", fp32_path))
    _write_tflite_diagnostics(fp32_path, diagnostics_dir / f"{fp32_path.stem}_op_inventory.txt")
    logger.info(f"Exported fp32 LiteRT model: {fp32_path}")

    _export_litert(
        model=model,
        sample_arg=fixed,
        out_path=fp16_path,
        lightweight_conversion=args.lightweight_conversion,
        quant_config=quant_recipes.full_fp16_recipe(),
    )
    variants.append(LiteRTVariant("fp16", fp16_path))
    _write_tflite_diagnostics(fp16_path, diagnostics_dir / f"{fp16_path.stem}_op_inventory.txt")
    logger.info(f"Exported fp16 LiteRT model: {fp16_path}")

    dynamic_path: Path | None = None
    dynamic_error = ""
    if args.dynamic_frames:
        dynamic_path = args.output_dir / "wavehax_fp16_dynamic_litert.tflite"
        try:
            _export_litert(
                model=model,
                sample_arg=fixed,
                out_path=dynamic_path,
                lightweight_conversion=args.lightweight_conversion,
                dynamic_shapes=(
                    {2: torch.export.Dim("frames", min=args.dynamic_frame_min, max=args.dynamic_frame_max)},
                ),
                quant_config=quant_recipes.full_fp16_recipe(),
            )
            _write_tflite_diagnostics(dynamic_path, diagnostics_dir / f"{dynamic_path.stem}_op_inventory.txt")
            logger.info(f"Exported dynamic-frame fp16 LiteRT model: {dynamic_path}")
        except Exception as exc:
            dynamic_error = f"{type(exc).__name__}: {exc}"
            dynamic_path = None
            (diagnostics_dir / "wavehax_fp16_dynamic_export_failure.txt").write_text(
                "Dynamic-frame Wavehax LiteRT export failed.\n\n"
                f"sample_shape={tuple(fixed.shape)}\n"
                f"dynamic_frame_min={args.dynamic_frame_min}\n"
                f"dynamic_frame_max={args.dynamic_frame_max}\n\n"
                f"{dynamic_error}\n",
                encoding="utf-8",
            )
            logger.warning(f"Dynamic-frame fp16 export failed: {dynamic_error}")

    multisig_path: Path | None = None
    multisig_error = ""
    if args.multisignature_static:
        multisig_path = args.output_dir / "wavehax_fp16_multisig_static_litert.tflite"
        try:
            _export_litert_multisignature_static(
                model=model,
                samples=samples,
                out_path=multisig_path,
                lightweight_conversion=args.lightweight_conversion,
                quant_config=quant_recipes.full_fp16_recipe(),
            )
            _write_tflite_diagnostics(multisig_path, diagnostics_dir / f"{multisig_path.stem}_op_inventory.txt")
            logger.info(f"Exported static multi-signature fp16 LiteRT model: {multisig_path}")
        except Exception as exc:
            multisig_error = f"{type(exc).__name__}: {exc}"
            multisig_path = None
            (diagnostics_dir / "wavehax_fp16_multisig_export_failure.txt").write_text(
                "Static multi-signature Wavehax LiteRT export failed.\n\n"
                f"samples={[sample.tag for sample in samples]}\n\n"
                f"{multisig_error}\n",
                encoding="utf-8",
            )
            logger.warning(f"Static multi-signature fp16 export failed: {multisig_error}")

    sample_length_models: list[Path] = []
    if args.export_all_sample_lengths:
        for sample in samples:
            frames = int(sample.features.shape[-1])
            sample_input = sample.features.float().unsqueeze(0)
            sample_path = args.output_dir / f"wavehax_fp16_{sample.tag}_{frames}f_litert.tflite"
            sample_model = copy.deepcopy(model).eval()
            _export_litert(
                model=sample_model,
                sample_arg=sample_input,
                out_path=sample_path,
                lightweight_conversion=args.lightweight_conversion,
                quant_config=quant_recipes.full_fp16_recipe(),
            )
            _write_tflite_diagnostics(sample_path, diagnostics_dir / f"{sample_path.stem}_op_inventory.txt")
            sample_length_models.append(sample_path)
            logger.info(f"Exported sample-length fp16 LiteRT model: {sample_path}")

    quality: list[dict[str, object]] = []
    if not args.skip_litert_validation:
        for sample in samples:
            fixed_sample = _trim_or_pad_features(sample.features.float(), fixed_frames)
            input_np = fixed_sample.unsqueeze(0).numpy().astype(np.float32)
            with torch.inference_mode():
                torch_pred = model(fixed_sample.unsqueeze(0)).detach().cpu().numpy().astype(np.float32)
            for variant in variants:
                pred = _run_litert_inference(variant.path, input_np)
                save_wav_16bit(wav_dir / f"{sample.tag}_{variant.name}_litert.wav", pred, sample.sample_rate)
                diff = pred.reshape(-1) - torch_pred.reshape(-1)
                quality.append(
                    {
                        "sample": sample.tag,
                        "variant": variant.name,
                        "output_shape": list(pred.shape),
                        "finite": bool(np.isfinite(pred).all()),
                        "rms": float(np.sqrt(np.mean(pred.reshape(-1) ** 2))),
                        "peak": float(np.max(np.abs(pred.reshape(-1)))),
                        "max_abs_error_vs_pytorch": float(np.max(np.abs(diff))),
                        "mean_abs_error_vs_pytorch": float(np.mean(np.abs(diff))),
                    }
                )
        (diagnostics_dir / "quality.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "output_dir": str(args.output_dir),
        "fixed_frames": fixed_frames,
        "input_channels": input_channels,
        "optimized_prior": bool(args.optimized_prior),
        "zero_pad_reflect_convs": bool(args.zero_pad_reflect_convs),
        "config": config,
        "samples": [
            {
                "tag": s.tag,
                "path": str(s.path),
                "features_shape": list(s.features.shape),
                "sample_rate": s.sample_rate,
                "frame_hop": s.frame_hop,
            }
            for s in samples
        ],
        "models": [{"name": v.name, "path": str(v.path), "bytes": v.path.stat().st_size} for v in variants],
        "sample_length_models": [
            {"path": str(path), "bytes": path.stat().st_size} for path in sample_length_models if path.exists()
        ],
        "dynamic_model": (
            {"path": str(dynamic_path), "bytes": dynamic_path.stat().st_size}
            if dynamic_path is not None and dynamic_path.exists()
            else None
        ),
        "dynamic_error": dynamic_error,
        "multisig_model": (
            {"path": str(multisig_path), "bytes": multisig_path.stat().st_size}
            if multisig_path is not None and multisig_path.exists()
            else None
        ),
        "multisig_error": multisig_error,
        "quality": quality,
        "elapsed_seconds": round(time.time() - start, 3),
        "max_rss_mb": round(_memory_mb(), 1),
    }
    (diagnostics_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    _append_progress(
        args.progress_file,
        [
            f"## Wavehax export - {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Checkpoint: `{args.checkpoint}`",
            f"- Output: `{args.output_dir}`",
            f"- Exported: `{fp32_path.name}`, `{fp16_path.name}`",
            f"- Optimized prior: {bool(args.optimized_prior)}",
            f"- Zero-pad reflect convs: {bool(args.zero_pad_reflect_convs)}",
            f"- Dynamic fp16: `{dynamic_path.name}`" if dynamic_path else f"- Dynamic fp16: not exported ({dynamic_error or 'not requested'})",
            f"- Multisig fp16: `{multisig_path.name}`" if multisig_path else f"- Multisig fp16: not exported ({multisig_error or 'not requested'})",
            f"- Sample-length fp16 models: {len(sample_length_models)}",
            f"- Samples: {', '.join(s.tag for s in samples)}",
            f"- Fixed export shape: `[1, {input_channels}, {fixed_frames}]`",
            f"- WAVs: `{wav_dir}`",
            f"- Diagnostics: `{diagnostics_dir}`",
            f"- Elapsed: {time.time() - start:.1f}s; max RSS: {_memory_mb():.1f} MB",
            "",
        ],
    )
    logger.info(f"Done. Outputs saved under {args.output_dir}; progress updated at {args.progress_file}")


if __name__ == "__main__":
    main()
