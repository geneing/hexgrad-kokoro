"""Decoder adapters for Kokoro PT/TF Vocos backends.

This module provides:
- `PTVocosDecoder`: PyTorch Vocos generator wrapper.
- `TFVocosDecoder`: TensorFlow Vocos generator wrapper callable from KModel.

Both consume Kokoro vocoder conditioning tensors `(asr, f0, noise, style)` and
return waveform audio.

Example config snippet for use with `KModel`:
{
  "decoder_type": "pt_vocos",
  "vocos": {
    "checkpoint_path": "/path/to/vocos_last.pt",
    "hop_length": 300,
    "n_fft": 1200
  }
}

Direct construction examples:
`uv run python -c "from kokoro.vocos_decoder import PTVocosDecoder; d=PTVocosDecoder(dim_in=512, style_dim=128, checkpoint_path='/path/vocos.pt'); print('ok')"`
`uv run python -c "from kokoro.vocos_decoder import TFVocosDecoder; d=TFVocosDecoder(dim_in=512, style_dim=128, checkpoint_path='/path/vocos.pt'); print('ok')"`
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import torch
from loguru import logger
from torch import nn
from vocos.heads import ISTFTHead
from vocos.models import VocosBackbone


def _strip_parallel_prefixes(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    fixed: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod.", "generator.", "decoder."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        fixed[new_key] = value
    return fixed


def _is_tensor_state_dict(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return all(torch.is_tensor(v) for v in value.values())


def _load_generator_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if _is_tensor_state_dict(raw):
        return _strip_parallel_prefixes(raw)
    if isinstance(raw, Mapping) and _is_tensor_state_dict(raw.get("generator")):
        return _strip_parallel_prefixes(raw["generator"])
    raise TypeError(f"Unsupported Vocos checkpoint format: {checkpoint_path}")


class PairedVocosGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_input_channels: int,
        backbone_dim: int,
        backbone_intermediate_dim: int,
        backbone_layers: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
    ):
        super().__init__()
        self.conditioner = nn.Sequential(
            nn.Conv1d(in_channels, model_input_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(model_input_channels, model_input_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(model_input_channels, model_input_channels, kernel_size=1),
        )
        self.backbone = VocosBackbone(
            input_channels=model_input_channels,
            dim=backbone_dim,
            intermediate_dim=backbone_intermediate_dim,
            num_layers=backbone_layers,
        )
        self.head = ISTFTHead(dim=backbone_dim, n_fft=n_fft, hop_length=hop_length, padding=padding)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.conditioner(features)
        x = self.backbone(x)
        return self.head(x)


def _assemble_vocos_features(
    asr: torch.Tensor,
    f0_curve: torch.Tensor,
    noise: torch.Tensor,
    style: torch.Tensor,
) -> torch.Tensor:
    total_frames = int(f0_curve.shape[-1])
    if asr.shape[-1] != total_frames:
        asr = torch.nn.functional.interpolate(asr, size=total_frames, mode="linear", align_corners=False)
    f0 = f0_curve.unsqueeze(1)
    n = noise.unsqueeze(1)
    style_t = style.unsqueeze(-1).expand(-1, style.shape[-1], total_frames)
    return torch.cat([asr, f0, n, style_t], dim=1)


class PTVocosDecoder(nn.Module):
    def __init__(
        self,
        dim_in: int,
        style_dim: int,
        model_input_channels: int = 192,
        backbone_dim: int = 384,
        backbone_intermediate_dim: int = 1152,
        backbone_layers: int = 8,
        n_fft: int = 1200,
        hop_length: int = 300,
        padding: str = "same",
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        in_channels = int(dim_in) + int(style_dim) + 2
        self.generator = PairedVocosGenerator(
            in_channels=in_channels,
            model_input_channels=model_input_channels,
            backbone_dim=backbone_dim,
            backbone_intermediate_dim=backbone_intermediate_dim,
            backbone_layers=backbone_layers,
            n_fft=n_fft,
            hop_length=hop_length,
            padding=padding,
        )
        if checkpoint_path:
            state = _load_generator_state(Path(checkpoint_path))
            self.generator.load_state_dict(state, strict=False)
        else:
            logger.warning("PTVocosDecoder initialized without checkpoint_path; decoder weights are random.")

    def forward(self, asr: torch.Tensor, f0_curve: torch.Tensor, noise: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        features = _assemble_vocos_features(asr, f0_curve, noise, style)
        return self.generator(features)


class TFVocosDecoder(nn.Module):
    def __init__(
        self,
        dim_in: int,
        style_dim: int,
        model_input_channels: int = 192,
        backbone_dim: int = 384,
        backbone_intermediate_dim: int = 1152,
        backbone_layers: int = 8,
        n_fft: int = 1200,
        hop_length: int = 300,
        padding: str = "same",
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        try:
            import tensorflow as tf
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("TensorFlow is required for decoder_type='tf_vocos'") from exc

        self._tf = tf
        self._hop_length = int(hop_length)
        in_channels = int(dim_in) + int(style_dim) + 2

        if checkpoint_path:
            from .tf.checkpoint_utils import (
                build_tf_generator,
                infer_tf_generator_config,
                load_pytorch_generator_state,
                load_pytorch_state_into_tf_generator,
            )

            state_dict, _ = load_pytorch_generator_state(Path(checkpoint_path))
            cfg = infer_tf_generator_config(state_dict, hop_length=hop_length, padding=padding)
            if cfg.in_channels != in_channels:
                raise ValueError(
                    f"Checkpoint expects in_channels={cfg.in_channels}, but KModel feature size is {in_channels}."
                )
            self.generator = build_tf_generator(cfg)
            load_pytorch_state_into_tf_generator(self.generator, state_dict)
        else:
            from .tf.model import PairedVocosGeneratorTF

            self.generator = PairedVocosGeneratorTF(
                in_channels=in_channels,
                model_input_channels=model_input_channels,
                backbone_dim=backbone_dim,
                backbone_intermediate_dim=backbone_intermediate_dim,
                backbone_layers=backbone_layers,
                n_fft=n_fft,
                hop_length=hop_length,
                padding=padding,
            )
            dummy = self._tf.zeros([1, in_channels, 16], dtype=self._tf.float32)
            _ = self.generator(dummy, training=False)
            logger.warning("TFVocosDecoder initialized without checkpoint_path; decoder weights are random.")

    def forward(self, asr: torch.Tensor, f0_curve: torch.Tensor, noise: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        features = _assemble_vocos_features(asr, f0_curve, noise, style)
        device = asr.device
        dtype = asr.dtype
        feat_np = features.detach().cpu().numpy()
        audio = self.generator(self._tf.convert_to_tensor(feat_np), training=False).numpy()
        audio_t = torch.from_numpy(audio).to(device=device, dtype=dtype)
        expected_samples = int(features.shape[-1]) * self._hop_length
        if expected_samples > 0 and audio_t.shape[-1] > expected_samples:
            audio_t = audio_t[..., :expected_samples]
        return audio_t
