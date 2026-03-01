"""Decoder adapters for Kokoro PT/TF Vocos backends.

This module provides:
- `PTVocosDecoder`: PyTorch Vocos generator wrapper.
- `StreamingPTVocosDecoder`: Stateful PyTorch streaming decoder.
- `TFVocosDecoder`: TensorFlow Vocos generator wrapper callable from KModel.
- `StreamingTFVocosDecoder`: Stateful TensorFlow streaming decoder.

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
`uv run python -c "from kokoro.vocos_decoder import StreamingPTVocosDecoder; d=StreamingPTVocosDecoder(dim_in=512, style_dim=128, checkpoint_path='/path/vocos.pt'); print('ok')"`
`uv run python -c "from kokoro.vocos_decoder import TFVocosDecoder; d=TFVocosDecoder(dim_in=512, style_dim=128, checkpoint_path='/path/vocos.pt'); print('ok')"`
`uv run python -c "from kokoro.vocos_decoder import StreamingTFVocosDecoder; d=StreamingTFVocosDecoder(dim_in=512, style_dim=128, checkpoint_path='/path/vocos.pt'); print('ok')"`
"""

from __future__ import annotations

import math
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


class StreamingPTVocosDecoder(PTVocosDecoder):
    """Stateful streaming Vocos decoder.

    Implements the rolling-cache chunked decoding pattern from streaming-vocos:
    maintain feature caches of `[chunk + 2*padding]` frames, decode when enough
    frames are available, trim leading/trailing overlap in waveform domain.
    """

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
        sample_rate: int = 24000,
        chunk_size_ms: int = 300,
        padding_ms: Optional[int] = 40,
    ):
        super().__init__(
            dim_in=dim_in,
            style_dim=style_dim,
            model_input_channels=model_input_channels,
            backbone_dim=backbone_dim,
            backbone_intermediate_dim=backbone_intermediate_dim,
            backbone_layers=backbone_layers,
            n_fft=n_fft,
            hop_length=hop_length,
            padding=padding,
            checkpoint_path=checkpoint_path,
        )
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        requested_chunk = max(1, int(chunk_size_ms / 1000.0 * self.sample_rate / self.hop_length))
        requested_pad_ms = int(padding_ms) if padding_ms is not None else 40
        requested_padding = max(0, int(requested_pad_ms / 1000.0 * self.sample_rate / self.hop_length))

        # Approximate feature receptive field in frames:
        # conditioner k1->k3->k1 plus embed k7 and L depthwise-k7 ConvNeXt blocks.
        receptive_frames = 9 + 6 * int(backbone_layers)
        istft_half_frames = max(1, int(round((n_fft / float(hop_length)) / 2.0)))
        min_padding = int(math.ceil((receptive_frames - 1) / 2.0)) + istft_half_frames
        min_chunk = max(receptive_frames, 2 * min_padding)

        self.chunk_size = max(requested_chunk, min_chunk)
        self.padding = max(requested_padding, min_padding)
        if self.chunk_size != requested_chunk or self.padding != requested_padding:
            logger.warning(
                "StreamingPTVocosDecoder adjusted chunk/padding for receptive field: "
                f"requested_chunk={requested_chunk} requested_padding={requested_padding} "
                f"-> chunk={self.chunk_size} padding={self.padding} "
                f"(receptive_frames={receptive_frames})"
            )
        self.overlap_samples = self.padding * self.hop_length
        self._feature_channels = int(dim_in) + int(style_dim) + 2
        self._cache_batch_size = 0
        self.cur_idx = -1
        self.caches: Optional[torch.Tensor] = None
        self._pending_chunk: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.cur_idx = -1
        self._cache_batch_size = 0
        self.caches = None
        self._pending_chunk = None

    def _ensure_caches(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        cache_frames = self.chunk_size + 2 * self.padding
        if (
            self.caches is None
            or self._cache_batch_size != batch_size
            or self.caches.device != device
            or self.caches.dtype != dtype
        ):
            self._cache_batch_size = int(batch_size)
            self.cur_idx = -1
            self.caches = torch.zeros(
                (batch_size, self._feature_channels, cache_frames),
                device=device,
                dtype=dtype,
            )

    def get_size(self) -> int:
        effective_size = self.cur_idx + 1 - self.padding
        if effective_size <= 0:
            return 0
        return effective_size % self.chunk_size or self.chunk_size

    def _smooth_boundary(self, prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        """Length-preserving boundary smoothing with mild gain matching."""
        if prev.ndim != 2 or cur.ndim != 2:
            return cur
        n = min(self.overlap_samples, prev.shape[-1], cur.shape[-1])
        if n <= 0:
            return cur
        prev_tail = prev[:, -n:]
        cur_head = cur[:, :n]

        # Mild boundary gain alignment to reduce loudness pumping.
        prev_rms = torch.sqrt(torch.mean(prev_tail * prev_tail, dim=-1, keepdim=True) + 1e-8)
        cur_rms = torch.sqrt(torch.mean(cur_head * cur_head, dim=-1, keepdim=True) + 1e-8)
        scale = torch.clamp(prev_rms / (cur_rms + 1e-8), 0.90, 1.10)
        cur = cur * scale
        cur_head = cur[:, :n]

        # De-click/phase smoothing without shortening duration:
        # blend only current chunk head toward previous tail.
        fade_in = torch.linspace(0.0, 1.0, n, device=cur.device, dtype=cur.dtype).unsqueeze(0)
        cur[:, :n] = prev_tail * (1.0 - fade_in) + cur_head * fade_in
        return cur

    def _stitch_length_preserving(self, chunk: torch.Tensor, is_last: bool):
        if chunk.ndim != 2:
            return
        if self._pending_chunk is None:
            if is_last:
                yield chunk
            else:
                self._pending_chunk = chunk
            return

        prev = self._pending_chunk
        cur = self._smooth_boundary(prev, chunk)
        yield prev
        if is_last:
            yield cur
            self._pending_chunk = None
        else:
            self._pending_chunk = cur

    @torch.no_grad()
    def decode_caches(self) -> torch.Tensor:
        if self.caches is None:
            return torch.empty(1, 0)
        cur_size = self.get_size()
        if cur_size == 0:
            return torch.empty(self.caches.shape[0], 0, device=self.caches.device, dtype=self.caches.dtype)
        audio = self.generator(self.caches)
        if self.padding > 0:
            audio = audio[:, self.padding * self.hop_length :]
        audio = audio[:, (self.chunk_size - cur_size) * self.hop_length :]
        stitched = list(self._stitch_length_preserving(audio, is_last=True))
        self.reset()
        if not stitched:
            return torch.empty(audio.shape[0], 0, device=audio.device, dtype=audio.dtype)
        return torch.cat(stitched, dim=-1)

    @torch.no_grad()
    def streaming_decode_features(self, features: torch.Tensor, is_last: bool = False):
        if features.ndim != 3:
            raise ValueError(f"Expected features shape [B,C,T], got {tuple(features.shape)}")
        bsz, _, t = features.shape
        self._ensure_caches(bsz, features.device, features.dtype)
        for idx, feature in enumerate(torch.unbind(features, dim=2)):
            self.caches = torch.roll(self.caches, shifts=-1, dims=2)
            self.caches[:, :, -1] = feature
            self.cur_idx += 1

            is_last_feature = bool(is_last and idx == t - 1)
            cur_size = self.get_size()
            if cur_size != self.chunk_size and not is_last_feature:
                continue

            audio = self.generator(self.caches)
            if self.padding > 0:
                audio = audio[:, self.padding * self.hop_length :]
            if cur_size != self.chunk_size:
                audio = audio[:, (self.chunk_size - cur_size) * self.hop_length :]
            if not is_last_feature:
                audio = audio[:, : self.chunk_size * self.hop_length]
            for out in self._stitch_length_preserving(audio, is_last=is_last_feature):
                yield out
            if is_last_feature:
                self.reset()

    @torch.no_grad()
    def streaming_decode(
        self,
        asr: torch.Tensor,
        f0_curve: torch.Tensor,
        noise: torch.Tensor,
        style: torch.Tensor,
        is_last: bool = False,
    ):
        features = _assemble_vocos_features(asr, f0_curve, noise, style)
        yield from self.streaming_decode_features(features, is_last=is_last)


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


class StreamingTFVocosDecoder(TFVocosDecoder):
    """Stateful streaming decoder using TensorFlow Vocos generator."""

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
        sample_rate: int = 24000,
        chunk_size_ms: int = 300,
        padding_ms: Optional[int] = 40,
    ):
        super().__init__(
            dim_in=dim_in,
            style_dim=style_dim,
            model_input_channels=model_input_channels,
            backbone_dim=backbone_dim,
            backbone_intermediate_dim=backbone_intermediate_dim,
            backbone_layers=backbone_layers,
            n_fft=n_fft,
            hop_length=hop_length,
            padding=padding,
            checkpoint_path=checkpoint_path,
        )
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        requested_chunk = max(1, int(chunk_size_ms / 1000.0 * self.sample_rate / self.hop_length))
        requested_pad_ms = int(padding_ms) if padding_ms is not None else 40
        requested_padding = max(0, int(requested_pad_ms / 1000.0 * self.sample_rate / self.hop_length))

        receptive_frames = 9 + 6 * int(backbone_layers)
        istft_half_frames = max(1, int(round((n_fft / float(hop_length)) / 2.0)))
        min_padding = int(math.ceil((receptive_frames - 1) / 2.0)) + istft_half_frames
        min_chunk = max(receptive_frames, 2 * min_padding)

        self.chunk_size = max(requested_chunk, min_chunk)
        self.padding = max(requested_padding, min_padding)
        if self.chunk_size != requested_chunk or self.padding != requested_padding:
            logger.warning(
                "StreamingTFVocosDecoder adjusted chunk/padding for receptive field: "
                f"requested_chunk={requested_chunk} requested_padding={requested_padding} "
                f"-> chunk={self.chunk_size} padding={self.padding} "
                f"(receptive_frames={receptive_frames})"
            )
        self.overlap_samples = self.padding * self.hop_length
        self._feature_channels = int(dim_in) + int(style_dim) + 2
        self._cache_batch_size = 0
        self.cur_idx = -1
        self.caches: Optional[torch.Tensor] = None
        self._pending_chunk: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.cur_idx = -1
        self._cache_batch_size = 0
        self.caches = None
        self._pending_chunk = None

    def _ensure_caches(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        cache_frames = self.chunk_size + 2 * self.padding
        if (
            self.caches is None
            or self._cache_batch_size != batch_size
            or self.caches.device != device
            or self.caches.dtype != dtype
        ):
            self._cache_batch_size = int(batch_size)
            self.cur_idx = -1
            self.caches = torch.zeros(
                (batch_size, self._feature_channels, cache_frames),
                device=device,
                dtype=dtype,
            )

    def get_size(self) -> int:
        effective_size = self.cur_idx + 1 - self.padding
        if effective_size <= 0:
            return 0
        return effective_size % self.chunk_size or self.chunk_size

    def _smooth_boundary(self, prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        """Length-preserving boundary smoothing with mild gain matching."""
        if prev.ndim != 2 or cur.ndim != 2:
            return cur
        n = min(self.overlap_samples, prev.shape[-1], cur.shape[-1])
        if n <= 0:
            return cur
        prev_tail = prev[:, -n:]
        cur_head = cur[:, :n]

        prev_rms = torch.sqrt(torch.mean(prev_tail * prev_tail, dim=-1, keepdim=True) + 1e-8)
        cur_rms = torch.sqrt(torch.mean(cur_head * cur_head, dim=-1, keepdim=True) + 1e-8)
        scale = torch.clamp(prev_rms / (cur_rms + 1e-8), 0.90, 1.10)
        cur = cur * scale
        cur_head = cur[:, :n]

        fade_in = torch.linspace(0.0, 1.0, n, device=cur.device, dtype=cur.dtype).unsqueeze(0)
        cur[:, :n] = prev_tail * (1.0 - fade_in) + cur_head * fade_in
        return cur

    def _stitch_length_preserving(self, chunk: torch.Tensor, is_last: bool):
        if chunk.ndim != 2:
            return
        if self._pending_chunk is None:
            if is_last:
                yield chunk
            else:
                self._pending_chunk = chunk
            return

        prev = self._pending_chunk
        cur = self._smooth_boundary(prev, chunk)
        yield prev
        if is_last:
            yield cur
            self._pending_chunk = None
        else:
            self._pending_chunk = cur

    @torch.no_grad()
    def _decode_cache_tensor(self) -> torch.Tensor:
        if self.caches is None:
            return torch.empty(1, 0)
        feat_np = self.caches.detach().cpu().numpy().astype("float32", copy=False)
        audio = self.generator(self._tf.convert_to_tensor(feat_np), training=False).numpy()
        return torch.from_numpy(audio).to(device=self.caches.device, dtype=self.caches.dtype)

    @torch.no_grad()
    def decode_caches(self) -> torch.Tensor:
        if self.caches is None:
            return torch.empty(1, 0)
        cur_size = self.get_size()
        if cur_size == 0:
            return torch.empty(self.caches.shape[0], 0, device=self.caches.device, dtype=self.caches.dtype)
        audio = self._decode_cache_tensor()
        if self.padding > 0:
            audio = audio[:, self.padding * self.hop_length :]
        audio = audio[:, (self.chunk_size - cur_size) * self.hop_length :]
        stitched = list(self._stitch_length_preserving(audio, is_last=True))
        self.reset()
        if not stitched:
            return torch.empty(audio.shape[0], 0, device=audio.device, dtype=audio.dtype)
        return torch.cat(stitched, dim=-1)

    @torch.no_grad()
    def streaming_decode_features(self, features: torch.Tensor, is_last: bool = False):
        if features.ndim != 3:
            raise ValueError(f"Expected features shape [B,C,T], got {tuple(features.shape)}")
        bsz, _, t = features.shape
        self._ensure_caches(bsz, features.device, features.dtype)
        for idx, feature in enumerate(torch.unbind(features, dim=2)):
            self.caches = torch.roll(self.caches, shifts=-1, dims=2)
            self.caches[:, :, -1] = feature
            self.cur_idx += 1

            is_last_feature = bool(is_last and idx == t - 1)
            cur_size = self.get_size()
            if cur_size != self.chunk_size and not is_last_feature:
                continue

            audio = self._decode_cache_tensor()
            if self.padding > 0:
                audio = audio[:, self.padding * self.hop_length :]
            if cur_size != self.chunk_size:
                audio = audio[:, (self.chunk_size - cur_size) * self.hop_length :]
            if not is_last_feature:
                audio = audio[:, : self.chunk_size * self.hop_length]
            for out in self._stitch_length_preserving(audio, is_last=is_last_feature):
                yield out
            if is_last_feature:
                self.reset()

    @torch.no_grad()
    def streaming_decode(
        self,
        asr: torch.Tensor,
        f0_curve: torch.Tensor,
        noise: torch.Tensor,
        style: torch.Tensor,
        is_last: bool = False,
    ):
        features = _assemble_vocos_features(asr, f0_curve, noise, style)
        yield from self.streaming_decode_features(features, is_last=is_last)
