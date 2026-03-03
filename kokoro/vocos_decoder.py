"""Decoder adapters for Kokoro PT/TF Vocos backends.

This module provides:
- `PTVocosDecoder`: PyTorch Vocos generator wrapper.
- `StreamingPTVocosDecoder`: Causal chunked PyTorch streaming decoder.
- `TFVocosDecoder`: TensorFlow Vocos generator wrapper callable from KModel.
- `StreamingTFVocosDecoder`: Causal chunked TensorFlow streaming decoder.

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

import contextlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import List, Mapping, Optional, Type

import numpy as np
import torch
from loguru import logger
from torch import nn
from vocos.heads import ISTFTHead as LegacyISTFTHead
from vocos.models import VocosBackbone as LegacyVocosBackbone


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


_STREAMING_VOCOS_CLASSES: Optional[tuple[Type[nn.Module], Type[nn.Module]]] = None


def resolve_streaming_vocos_classes(repo_root: Optional[Path]) -> tuple[Type[nn.Module], Type[nn.Module]]:
    """Resolve streaming Vocos backbone/head classes from warisqr007/vocos layout."""
    global _STREAMING_VOCOS_CLASSES
    if _STREAMING_VOCOS_CLASSES is not None:
        return _STREAMING_VOCOS_CLASSES

    tried: List[str] = []
    try:
        mod = importlib.import_module("src.components")
        _STREAMING_VOCOS_CLASSES = (mod.VocosBackbone, mod.ISTFTHead)
        return _STREAMING_VOCOS_CLASSES
    except Exception as exc:  # noqa: BLE001
        tried.append(f"import src.components failed: {exc}")

    if repo_root is not None:
        repo_root = repo_root.resolve()
        src_root = repo_root / "src"
        if src_root.exists():
            package_name = "streaming_vocos_src"
            try:
                pkg = sys.modules.get(package_name)
                if pkg is None:
                    spec = importlib.util.spec_from_file_location(
                        package_name,
                        src_root / "__init__.py",
                        submodule_search_locations=[str(src_root)],
                    )
                    if spec is None or spec.loader is None:
                        raise RuntimeError(f"Could not build module spec for {src_root}")
                    pkg = importlib.util.module_from_spec(spec)
                    sys.modules[package_name] = pkg
                    spec.loader.exec_module(pkg)
                # Upstream streaming-vocos uses absolute imports like `from src.utils ...`.
                sys.modules["src"] = pkg
                utils_root = src_root / "utils"
                if utils_root.exists():
                    for pkg_name in (f"{package_name}.utils", "src.utils"):
                        if pkg_name not in sys.modules:
                            m = types.ModuleType(pkg_name)
                            m.__path__ = [str(utils_root)]  # type: ignore[attr-defined]
                            sys.modules[pkg_name] = m
                    for mod_name in (f"{package_name}.utils.compile", "src.utils.compile"):
                        if mod_name not in sys.modules:
                            c_spec = importlib.util.spec_from_file_location(mod_name, utils_root / "compile.py")
                            if c_spec is None or c_spec.loader is None:
                                raise RuntimeError(f"Could not load {utils_root / 'compile.py'}")
                            c_mod = importlib.util.module_from_spec(c_spec)
                            sys.modules[mod_name] = c_mod
                            c_spec.loader.exec_module(c_mod)
                mod = importlib.import_module(f"{package_name}.components")
                _STREAMING_VOCOS_CLASSES = (mod.VocosBackbone, mod.ISTFTHead)
                return _STREAMING_VOCOS_CLASSES
            except Exception as exc:  # noqa: BLE001
                tried.append(f"import via package loader repo_root={repo_root} failed: {exc}")
            repo_path = str(repo_root)
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
            try:
                mod = importlib.import_module("src.components")
                _STREAMING_VOCOS_CLASSES = (mod.VocosBackbone, mod.ISTFTHead)
                return _STREAMING_VOCOS_CLASSES
            except Exception as exc:  # noqa: BLE001
                tried.append(f"import via repo_root={repo_root} failed: {exc}")

    raise RuntimeError(
        "Could not import streaming Vocos classes from warisqr007/vocos. "
        "Set streaming_vocos_repo to a local clone path containing src/components "
        f"(attempts: {' | '.join(tried)})"
    )


def _infer_vocos_impl(state_dict: Optional[Mapping[str, torch.Tensor]], requested: str) -> str:
    req = str(requested).lower()
    if req in {"streaming", "legacy"}:
        return req
    if state_dict is None:
        return "streaming"
    if "backbone.embed.conv.conv.weight_v" in state_dict or "backbone.convnext.0.dwconv.conv.conv.weight_v" in state_dict:
        return "streaming"
    return "legacy"


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
        vocos_impl: str = "streaming",
        streaming_vocos_repo: Optional[Path] = None,
        backbone_causal: bool = True,
        backbone_pad_mode: str = "constant",
        backbone_norm: str = "weight_norm",
    ):
        super().__init__()
        self.conditioner = nn.Sequential(
            nn.Conv1d(in_channels, model_input_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(model_input_channels, model_input_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(model_input_channels, model_input_channels, kernel_size=1),
        )
        self._head_has_channel_axis = False
        impl = str(vocos_impl).lower()
        if impl == "streaming":
            streaming_backbone_cls, streaming_head_cls = resolve_streaming_vocos_classes(streaming_vocos_repo)
            inferred_mlp_ratio = float(backbone_intermediate_dim) / float(max(1, backbone_dim))
            self.backbone = streaming_backbone_cls(
                input_channels=model_input_channels,
                dim=backbone_dim,
                mlp_ratio=inferred_mlp_ratio,
                kernel_size=7,
                dilation=1,
                norm=backbone_norm,
                causal=bool(backbone_causal),
                pad_mode=backbone_pad_mode,
                num_layers=backbone_layers,
            )
            self.head = streaming_head_cls(dim=backbone_dim, n_fft=n_fft, hop_length=hop_length)
            self._head_has_channel_axis = True
        elif impl == "legacy":
            self.backbone = LegacyVocosBackbone(
                input_channels=model_input_channels,
                dim=backbone_dim,
                intermediate_dim=backbone_intermediate_dim,
                num_layers=backbone_layers,
            )
            self.head = LegacyISTFTHead(dim=backbone_dim, n_fft=n_fft, hop_length=hop_length, padding=padding)
        else:
            raise ValueError(f"Unsupported vocos_impl={vocos_impl}. Expected one of: streaming, legacy")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.conditioner(features)
        x = self.backbone(x)
        y = self.head(x)
        if self._head_has_channel_axis and y.ndim == 3 and y.shape[1] == 1:
            return y[:, 0, :]
        return y


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
        vocos_impl: str = "auto",
        streaming_vocos_repo: Optional[str] = "third_party/vocos_streaming",
        backbone_causal: bool = True,
        backbone_pad_mode: str = "constant",
        backbone_norm: str = "weight_norm",
    ):
        super().__init__()
        in_channels = int(dim_in) + int(style_dim) + 2
        state: Optional[dict[str, torch.Tensor]] = None
        impl = str(vocos_impl).lower()
        if checkpoint_path:
            state = _load_generator_state(Path(checkpoint_path))
            impl = _infer_vocos_impl(state, impl)
        elif impl == "auto":
            impl = "streaming"
        self.generator = PairedVocosGenerator(
            in_channels=in_channels,
            model_input_channels=model_input_channels,
            backbone_dim=backbone_dim,
            backbone_intermediate_dim=backbone_intermediate_dim,
            backbone_layers=backbone_layers,
            n_fft=n_fft,
            hop_length=hop_length,
            padding=padding,
            vocos_impl=impl,
            streaming_vocos_repo=(Path(streaming_vocos_repo).resolve() if streaming_vocos_repo else None),
            backbone_causal=backbone_causal,
            backbone_pad_mode=backbone_pad_mode,
            backbone_norm=backbone_norm,
        )
        if state is not None:
            load = self.generator.load_state_dict(state, strict=False)
            if load.missing_keys or load.unexpected_keys:
                raise RuntimeError(
                    "Generator state_dict mismatch. "
                    f"missing={len(load.missing_keys)} unexpected={len(load.unexpected_keys)} "
                    f"(sample missing={load.missing_keys[:4]}, sample unexpected={load.unexpected_keys[:4]})"
                )
        else:
            logger.warning("PTVocosDecoder initialized without checkpoint_path; decoder weights are random.")

    def forward(self, asr: torch.Tensor, f0_curve: torch.Tensor, noise: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        features = _assemble_vocos_features(asr, f0_curve, noise, style)
        return self.generator(features)


class StreamingPTVocosDecoder(PTVocosDecoder):
    """Causal streaming Vocos decoder.

    Causal chunked decode using streaming-vocos architecture without wrapper-side
    rolling cache/state management.
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
        checkpoint_path: Optional[str] = None,
        vocos_impl: str = "auto",
        streaming_vocos_repo: Optional[str] = "third_party/vocos_streaming",
        backbone_causal: bool = True,
        backbone_pad_mode: str = "constant",
        backbone_norm: str = "weight_norm",
        sample_rate: int = 24000,
        chunk_size_ms: int = 300,
    ):
        _ = (backbone_layers, n_fft)
        super().__init__(
            dim_in=dim_in,
            style_dim=style_dim,
            model_input_channels=model_input_channels,
            backbone_dim=backbone_dim,
            backbone_intermediate_dim=backbone_intermediate_dim,
            backbone_layers=backbone_layers,
            n_fft=n_fft,
            hop_length=hop_length,
            checkpoint_path=checkpoint_path,
            vocos_impl=vocos_impl,
            streaming_vocos_repo=streaming_vocos_repo,
            backbone_causal=backbone_causal,
            backbone_pad_mode=backbone_pad_mode,
            backbone_norm=backbone_norm,
        )
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        self.chunk_size = max(1, int(chunk_size_ms / 1000.0 * self.sample_rate / self.hop_length))
        self._stream_ctx: Optional[contextlib.ExitStack] = None
        self._stream_batch_size = 0

    def reset(self) -> None:
        if self._stream_ctx is not None:
            self._stream_ctx.close()
        self._stream_ctx = None
        self._stream_batch_size = 0

    def _ensure_streaming_context(self, batch_size: int) -> None:
        if self._stream_ctx is not None and self._stream_batch_size == int(batch_size):
            return
        self.reset()
        stack = contextlib.ExitStack()
        if hasattr(self.generator.backbone, "streaming"):
            stack.enter_context(self.generator.backbone.streaming(int(batch_size)))
        if hasattr(self.generator.head, "streaming"):
            stack.enter_context(self.generator.head.streaming(int(batch_size)))
        self._stream_ctx = stack
        self._stream_batch_size = int(batch_size)

    @torch.no_grad()
    def decode_caches(self) -> torch.Tensor:
        device = next(self.generator.parameters()).device
        dtype = next(self.generator.parameters()).dtype
        bsz = max(1, int(self._stream_batch_size))
        self.reset()
        return torch.empty((bsz, 0), device=device, dtype=dtype)

    @torch.no_grad()
    def streaming_decode_features(self, features: torch.Tensor, is_last: bool = False):
        if features.ndim != 3:
            raise ValueError(f"Expected features shape [B,C,T], got {tuple(features.shape)}")
        bsz = int(features.shape[0])
        self._ensure_streaming_context(bsz)
        # The conditioner contains a non-streaming k=3 conv. Running it per-chunk
        # introduces artificial boundary transients. Condition once, then stream.
        conditioned = self.generator.conditioner(features)
        for feat_chunk in conditioned.split(self.chunk_size, dim=2):
            if feat_chunk.shape[-1] == 0:
                continue
            x = self.generator.backbone(feat_chunk)
            audio = self.generator.head(x)
            if getattr(self.generator, "_head_has_channel_axis", False) and audio.ndim == 3 and audio.shape[1] == 1:
                audio = audio[:, 0, :]
            yield audio
        if is_last:
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
        vocos_impl: str = "auto",
        streaming_vocos_repo: Optional[str] = "third_party/vocos_streaming",
        backbone_causal: bool = True,
        backbone_pad_mode: str = "constant",
        backbone_norm: str = "weight_norm",
    ):
        super().__init__()
        _ = (vocos_impl, streaming_vocos_repo, backbone_causal, backbone_pad_mode, backbone_norm)
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
    """Causal chunked decode using TensorFlow streaming-vocos architecture."""

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
        checkpoint_path: Optional[str] = None,
        vocos_impl: str = "auto",
        streaming_vocos_repo: Optional[str] = "third_party/vocos_streaming",
        backbone_causal: bool = True,
        backbone_pad_mode: str = "constant",
        backbone_norm: str = "weight_norm",
        sample_rate: int = 24000,
        chunk_size_ms: int = 300,
    ):
        _ = (backbone_layers, n_fft)
        super().__init__(
            dim_in=dim_in,
            style_dim=style_dim,
            model_input_channels=model_input_channels,
            backbone_dim=backbone_dim,
            backbone_intermediate_dim=backbone_intermediate_dim,
            backbone_layers=backbone_layers,
            n_fft=n_fft,
            hop_length=hop_length,
            checkpoint_path=checkpoint_path,
            vocos_impl=vocos_impl,
            streaming_vocos_repo=streaming_vocos_repo,
            backbone_causal=backbone_causal,
            backbone_pad_mode=backbone_pad_mode,
            backbone_norm=backbone_norm,
        )
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        self.chunk_size = max(1, int(chunk_size_ms / 1000.0 * self.sample_rate / self.hop_length))

    def reset(self) -> None:
        return

    @torch.no_grad()
    def decode_caches(self) -> torch.Tensor:
        return torch.empty((1, 0))

    @torch.no_grad()
    def streaming_decode_features(self, features: torch.Tensor, is_last: bool = False):
        _ = is_last
        if features.ndim != 3:
            raise ValueError(f"Expected features shape [B,C,T], got {tuple(features.shape)}")
        device = features.device
        dtype = features.dtype
        # Match PT streaming chunk semantics: run non-streaming conditioner once,
        # then chunk only backbone+head.
        feat_np = features.detach().cpu().numpy().astype(np.float32, copy=False)
        feat_tf = self._tf.convert_to_tensor(feat_np)
        conditioned = self.generator.conditioner(self._tf.transpose(feat_tf, [0, 2, 1]), training=False)
        conditioned = self._tf.transpose(conditioned, [0, 2, 1])
        total_frames = int(conditioned.shape[-1])
        for start in range(0, total_frames, self.chunk_size):
            end = min(total_frames, start + self.chunk_size)
            feat_chunk = conditioned[:, :, start:end]
            if int(feat_chunk.shape[-1]) == 0:
                continue
            x = self.generator.backbone(feat_chunk, training=False)
            audio_np = self.generator.head(x, training=False).numpy()
            audio_t = torch.from_numpy(audio_np).to(device=device, dtype=dtype)
            yield audio_t

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
