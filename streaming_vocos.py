"""Self-contained streaming Vocos inference module.

This module implements:
1) Loading saved PyTorch streaming-vocos model weights (fp32/fp16/int8-like).
2) Stateful chunked synthesis with fixed chunk size (frames) set at init time.
3) No dependency on project-local third_party streaming-vocos sources.

Example usage:
```python
import torch
from streaming_vocos import StreamingVocos

# Load a saved inference checkpoint with a static chunk size of 24 frames.
decoder = StreamingVocos.from_checkpoint(
    checkpoint_path="output/saved_infer_weights/vocos.pt",
    chunk_frames=24,
    device="cuda",
)

# Features are [batch, channels, frames].
features = torch.randn(1, 514, 240, device="cuda")

# One-shot synthesis (internally chunked into 24-frame blocks).
audio = decoder.synthesize(features, is_last=True)  # [1, 240 * hop_length]
```

Streaming usage:
```python
import torch
from streaming_vocos import StreamingVocos

decoder = StreamingVocos.from_checkpoint(
    checkpoint_path="output/saved_infer_weights/vocos_fp16.pt",
    chunk_frames=16,
    device="cuda",
    use_fp16=True,
)

# Feed incoming feature windows as they arrive.
chunk1 = torch.randn(1, 514, 16, device="cuda")
chunk2 = torch.randn(1, 514, 16, device="cuda")

audio1 = decoder.synthesize_chunk(chunk1)  # keeps internal streaming state
audio2 = decoder.synthesize_chunk(chunk2)

# Reset state when stream ends.
decoder.reset()
```
"""

from __future__ import annotations

import contextlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import weight_norm


def _is_tensor_state_dict(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return all(torch.is_tensor(v) for v in value.values())


def _strip_parallel_prefixes(state_dict: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    fixed: Dict[str, Tensor] = {}
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


def _load_raw_state_dict(path: Path) -> Dict[str, Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if _is_tensor_state_dict(raw):
        return _strip_parallel_prefixes(raw)
    if isinstance(raw, Mapping):
        for key in ("state_dict", "generator", "model"):
            value = raw.get(key)
            if _is_tensor_state_dict(value):
                return _strip_parallel_prefixes(value)
    raise TypeError(f"Unsupported checkpoint format: {path}")


def _remap_quantized_inference_state(raw: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    """Map prepare_weights-style QuantizedVocosInference keys back to generator keys.

    The minimal runtime here is pure PyTorch and does not rebuild PT2E quantized graph
    modules. If slim q8 artifacts only contain graph-internal frozen params, this mapping
    cannot recover the ConvNeXt core and loading will fail with missing keys.
    """

    mapped: Dict[str, Tensor] = {}
    has_wrapped_keys = any(k.startswith(("pre.", "core.", "post.")) for k in raw.keys())
    if not has_wrapped_keys:
        return dict(raw)

    for key, value in raw.items():
        out_key: Optional[str] = None
        if key.startswith("pre.conditioner."):
            out_key = "conditioner." + key[len("pre.conditioner.") :]
        elif key.startswith("pre.embed."):
            out_key = "backbone.embed." + key[len("pre.embed.") :]
        elif key.startswith("pre.norm."):
            out_key = "backbone.norm." + key[len("pre.norm.") :]
        elif key.startswith("core.convnext."):
            out_key = "backbone.convnext." + key[len("core.convnext.") :]
        elif key.startswith("post.final_layer_norm."):
            out_key = "backbone.final_layer_norm." + key[len("post.final_layer_norm.") :]
        elif key.startswith("post.head."):
            out_key = "head." + key[len("post.head.") :]
        elif key.startswith(("pre.", "core.", "post.")):
            out_key = None
        else:
            out_key = key

        if out_key is not None:
            mapped[out_key] = value
    return mapped


def _dequantize_if_needed(value: Tensor) -> Tensor:
    if value.is_quantized:
        return value.dequantize()
    if value.dtype in (torch.int8, torch.uint8, torch.int16, torch.int32):
        # Fallback path for int-only tensors in "int8-like" checkpoints.
        return value.float()
    return value


def _normalize_checkpoint_state(path: Path) -> Dict[str, Tensor]:
    raw = _load_raw_state_dict(path)
    mapped = _remap_quantized_inference_state(raw)
    normalized: Dict[str, Tensor] = {}
    for key, value in mapped.items():
        normalized[key] = _dequantize_if_needed(value)
    return normalized


def _infer_int_key_from_weight_shape(state_dict: Mapping[str, Tensor], key: str, index: int) -> int:
    tensor = state_dict[key]
    return int(tensor.shape[index])


@dataclass
class StreamingVocosConfig:
    in_channels: int
    model_input_channels: int
    backbone_dim: int
    backbone_intermediate_dim: int
    backbone_layers: int
    n_fft: int
    hop_length: int = 300
    backbone_causal: bool = True
    backbone_pad_mode: str = "constant"
    backbone_norm: str = "weight_norm"

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Tensor],
        hop_length: int = 300,
        backbone_causal: bool = True,
        backbone_pad_mode: str = "constant",
        backbone_norm: str = "weight_norm",
    ) -> "StreamingVocosConfig":
        in_channels = _infer_int_key_from_weight_shape(state_dict, "conditioner.0.weight", 1)
        model_input_channels = _infer_int_key_from_weight_shape(state_dict, "conditioner.0.weight", 0)
        if "backbone.embed.conv.conv.weight_v" in state_dict:
            backbone_dim = _infer_int_key_from_weight_shape(state_dict, "backbone.embed.conv.conv.weight_v", 0)
        else:
            backbone_dim = _infer_int_key_from_weight_shape(state_dict, "backbone.embed.conv.conv.weight", 0)
        backbone_intermediate_dim = _infer_int_key_from_weight_shape(state_dict, "backbone.convnext.0.pwconv1.weight", 0)
        n_fft = int(state_dict["head.out.weight"].shape[0]) - 2

        layer_ids = set()
        pattern = re.compile(r"^backbone\.convnext\.(\d+)\.")
        for key in state_dict.keys():
            m = pattern.match(key)
            if m:
                layer_ids.add(int(m.group(1)))
        if not layer_ids:
            raise RuntimeError("Could not infer backbone layer count from checkpoint keys.")

        return cls(
            in_channels=in_channels,
            model_input_channels=model_input_channels,
            backbone_dim=backbone_dim,
            backbone_intermediate_dim=backbone_intermediate_dim,
            backbone_layers=max(layer_ids) + 1,
            n_fft=n_fft,
            hop_length=int(hop_length),
            backbone_causal=bool(backbone_causal),
            backbone_pad_mode=str(backbone_pad_mode),
            backbone_norm=str(backbone_norm),
        )


@dataclass
class _StreamingConvState:
    previous: Optional[Tensor] = None

    def reset(self) -> None:
        self.previous = None


@dataclass
class _StreamingConvPadState:
    padding_to_add: int
    original_padding_to_add: int

    def reset(self) -> None:
        self.padding_to_add = self.original_padding_to_add


@dataclass
class _StreamingAddState:
    previous_x: Optional[Tensor] = None
    previous_y: Optional[Tensor] = None

    def reset(self) -> None:
        self.previous_x = None
        self.previous_y = None


@dataclass
class _StreamingISTFTState:
    prev_buffer: Tensor

    def reset(self) -> None:
        self.prev_buffer.zero_()


class _StreamingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: Any = None

    @property
    def is_streaming(self) -> bool:
        return self._streaming_state is not None

    def _start_streaming(self, batch_size: int) -> None:
        for module in self.modules():
            if isinstance(module, _StreamingModule):
                module._streaming_state = module._init_streaming_state(batch_size)

    def _stop_streaming(self) -> None:
        for module in self.modules():
            if isinstance(module, _StreamingModule):
                module._streaming_state = None

    @contextlib.contextmanager
    def streaming(self, batch_size: int):
        self._start_streaming(batch_size)
        try:
            yield
        finally:
            self._stop_streaming()

    def _init_streaming_state(self, batch_size: int) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


def _get_extra_padding_for_conv1d(x: Tensor, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)


def _pad1d(x: Tensor, paddings: tuple[int, int], mode: str = "constant", value: float = 0.0) -> Tensor:
    padding_left, padding_right = paddings
    if mode == "reflect":
        length = x.shape[-1]
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        return padded[..., : padded.shape[-1] - extra_pad]
    return F.pad(x, paddings, mode, value)


class _RawStreamingConv1d(nn.Conv1d, _StreamingModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.padding[0] != 0:
            raise ValueError("Raw streaming conv expects external padding.")

    def _init_streaming_state(self, batch_size: int) -> _StreamingConvState:
        _ = batch_size
        return _StreamingConvState()

    def forward(self, x: Tensor) -> Tensor:
        stride = self.stride[0]
        kernel = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        if self._streaming_state is None:
            return super().forward(x)
        previous = self._streaming_state.previous
        if previous is not None:
            x = torch.cat([previous, x], dim=-1)
        bsz = x.shape[0]
        length = x.shape[-1]
        num_frames = max(0, int(math.floor((length - kernel) / stride) + 1))
        offset = num_frames * stride
        self._streaming_state.previous = x[..., offset:]
        if num_frames <= 0:
            return torch.empty(bsz, self.out_channels, 0, device=x.device, dtype=x.dtype)
        in_len = (num_frames - 1) * stride + kernel
        return super().forward(x[..., :in_len])


class _NormConv1d(nn.Module):
    def __init__(
        self,
        *args,
        norm: str = "none",
        **kwargs,
    ):
        super().__init__()
        conv = _RawStreamingConv1d(*args, **kwargs)
        self.conv = weight_norm(conv) if norm == "weight_norm" else conv

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _StreamingConv1d(_StreamingModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = True,
        norm: str = "weight_norm",
        pad_mode: str = "constant",
    ):
        super().__init__()
        self.conv = _NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            norm=norm,
        )
        self.causal = bool(causal)
        self.pad_mode = str(pad_mode)

    @property
    def _stride(self) -> int:
        return int(self.conv.conv.stride[0])

    @property
    def _kernel_size(self) -> int:
        return int(self.conv.conv.kernel_size[0])

    @property
    def _effective_kernel_size(self) -> int:
        dilation = int(self.conv.conv.dilation[0])
        return (self._kernel_size - 1) * dilation + 1

    @property
    def _padding_total(self) -> int:
        return self._effective_kernel_size - self._stride

    def _init_streaming_state(self, batch_size: int) -> _StreamingConvPadState:
        _ = batch_size
        if self.causal:
            return _StreamingConvPadState(self._padding_total, self._padding_total)
        return _StreamingConvPadState(self._padding_total // 2, self._padding_total // 2)

    def forward(self, x: Tensor) -> Tensor:
        padding_total = self._padding_total
        extra_padding = _get_extra_padding_for_conv1d(x, self._effective_kernel_size, self._stride, padding_total)
        state = self._streaming_state
        if state is None:
            if self.causal:
                x = _pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
            else:
                pad_right = padding_total // 2
                pad_left = padding_total - pad_right
                x = _pad1d(x, (pad_left, pad_right + extra_padding), mode=self.pad_mode)
        else:
            if state.padding_to_add > 0 and x.shape[-1] > 0:
                x = _pad1d(x, (state.padding_to_add, 0), mode=self.pad_mode)
                state.padding_to_add = 0
        return self.conv(x)


class _StreamingAdd(_StreamingModule):
    def _init_streaming_state(self, batch_size: int) -> _StreamingAddState:
        _ = batch_size
        return _StreamingAddState()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self._streaming_state is None:
            return x + y
        prev_x = self._streaming_state.previous_x
        prev_y = self._streaming_state.previous_y
        if prev_x is not None:
            x = torch.cat([prev_x, x], dim=-1)
        if prev_y is not None:
            y = torch.cat([prev_y, y], dim=-1)
        m = min(x.shape[-1], y.shape[-1])
        self._streaming_state.previous_x = x[..., m:]
        self._streaming_state.previous_y = y[..., m:]
        return x[..., :m] + y[..., :m]


class _LayerNormCN(nn.Module):
    # ConvNeXt-style channels-last LN used by the third-party reference implementation.
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = float(eps)
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class _ConvNeXtBlock(_StreamingModule):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        kernel_size: int,
        causal: bool,
        pad_mode: str,
        norm: str,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dwconv = _StreamingConv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            dilation=1,
            groups=dim,
            bias=True,
            causal=causal,
            norm=norm,
            pad_mode=pad_mode,
        )
        self.norm = _LayerNormCN(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)))
        self.add = _StreamingAdd()

    def _init_streaming_state(self, batch_size: int) -> None:
        _ = batch_size
        return None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 2, 1)
        return self.add(identity, x)


class _VocosBackbone(_StreamingModule):
    def __init__(
        self,
        input_channels: int,
        dim: int,
        mlp_ratio: float,
        num_layers: int,
        kernel_size: int = 7,
        causal: bool = True,
        pad_mode: str = "constant",
        norm: str = "weight_norm",
    ):
        super().__init__()
        self.embed = _StreamingConv1d(
            input_channels,
            dim,
            kernel_size=kernel_size,
            dilation=1,
            groups=1,
            bias=True,
            causal=causal,
            norm=norm,
            pad_mode=pad_mode,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        scale = 1e-6 if num_layers <= 0 else 1.0 / num_layers
        self.convnext = nn.ModuleList(
            [
                _ConvNeXtBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    kernel_size=kernel_size,
                    causal=causal,
                    pad_mode=pad_mode,
                    norm=norm,
                    layer_scale_init_value=scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def _init_streaming_state(self, batch_size: int) -> None:
        _ = batch_size
        return None

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        x = self.final_layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return x


class _OverlapAdd1d(nn.Module):
    def __init__(self, win_length: int, hop: int):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(win_length, 1, kernel_size=win_length, stride=hop, bias=False)
        weight = torch.zeros(win_length, 1, win_length)
        for i in range(win_length):
            weight[i, 0, i] = 1.0
        self.deconv.weight.data.copy_(weight)
        self.deconv.weight.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.deconv(x).squeeze(1)


class _StreamingISTFT(_StreamingModule):
    def __init__(self, n_fft: int, hop_length: int, win_length: Optional[int] = None):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop = int(hop_length)
        self.win_length = int(win_length or n_fft)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.overlap_add = _OverlapAdd1d(self.win_length, self.hop)
        self.tail = self.win_length - self.hop
        if self.tail < 0:
            raise ValueError("hop_length must be <= win_length")

    def _init_streaming_state(self, batch_size: int) -> _StreamingISTFTState:
        buf = torch.zeros(batch_size, self.tail, device=self.window.device)
        return _StreamingISTFTState(prev_buffer=buf)

    def forward(self, s: Tensor) -> Tensor:
        bsz, _, frames = s.shape
        if frames == 0:
            return torch.empty(bsz, 0, device=s.device, dtype=torch.float32)
        spec = s.transpose(1, 2)
        time_frames = torch.fft.irfft(spec, n=self.n_fft)
        time_frames = time_frames[..., : self.win_length] * self.window
        x = time_frames.transpose(1, 2)
        out = self.overlap_add(x)
        state = self._streaming_state
        if state is None:
            return out[:, : frames * self.hop]
        out[:, : self.tail] += state.prev_buffer
        ready = out[:, : frames * self.hop]
        state.prev_buffer = out[:, frames * self.hop :]
        return ready


class _ISTFTHead(_StreamingModule):
    def __init__(self, dim: int, n_fft: int, hop_length: int):
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = _StreamingISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft)

    def _init_streaming_state(self, batch_size: int) -> None:
        _ = batch_size
        return None

    def forward(self, x: Tensor) -> Tensor:
        x = self.out(x.transpose(1, 2)).transpose(1, 2)
        mag, phase = x.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)
        real = torch.cos(phase)
        imag = torch.sin(phase)
        spec = mag * (real + 1j * imag)
        audio = self.istft(spec)
        return audio.unsqueeze(1)


class StreamingVocosGenerator(_StreamingModule):
    def __init__(self, cfg: StreamingVocosConfig):
        super().__init__()
        self.conditioner = nn.Sequential(
            nn.Conv1d(cfg.in_channels, cfg.model_input_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(cfg.model_input_channels, cfg.model_input_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(cfg.model_input_channels, cfg.model_input_channels, kernel_size=1),
        )
        mlp_ratio = float(cfg.backbone_intermediate_dim) / float(max(1, cfg.backbone_dim))
        self.backbone = _VocosBackbone(
            input_channels=cfg.model_input_channels,
            dim=cfg.backbone_dim,
            mlp_ratio=mlp_ratio,
            num_layers=cfg.backbone_layers,
            kernel_size=7,
            causal=cfg.backbone_causal,
            pad_mode=cfg.backbone_pad_mode,
            norm=cfg.backbone_norm,
        )
        self.head = _ISTFTHead(dim=cfg.backbone_dim, n_fft=cfg.n_fft, hop_length=cfg.hop_length)

    def _init_streaming_state(self, batch_size: int) -> None:
        _ = batch_size
        return None

    def forward(self, features: Tensor) -> Tensor:
        x = self.conditioner(features)
        x = self.backbone(x)
        y = self.head(x)
        if y.ndim == 3 and y.shape[1] == 1:
            return y[:, 0, :]
        return y


class StreamingVocos:
    """High-level inference wrapper with fixed-frame chunked synthesis."""

    def __init__(
        self,
        model: StreamingVocosGenerator,
        config: StreamingVocosConfig,
        chunk_frames: int,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.chunk_frames = int(chunk_frames)
        if self.chunk_frames <= 0:
            raise ValueError("chunk_frames must be > 0")
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self._stream_ctx: Optional[contextlib.ExitStack] = None
        self._stream_batch_size: int = 0

    @property
    def _model_dtype(self) -> torch.dtype:
        return self.model.conditioner[0].weight.dtype

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        chunk_frames: int,
        *,
        device: str | torch.device = "cpu",
        hop_length: int = 300,
        backbone_causal: bool = True,
        backbone_pad_mode: str = "constant",
        backbone_norm: str = "weight_norm",
        use_fp16: bool = False,
    ) -> "StreamingVocos":
        path = Path(checkpoint_path)
        state = _normalize_checkpoint_state(path)
        cfg = StreamingVocosConfig.from_state_dict(
            state,
            hop_length=hop_length,
            backbone_causal=backbone_causal,
            backbone_pad_mode=backbone_pad_mode,
            backbone_norm=backbone_norm,
        )
        model = StreamingVocosGenerator(cfg)
        load = model.load_state_dict(state, strict=False)
        if load.missing_keys or load.unexpected_keys:
            missing_preview = ", ".join(load.missing_keys[:6])
            unexpected_preview = ", ".join(load.unexpected_keys[:6])
            raise RuntimeError(
                "Checkpoint/model mismatch. "
                f"missing={len(load.missing_keys)} [{missing_preview}] "
                f"unexpected={len(load.unexpected_keys)} [{unexpected_preview}]"
            )

        dtype = torch.float16 if use_fp16 else torch.float32
        model = model.to(device=torch.device(device), dtype=dtype)
        return cls(model=model, config=cfg, chunk_frames=chunk_frames, device=torch.device(device))

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
        stack.enter_context(self.model.backbone.streaming(int(batch_size)))
        stack.enter_context(self.model.head.streaming(int(batch_size)))
        self._stream_ctx = stack
        self._stream_batch_size = int(batch_size)

    @torch.inference_mode()
    def synthesize_chunk(self, features_chunk: Tensor) -> Tensor:
        """Synthesize one fixed-size chunk.

        Args:
            features_chunk: [B, C, T] where T == self.chunk_frames.
        Returns:
            audio chunk: [B, T * hop_length]
        """
        if features_chunk.ndim != 3:
            raise ValueError(f"Expected features_chunk to have shape [B,C,T], got {tuple(features_chunk.shape)}")
        if int(features_chunk.shape[-1]) != self.chunk_frames:
            raise ValueError(
                f"Expected chunk length {self.chunk_frames} frames, got {int(features_chunk.shape[-1])}. "
                "Use synthesize() for variable-length input."
            )
        features_chunk = features_chunk.to(device=self.device, dtype=self._model_dtype)
        self._ensure_streaming_context(int(features_chunk.shape[0]))
        conditioned = self.model.conditioner(features_chunk)
        x = self.model.backbone(conditioned)
        y = self.model.head(x)
        if y.ndim == 3 and y.shape[1] == 1:
            y = y[:, 0, :]
        return y

    @torch.inference_mode()
    def synthesize(self, features: Tensor, is_last: bool = True) -> Tensor:
        """Synthesize variable-length features by fixed-size chunking.

        The final partial chunk is zero-padded to self.chunk_frames and trimmed in audio domain.
        """
        if features.ndim != 3:
            raise ValueError(f"Expected features to have shape [B,C,T], got {tuple(features.shape)}")
        bsz, _, total_frames = features.shape
        self._ensure_streaming_context(int(bsz))

        outputs = []
        pos = 0
        while pos < total_frames:
            end = min(total_frames, pos + self.chunk_frames)
            chunk = features[..., pos:end]
            valid_frames = int(end - pos)
            if valid_frames < self.chunk_frames:
                pad = self.chunk_frames - valid_frames
                chunk = F.pad(chunk, (0, pad))
            chunk_audio = self.synthesize_chunk(chunk.to(self.device))
            outputs.append(chunk_audio[..., : valid_frames * self.config.hop_length])
            pos = end

        if not outputs:
                audio = torch.empty(int(bsz), 0, device=self.device, dtype=self._model_dtype)
        else:
            audio = torch.cat(outputs, dim=-1)
        if is_last:
            self.reset()
        return audio

    @torch.inference_mode()
    def stream_chunks(self, features: Tensor, is_last: bool = True) -> Iterator[Tensor]:
        """Yield synthesized audio per chunk for streaming pipelines."""
        if features.ndim != 3:
            raise ValueError(f"Expected features to have shape [B,C,T], got {tuple(features.shape)}")
        bsz, _, total_frames = features.shape
        self._ensure_streaming_context(int(bsz))

        pos = 0
        while pos < total_frames:
            end = min(total_frames, pos + self.chunk_frames)
            chunk = features[..., pos:end]
            valid_frames = int(end - pos)
            if valid_frames < self.chunk_frames:
                chunk = F.pad(chunk, (0, self.chunk_frames - valid_frames))
            out = self.synthesize_chunk(chunk.to(self.device))
            yield out[..., : valid_frames * self.config.hop_length]
            pos = end

        if is_last:
            self.reset()
