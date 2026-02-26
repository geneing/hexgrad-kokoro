"""TensorFlow/PyTorch checkpoint interop helpers for Vocos decoder modules.

This module centralizes framework-bridge operations used across TF workflows:
- Read PyTorch checkpoint payloads and normalize state-dict key prefixes.
- Infer TensorFlow generator config directly from PyTorch tensor shapes.
- Build TensorFlow generator/export models with matching architecture.
- Assign mapped PyTorch weights into TensorFlow layers with correct layout.
- Save converted TF checkpoints/weights and utility audio/features for tests.

The goal is deterministic, explicit conversion behavior so training, conversion,
quantization, and validation scripts all share the exact same mapping logic.
"""

from __future__ import annotations

import json
import re
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import tensorflow as tf
import torch

from .model import DEFAULT_FIXED_FRAMES, PairedVocosGeneratorExportTF, PairedVocosGeneratorTF


@dataclass
class TFVocosGeneratorConfig:
    in_channels: int
    model_input_channels: int
    backbone_dim: int
    backbone_intermediate_dim: int
    backbone_layers: int
    n_fft: int
    hop_length: int
    padding: str = "same"


def _strip_parallel_prefixes(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    fixed: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        fixed[new_key] = value
    return fixed


def _is_tensor_state_dict(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return all(torch.is_tensor(v) for v in value.values())


def load_pytorch_generator_state(checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if _is_tensor_state_dict(raw):
        return _strip_parallel_prefixes(raw), {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(raw)}")
    generator = raw.get("generator")
    if not _is_tensor_state_dict(generator):
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain a valid 'generator' state_dict")
    metadata = {k: raw.get(k) for k in ("step", "epoch", "frame_cap") if k in raw}
    return _strip_parallel_prefixes(generator), metadata


def infer_tf_generator_config(
    state_dict: Mapping[str, torch.Tensor], hop_length: int, padding: str = "same"
) -> TFVocosGeneratorConfig:
    try:
        model_input_channels = int(state_dict["conditioner.0.weight"].shape[0])
        in_channels = int(state_dict["conditioner.0.weight"].shape[1])
        backbone_dim = int(state_dict["backbone.embed.weight"].shape[0])
        backbone_intermediate_dim = int(state_dict["backbone.convnext.0.pwconv1.weight"].shape[0])
        n_fft = int(state_dict["head.out.weight"].shape[0]) - 2
    except KeyError as exc:
        raise KeyError(f"Could not infer generator config, missing key: {exc}") from exc

    layer_ids = set()
    pattern = re.compile(r"^backbone\.convnext\.(\d+)\.dwconv\.weight$")
    for key in state_dict.keys():
        match = pattern.match(key)
        if match:
            layer_ids.add(int(match.group(1)))
    if not layer_ids:
        raise RuntimeError("Could not infer backbone layer count from state_dict keys")
    backbone_layers = max(layer_ids) + 1

    return TFVocosGeneratorConfig(
        in_channels=in_channels,
        model_input_channels=model_input_channels,
        backbone_dim=backbone_dim,
        backbone_intermediate_dim=backbone_intermediate_dim,
        backbone_layers=backbone_layers,
        n_fft=n_fft,
        hop_length=int(hop_length),
        padding=padding,
    )


def build_tf_generator(config: TFVocosGeneratorConfig) -> PairedVocosGeneratorTF:
    model = PairedVocosGeneratorTF(
        in_channels=config.in_channels,
        model_input_channels=config.model_input_channels,
        backbone_dim=config.backbone_dim,
        backbone_intermediate_dim=config.backbone_intermediate_dim,
        backbone_layers=config.backbone_layers,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        padding=config.padding,
    )
    # Build model variables.
    dummy = tf.zeros([1, config.in_channels, 16], dtype=tf.float32)
    _ = model(dummy, training=False)
    return model


def build_tf_export_generator(
    config: TFVocosGeneratorConfig,
    fixed_frames: int = DEFAULT_FIXED_FRAMES,
) -> PairedVocosGeneratorExportTF:
    model = PairedVocosGeneratorExportTF(
        in_channels=config.in_channels,
        model_input_channels=config.model_input_channels,
        backbone_dim=config.backbone_dim,
        backbone_intermediate_dim=config.backbone_intermediate_dim,
        backbone_layers=config.backbone_layers,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        fixed_frames=fixed_frames,
        padding=config.padding,
    )
    dummy = tf.zeros([1, config.in_channels, int(fixed_frames)], dtype=tf.float32)
    _ = model(dummy, training=False)
    return model


def _np_from_torch(x: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _assign_conv1d(layer: tf.keras.layers.Conv1D, w: torch.Tensor, b: torch.Tensor) -> None:
    kernel = np.transpose(_np_from_torch(w), (2, 1, 0))
    bias = _np_from_torch(b)
    layer.kernel.assign(kernel.astype(tf.as_dtype(layer.kernel.dtype).as_numpy_dtype))
    layer.bias.assign(bias.astype(tf.as_dtype(layer.bias.dtype).as_numpy_dtype))


def _assign_depthwise_conv1d(layer: tf.keras.layers.DepthwiseConv1D, w: torch.Tensor, b: torch.Tensor) -> None:
    # PyTorch depthwise conv [C, 1, K] -> Keras [K, C, 1]
    kernel = np.transpose(_np_from_torch(w), (2, 0, 1))
    bias = _np_from_torch(b)
    layer.kernel.assign(kernel.astype(tf.as_dtype(layer.kernel.dtype).as_numpy_dtype))
    layer.bias.assign(bias.astype(tf.as_dtype(layer.bias.dtype).as_numpy_dtype))


def _assign_dense(layer: tf.keras.layers.Dense, w: torch.Tensor, b: torch.Tensor) -> None:
    kernel = np.transpose(_np_from_torch(w), (1, 0))
    bias = _np_from_torch(b)
    layer.kernel.assign(kernel.astype(tf.as_dtype(layer.kernel.dtype).as_numpy_dtype))
    layer.bias.assign(bias.astype(tf.as_dtype(layer.bias.dtype).as_numpy_dtype))


def _assign_layer_norm(layer: tf.keras.layers.LayerNormalization, w: torch.Tensor, b: torch.Tensor) -> None:
    gamma = _np_from_torch(w)
    beta = _np_from_torch(b)
    layer.gamma.assign(gamma.astype(tf.as_dtype(layer.gamma.dtype).as_numpy_dtype))
    layer.beta.assign(beta.astype(tf.as_dtype(layer.beta.dtype).as_numpy_dtype))


def load_pytorch_state_into_tf_generator(
    model: PairedVocosGeneratorTF, state_dict: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    used: set[str] = set()

    conditioner = model.conditioner.layers
    _assign_conv1d(conditioner[0], state_dict["conditioner.0.weight"], state_dict["conditioner.0.bias"])
    used.update({"conditioner.0.weight", "conditioner.0.bias"})
    _assign_conv1d(conditioner[2], state_dict["conditioner.2.weight"], state_dict["conditioner.2.bias"])
    used.update({"conditioner.2.weight", "conditioner.2.bias"})
    _assign_conv1d(conditioner[4], state_dict["conditioner.4.weight"], state_dict["conditioner.4.bias"])
    used.update({"conditioner.4.weight", "conditioner.4.bias"})

    _assign_conv1d(model.backbone.embed, state_dict["backbone.embed.weight"], state_dict["backbone.embed.bias"])
    used.update({"backbone.embed.weight", "backbone.embed.bias"})
    _assign_layer_norm(model.backbone.norm, state_dict["backbone.norm.weight"], state_dict["backbone.norm.bias"])
    used.update({"backbone.norm.weight", "backbone.norm.bias"})

    for i, block in enumerate(model.backbone.blocks):
        prefix = f"backbone.convnext.{i}"
        _assign_depthwise_conv1d(
            block.dwconv,
            state_dict[f"{prefix}.dwconv.weight"],
            state_dict[f"{prefix}.dwconv.bias"],
        )
        _assign_layer_norm(block.norm, state_dict[f"{prefix}.norm.weight"], state_dict[f"{prefix}.norm.bias"])
        _assign_dense(block.pwconv1, state_dict[f"{prefix}.pwconv1.weight"], state_dict[f"{prefix}.pwconv1.bias"])
        _assign_dense(block.pwconv2, state_dict[f"{prefix}.pwconv2.weight"], state_dict[f"{prefix}.pwconv2.bias"])
        gamma = _np_from_torch(state_dict[f"{prefix}.gamma"]).astype(tf.as_dtype(block.gamma.dtype).as_numpy_dtype)
        block.gamma.assign(gamma)
        used.update(
            {
                f"{prefix}.dwconv.weight",
                f"{prefix}.dwconv.bias",
                f"{prefix}.norm.weight",
                f"{prefix}.norm.bias",
                f"{prefix}.pwconv1.weight",
                f"{prefix}.pwconv1.bias",
                f"{prefix}.pwconv2.weight",
                f"{prefix}.pwconv2.bias",
                f"{prefix}.gamma",
            }
        )

    _assign_layer_norm(
        model.backbone.final_layer_norm,
        state_dict["backbone.final_layer_norm.weight"],
        state_dict["backbone.final_layer_norm.bias"],
    )
    used.update({"backbone.final_layer_norm.weight", "backbone.final_layer_norm.bias"})

    _assign_dense(model.head.out, state_dict["head.out.weight"], state_dict["head.out.bias"])
    used.update({"head.out.weight", "head.out.bias"})

    ignored = {"head.istft.window"}
    missing = sorted((set(state_dict.keys()) - ignored) - used)
    unexpected_ignored = sorted((set(state_dict.keys()) & ignored))

    if missing:
        raise RuntimeError(f"Unmapped PyTorch generator keys: {missing[:16]}")

    return {
        "num_loaded_keys": len(used),
        "ignored_keys": unexpected_ignored,
    }


def load_pytorch_checkpoint_into_tf_generator(
    model: PairedVocosGeneratorTF, checkpoint_path: Path
) -> dict[str, object]:
    state_dict, metadata = load_pytorch_generator_state(checkpoint_path)
    report = load_pytorch_state_into_tf_generator(model, state_dict)
    report["metadata"] = metadata
    return report


def convert_pytorch_checkpoint_to_tf(
    pytorch_checkpoint: Path,
    output_dir: Path,
    hop_length: int,
    padding: str = "same",
) -> dict[str, object]:
    state_dict, metadata = load_pytorch_generator_state(pytorch_checkpoint)
    config = infer_tf_generator_config(state_dict, hop_length=hop_length, padding=padding)
    model = build_tf_generator(config)
    report = load_pytorch_state_into_tf_generator(model, state_dict)

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "generator.weights.h5"
    config_path = output_dir / "generator_config.json"
    ckpt_prefix = output_dir / "generator_tf_ckpt"

    model.save_weights(str(weights_path))
    tf_ckpt = tf.train.Checkpoint(generator=model)
    ckpt_written = tf_ckpt.save(file_prefix=str(ckpt_prefix))
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    return {
        "weights_path": str(weights_path),
        "checkpoint_prefix": str(ckpt_written),
        "config_path": str(config_path),
        "config": asdict(config),
        "report": report,
        "metadata": metadata,
    }


def build_feature_from_pair_payload(pair: Mapping[str, object]) -> np.ndarray:
    asr = pair["asr"].float().cpu().numpy().astype(np.float32)
    f0 = pair["f0"].float().cpu().numpy().astype(np.float32)
    noise = pair["noise"].float().cpu().numpy().astype(np.float32)
    style = pair["style"].float().cpu().numpy().astype(np.float32)

    total_frames = int(f0.shape[-1])
    if asr.shape[-1] != total_frames:
        src_x = np.linspace(0.0, 1.0, asr.shape[-1], dtype=np.float32)
        dst_x = np.linspace(0.0, 1.0, total_frames, dtype=np.float32)
        asr = np.stack([np.interp(dst_x, src_x, ch).astype(np.float32) for ch in asr], axis=0)

    f0 = f0[None, :]
    noise = noise[None, :]
    style = np.repeat(style[:, None], total_frames, axis=1)
    feat = np.concatenate([asr, f0, noise, style], axis=0)
    return feat[None, ...].astype(np.float32)


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
