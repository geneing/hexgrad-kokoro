"""Smoke test: compare PyTorch and TensorFlow Vocos outputs from one checkpoint.

This utility verifies model-implementation and weight-mapping parity by running the
same conditioning features through:
- A PyTorch Vocos generator loaded directly from a PyTorch checkpoint.
- A TensorFlow Vocos generator initialized from the same checkpoint.

It reports waveform error metrics and can enforce pass/fail thresholds.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from loguru import logger
from torch import nn
from vocos.heads import ISTFTHead
from vocos.models import VocosBackbone

try:
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    plt = None

from .checkpoint_utils import (
    build_feature_from_pair_payload,
    build_tf_generator,
    infer_tf_generator_config,
    load_pytorch_generator_state,
    load_pytorch_state_into_tf_generator,
    save_wav_16bit,
)


class PairedVocosGeneratorPT(nn.Module):
    """Minimal PyTorch paired-feature Vocos generator matching training architecture."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compare PyTorch and TensorFlow Vocos outputs for one checkpoint")
    parser.add_argument("--pytorch-checkpoint", type=Path, default=Path("output/checkpoints/last.pt"))
    parser.add_argument("--pair-path", type=Path, default=None)
    parser.add_argument("--pairs-root", type=Path, default=Path("inputs/pairs"))
    parser.add_argument("--hop-length", type=int, default=300)
    parser.add_argument("--padding", type=str, default="same", choices=["same", "center"])
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-abs-threshold", type=float, default=2e-3)
    parser.add_argument("--rmse-threshold", type=float, default=5e-4)
    parser.add_argument("--save-audio-dir", type=Path, default=Path("output/pt_tf_compare"))
    parser.add_argument("--no-save-audio", action="store_true")
    parser.add_argument("--no-save-plot", action="store_true")
    return parser.parse_args()


def _pick_pair_path(pair_path: Path | None, pairs_root: Path) -> Path:
    if pair_path is not None:
        if not pair_path.exists():
            raise FileNotFoundError(f"Pair file not found: {pair_path}")
        return pair_path
    if not pairs_root.exists():
        raise FileNotFoundError(f"Pairs root not found: {pairs_root}")
    found = sorted(pairs_root.rglob("*.pt"))
    if not found:
        raise FileNotFoundError(f"No pair files found under {pairs_root}")
    return found[0]


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    n = int(min(len(a), len(b)))
    if n <= 0:
        return {"samples": 0.0, "max_abs": math.nan, "mae": math.nan, "rmse": math.nan, "corr": math.nan}
    x = a[:n].astype(np.float64, copy=False)
    y = b[:n].astype(np.float64, copy=False)
    diff = x - y
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom_x = float(np.std(x))
    denom_y = float(np.std(y))
    if denom_x > 1e-12 and denom_y > 1e-12:
        corr = float(np.corrcoef(x, y)[0, 1])
    else:
        corr = math.nan
    return {
        "samples": float(n),
        "max_abs": float(np.max(np.abs(diff))),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": rmse,
        "corr": corr,
    }


def _plot_waveforms(
    pt_audio: np.ndarray,
    tf_audio: np.ndarray,
    sample_rate: int,
    output_path: Path,
) -> None:
    if plt is None:
        logger.warning("matplotlib is unavailable; skipping waveform plot.")
        return

    n = min(len(pt_audio), len(tf_audio))
    if n <= 0:
        logger.warning("No samples available for waveform plot; skipping.")
        return

    pt = pt_audio[:n]
    tf_w = tf_audio[:n]
    diff = pt - tf_w
    t = np.arange(n, dtype=np.float32) / float(sample_rate)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t, pt, linewidth=0.8, color="#1f77b4")
    axes[0].set_title("PyTorch Output")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, tf_w, linewidth=0.8, color="#2ca02c")
    axes[1].set_title("TensorFlow Output")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, diff, linewidth=0.8, color="#d62728")
    axes[2].set_title("Difference (PT - TF)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.pytorch_checkpoint.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {args.pytorch_checkpoint}")

    pair_path = _pick_pair_path(args.pair_path, args.pairs_root)
    device = _resolve_device(args.device)
    logger.info(f"Comparing PT vs TF with checkpoint={args.pytorch_checkpoint} pair={pair_path} device={device}")

    state_dict, metadata = load_pytorch_generator_state(args.pytorch_checkpoint)
    cfg = infer_tf_generator_config(state_dict, hop_length=args.hop_length, padding=args.padding)

    tf_model = build_tf_generator(cfg)
    tf_report = load_pytorch_state_into_tf_generator(tf_model, state_dict)
    logger.info(
        f"Loaded TensorFlow generator from checkpoint. keys={tf_report['num_loaded_keys']} ignored={tf_report['ignored_keys']}"
    )

    pt_model = PairedVocosGeneratorPT(
        in_channels=cfg.in_channels,
        model_input_channels=cfg.model_input_channels,
        backbone_dim=cfg.backbone_dim,
        backbone_intermediate_dim=cfg.backbone_intermediate_dim,
        backbone_layers=cfg.backbone_layers,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        padding=cfg.padding,
    ).to(device)
    load_result = pt_model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        raise RuntimeError(f"PyTorch model load missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        raise RuntimeError(f"PyTorch model load unexpected keys: {load_result.unexpected_keys}")
    pt_model.eval()

    if metadata:
        logger.info(f"Checkpoint metadata: {metadata}")

    pair = torch.load(pair_path, map_location="cpu", weights_only=False)
    feat_np = build_feature_from_pair_payload(pair)

    with torch.inference_mode():
        feat_pt = torch.from_numpy(feat_np).to(device=device, dtype=torch.float32)
        pt_audio = pt_model(feat_pt).detach().cpu().numpy()[0].astype(np.float32)

    tf_audio = tf_model(tf.convert_to_tensor(feat_np, dtype=tf.float32), training=False).numpy()[0].astype(np.float32)

    expected_samples = int(feat_np.shape[-1] * args.hop_length)
    if expected_samples > 0:
        pt_audio = pt_audio[:expected_samples]
        tf_audio = tf_audio[:expected_samples]

    metrics = _compute_metrics(pt_audio, tf_audio)
    logger.info(
        "Parity metrics: "
        f"samples={int(metrics['samples'])} "
        f"max_abs={metrics['max_abs']:.8f} "
        f"mae={metrics['mae']:.8f} "
        f"rmse={metrics['rmse']:.8f} "
        f"corr={metrics['corr']:.8f}"
    )

    if not args.no_save_audio:
        args.save_audio_dir.mkdir(parents=True, exist_ok=True)
        pt_wav = args.save_audio_dir / "pytorch.wav"
        tf_wav = args.save_audio_dir / "tensorflow.wav"
        diff_wav = args.save_audio_dir / "abs_diff.wav"
        save_wav_16bit(pt_wav, pt_audio, sample_rate=args.sample_rate)
        save_wav_16bit(tf_wav, tf_audio, sample_rate=args.sample_rate)
        # Scale absolute error for audibility.
        n = min(len(pt_audio), len(tf_audio))
        abs_diff = np.clip(np.abs(pt_audio[:n] - tf_audio[:n]) * 20.0, 0.0, 1.0)
        save_wav_16bit(diff_wav, abs_diff, sample_rate=args.sample_rate)
        logger.info(f"Wrote comparison audio: {pt_wav}, {tf_wav}, {diff_wav}")

    if not args.no_save_plot:
        plot_path = args.save_audio_dir / "waveform_compare.png"
        _plot_waveforms(
            pt_audio=pt_audio,
            tf_audio=tf_audio,
            sample_rate=args.sample_rate,
            output_path=plot_path,
        )
        logger.info(f"Wrote waveform plot: {plot_path}")

    if metrics["max_abs"] > float(args.max_abs_threshold) or metrics["rmse"] > float(args.rmse_threshold):
        raise RuntimeError(
            "PT/TF parity check failed: "
            f"max_abs={metrics['max_abs']:.8f} (threshold={args.max_abs_threshold}), "
            f"rmse={metrics['rmse']:.8f} (threshold={args.rmse_threshold})"
        )

    logger.info("PT/TF parity check passed.")


if __name__ == "__main__":
    main()
