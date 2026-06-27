from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
WAVEHAX_ROOT = Path(__file__).resolve().parent
for path in (ROOT, WAVEHAX_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from third_party.kokoro_vocoder_distill import KokoroFeatureConditioner, add_common_args, train_decoder
from wavehax.generators.wavehax import MultiScaleWavehaxGenerator


class KokoroMultiScaleWavehaxGenerator(nn.Module):
    def __init__(
        self,
        model_input_channels: int,
        n_fft: int,
        hop_length: int,
        sample_rate: int,
        channels: int,
        mult_channels: int,
        kernel_size: int,
        num_blocks: int,
        decomposer: str,
        num_splits: int,
        prior_type: str,
        control_channels: int,
        control_layers: int,
        drop_prob: float,
        use_gradient_checkpointing: bool,
        norm_type: str,
        padding_mode: str,
        export_safe_ops: bool,
    ):
        super().__init__()
        self.conditioner = KokoroFeatureConditioner(
            out_channels=model_input_channels,
            control_channels=control_channels,
            control_layers=control_layers,
        )
        self.generator = MultiScaleWavehaxGenerator(
            in_channels=model_input_channels,
            channels=channels,
            mult_channels=mult_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            decomposer=decomposer,
            num_splits=num_splits,
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate,
            prior_type=prior_type,
            drop_prob=drop_prob,
            framewise_norm=True,
            use_gradient_checkpointing=use_gradient_checkpointing,
            norm_type=norm_type,
            padding_mode=padding_mode,
            export_safe_ops=export_safe_ops,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        cond = self.conditioner(features)
        f0 = features[:, 512:513, :]
        audio, _prior = self.generator(cond, f0)
        return audio[:, 0, :] if audio.ndim == 3 and audio.shape[1] == 1 else audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train third_party/wavehax MultiScaleWavehax as a Kokoro decoder")
    add_common_args(parser)
    parser.set_defaults(
        batch_size=16,
        min_batch_size=1,
        frame_cap=240,
        min_frame_cap=32,
        n_fft=120,
        val_steps=1,
        sample_count=1,
        pretrain_steps=10000,
        gen_lr=2e-4,
        disc_lr=1e-4,
        gan_loss_coeff=0.1,
        fm_loss_coeff=0.5,
    )
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--mult-channels", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--decomposer", type=str, default="MultiStream1d")
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument("--prior-type", type=str, default="pcph_closed_form")
    parser.add_argument("--drop-prob", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--chunk-frames", type=int, default=24)
    parser.add_argument("--norm-type", choices=("batch",), default="batch")
    parser.add_argument("--padding-mode", choices=("zeros",), default="zeros")
    parser.add_argument("--export-safe-ops", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def assert_no_layer_norm(module: nn.Module) -> None:
    offenders = [name for name, child in module.named_modules() if child.__class__.__name__.startswith("LayerNorm")]
    if offenders:
        raise RuntimeError(f"Strict-BN Wavehax must not contain LayerNorm modules: {offenders[:8]}")


def build_generator(args: argparse.Namespace) -> nn.Module:
    model = KokoroMultiScaleWavehaxGenerator(
        model_input_channels=args.model_input_channels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        channels=args.channels,
        mult_channels=args.mult_channels,
        kernel_size=args.kernel_size,
        num_blocks=args.num_blocks,
        decomposer=args.decomposer,
        num_splits=args.num_splits,
        prior_type=args.prior_type,
        control_channels=args.control_channels,
        control_layers=args.control_layers,
        drop_prob=args.drop_prob,
        use_gradient_checkpointing=args.gradient_checkpointing,
        norm_type=args.norm_type,
        padding_mode=args.padding_mode,
        export_safe_ops=args.export_safe_ops,
    )
    assert_no_layer_norm(model)
    return model


def main() -> None:
    args = parse_args()
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_config = resume.get("backend_config", {}) if isinstance(resume, dict) else {}
        if resume_config.get("norm_type") != "batch" or not bool(resume_config.get("export_safe_ops")):
            raise ValueError("Strict-BN export-friendly Wavehax training cannot resume from an older LN/non-export-safe checkpoint.")
    backend_config: Dict[str, object] = {
        "model": "third_party/wavehax/mswavehax_export_safe_streaming",
        "model_input_channels": args.model_input_channels,
        "channels": args.channels,
        "mult_channels": args.mult_channels,
        "kernel_size": args.kernel_size,
        "num_blocks": args.num_blocks,
        "decomposer": args.decomposer,
        "num_splits": args.num_splits,
        "prior_type": args.prior_type,
        "drop_prob": args.drop_prob,
        "framewise_norm": True,
        "norm_type": args.norm_type,
        "padding_mode": args.padding_mode,
        "export_safe_ops": args.export_safe_ops,
        "chunk_frames": args.chunk_frames,
        "use_gradient_checkpointing": args.gradient_checkpointing,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "control_channels": args.control_channels,
        "control_layers": args.control_layers,
    }
    train_decoder(args, "mswavehax", build_generator, backend_config)


if __name__ == "__main__":
    main()
