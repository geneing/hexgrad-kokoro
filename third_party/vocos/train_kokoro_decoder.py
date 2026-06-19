from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from third_party.kokoro_vocoder_distill import KokoroFeatureConditioner, add_common_args, train_decoder
from vocos.heads import ISTFTHead
from vocos.models import VocosBackbone


class KokoroVocosGenerator(nn.Module):
    def __init__(
        self,
        model_input_channels: int,
        backbone_dim: int,
        backbone_intermediate_dim: int,
        backbone_layers: int,
        n_fft: int,
        hop_length: int,
        control_channels: int,
        control_layers: int,
    ):
        super().__init__()
        self.conditioner = KokoroFeatureConditioner(
            out_channels=model_input_channels,
            control_channels=control_channels,
            control_layers=control_layers,
        )
        self.backbone = VocosBackbone(
            input_channels=model_input_channels,
            dim=backbone_dim,
            intermediate_dim=backbone_intermediate_dim,
            num_layers=backbone_layers,
        )
        self.head = ISTFTHead(dim=backbone_dim, n_fft=n_fft, hop_length=hop_length, padding="same")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.conditioner(features)
        x = self.backbone(x)
        return self.head(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train third_party/vocos as a Kokoro decoder")
    add_common_args(parser)
    parser.add_argument("--backbone-dim", type=int, default=384)
    parser.add_argument("--backbone-intermediate-dim", type=int, default=1152)
    parser.add_argument("--backbone-layers", type=int, default=8)
    return parser.parse_args()


def build_generator(args: argparse.Namespace) -> nn.Module:
    return KokoroVocosGenerator(
        model_input_channels=args.model_input_channels,
        backbone_dim=args.backbone_dim,
        backbone_intermediate_dim=args.backbone_intermediate_dim,
        backbone_layers=args.backbone_layers,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        control_channels=args.control_channels,
        control_layers=args.control_layers,
    )


def main() -> None:
    args = parse_args()
    backend_config: Dict[str, object] = {
        "model": "third_party/vocos",
        "model_input_channels": args.model_input_channels,
        "backbone_dim": args.backbone_dim,
        "backbone_intermediate_dim": args.backbone_intermediate_dim,
        "backbone_layers": args.backbone_layers,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "control_channels": args.control_channels,
        "control_layers": args.control_layers,
    }
    train_decoder(args, "vocos", build_generator, backend_config)


if __name__ == "__main__":
    main()
