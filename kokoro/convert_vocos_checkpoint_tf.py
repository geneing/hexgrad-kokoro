from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from .tf_checkpoint_utils import convert_pytorch_checkpoint_to_tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Convert PyTorch Vocos checkpoint to TensorFlow checkpoint/weights")
    parser.add_argument(
        "--pytorch-checkpoint",
        type=Path,
        default=Path("output/checkpoints/last.pt"),
        help="Path to PyTorch checkpoint produced by kokoro/train_vocos.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/tf_checkpoints"),
        help="Directory where TensorFlow checkpoint + weights + config will be written",
    )
    parser.add_argument("--hop-length", type=int, default=300)
    parser.add_argument("--padding", type=str, default="same", choices=["same", "center"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pytorch_checkpoint.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {args.pytorch_checkpoint}")

    report = convert_pytorch_checkpoint_to_tf(
        pytorch_checkpoint=args.pytorch_checkpoint,
        output_dir=args.output_dir,
        hop_length=args.hop_length,
        padding=args.padding,
    )
    logger.info(f"Converted PyTorch checkpoint: {args.pytorch_checkpoint}")
    logger.info(f"TensorFlow weights: {report['weights_path']}")
    logger.info(f"TensorFlow checkpoint prefix: {report['checkpoint_prefix']}")
    logger.info(f"Generator config: {report['config_path']}")
    logger.info(
        "Loaded keys: "
        f"{report['report']['num_loaded_keys']} | "
        f"Ignored keys: {report['report']['ignored_keys']}"
    )
    if report.get("metadata"):
        logger.info(f"Checkpoint metadata: {report['metadata']}")


if __name__ == "__main__":
    main()
