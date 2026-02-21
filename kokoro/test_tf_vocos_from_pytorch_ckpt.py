from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from loguru import logger

from .tf_checkpoint_utils import (
    build_feature_from_pair_payload,
    build_tf_generator,
    infer_tf_generator_config,
    load_pytorch_generator_state,
    load_pytorch_state_into_tf_generator,
    save_wav_16bit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate WAV using TensorFlow Vocos loaded from PyTorch checkpoint")
    parser.add_argument(
        "--pytorch-checkpoint",
        type=Path,
        default=Path("output/checkpoints/last.pt"),
        help="Path to PyTorch checkpoint with 'generator' state",
    )
    parser.add_argument(
        "--pair-path",
        type=Path,
        default=None,
        help="Path to one vocoder pair .pt file. If omitted, picks first under --pairs-root.",
    )
    parser.add_argument(
        "--pairs-root",
        type=Path,
        default=Path("inputs/pairs"),
        help="Root searched for pair files when --pair-path is omitted.",
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=Path("output/tf_from_pytorch_checkpoint.wav"),
        help="Path to output generated WAV",
    )
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=300)
    parser.add_argument("--padding", type=str, default="same", choices=["same", "center"])
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


def main() -> None:
    args = parse_args()
    if not args.pytorch_checkpoint.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {args.pytorch_checkpoint}")

    pair_path = _pick_pair_path(args.pair_path, args.pairs_root)
    logger.info(f"Using pair input: {pair_path}")

    state_dict, metadata = load_pytorch_generator_state(args.pytorch_checkpoint)
    config = infer_tf_generator_config(state_dict, hop_length=args.hop_length, padding=args.padding)
    model = build_tf_generator(config)
    report = load_pytorch_state_into_tf_generator(model, state_dict)
    logger.info(
        f"Loaded PyTorch weights into TensorFlow model. keys={report['num_loaded_keys']} ignored={report['ignored_keys']}"
    )
    if metadata:
        logger.info(f"Checkpoint metadata: {metadata}")

    pair = torch.load(pair_path, map_location="cpu", weights_only=False)
    feat = build_feature_from_pair_payload(pair)
    audio = model(tf.convert_to_tensor(feat, dtype=tf.float32), training=False)
    audio_np = np.asarray(audio.numpy()[0], dtype=np.float32)

    expected_samples = int(feat.shape[-1] * args.hop_length)
    if expected_samples > 0:
        audio_np = audio_np[:expected_samples]
    save_wav_16bit(args.output_wav, audio_np, sample_rate=args.sample_rate)
    logger.info(
        f"Wrote generated WAV: {args.output_wav} | samples={len(audio_np)} "
        f"| duration_sec={len(audio_np) / float(args.sample_rate):.3f}"
    )


if __name__ == "__main__":
    main()
