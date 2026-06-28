#!/usr/bin/env python3
"""Generate streaming Wavehax WAVs from saved Kokoro feature tensors."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch

from wavehax_streaming import (
    StreamingKokoroWavehax,
    audio_stats,
    comparison_metrics,
    compose_features_from_pt,
    feature_paths_from_globs,
    load_wavehax_model,
    save_wav_16bit,
    select_device,
)


def checkpoint_backend_config(path: Path) -> dict[str, object]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, Mapping):
        raise TypeError(f"Expected checkpoint mapping in {path}, got {type(ckpt)}")
    config = ckpt.get("backend_config", {})
    if not isinstance(config, Mapping):
        raise TypeError(f"Expected backend_config mapping in {path}, got {type(config)}")
    return dict(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate WAVs with the trained streaming Wavehax model")
    parser.add_argument("--checkpoint", type=Path, default=Path("data/training/wavehax/checkpoints/last.pt"))
    parser.add_argument("--input-feature-glob", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("test_output"))
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--chunk-frames", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_feature_glob is None:
        args.input_feature_glob = ["data/af*.pt"]

    checkpoint = args.checkpoint.resolve()
    config = checkpoint_backend_config(checkpoint)
    sample_rate = int(args.sample_rate if args.sample_rate is not None else config.get("sample_rate", 24000))
    hop_length = int(args.hop_length if args.hop_length is not None else config.get("hop_length", 300))
    chunk_frames = int(args.chunk_frames if args.chunk_frames is not None else config.get("chunk_frames", 24))

    device = select_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = feature_paths_from_globs(args.input_feature_glob)
    if not paths:
        raise RuntimeError(f"No feature files matched: {args.input_feature_glob}")

    torch.set_grad_enabled(False)
    model = load_wavehax_model(checkpoint, sample_rate, device)
    streamer = StreamingKokoroWavehax(
        model=model,
        chunk_frames=chunk_frames,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )

    print(
        f"Using checkpoint={checkpoint} sample_rate={sample_rate} "
        f"hop_length={hop_length} chunk_frames={chunk_frames} device={device}"
    )
    with torch.inference_mode():
        for path in paths:
            features = compose_features_from_pt(path).unsqueeze(0).to(device)
            full_audio = model(features).detach().cpu().numpy()
            full_path = output_dir / f"{path.stem}_wavehax_full.wav"
            save_wav_16bit(full_path, full_audio, sample_rate)
            full_rms, full_peak = audio_stats(full_audio)
            print(f"Wrote {full_path} shape={tuple(full_audio.shape)} rms={full_rms:.6f} peak={full_peak:.6f}")

            stream_audio = streamer.synthesize(features).detach().cpu().numpy()
            stream_path = output_dir / f"{path.stem}_wavehax_streaming.wav"
            save_wav_16bit(stream_path, stream_audio, sample_rate)
            stream_rms, stream_peak = audio_stats(stream_audio)
            metrics = comparison_metrics(full_audio, stream_audio, chunk_frames * hop_length)
            print(
                f"Wrote {stream_path} shape={tuple(stream_audio.shape)} "
                f"rms={stream_rms:.6f} peak={stream_peak:.6f} "
                f"mae={metrics['mae']:.6f} rmse={metrics['rmse']:.6f} "
                f"corr={metrics['corr']:.6f} boundary={metrics['boundary_click']:.6f}"
            )


if __name__ == "__main__":
    main()
