"""Smoke test: compare PyTorch and TensorFlow Vocos outputs from one checkpoint.

This utility verifies PT/TF parity for Kokoro Vocos checkpoints by running the
same paired conditioning features (`asr`, `f0`, `noise`, `style`) through:
1) Full-forward inference:
   - PyTorch generator (from checkpoint)
   - TensorFlow generator loaded from same checkpoint via mapping utilities
2) Optional chunked streaming inference comparison:
   - PT side uses corrected cache-window chunked decoding
   - TF side uses the same corrected cache-window chunked decoding

What it checks:
- Waveform-level parity metrics: max_abs, MAE, RMSE, correlation.
- Independent thresholds for full-forward and chunked-streaming parity.
- Optional waveform/audio artifact dumps for qualitative inspection.

Key behavior:
- Streaming PT path is the default and is expected to use the new
  `third_party/vocos_streaming/src/modules/__init__.py` package layout.
- `--vocos-impl auto` can still infer streaming vs legacy PT backbone from
  checkpoint keys.
- For streaming checkpoints, PT reference path uses the same streaming generator
  implementation used by training.
- Chunk size is configured in ms and converted to frames using sample rate/hop.

Outputs (if not disabled):
- `pytorch.wav`, `tensorflow.wav`, `abs_diff.wav`
- `pytorch_streaming.wav`, `tensorflow_streaming.wav`, `abs_diff_streaming.wav`
- `waveform_compare.png`, `waveform_compare_streaming.png`

Examples:

1) Basic parity check (streaming backend default)
   uv run python -m kokoro.tf.smoke_compare_pt_tf \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --pairs-root inputs/pairs

2) Explicit streaming PT path
   uv run python -m kokoro.tf.smoke_compare_pt_tf \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --pairs-root inputs/pairs \
     --vocos-impl streaming \
     --streaming-vocos-repo third_party/vocos_streaming

3) Enable chunked streaming comparison with custom chunking
   uv run python -m kokoro.tf.smoke_compare_pt_tf \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --pairs-root inputs/pairs \
     --stream-chunk-size-ms 200 \
     --stream-padding-ms 40

4) Stricter thresholds and artifact export
   uv run python -m kokoro.tf.smoke_compare_pt_tf \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --pairs-root inputs/pairs \
     --max-abs-threshold 0.0015 \
     --rmse-threshold 0.0004 \
     --streaming-max-abs-threshold 0.0025 \
     --streaming-rmse-threshold 0.0006 \
     --save-audio-dir output/pt_tf_compare_strict

5) Full-forward only (skip chunked comparison)
   uv run python -m kokoro.tf.smoke_compare_pt_tf \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --pairs-root inputs/pairs \
     --no-compare-streaming-chunked
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf
import torch
from loguru import logger

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
from ..train_vocos import PairedVocosGenerator


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
    parser.add_argument("--streaming-max-abs-threshold", type=float, default=3e-3)
    parser.add_argument("--streaming-rmse-threshold", type=float, default=8e-4)
    parser.add_argument("--save-audio-dir", type=Path, default=Path("output/pt_tf_compare"))
    parser.add_argument("--no-save-audio", action="store_true")
    parser.add_argument("--no-save-plot", action="store_true")
    parser.add_argument("--compare-streaming-chunked", dest="compare_streaming_chunked", action="store_true")
    parser.add_argument("--no-compare-streaming-chunked", dest="compare_streaming_chunked", action="store_false")
    parser.set_defaults(compare_streaming_chunked=True)
    parser.add_argument("--stream-chunk-size-ms", type=int, default=300)
    parser.add_argument("--stream-padding-ms", type=int, default=40)
    parser.add_argument(
        "--vocos-impl",
        type=str,
        choices=["auto", "streaming", "legacy"],
        default="streaming",
        help="PyTorch reference model backend. streaming is default; auto infers from checkpoint keys.",
    )
    parser.add_argument(
        "--streaming-vocos-repo",
        type=Path,
        default=Path("third_party/vocos_streaming"),
        help="Path containing streaming-vocos src/components when --vocos-impl=streaming.",
    )
    parser.add_argument("--backbone-causal", dest="backbone_causal", action="store_true")
    parser.add_argument("--no-backbone-causal", dest="backbone_causal", action="store_false")
    parser.set_defaults(backbone_causal=True)
    parser.add_argument("--backbone-pad-mode", type=str, default="constant")
    parser.add_argument("--backbone-norm", type=str, default="weight_norm")
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


def _infer_pt_vocos_impl(state_dict: dict[str, torch.Tensor], requested: str) -> str:
    if requested in {"streaming", "legacy"}:
        return requested
    # streaming checkpoints from kokoro.train_vocos include these nested conv+weight_norm keys.
    if "backbone.embed.conv.conv.weight_v" in state_dict or "backbone.convnext.0.dwconv.conv.conv.weight_v" in state_dict:
        return "streaming"
    return "legacy"


def _validate_streaming_repo_layout(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    required = repo_root / "src" / "modules" / "__init__.py"
    if not required.exists():
        raise FileNotFoundError(
            f"Streaming vocos repo layout check failed: expected {required} "
            "(new implementation entrypoint from third_party/vocos_streaming/src/modules/__init__.py)"
        )


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


def _compute_streaming_window(
    *,
    backbone_layers: int,
    n_fft: int,
    hop_length: int,
    sample_rate: int,
    chunk_size_ms: int,
    padding_ms: int,
) -> tuple[int, int]:
    requested_chunk = max(1, int(chunk_size_ms / 1000.0 * sample_rate / hop_length))
    requested_padding = max(0, int(padding_ms / 1000.0 * sample_rate / hop_length))
    receptive_frames = 9 + 6 * int(backbone_layers)
    istft_half_frames = max(1, int(round((n_fft / float(hop_length)) / 2.0)))
    min_padding = int(math.ceil((receptive_frames - 1) / 2.0)) + istft_half_frames
    min_chunk = max(receptive_frames, 2 * min_padding)
    return max(requested_chunk, min_chunk), max(requested_padding, min_padding)


def _streaming_decode_features(
    features: torch.Tensor,
    *,
    chunk_size: int,
    padding: int,
    hop_length: int,
    run_decode: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError(f"Expected [B,C,T], got {tuple(features.shape)}")
    bsz, ch, total_frames = features.shape
    if total_frames <= 0:
        return torch.empty((bsz, 0), dtype=features.dtype, device=features.device)

    cache = torch.zeros((bsz, ch, chunk_size + 2 * padding), dtype=features.dtype, device=features.device)
    cur_idx = -1
    chunks: list[torch.Tensor] = []

    def _get_size(idx: int) -> int:
        effective_size = idx + 1 - padding
        if effective_size <= 0:
            return 0
        return effective_size % chunk_size or chunk_size

    for idx, frame in enumerate(torch.unbind(features, dim=2)):
        cache = torch.roll(cache, shifts=-1, dims=2)
        cache[:, :, -1] = frame
        cur_idx += 1

        is_last_feature = idx == (total_frames - 1)
        cur_size = _get_size(cur_idx)
        if cur_size != chunk_size and not is_last_feature:
            continue

        audio = run_decode(cache)
        if padding > 0:
            audio = audio[:, padding * hop_length :]
        if cur_size != chunk_size:
            audio = audio[:, (chunk_size - cur_size) * hop_length :]
        if not is_last_feature:
            audio = audio[:, : chunk_size * hop_length]
        chunks.append(audio)

    if not chunks:
        return torch.empty((bsz, 0), dtype=features.dtype, device=features.device)
    return torch.cat(chunks, dim=-1)


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

    pt_impl = _infer_pt_vocos_impl(state_dict, args.vocos_impl)
    logger.info(f"Using PyTorch reference vocos_impl={pt_impl}")
    if pt_impl == "streaming":
        _validate_streaming_repo_layout(args.streaming_vocos_repo)
    pt_model = PairedVocosGenerator(
        in_channels=cfg.in_channels,
        model_input_channels=cfg.model_input_channels,
        backbone_dim=cfg.backbone_dim,
        backbone_intermediate_dim=cfg.backbone_intermediate_dim,
        backbone_layers=cfg.backbone_layers,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        padding=cfg.padding,
        vocos_impl=pt_impl,
        streaming_vocos_repo=args.streaming_vocos_repo,
        backbone_causal=args.backbone_causal,
        backbone_pad_mode=args.backbone_pad_mode,
        backbone_norm=args.backbone_norm,
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
    feat_pt = torch.from_numpy(feat_np).to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        pt_audio = pt_model(feat_pt).detach().cpu().numpy()[0].astype(np.float32)

    tf_audio = tf_model(tf.convert_to_tensor(feat_np, dtype=tf.float32), training=False).numpy()[0].astype(np.float32)

    expected_samples = int(feat_np.shape[-1] * args.hop_length)
    if expected_samples > 0:
        pt_audio = pt_audio[:expected_samples]
        tf_audio = tf_audio[:expected_samples]

    metrics = _compute_metrics(pt_audio, tf_audio)
    logger.info(
        "Full-forward parity metrics: "
        f"samples={int(metrics['samples'])} "
        f"max_abs={metrics['max_abs']:.8f} "
        f"mae={metrics['mae']:.8f} "
        f"rmse={metrics['rmse']:.8f} "
        f"corr={metrics['corr']:.8f}"
    )

    streaming_metrics: dict[str, float] | None = None
    pt_audio_stream_np: np.ndarray | None = None
    tf_audio_stream_np: np.ndarray | None = None
    if args.compare_streaming_chunked:
        chunk_size_cache, padding = _compute_streaming_window(
            backbone_layers=cfg.backbone_layers,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            sample_rate=args.sample_rate,
            chunk_size_ms=args.stream_chunk_size_ms,
            padding_ms=args.stream_padding_ms,
        )
        logger.info(
            "Running chunked streaming parity: "
            f"cache-window chunk_frames={chunk_size_cache}, padding_frames={padding}, hop={cfg.hop_length}"
        )

        with torch.inference_mode():
            pt_stream = _streaming_decode_features(
                feat_pt,
                chunk_size=chunk_size_cache,
                padding=padding,
                hop_length=cfg.hop_length,
                run_decode=lambda cache: pt_model(cache),
            )

        def _tf_decode(cache: torch.Tensor) -> torch.Tensor:
            cache_np = cache.detach().cpu().numpy().astype(np.float32, copy=False)
            out_np = tf_model(tf.convert_to_tensor(cache_np, dtype=tf.float32), training=False).numpy()
            return torch.from_numpy(out_np).to(device=cache.device, dtype=cache.dtype)

        tf_stream = _streaming_decode_features(
            feat_pt,
            chunk_size=chunk_size_cache,
            padding=padding,
            hop_length=cfg.hop_length,
            run_decode=_tf_decode,
        )

        pt_audio_stream_np = pt_stream.detach().cpu().numpy()[0].astype(np.float32, copy=False)
        tf_audio_stream_np = tf_stream.detach().cpu().numpy()[0].astype(np.float32, copy=False)
        if expected_samples > 0:
            pt_audio_stream_np = pt_audio_stream_np[:expected_samples]
            tf_audio_stream_np = tf_audio_stream_np[:expected_samples]

        streaming_metrics = _compute_metrics(pt_audio_stream_np, tf_audio_stream_np)
        logger.info(
            "Chunked-streaming parity metrics: "
            f"samples={int(streaming_metrics['samples'])} "
            f"max_abs={streaming_metrics['max_abs']:.8f} "
            f"mae={streaming_metrics['mae']:.8f} "
            f"rmse={streaming_metrics['rmse']:.8f} "
            f"corr={streaming_metrics['corr']:.8f}"
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

        if pt_audio_stream_np is not None and tf_audio_stream_np is not None:
            pt_stream_wav = args.save_audio_dir / "pytorch_streaming.wav"
            tf_stream_wav = args.save_audio_dir / "tensorflow_streaming.wav"
            diff_stream_wav = args.save_audio_dir / "abs_diff_streaming.wav"
            save_wav_16bit(pt_stream_wav, pt_audio_stream_np, sample_rate=args.sample_rate)
            save_wav_16bit(tf_stream_wav, tf_audio_stream_np, sample_rate=args.sample_rate)
            n_stream = min(len(pt_audio_stream_np), len(tf_audio_stream_np))
            abs_diff_stream = np.clip(
                np.abs(pt_audio_stream_np[:n_stream] - tf_audio_stream_np[:n_stream]) * 20.0, 0.0, 1.0
            )
            save_wav_16bit(diff_stream_wav, abs_diff_stream, sample_rate=args.sample_rate)
            logger.info(f"Wrote streaming comparison audio: {pt_stream_wav}, {tf_stream_wav}, {diff_stream_wav}")

    if not args.no_save_plot:
        plot_path = args.save_audio_dir / "waveform_compare.png"
        _plot_waveforms(
            pt_audio=pt_audio,
            tf_audio=tf_audio,
            sample_rate=args.sample_rate,
            output_path=plot_path,
        )
        logger.info(f"Wrote waveform plot: {plot_path}")
        if pt_audio_stream_np is not None and tf_audio_stream_np is not None:
            stream_plot_path = args.save_audio_dir / "waveform_compare_streaming.png"
            _plot_waveforms(
                pt_audio=pt_audio_stream_np,
                tf_audio=tf_audio_stream_np,
                sample_rate=args.sample_rate,
                output_path=stream_plot_path,
            )
            logger.info(f"Wrote streaming waveform plot: {stream_plot_path}")

    if metrics["max_abs"] > float(args.max_abs_threshold) or metrics["rmse"] > float(args.rmse_threshold):
        raise RuntimeError(
            "PT/TF full-forward parity check failed: "
            f"max_abs={metrics['max_abs']:.8f} (threshold={args.max_abs_threshold}), "
            f"rmse={metrics['rmse']:.8f} (threshold={args.rmse_threshold})"
        )

    if streaming_metrics is not None and (
        streaming_metrics["max_abs"] > float(args.streaming_max_abs_threshold)
        or streaming_metrics["rmse"] > float(args.streaming_rmse_threshold)
    ):
        raise RuntimeError(
            "PT/TF chunked-streaming parity check failed: "
            f"max_abs={streaming_metrics['max_abs']:.8f} (threshold={args.streaming_max_abs_threshold}), "
            f"rmse={streaming_metrics['rmse']:.8f} (threshold={args.streaming_rmse_threshold})"
        )

    logger.info("PT/TF parity check passed.")


if __name__ == "__main__":
    main()
