"""Investigate TF full vs chunked causal Vocos quality using converted TF weights.

Purpose
- Load TensorFlow generator weights produced by `kokoro.tf.convert`:
  `generator_config.json` + `generator.weights.h5`.
- Compare full-forward TF decode vs chunked TF decode on random English pairs.
- Save waveform plots and boundary-focused diagnostics to explain audible artifacts.

Chunked variants
- `naive`: split raw features into chunks and run full TF generator per chunk.
- `conditioned`: run conditioner once on full features, then chunk backbone+head.
- `both` (default): run both variants for side-by-side analysis.

Outputs
- `report.json` with per-sample metrics and aggregate stats by `(variant, chunk_ms)`.
- Optional wavs:
  - `full_tf.wav`
  - `stream_<variant>_chunk_<N>ms.wav`
- Optional plots:
  - `plot_<variant>_chunk_<N>ms.png` with full/chunked overlay and boundary zoom.

Example
`uv run python -m kokoro.tf.investigate_chunked_streaming_vocos_tf \
  --tf-config output/tf_checkpoints/generator_config.json \
  --tf-weights output/tf_checkpoints/generator.weights.h5 \
  --data-root /export/eingerman/audio/vocoder \
  --out-dir output/tf_chunked_vocos_compare \
  --chunk-sizes-ms 40,80,160,300 \
  --chunked-variant both`
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import tensorflow as tf
import torch
from loguru import logger

from .checkpoint_utils import TFVocosGeneratorConfig, build_feature_from_pair_payload, build_tf_generator, save_wav_16bit

ENGLISH_VOICE_PREFIXES = ("af_", "am_", "bf_", "bm_")


def _parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("expected at least one integer value")
    return out


def _parse_variants(variant: str) -> List[str]:
    v = variant.strip().lower()
    if v == "both":
        return ["naive", "conditioned"]
    if v not in {"naive", "conditioned"}:
        raise ValueError("--chunked-variant must be one of: naive, conditioned, both")
    return [v]


def _is_english_voice(name: str) -> bool:
    n = name.lower()
    return any(n.startswith(prefix) for prefix in ENGLISH_VOICE_PREFIXES)


def _select_random_english_pairs(pairs_root: Path, num_samples: int, seed: int) -> List[Path]:
    voice_dirs = [p for p in pairs_root.iterdir() if p.is_dir() and _is_english_voice(p.name)]
    if not voice_dirs:
        raise RuntimeError(f"No English voice directories found under {pairs_root}")
    rng = random.Random(seed)
    rng.shuffle(voice_dirs)
    picks: List[Path] = []
    for voice_dir in voice_dirs:
        files = sorted(voice_dir.glob("*.pt"))
        if not files:
            continue
        picks.append(rng.choice(files))
        if len(picks) >= num_samples:
            break
    if len(picks) < num_samples:
        raise RuntimeError(f"Requested {num_samples} samples but found only {len(picks)} eligible voices in {pairs_root}")
    return picks


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(a.shape[-1], b.shape[-1])
    return a[:n], b[:n]


def _stft_logmag_distance(a: np.ndarray, b: np.ndarray, n_fft: int = 1024, hop_length: int = 256) -> float:
    a_t = torch.from_numpy(a).float()
    b_t = torch.from_numpy(b).float()
    window = torch.hann_window(n_fft, dtype=a_t.dtype)
    sa = torch.stft(a_t, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    sb = torch.stft(b_t, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    return float(torch.mean(torch.abs(torch.log1p(sa.abs()) - torch.log1p(sb.abs()))).item())


def _metrics(ref: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    ref, pred = _align(ref, pred)
    if ref.size == 0:
        return {"samples": 0, "l1": 0.0, "rmse": 0.0, "snr_db": 0.0, "stft_logmag_l1": 0.0}
    err = pred - ref
    l1 = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    num = float(np.sum(ref * ref))
    den = float(np.sum(err * err) + 1e-12)
    snr_db = float(10.0 * np.log10(num / den))
    return {
        "samples": int(ref.shape[-1]),
        "l1": l1,
        "rmse": rmse,
        "snr_db": snr_db,
        "stft_logmag_l1": _stft_logmag_distance(ref, pred),
    }


def _boundary_click_score(audio: np.ndarray, chunk_samples: int) -> float:
    if chunk_samples <= 0 or audio.size == 0:
        return 0.0
    idx = np.arange(chunk_samples, audio.shape[-1], chunk_samples, dtype=np.int64)
    if idx.size == 0:
        return 0.0
    jumps = np.abs(audio[idx] - audio[idx - 1])
    return float(np.mean(jumps))


def _clip_fraction(audio: np.ndarray, threshold: float = 0.999) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= threshold))


def _boundary_error_metrics(ref: np.ndarray, pred: np.ndarray, chunk_samples: int, window_samples: int) -> Dict[str, float]:
    ref, pred = _align(ref, pred)
    if ref.size == 0:
        return {"boundary_err_rmse": 0.0, "boundary_err_peak": 0.0}
    err = pred - ref
    if chunk_samples <= 0:
        return {
            "boundary_err_rmse": float(np.sqrt(np.mean(err * err))),
            "boundary_err_peak": float(np.max(np.abs(err))),
        }
    idx = np.arange(chunk_samples, err.shape[-1], chunk_samples, dtype=np.int64)
    if idx.size == 0:
        return {"boundary_err_rmse": 0.0, "boundary_err_peak": 0.0}
    w = max(1, int(window_samples))
    segments: List[np.ndarray] = []
    for i in idx.tolist():
        lo = max(0, int(i) - w)
        hi = min(err.shape[-1], int(i) + w)
        if hi > lo:
            segments.append(err[lo:hi])
    if not segments:
        return {"boundary_err_rmse": 0.0, "boundary_err_peak": 0.0}
    cat = np.concatenate(segments, axis=0)
    return {
        "boundary_err_rmse": float(np.sqrt(np.mean(cat * cat))),
        "boundary_err_peak": float(np.max(np.abs(cat))),
    }


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(x)), "std": float(np.std(x))}


def _save_waveform_plot(
    out_path: Path,
    ref: np.ndarray,
    pred: np.ndarray,
    sample_rate: int,
    chunk_samples: int,
    title: str,
    boundary_window_ms: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not import matplotlib for plots: {exc}")
        return

    ref, pred = _align(ref.astype(np.float32, copy=False), pred.astype(np.float32, copy=False))
    if ref.size == 0:
        return
    err = pred - ref
    n = ref.shape[-1]
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    boundaries = np.arange(chunk_samples, n, chunk_samples, dtype=np.int64) if chunk_samples > 0 else np.array([], dtype=np.int64)
    window_samples = max(1, int(boundary_window_ms * sample_rate / 1000.0))
    worst_boundary = int(boundaries[0]) if boundaries.size > 0 else n // 2
    if boundaries.size > 0:
        scores = []
        for b in boundaries.tolist():
            lo = max(0, b - window_samples)
            hi = min(n, b + window_samples)
            scores.append(float(np.max(np.abs(err[lo:hi])) if hi > lo else 0.0))
        worst_boundary = int(boundaries[int(np.argmax(np.asarray(scores)))])
    lo = max(0, worst_boundary - window_samples)
    hi = min(n, worst_boundary + window_samples)
    tz = t[lo:hi]

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=True)
    axes[0].plot(t, ref, linewidth=0.6, label="full_tf", alpha=0.9)
    axes[0].plot(t, pred, linewidth=0.6, label="chunked_tf", alpha=0.75)
    if boundaries.size <= 120:
        for b in boundaries.tolist():
            axes[0].axvline(float(b) / float(sample_rate), color="0.75", linewidth=0.4)
    axes[0].set_title(title)
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, err, color="tab:red", linewidth=0.5)
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Error")
    axes[1].set_title("Full - Chunked Error")

    axes[2].plot(tz, ref[lo:hi], linewidth=0.9, label="full_tf", alpha=0.9)
    axes[2].plot(tz, pred[lo:hi], linewidth=0.9, label="chunked_tf", alpha=0.8)
    axes[2].axvline(float(worst_boundary) / float(sample_rate), color="tab:orange", linewidth=1.0, label="worst boundary")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_title(f"Boundary Zoom (+/- {boundary_window_ms:.1f} ms)")
    axes[2].legend(loc="upper right")

    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _decode_full(model: tf.keras.Model, features_bct: np.ndarray) -> np.ndarray:
    audio = model(tf.convert_to_tensor(features_bct, dtype=tf.float32), training=False).numpy()
    return np.asarray(audio[0], dtype=np.float32)


def _split_full_and_tail(features_bct: np.ndarray, chunk_frames: int) -> tuple[np.ndarray, np.ndarray]:
    total = int(features_bct.shape[-1])
    chunk = int(chunk_frames)
    n_full = total // chunk
    tail = total - (n_full * chunk)
    if n_full > 0:
        c = int(features_bct.shape[1])
        full = features_bct[:, :, : n_full * chunk]
        # [1, C, n_full * chunk] -> [n_full, C, chunk]
        full_chunks = full.reshape(1, c, n_full, chunk)[0].transpose(1, 0, 2).astype(np.float32, copy=False)
    else:
        full_chunks = np.zeros((0, int(features_bct.shape[1]), chunk), dtype=np.float32)
    if tail > 0:
        tail_chunk = features_bct[:, :, -tail:].astype(np.float32, copy=False)
    else:
        tail_chunk = np.zeros((0, int(features_bct.shape[1]), 0), dtype=np.float32)
    return full_chunks, tail_chunk


def _decode_chunked_naive(
    features_bct: np.ndarray,
    chunk_frames: int,
    run_full_decode: tf.types.experimental.ConcreteFunction | object,
    hop_length: int,
) -> np.ndarray:
    full_chunks, tail_chunk = _split_full_and_tail(features_bct, chunk_frames)
    out: List[np.ndarray] = []
    if full_chunks.shape[0] > 0:
        audio_full = run_full_decode(tf.convert_to_tensor(full_chunks, dtype=tf.float32)).numpy()
        out.append(np.asarray(audio_full.reshape(-1), dtype=np.float32))
    if tail_chunk.shape[0] > 0:
        audio_tail = run_full_decode(tf.convert_to_tensor(tail_chunk, dtype=tf.float32)).numpy()
        tail_samples = int(tail_chunk.shape[-1]) * int(hop_length)
        out.append(np.asarray(audio_tail[0, :tail_samples], dtype=np.float32))
    if not out:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)


def _decode_chunked_conditioned(
    features_bct: np.ndarray,
    chunk_frames: int,
    run_conditioner: tf.types.experimental.ConcreteFunction | object,
    run_backbone_head: tf.types.experimental.ConcreteFunction | object,
    hop_length: int,
) -> np.ndarray:
    conditioned = run_conditioner(tf.convert_to_tensor(features_bct, dtype=tf.float32)).numpy()
    full_chunks, tail_chunk = _split_full_and_tail(np.asarray(conditioned, dtype=np.float32), chunk_frames)
    out: List[np.ndarray] = []
    if full_chunks.shape[0] > 0:
        audio_full = run_backbone_head(tf.convert_to_tensor(full_chunks, dtype=tf.float32)).numpy()
        out.append(np.asarray(audio_full.reshape(-1), dtype=np.float32))
    if tail_chunk.shape[0] > 0:
        audio_tail = run_backbone_head(tf.convert_to_tensor(tail_chunk, dtype=tf.float32)).numpy()
        tail_samples = int(tail_chunk.shape[-1]) * int(hop_length)
        out.append(np.asarray(audio_tail[0, :tail_samples], dtype=np.float32))
    if not out:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)


def run(args: argparse.Namespace) -> None:
    pairs_root = (args.pairs_root if args.pairs_root else (args.data_root / "pairs")).resolve()
    selected_pairs = _select_random_english_pairs(pairs_root, args.num_samples, args.seed)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_data = json.loads(args.tf_config.read_text(encoding="utf-8"))
    cfg = TFVocosGeneratorConfig(**cfg_data)
    model = build_tf_generator(cfg)
    model.load_weights(str(args.tf_weights))
    # Cache graph traces once and avoid per-chunk retracing/launch overhead.
    in_ch = int(cfg.in_channels)
    cond_ch = int(cfg.model_input_channels)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, in_ch, None], dtype=tf.float32)], reduce_retracing=True)
    def run_full_decode(x: tf.Tensor) -> tf.Tensor:
        return model(x, training=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, in_ch, None], dtype=tf.float32)], reduce_retracing=True)
    def run_conditioner(x: tf.Tensor) -> tf.Tensor:
        return tf.transpose(model.conditioner(tf.transpose(x, [0, 2, 1]), training=False), [0, 2, 1])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, cond_ch, None], dtype=tf.float32)], reduce_retracing=True)
    def run_backbone_head(x: tf.Tensor) -> tf.Tensor:
        return model.head(model.backbone(x, training=False), training=False)

    chunk_sizes = _parse_csv_ints(args.chunk_sizes_ms)
    variants = _parse_variants(args.chunked_variant)

    report: Dict[str, object] = {
        "tf_config": str(args.tf_config),
        "tf_weights": str(args.tf_weights),
        "pairs_root": str(pairs_root),
        "selected_pairs": [str(p) for p in selected_pairs],
        "variants": variants,
        "chunk_sizes_ms": chunk_sizes,
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "samples": [],
        "aggregate_by_chunk_variant": {},
    }

    for sample_idx, pair_path in enumerate(selected_pairs):
        pair = torch.load(pair_path, map_location="cpu", weights_only=False)
        features = build_feature_from_pair_payload(pair)
        voice = pair_path.parent.name
        stem = pair_path.stem

        full_ref = _decode_full(model, features)

        if args.save_audio:
            sample_dir = out_dir / f"{sample_idx:02d}_{voice}_{stem}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            save_wav_16bit(sample_dir / "full_tf.wav", full_ref, sample_rate=args.sample_rate)
        else:
            sample_dir = out_dir

        sample_entry: Dict[str, object] = {
            "pair": str(pair_path),
            "voice": voice,
            "full_num_samples": int(full_ref.shape[-1]),
            "experiments": [],
        }

        for chunk_ms in chunk_sizes:
            chunk_frames = max(1, int(chunk_ms / 1000.0 * args.sample_rate / args.hop_length))
            chunk_samples = chunk_frames * int(args.hop_length)
            for variant in variants:
                if variant == "naive":
                    stream_audio = _decode_chunked_naive(
                        features,
                        chunk_frames,
                        run_full_decode=run_full_decode,
                        hop_length=args.hop_length,
                    )
                else:
                    stream_audio = _decode_chunked_conditioned(
                        features,
                        chunk_frames,
                        run_conditioner=run_conditioner,
                        run_backbone_head=run_backbone_head,
                        hop_length=args.hop_length,
                    )

                if args.save_audio:
                    save_wav_16bit(
                        sample_dir / f"stream_{variant}_chunk_{chunk_ms}ms.wav",
                        stream_audio,
                        sample_rate=args.sample_rate,
                    )
                if args.save_plots:
                    _save_waveform_plot(
                        out_path=sample_dir / f"plot_{variant}_chunk_{chunk_ms}ms.png",
                        ref=full_ref,
                        pred=stream_audio,
                        sample_rate=args.sample_rate,
                        chunk_samples=chunk_samples,
                        title=f"{voice}/{stem} - {variant} chunk={chunk_ms}ms",
                        boundary_window_ms=float(args.boundary_window_ms),
                    )

                m = _metrics(full_ref, stream_audio)
                ref_aligned, pred_aligned = _align(full_ref, stream_audio)
                err = pred_aligned - ref_aligned
                m["variant"] = variant
                m["chunk_ms"] = int(chunk_ms)
                m["effective_chunk_frames"] = int(chunk_frames)
                m["effective_padding_frames"] = 0
                m["boundary_click_score"] = _boundary_click_score(pred_aligned, chunk_samples)
                m["boundary_err_click_score"] = _boundary_click_score(np.abs(err), chunk_samples)
                m["pred_peak_abs"] = float(np.max(np.abs(pred_aligned))) if pred_aligned.size > 0 else 0.0
                m["pred_clip_fraction"] = _clip_fraction(pred_aligned)
                m["err_peak_abs"] = float(np.max(np.abs(err))) if err.size > 0 else 0.0
                m["err_p99_abs"] = float(np.percentile(np.abs(err), 99.0)) if err.size > 0 else 0.0
                m.update(
                    _boundary_error_metrics(
                        ref_aligned,
                        pred_aligned,
                        chunk_samples=chunk_samples,
                        window_samples=max(1, int(args.boundary_window_ms * args.sample_rate / 1000.0)),
                    )
                )
                sample_entry["experiments"].append(m)
        report["samples"].append(sample_entry)

    aggregate: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for variant in variants:
        aggregate[variant] = {}
        for chunk_ms in chunk_sizes:
            rows = []
            for sample in report["samples"]:
                for exp in sample["experiments"]:
                    if str(exp["variant"]) == variant and int(exp["chunk_ms"]) == int(chunk_ms):
                        rows.append(exp)
            aggregate[variant][str(chunk_ms)] = {
                "snr_db": _mean_std([float(r["snr_db"]) for r in rows]),
                "l1": _mean_std([float(r["l1"]) for r in rows]),
                "rmse": _mean_std([float(r["rmse"]) for r in rows]),
                "stft_logmag_l1": _mean_std([float(r["stft_logmag_l1"]) for r in rows]),
                "boundary_click_score": _mean_std([float(r["boundary_click_score"]) for r in rows]),
                "boundary_err_click_score": _mean_std([float(r["boundary_err_click_score"]) for r in rows]),
                "boundary_err_rmse": _mean_std([float(r["boundary_err_rmse"]) for r in rows]),
                "boundary_err_peak": _mean_std([float(r["boundary_err_peak"]) for r in rows]),
                "pred_peak_abs": _mean_std([float(r["pred_peak_abs"]) for r in rows]),
                "pred_clip_fraction": _mean_std([float(r["pred_clip_fraction"]) for r in rows]),
                "err_peak_abs": _mean_std([float(r["err_peak_abs"]) for r in rows]),
                "err_p99_abs": _mean_std([float(r["err_p99_abs"]) for r in rows]),
            }
    report["aggregate_by_chunk_variant"] = aggregate

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Selected {len(selected_pairs)} English samples from distinct voices under: {pairs_root}")
    for p in selected_pairs:
        print(f"  - {p.parent.name}: {p.name}")
    for variant in variants:
        for chunk_ms in chunk_sizes:
            agg = aggregate[variant][str(chunk_ms)]
            print(
                f"variant={variant} chunk={chunk_ms}ms snr={agg['snr_db']['mean']:.2f}±{agg['snr_db']['std']:.2f}dB "
                f"l1={agg['l1']['mean']:.6f}±{agg['l1']['std']:.6f} "
                f"lsd={agg['stft_logmag_l1']['mean']:.6f}±{agg['stft_logmag_l1']['std']:.6f} "
                f"jump={agg['boundary_click_score']['mean']:.6f}±{agg['boundary_click_score']['std']:.6f} "
                f"b_rmse={agg['boundary_err_rmse']['mean']:.6f}±{agg['boundary_err_rmse']['std']:.6f} "
                f"clip={agg['pred_clip_fraction']['mean']:.6f}±{agg['pred_clip_fraction']['std']:.6f}"
            )
    print(f"Saved report: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Investigate TF chunked vs full streaming-vocos decode quality")
    p.add_argument("--tf-config", type=Path, default=Path("output/tf_checkpoints/generator_config.json"))
    p.add_argument("--tf-weights", type=Path, default=Path("output/tf_checkpoints/generator.weights.h5"))
    p.add_argument("--data-root", type=Path, default=Path("inputs/"))
    p.add_argument("--pairs-root", type=Path, default=None, help="Override pairs root; defaults to <data-root>/pairs")
    p.add_argument("--num-samples", type=int, default=20, help="Number of random English samples from distinct voices")
    p.add_argument("--seed", type=int, default=4444)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--hop-length", type=int, default=300)
    p.add_argument("--chunk-sizes-ms", type=str, default="40,80,160,300")
    p.add_argument("--chunked-variant", type=str, default="both", choices=["naive", "conditioned", "both"])
    p.add_argument("--save-audio", dest="save_audio", action="store_true")
    p.add_argument("--no-save-audio", dest="save_audio", action="store_false")
    p.set_defaults(save_audio=True)
    p.add_argument("--save-plots", dest="save_plots", action="store_true")
    p.add_argument("--no-save-plots", dest="save_plots", action="store_false")
    p.set_defaults(save_plots=True)
    p.add_argument("--boundary-window-ms", type=float, default=20.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
