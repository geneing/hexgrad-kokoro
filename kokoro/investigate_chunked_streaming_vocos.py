"""Investigate quality gaps between full and chunked Streaming Vocos decoding.

Purpose
- Compare non-chunked decoding (`PTVocosDecoder`) against chunked streaming
  decoding (`StreamingPTVocosDecoder`) using the same conditioning inputs.
- Compare chunked outputs across quantization variants (`fp32`, `fp16`, `int8`)
  against fp32 chunked reference.
- Export quantized PT weights for causal streaming vocos.
- Measure objective deltas that correlate with perceived quality loss:
  `snr_db`, `l1`, `rmse`, `stft_logmag_l1`, and a chunk-boundary click score.
- Identify whether chunk size or padding settings are causing regressions.

Sample selection
- This script automatically selects random English samples from distinct voices.
- English voices are detected by voice directory prefixes:
  `af_`, `am_`, `bf_`, `bm_`.
- Default selection is 20 samples (`--num-samples 20`), one random `.pt` per
  eligible voice directory, seeded by `--seed`.

Inputs
- `--checkpoint`: path to vocoder checkpoint/state dict.
- `--data-root`: dataset root containing `pairs/` (default:
  `inputs/`).
- Optional `--pairs-root` to override `<data-root>/pairs`.
- `--quantization-variants`: comma-separated subset from `fp32,fp16,int8`.
- Optional `--export-weights-dir`: writes `vocos_fp32.pt`, `vocos_fp16.pt`,
  and `vocos_int8_qdq.pt`.

Outputs
- `report.json` with:
  - selected sample list,
  - per-sample metrics for each `(variant, chunk size)`,
  - aggregate mean/std metrics per `(variant, chunk size)`.
- Audio files are saved by default (`--save-audio`), with one folder per sample:
  - `full_fp32.wav`, plus `full_<variant>.wav` for quantized variants
  - `stream_<variant>_chunk_<N>ms.wav` for each requested chunk size.

Examples
1) Default run (20 random English samples, default chunk sizes):
`uv run python -m kokoro.investigate_chunked_streaming_vocos \
  --checkpoint output/checkpoints/last.pt \
  --data-root inputs/ \
  --out-dir output/chunked_vocos_debug`

2) Custom chunk sizes and deterministic seed:
`uv run python -m kokoro.investigate_chunked_streaming_vocos \
  --checkpoint output/checkpoints/last.pt \
  --data-root inputs/ \
  --out-dir output/chunked_vocos_debug_c40_80_120 \
  --chunk-sizes-ms 40,80,120 \
  --seed 1234`

3) Use explicit pairs root, fewer samples, and skip wav output:
`uv run python -m kokoro.investigate_chunked_streaming_vocos \
  --checkpoint output/checkpoints/last.pt \
  --pairs-root inputs//pairs \
  --num-samples 12 \
  --no-save-audio \
  --out-dir output/chunked_vocos_metrics_only`

4) Force streaming backend + local streaming-vocos repo:
`uv run python -m kokoro.investigate_chunked_streaming_vocos \
  --checkpoint output/checkpoints/last.pt \
  --data-root inputs/ \
  --out-dir output/chunked_vocos_streaming_impl \
  --vocos-impl streaming \
  --streaming-vocos-repo third_party/vocos_streaming`

5) Export quantized PT weights and compare all variants:
`uv run python -m kokoro.investigate_chunked_streaming_vocos \
  --checkpoint output/checkpoints/last.pt \
  --data-root inputs/ \
  --out-dir output/chunked_vocos_quant_compare \
  --export-weights-dir output/saved_infer_weights_quant \
  --quantization-variants fp32,fp16,int8`
"""

from __future__ import annotations

import argparse
import json
import random
import wave
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from loguru import logger

from .vocos_decoder import PTVocosDecoder, StreamingPTVocosDecoder

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


def _parse_csv_variants(text: str) -> List[str]:
    allowed = {"fp32", "fp16", "int8"}
    out: List[str] = []
    for part in text.split(","):
        v = part.strip().lower()
        if not v:
            continue
        if v not in allowed:
            raise ValueError(f"Unsupported quantization variant '{v}'. Expected one of {sorted(allowed)}")
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError("Expected at least one quantization variant")
    if "fp32" not in out:
        out = ["fp32"] + out
    return out


def _load_pair(path: Path) -> Dict[str, torch.Tensor]:
    row = torch.load(path, map_location="cpu", weights_only=False)
    asr = row["asr"].float()
    f0 = row["f0"].float()
    noise = row["noise"].float()
    style = row["style"].float()
    if asr.ndim == 2:
        asr = asr.unsqueeze(0)
    if f0.ndim == 1:
        f0 = f0.unsqueeze(0)
    if noise.ndim == 1:
        noise = noise.unsqueeze(0)
    if style.ndim == 1:
        style = style.unsqueeze(0)
    return {"asr": asr, "f0": f0, "noise": noise, "style": style}


def _align_audio(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(int(a.shape[-1]), int(b.shape[-1]))
    return a[..., :n], b[..., :n]


def _stft_logmag_distance(a: torch.Tensor, b: torch.Tensor, n_fft: int = 1024, hop_length: int = 256) -> float:
    wa = torch.hann_window(n_fft, device=a.device, dtype=a.dtype)
    sa = torch.stft(a, n_fft=n_fft, hop_length=hop_length, window=wa, return_complex=True)
    sb = torch.stft(b, n_fft=n_fft, hop_length=hop_length, window=wa, return_complex=True)
    return float(torch.mean(torch.abs(torch.log1p(sa.abs()) - torch.log1p(sb.abs()))).item())


def _metrics(ref: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    ref, pred = _align_audio(ref, pred)
    err = pred - ref
    l1 = float(torch.mean(torch.abs(err)).item())
    rmse = float(torch.sqrt(torch.mean(err * err)).item())
    num = torch.sum(ref * ref)
    den = torch.sum(err * err) + 1e-12
    snr_db = float(10.0 * torch.log10(num / den).item())
    lsd = _stft_logmag_distance(ref, pred)
    return {
        "samples": int(ref.shape[-1]),
        "l1": l1,
        "rmse": rmse,
        "snr_db": snr_db,
        "stft_logmag_l1": lsd,
    }


def _boundary_click_score(audio: torch.Tensor, chunk_samples: int) -> float:
    if chunk_samples <= 0:
        return 0.0
    t = int(audio.shape[-1])
    idx = torch.arange(chunk_samples, t, chunk_samples, device=audio.device)
    if idx.numel() == 0:
        return 0.0
    jumps = torch.abs(audio[idx] - audio[idx - 1])
    return float(jumps.mean().item())


def _is_english_voice(voice: str) -> bool:
    low = voice.lower()
    return any(low.startswith(prefix) for prefix in ENGLISH_VOICE_PREFIXES)


def _select_random_english_pairs(pairs_root: Path, num_samples: int, seed: int) -> List[Path]:
    voice_dirs = [p for p in pairs_root.iterdir() if p.is_dir() and _is_english_voice(p.name)]
    if not voice_dirs:
        raise RuntimeError(f"No English voice directories found under {pairs_root}")

    rng = random.Random(seed)
    rng.shuffle(voice_dirs)
    candidates: List[Path] = []
    for vdir in voice_dirs:
        files = sorted(vdir.glob("*.pt"))
        if not files:
            continue
        candidates.append(rng.choice(files))
        if len(candidates) >= num_samples:
            break

    if len(candidates) < num_samples:
        raise RuntimeError(
            f"Requested {num_samples} distinct English-voice samples but only found {len(candidates)} "
            f"voices with pair files in {pairs_root}"
        )
    return candidates


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    t = torch.tensor(values, dtype=torch.float32)
    return {"mean": float(t.mean().item()), "std": float(t.std(unbiased=False).item())}


def _copy_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _fp16_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if torch.is_tensor(v) and v.is_floating_point():
            out[k] = v.detach().cpu().half()
        else:
            out[k] = v.detach().cpu().clone()
    return out


def _fp16_qdq_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if torch.is_tensor(v) and v.is_floating_point():
            out[k] = v.detach().cpu().half().float()
        else:
            out[k] = v.detach().cpu().clone()
    return out


def _int8_qdq_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if torch.is_tensor(v) and v.is_floating_point():
            x = v.detach().cpu().float()
            max_abs = float(x.abs().max().item()) if x.numel() > 0 else 0.0
            if max_abs < 1e-12:
                out[k] = x
            else:
                scale = max_abs / 127.0
                q = torch.clamp(torch.round(x / scale), -127, 127)
                out[k] = (q * scale).float()
        else:
            out[k] = v.detach().cpu().clone()
    return out


def _save_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    x = audio.detach().cpu().float()
    if x.ndim == 2:
        x = x[0]
    x = torch.clamp(x, -1.0, 1.0)
    pcm16 = (x.numpy() * 32767.0).round().astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def run(args: argparse.Namespace) -> None:
    if args.device == "auto" and torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            min_free_bytes = 2 * 1024 * 1024 * 1024  # 2 GiB safety floor for multi-variant decoding
            if free_bytes < min_free_bytes:
                logger.warning(
                    "CUDA available but low free VRAM "
                    f"({free_bytes / (1024**3):.2f} GiB free / {total_bytes / (1024**3):.2f} GiB total); "
                    "falling back to CPU. Use --device cuda to force CUDA."
                )
                device = torch.device("cpu")
            else:
                device = torch.device("cuda")
        except Exception:
            logger.warning("Could not query CUDA free memory; falling back to CPU for safety in auto mode.")
            device = torch.device("cpu")
    else:
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    pairs_root = (args.pairs_root if args.pairs_root else (args.data_root / "pairs")).resolve()
    selected_pairs = _select_random_english_pairs(pairs_root, args.num_samples, args.seed)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    first_pair = _load_pair(selected_pairs[0])
    base_full_decoder = PTVocosDecoder(
        dim_in=int(first_pair["asr"].shape[1]),
        style_dim=int(first_pair["style"].shape[1]),
        model_input_channels=args.model_input_channels,
        backbone_dim=args.backbone_dim,
        backbone_intermediate_dim=args.backbone_intermediate_dim,
        backbone_layers=args.backbone_layers,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        checkpoint_path=str(args.checkpoint),
        vocos_impl=args.vocos_impl,
        streaming_vocos_repo=str(args.streaming_vocos_repo),
        backbone_causal=args.backbone_causal,
        backbone_pad_mode=args.backbone_pad_mode,
        backbone_norm=args.backbone_norm,
    ).to(device).eval()
    base_state = _copy_state(base_full_decoder.generator.state_dict())

    variant_names = _parse_csv_variants(args.quantization_variants)
    variant_states: Dict[str, Dict[str, torch.Tensor]] = {"fp32": _copy_state(base_state)}
    if "fp16" in variant_names:
        variant_states["fp16"] = _fp16_qdq_state(base_state)
    if "int8" in variant_names:
        variant_states["int8"] = _int8_qdq_state(base_state)

    if args.export_weights_dir is not None:
        export_dir = args.export_weights_dir.resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        torch.save(_copy_state(base_state), export_dir / "vocos_fp32.pt")
        if "fp16" in variant_names:
            torch.save(_fp16_state(base_state), export_dir / "vocos_fp16.pt")
        if "int8" in variant_names:
            torch.save(_int8_qdq_state(base_state), export_dir / "vocos_int8_qdq.pt")

    full_decoders: Dict[str, PTVocosDecoder] = {}
    stream_decoders_by_variant: Dict[str, Dict[int, StreamingPTVocosDecoder]] = {}
    for variant in variant_names:
        d_full = PTVocosDecoder(
            dim_in=int(first_pair["asr"].shape[1]),
            style_dim=int(first_pair["style"].shape[1]),
            model_input_channels=args.model_input_channels,
            backbone_dim=args.backbone_dim,
            backbone_intermediate_dim=args.backbone_intermediate_dim,
            backbone_layers=args.backbone_layers,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            vocos_impl=args.vocos_impl,
            streaming_vocos_repo=str(args.streaming_vocos_repo),
            backbone_causal=args.backbone_causal,
            backbone_pad_mode=args.backbone_pad_mode,
            backbone_norm=args.backbone_norm,
        ).to(device).eval()
        d_full.generator.load_state_dict(variant_states[variant], strict=True)
        full_decoders[variant] = d_full

    chunk_sizes = _parse_csv_ints(args.chunk_sizes_ms)
    for variant in variant_names:
        local: Dict[int, StreamingPTVocosDecoder] = {}
        for chunk_ms in chunk_sizes:
            d_stream = StreamingPTVocosDecoder(
                dim_in=int(first_pair["asr"].shape[1]),
                style_dim=int(first_pair["style"].shape[1]),
                model_input_channels=args.model_input_channels,
                backbone_dim=args.backbone_dim,
                backbone_intermediate_dim=args.backbone_intermediate_dim,
                backbone_layers=args.backbone_layers,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                vocos_impl=args.vocos_impl,
                streaming_vocos_repo=str(args.streaming_vocos_repo),
                backbone_causal=args.backbone_causal,
                backbone_pad_mode=args.backbone_pad_mode,
                backbone_norm=args.backbone_norm,
                sample_rate=args.sample_rate,
                chunk_size_ms=chunk_ms,
            ).to(device).eval()
            d_stream.generator.load_state_dict(variant_states[variant], strict=True)
            local[chunk_ms] = d_stream
        stream_decoders_by_variant[variant] = local

    report: Dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "variants": variant_names,
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "pairs_root": str(pairs_root),
        "selected_pairs": [str(p) for p in selected_pairs],
        "chunk_sizes_ms": chunk_sizes,
        "samples": [],
        "aggregate_by_chunk_variant": {},
    }

    for sample_idx, pair_path in enumerate(selected_pairs):
        pair = _load_pair(pair_path)
        asr = pair["asr"].to(device)
        f0 = pair["f0"].to(device)
        noise = pair["noise"].to(device)
        style = pair["style"].to(device)
        voice = pair_path.parent.name
        stem = pair_path.stem

        with torch.no_grad():
            full_ref = full_decoders["fp32"](asr, f0, noise, style)[0].detach().cpu()

        if args.save_audio:
            sample_dir = out_dir / f"{sample_idx:02d}_{voice}_{stem}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_wav(sample_dir / "full_fp32.wav", full_ref, sample_rate=args.sample_rate)
        else:
            sample_dir = out_dir

        sample_entry: Dict[str, object] = {
            "pair": str(pair_path),
            "voice": voice,
            "full_num_samples": int(full_ref.numel()),
            "experiments": [],
        }

        chunked_ref: Dict[int, torch.Tensor] = {}
        for chunk_ms in chunk_sizes:
            stream_decoder = stream_decoders_by_variant["fp32"][chunk_ms]
            chunks: List[torch.Tensor] = []
            with torch.no_grad():
                for part in stream_decoder.streaming_decode(asr, f0, noise, style, is_last=True):
                    if part.numel() > 0:
                        chunks.append(part.detach().cpu())
            stream_audio = torch.cat([c[0] for c in chunks], dim=-1) if chunks else torch.empty(0, dtype=full_ref.dtype)
            chunked_ref[chunk_ms] = stream_audio
            if args.save_audio:
                _save_wav(sample_dir / f"stream_fp32_chunk_{chunk_ms}ms.wav", stream_audio, sample_rate=args.sample_rate)

        for variant in variant_names:
            if args.save_audio and variant != "fp32":
                with torch.no_grad():
                    full_variant = full_decoders[variant](asr, f0, noise, style)[0].detach().cpu()
                _save_wav(sample_dir / f"full_{variant}.wav", full_variant, sample_rate=args.sample_rate)
            for chunk_ms in chunk_sizes:
                stream_decoder = stream_decoders_by_variant[variant][chunk_ms]
                chunks: List[torch.Tensor] = []
                with torch.no_grad():
                    for part in stream_decoder.streaming_decode(asr, f0, noise, style, is_last=True):
                        if part.numel() > 0:
                            chunks.append(part.detach().cpu())
                stream_audio = torch.cat([c[0] for c in chunks], dim=-1) if chunks else torch.empty(0, dtype=full_ref.dtype)
                if args.save_audio and variant != "fp32":
                    _save_wav(sample_dir / f"stream_{variant}_chunk_{chunk_ms}ms.wav", stream_audio, sample_rate=args.sample_rate)

                m = _metrics(chunked_ref[chunk_ms], stream_audio)
                m["variant"] = variant
                m["chunk_ms"] = chunk_ms
                m["effective_chunk_frames"] = int(stream_decoder.chunk_size)
                m["effective_padding_frames"] = 0
                m["boundary_click_score"] = _boundary_click_score(
                    torch.abs(stream_audio - chunked_ref[chunk_ms]),
                    stream_decoder.chunk_size * args.hop_length,
                )
                sample_entry["experiments"].append(m)

        report["samples"].append(sample_entry)

    aggregate: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for variant in variant_names:
        aggregate[variant] = {}
        for chunk_ms in chunk_sizes:
            rows = []
            for sample in report["samples"]:
                for exp in sample["experiments"]:
                    if str(exp["variant"]) == variant and int(exp["chunk_ms"]) == chunk_ms:
                        rows.append(exp)
            aggregate[variant][str(chunk_ms)] = {
                "snr_db": _mean_std([float(r["snr_db"]) for r in rows]),
                "l1": _mean_std([float(r["l1"]) for r in rows]),
                "rmse": _mean_std([float(r["rmse"]) for r in rows]),
                "stft_logmag_l1": _mean_std([float(r["stft_logmag_l1"]) for r in rows]),
                "boundary_click_score": _mean_std([float(r["boundary_click_score"]) for r in rows]),
            }
    report["aggregate_by_chunk_variant"] = aggregate

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Selected {len(selected_pairs)} English samples from distinct voices under: {pairs_root}")
    for p in selected_pairs:
        print(f"  - {p.parent.name}: {p.name}")
    for variant in variant_names:
        for chunk_ms in chunk_sizes:
            agg = aggregate[variant][str(chunk_ms)]
            print(
                f"variant={variant} chunk={chunk_ms}ms snr={agg['snr_db']['mean']:.2f}±{agg['snr_db']['std']:.2f}dB "
                f"l1={agg['l1']['mean']:.6f}±{agg['l1']['std']:.6f} "
                f"lsd={agg['stft_logmag_l1']['mean']:.6f}±{agg['stft_logmag_l1']['std']:.6f} "
                f"click={agg['boundary_click_score']['mean']:.6f}±{agg['boundary_click_score']['std']:.6f}"
            )
    print(f"Saved report: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Investigate chunked vs full streaming-vocos decode quality")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to vocos checkpoint/state_dict")
    p.add_argument("--data-root", type=Path, default=Path("inputs/"))
    p.add_argument("--pairs-root", type=Path, default=None, help="Override pairs root; defaults to <data-root>/pairs")
    p.add_argument("--num-samples", type=int, default=20, help="Number of random English samples from distinct voices")
    p.add_argument("--seed", type=int, default=4444)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--export-weights-dir", type=Path, default=None, help="Optional directory to export fp32/fp16/int8 PT weights")
    p.add_argument("--quantization-variants", type=str, default="fp32,fp16,int8")
    p.add_argument("--save-audio", dest="save_audio", action="store_true")
    p.add_argument("--no-save-audio", dest="save_audio", action="store_false")
    p.set_defaults(save_audio=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--hop-length", type=int, default=300)
    p.add_argument("--n-fft", type=int, default=1200)
    p.add_argument("--model-input-channels", type=int, default=192)
    p.add_argument("--backbone-dim", type=int, default=384)
    p.add_argument("--backbone-intermediate-dim", type=int, default=1152)
    p.add_argument("--backbone-layers", type=int, default=8)

    p.add_argument("--vocos-impl", type=str, default="auto", choices=["auto", "streaming", "legacy"])
    p.add_argument("--streaming-vocos-repo", type=Path, default=Path("third_party/vocos_streaming"))
    p.add_argument("--backbone-causal", dest="backbone_causal", action="store_true")
    p.add_argument("--no-backbone-causal", dest="backbone_causal", action="store_false")
    p.set_defaults(backbone_causal=True)
    p.add_argument("--backbone-pad-mode", type=str, default="constant")
    p.add_argument("--backbone-norm", type=str, default="weight_norm")

    p.add_argument("--chunk-sizes-ms", type=str, default="40,80,160,300")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
