"""Integration test for KModel decoder variants on vocoder validation pairs.

This utility:
1) Verifies KModel decoder selection (`istft`, `pt_vocos`, `streaming_pt_vocos`, `tf_vocos`, `streaming_tf_vocos`).
2) Generates audio with all decoders on N validation entries (default: 20).
3) Compares Vocos outputs to iSTFT using MFCC/log-mel cosine similarity proxies.
4) Writes WAV files, per-entry waveform plots, and aggregate metric reports.

Primary usage:
`uv run kokoro-test-kmodel-decoders --config-json /path/config.json --kokoro-checkpoint /path/kokoro.pth --vocos-checkpoint /path/vocos.pt --data-root /export/eingerman/audio/vocoder --num-entries 20 --output-dir output/decoder_compare`

Alternative module usage:
uv run python -m kokoro.test_kmodel_decoder_compare --config-json ../Kokoro/checkpoints/config.json --kokoro-checkpoint ../Kokoro/checkpoints/kokoro-v1_0.pth --vocos-checkpoint output/checkpoints/last.pt --data-root inputs/ --num-entries 20


Outputs:
- `audio/istft/*.wav`, `audio/pt_vocos/*.wav`, `audio/streaming_pt_vocos/*.wav`, `audio/tf_vocos/*.wav`, `audio/streaming_tf_vocos/*.wav`
- `plots/*.png` waveform comparisons
- `metrics.csv` per-sample similarity values
- `summary.json` aggregate means/stddev
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from loguru import logger

from .model import KModel
from .vocos_decoder import PTVocosDecoder, StreamingPTVocosDecoder, StreamingTFVocosDecoder, TFVocosDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compare KModel decoder variants on validation vocoder pairs")
    parser.add_argument("--config-json", type=Path, required=True, help="Kokoro config.json path")
    parser.add_argument("--kokoro-checkpoint", type=Path, required=True, help="Kokoro model .pth path")
    parser.add_argument(
        "--vocos-checkpoint",
        type=Path,
        required=True,
        help="Vocos generator checkpoint path from kokoro/train_vocos.py",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Vocoder data root containing audio/pairs/filelists")
    parser.add_argument("--val-filelist", type=Path, default=None, help="Defaults to <data-root>/filelists/vocos.val.txt")
    parser.add_argument("--output-dir", type=Path, default=Path("output/decoder_compare"))
    parser.add_argument("--num-entries", type=int, default=20)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_lines(path: Path) -> List[Path]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [Path(line).resolve() for line in lines]


def _derive_pair_path(data_root: Path, wav_path: Path) -> Path:
    audio_root = (data_root / "audio").resolve()
    pair_root = (data_root / "pairs").resolve()
    rel = wav_path.resolve().relative_to(audio_root)
    return pair_root / rel.with_suffix(".pt")


def _save_wav_16bit(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm16 = (waveform * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def _to_numpy_audio(audio_t: torch.Tensor, expected_samples: int) -> np.ndarray:
    audio_np = audio_t.detach().float().cpu().numpy().reshape(-1)
    if expected_samples > 0:
        audio_np = audio_np[:expected_samples]
    return audio_np.astype(np.float32, copy=False)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return math.nan
    return float(np.dot(a, b) / denom)


def _embed_mfcc(audio: np.ndarray, mfcc_tf: torchaudio.transforms.MFCC) -> np.ndarray:
    wav = torch.from_numpy(audio).float().unsqueeze(0)
    feat = mfcc_tf(wav).squeeze(0)
    emb = feat.mean(dim=-1)
    emb = (emb - emb.mean()) / (emb.std() + 1e-6)
    return emb.numpy().astype(np.float32, copy=False)


def _embed_logmel(audio: np.ndarray, mel_tf: torchaudio.transforms.MelSpectrogram) -> np.ndarray:
    wav = torch.from_numpy(audio).float().unsqueeze(0)
    mel = mel_tf(wav).squeeze(0).clamp_min(1e-7).log()
    emb = mel.mean(dim=-1)
    emb = (emb - emb.mean()) / (emb.std() + 1e-6)
    return emb.numpy().astype(np.float32, copy=False)


def _frame_rms_db(audio: np.ndarray, frame_length: int = 1200, hop_length: int = 300) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if x.size < frame_length:
        rms = float(np.sqrt(np.mean(x * x) + 1e-8))
        return np.asarray([20.0 * np.log10(rms + 1e-8)], dtype=np.float32)
    vals = []
    for start in range(0, x.size - frame_length + 1, hop_length):
        seg = x[start : start + frame_length]
        rms = float(np.sqrt(np.mean(seg * seg) + 1e-8))
        vals.append(20.0 * np.log10(rms + 1e-8))
    return np.asarray(vals, dtype=np.float32)


def _loudness_delta_std_db(ref_audio: np.ndarray, test_audio: np.ndarray) -> float:
    ref_db = _frame_rms_db(ref_audio)
    test_db = _frame_rms_db(test_audio)
    n = min(ref_db.size, test_db.size)
    if n <= 0:
        return math.nan
    delta = test_db[:n] - ref_db[:n]
    return float(np.std(delta))


def _boundary_click_score(audio: np.ndarray, chunk_samples: int, context: int = 128) -> float:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < 2 or chunk_samples <= 0:
        return math.nan
    boundaries = np.arange(chunk_samples, x.size, chunk_samples, dtype=np.int64)
    if boundaries.size == 0:
        return math.nan
    vals = []
    for b in boundaries:
        jump = abs(float(x[b] - x[b - 1]))
        lo = max(0, int(b) - context)
        hi = min(x.size, int(b) + context)
        local = x[lo:hi]
        local_rms = float(np.sqrt(np.mean(local * local) + 1e-8))
        vals.append(jump / max(local_rms, 1e-5))
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _phase_coherence(ref_audio: np.ndarray, test_audio: np.ndarray, n_fft: int = 1200, hop_length: int = 300) -> float:
    n = min(len(ref_audio), len(test_audio))
    if n <= 0:
        return math.nan
    x = torch.from_numpy(ref_audio[:n]).float()
    y = torch.from_numpy(test_audio[:n]).float()
    win = torch.hann_window(n_fft, dtype=torch.float32)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=win, return_complex=True, center=True)
    Y = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=win, return_complex=True, center=True)
    num = torch.abs((X * torch.conj(Y)).sum()).item()
    den = torch.sqrt((X.abs().pow(2).sum() * Y.abs().pow(2).sum())).item() + 1e-8
    return float(num / den)


def _boundary_phase_mismatch_mae_rad(
    ref_audio: np.ndarray,
    test_audio: np.ndarray,
    chunk_samples: int,
    n_fft: int = 1200,
    hop_length: int = 300,
    frame_radius: int = 2,
) -> float:
    n = min(len(ref_audio), len(test_audio))
    if n <= 0 or chunk_samples <= 0:
        return math.nan
    boundaries = np.arange(chunk_samples, n, chunk_samples, dtype=np.int64)
    if boundaries.size == 0:
        return math.nan

    x = torch.from_numpy(ref_audio[:n]).float()
    y = torch.from_numpy(test_audio[:n]).float()
    win = torch.hann_window(n_fft, dtype=torch.float32)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=win, return_complex=True, center=True)
    Y = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=win, return_complex=True, center=True)
    phase_diff = torch.angle(Y) - torch.angle(X)
    # Wrap to [-pi, pi]
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff)).abs()

    t_frames = X.shape[1]
    boundary_frames = np.unique(np.clip((boundaries // hop_length).astype(np.int64), 0, max(0, t_frames - 1)))
    mask = torch.zeros((t_frames,), dtype=torch.bool)
    for bf in boundary_frames.tolist():
        lo = max(0, int(bf) - frame_radius)
        hi = min(t_frames, int(bf) + frame_radius + 1)
        mask[lo:hi] = True
    if not torch.any(mask):
        return math.nan

    # Weight by shared magnitude so low-energy bins do not dominate.
    weights = torch.minimum(X.abs(), Y.abs())[:, mask]
    vals = phase_diff[:, mask]
    num = torch.sum(vals * weights).item()
    den = torch.sum(weights).item() + 1e-8
    return float(num / den)


def _artifact_score(ref_audio: np.ndarray, test_audio: np.ndarray, chunk_samples: int) -> float:
    loud_std = _loudness_delta_std_db(ref_audio, test_audio)
    click = _boundary_click_score(test_audio, chunk_samples=chunk_samples)
    coh = _phase_coherence(ref_audio, test_audio)
    if math.isnan(loud_std) or math.isnan(click) or math.isnan(coh):
        return math.nan
    return float(loud_std + click + 10.0 * (1.0 - coh))


def _plot_waveforms(
    output_path: Path,
    sample_rate: int,
    istft: np.ndarray,
    pt_vocos: np.ndarray,
    streaming_pt_vocos: np.ndarray,
    tf_vocos: np.ndarray,
    streaming_tf_vocos: np.ndarray,
) -> None:
    n = min(len(istft), len(pt_vocos), len(streaming_pt_vocos), len(tf_vocos), len(streaming_tf_vocos))
    if n <= 0:
        return
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(t, istft[:n], color="#1f77b4", linewidth=0.8)
    axes[0].set_title("iSTFT")
    axes[0].set_ylabel("Amp")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, pt_vocos[:n], color="#2ca02c", linewidth=0.8)
    axes[1].set_title("PyTorch Vocos")
    axes[1].set_ylabel("Amp")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, streaming_pt_vocos[:n], color="#d62728", linewidth=0.8)
    axes[2].set_title("Streaming PT Vocos")
    axes[2].set_ylabel("Amp")
    axes[2].grid(alpha=0.25)

    axes[3].plot(t, tf_vocos[:n], color="#9467bd", linewidth=0.8)
    axes[3].set_title("TensorFlow Vocos")
    axes[3].set_ylabel("Amp")
    axes[3].grid(alpha=0.25)

    axes[4].plot(t, streaming_tf_vocos[:n], color="#8c564b", linewidth=0.8)
    axes[4].set_title("Streaming TF Vocos")
    axes[4].set_ylabel("Amp")
    axes[4].set_xlabel("Time (s)")
    axes[4].grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_model(
    base_config: Dict,
    decoder_type: str,
    kokoro_checkpoint: Path,
    vocos_checkpoint: Path,
    device: torch.device,
) -> KModel:
    cfg = copy.deepcopy(base_config)
    if decoder_type == "streaming_pt_vocos":
        cfg["decoder_type"] = "pt_vocos"
        cfg.setdefault("vocos", {})
        cfg["vocos"]["streaming"] = True
    elif decoder_type == "streaming_tf_vocos":
        cfg["decoder_type"] = "tf_vocos"
        cfg.setdefault("vocos", {})
        cfg["vocos"]["streaming"] = True
    else:
        cfg["decoder_type"] = decoder_type
    if decoder_type in {"pt_vocos", "streaming_pt_vocos", "tf_vocos", "streaming_tf_vocos"}:
        cfg.setdefault("vocos", {})
        cfg["vocos"]["checkpoint_path"] = str(vocos_checkpoint)

    model = KModel(config=cfg, model=str(kokoro_checkpoint), repo_id="hexgrad/Kokoro-82M").eval()
    if decoder_type != "tf_vocos":
        model = model.to(device)
    return model


def _streaming_decode_to_numpy(
    decoder,
    asr: torch.Tensor,
    f0: torch.Tensor,
    noise: torch.Tensor,
    style: torch.Tensor,
    expected_samples: int,
) -> np.ndarray:
    decoder.reset()
    chunks = list(decoder.streaming_decode(asr, f0, noise, style, is_last=True))
    if not chunks:
        audio_t = decoder.decode_caches()
    else:
        audio_t = torch.cat(chunks, dim=-1)
    return _to_numpy_audio(audio_t, expected_samples=expected_samples)


def main() -> None:
    args = parse_args()
    if not args.config_json.exists():
        raise FileNotFoundError(f"Missing config: {args.config_json}")
    if not args.kokoro_checkpoint.exists():
        raise FileNotFoundError(f"Missing Kokoro checkpoint: {args.kokoro_checkpoint}")
    if not args.vocos_checkpoint.exists():
        raise FileNotFoundError(f"Missing Vocos checkpoint: {args.vocos_checkpoint}")

    data_root = args.data_root.resolve()
    val_filelist = args.val_filelist or (data_root / "filelists" / "vocos.val.txt")
    if not val_filelist.exists():
        raise FileNotFoundError(f"Missing val filelist: {val_filelist}")

    wav_paths = _load_lines(val_filelist)
    if not wav_paths:
        raise RuntimeError(f"No validation entries in {val_filelist}")
    wav_paths = wav_paths[: max(1, int(args.num_entries))]
    pair_paths = [_derive_pair_path(data_root, wav) for wav in wav_paths]
    missing = [p for p in pair_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing pair files for {len(missing)} entries. Example: {missing[0]}")

    base_config = json.loads(args.config_json.read_text(encoding="utf-8"))
    device = _resolve_device(args.device)
    logger.info(f"Running decoder comparison on {len(pair_paths)} entries | device={device}")

    decoders = {}
    for decoder_type in ("istft", "pt_vocos", "streaming_pt_vocos", "tf_vocos", "streaming_tf_vocos"):
        km = _build_model(
            base_config=base_config,
            decoder_type=decoder_type,
            kokoro_checkpoint=args.kokoro_checkpoint,
            vocos_checkpoint=args.vocos_checkpoint,
            device=device,
        )
        if decoder_type == "pt_vocos" and not isinstance(km.decoder, PTVocosDecoder):
            raise RuntimeError("decoder_type='pt_vocos' did not build a PTVocosDecoder")
        if decoder_type == "streaming_pt_vocos" and not isinstance(km.decoder, StreamingPTVocosDecoder):
            raise RuntimeError("decoder_type='streaming_pt_vocos' did not build a StreamingPTVocosDecoder")
        if decoder_type == "tf_vocos" and not isinstance(km.decoder, TFVocosDecoder):
            raise RuntimeError("decoder_type='tf_vocos' did not build a TFVocosDecoder")
        if decoder_type == "streaming_tf_vocos" and not isinstance(km.decoder, StreamingTFVocosDecoder):
            raise RuntimeError("decoder_type='streaming_tf_vocos' did not build a StreamingTFVocosDecoder")
        decoders[decoder_type] = km.decoder.eval()
        del km
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mfcc_tf = torchaudio.transforms.MFCC(
        sample_rate=int(args.sample_rate),
        n_mfcc=40,
        melkwargs={"n_fft": 1200, "hop_length": 300, "n_mels": 80, "center": True},
    )
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(args.sample_rate),
        n_fft=1200,
        hop_length=300,
        n_mels=80,
        center=True,
    )

    rows: List[Dict[str, object]] = []
    output_dir = args.output_dir.resolve()
    (output_dir / "audio").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    stream_pt_chunk_samples = int(decoders["streaming_pt_vocos"].chunk_size * decoders["streaming_pt_vocos"].hop_length)
    stream_tf_chunk_samples = int(decoders["streaming_tf_vocos"].chunk_size * decoders["streaming_tf_vocos"].hop_length)

    for i, (wav_path, pair_path) in enumerate(zip(wav_paths, pair_paths), start=1):
        pair = torch.load(pair_path, map_location="cpu", weights_only=False)
        asr = pair["asr"].float().unsqueeze(0).to(device)
        f0 = pair["f0"].float().unsqueeze(0).to(device)
        noise = pair["noise"].float().unsqueeze(0).to(device)
        style = pair["style"].float().unsqueeze(0).to(device)
        expected_samples = int(f0.shape[-1]) * 300

        with torch.inference_mode():
            istft_audio = _to_numpy_audio(decoders["istft"](asr, f0, noise, style), expected_samples=expected_samples)
            pt_audio = _to_numpy_audio(decoders["pt_vocos"](asr, f0, noise, style), expected_samples=expected_samples)
            stream_pt_audio = _streaming_decode_to_numpy(
                decoders["streaming_pt_vocos"],
                asr=asr,
                f0=f0,
                noise=noise,
                style=style,
                expected_samples=expected_samples,
            )
            tf_audio = _to_numpy_audio(decoders["tf_vocos"](asr, f0, noise, style), expected_samples=expected_samples)
            stream_tf_audio = _streaming_decode_to_numpy(
                decoders["streaming_tf_vocos"],
                asr=asr,
                f0=f0,
                noise=noise,
                style=style,
                expected_samples=expected_samples,
            )

        stem = f"{i:03d}_{wav_path.stem}"
        _save_wav_16bit(output_dir / "audio" / "istft" / f"{stem}.wav", istft_audio, int(args.sample_rate))
        _save_wav_16bit(output_dir / "audio" / "pt_vocos" / f"{stem}.wav", pt_audio, int(args.sample_rate))
        _save_wav_16bit(output_dir / "audio" / "streaming_pt_vocos" / f"{stem}.wav", stream_pt_audio, int(args.sample_rate))
        _save_wav_16bit(output_dir / "audio" / "tf_vocos" / f"{stem}.wav", tf_audio, int(args.sample_rate))
        _save_wav_16bit(output_dir / "audio" / "streaming_tf_vocos" / f"{stem}.wav", stream_tf_audio, int(args.sample_rate))
        _plot_waveforms(
            output_dir / "plots" / f"{stem}.png",
            int(args.sample_rate),
            istft_audio,
            pt_audio,
            stream_pt_audio,
            tf_audio,
            stream_tf_audio,
        )

        mfcc_istft = _embed_mfcc(istft_audio, mfcc_tf)
        mfcc_pt = _embed_mfcc(pt_audio, mfcc_tf)
        mfcc_stream_pt = _embed_mfcc(stream_pt_audio, mfcc_tf)
        mfcc_tf_emb = _embed_mfcc(tf_audio, mfcc_tf)
        mfcc_stream_tf = _embed_mfcc(stream_tf_audio, mfcc_tf)
        mel_istft = _embed_logmel(istft_audio, mel_tf)
        mel_pt = _embed_logmel(pt_audio, mel_tf)
        mel_stream_pt = _embed_logmel(stream_pt_audio, mel_tf)
        mel_tf_emb = _embed_logmel(tf_audio, mel_tf)
        mel_stream_tf = _embed_logmel(stream_tf_audio, mel_tf)

        row = {
            "index": i,
            "wav_path": str(wav_path),
            "pair_path": str(pair_path),
            "mfcc_cos_pt_vs_istft": _cosine_similarity(mfcc_pt, mfcc_istft),
            "mfcc_cos_streaming_pt_vs_istft": _cosine_similarity(mfcc_stream_pt, mfcc_istft),
            "mfcc_cos_tf_vs_istft": _cosine_similarity(mfcc_tf_emb, mfcc_istft),
            "mfcc_cos_streaming_tf_vs_istft": _cosine_similarity(mfcc_stream_tf, mfcc_istft),
            "logmel_cos_pt_vs_istft": _cosine_similarity(mel_pt, mel_istft),
            "logmel_cos_streaming_pt_vs_istft": _cosine_similarity(mel_stream_pt, mel_istft),
            "logmel_cos_tf_vs_istft": _cosine_similarity(mel_tf_emb, mel_istft),
            "logmel_cos_streaming_tf_vs_istft": _cosine_similarity(mel_stream_tf, mel_istft),
            "streaming_pt_loudness_delta_std_db": _loudness_delta_std_db(istft_audio, stream_pt_audio),
            "streaming_tf_loudness_delta_std_db": _loudness_delta_std_db(istft_audio, stream_tf_audio),
            "streaming_pt_boundary_click_score": _boundary_click_score(stream_pt_audio, chunk_samples=stream_pt_chunk_samples),
            "streaming_tf_boundary_click_score": _boundary_click_score(stream_tf_audio, chunk_samples=stream_tf_chunk_samples),
            "streaming_pt_phase_coherence_vs_istft": _phase_coherence(istft_audio, stream_pt_audio),
            "streaming_tf_phase_coherence_vs_istft": _phase_coherence(istft_audio, stream_tf_audio),
            "streaming_pt_boundary_phase_mismatch_mae_rad": _boundary_phase_mismatch_mae_rad(
                istft_audio, stream_pt_audio, chunk_samples=stream_pt_chunk_samples
            ),
            "streaming_tf_boundary_phase_mismatch_mae_rad": _boundary_phase_mismatch_mae_rad(
                istft_audio, stream_tf_audio, chunk_samples=stream_tf_chunk_samples
            ),
            "streaming_pt_artifact_score": _artifact_score(
                istft_audio,
                stream_pt_audio,
                chunk_samples=stream_pt_chunk_samples,
            ),
            "streaming_tf_artifact_score": _artifact_score(
                istft_audio,
                stream_tf_audio,
                chunk_samples=stream_tf_chunk_samples,
            ),
        }
        rows.append(row)
        logger.info(
            f"[{i}/{len(pair_paths)}] {wav_path.name} "
            f"mfcc(pt/istft)={row['mfcc_cos_pt_vs_istft']:.4f} "
            f"mfcc(stream-pt/istft)={row['mfcc_cos_streaming_pt_vs_istft']:.4f} "
            f"mfcc(tf/istft)={row['mfcc_cos_tf_vs_istft']:.4f} "
            f"mfcc(stream-tf/istft)={row['mfcc_cos_streaming_tf_vs_istft']:.4f}"
        )

    metric_keys = [
        "mfcc_cos_pt_vs_istft",
        "mfcc_cos_streaming_pt_vs_istft",
        "mfcc_cos_tf_vs_istft",
        "mfcc_cos_streaming_tf_vs_istft",
        "logmel_cos_pt_vs_istft",
        "logmel_cos_streaming_pt_vs_istft",
        "logmel_cos_tf_vs_istft",
        "logmel_cos_streaming_tf_vs_istft",
        "streaming_pt_loudness_delta_std_db",
        "streaming_tf_loudness_delta_std_db",
        "streaming_pt_boundary_click_score",
        "streaming_tf_boundary_click_score",
        "streaming_pt_phase_coherence_vs_istft",
        "streaming_tf_phase_coherence_vs_istft",
        "streaming_pt_boundary_phase_mismatch_mae_rad",
        "streaming_tf_boundary_phase_mismatch_mae_rad",
        "streaming_pt_artifact_score",
        "streaming_tf_artifact_score",
    ]
    summary = {"num_entries": len(rows)}
    for key in metric_keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.nanmean(vals))
        summary[f"{key}_std"] = float(np.nanstd(vals))

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(f"Wrote metrics: {csv_path}")
    logger.info(f"Wrote summary: {json_path}")
    logger.info(
        "Aggregate similarity | "
        f"MFCC pt-vs-istft={summary['mfcc_cos_pt_vs_istft_mean']:.4f} "
        f"MFCC stream-pt-vs-istft={summary['mfcc_cos_streaming_pt_vs_istft_mean']:.4f} "
        f"MFCC tf-vs-istft={summary['mfcc_cos_tf_vs_istft_mean']:.4f} "
        f"MFCC stream-tf-vs-istft={summary['mfcc_cos_streaming_tf_vs_istft_mean']:.4f} "
        f"logmel pt-vs-istft={summary['logmel_cos_pt_vs_istft_mean']:.4f} "
        f"logmel stream-pt-vs-istft={summary['logmel_cos_streaming_pt_vs_istft_mean']:.4f} "
        f"logmel tf-vs-istft={summary['logmel_cos_tf_vs_istft_mean']:.4f} "
        f"logmel stream-tf-vs-istft={summary['logmel_cos_streaming_tf_vs_istft_mean']:.4f} "
        f"| artifact(stream-pt)={summary['streaming_pt_artifact_score_mean']:.4f} "
        f"artifact(stream-tf)={summary['streaming_tf_artifact_score_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
