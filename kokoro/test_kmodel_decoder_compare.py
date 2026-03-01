"""Integration test for KModel decoder variants on vocoder validation pairs.

This utility:
1) Verifies KModel decoder selection (`istft`, `pt_vocos`, `tf_vocos`).
2) Generates audio with all three decoders on N validation entries (default: 20).
3) Compares Vocos outputs to iSTFT using MFCC/log-mel cosine similarity proxies.
4) Writes WAV files, per-entry waveform plots, and aggregate metric reports.

Primary usage:
`uv run kokoro-test-kmodel-decoders --config-json /path/config.json --kokoro-checkpoint /path/kokoro.pth --vocos-checkpoint /path/vocos.pt --data-root /export/eingerman/audio/vocoder --num-entries 20 --output-dir output/decoder_compare`

Alternative module usage:
uv run python -m kokoro.test_kmodel_decoder_compare --config-json ../Kokoro/checkpoints/config.json --kokoro-checkpoint ../Kokoro/checkpoints/kokoro-v1_0.pth --vocos-checkpoint output/checkpoints/last.pt --data-root inputs/ --num-entries 20


Outputs:
- `audio/istft/*.wav`, `audio/pt_vocos/*.wav`, `audio/tf_vocos/*.wav`
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
from .vocos_decoder import PTVocosDecoder, TFVocosDecoder


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


def _plot_waveforms(output_path: Path, sample_rate: int, istft: np.ndarray, pt_vocos: np.ndarray, tf_vocos: np.ndarray) -> None:
    n = min(len(istft), len(pt_vocos), len(tf_vocos))
    if n <= 0:
        return
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, istft[:n], color="#1f77b4", linewidth=0.8)
    axes[0].set_title("iSTFT")
    axes[0].set_ylabel("Amp")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, pt_vocos[:n], color="#2ca02c", linewidth=0.8)
    axes[1].set_title("PyTorch Vocos")
    axes[1].set_ylabel("Amp")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, tf_vocos[:n], color="#d62728", linewidth=0.8)
    axes[2].set_title("TensorFlow Vocos")
    axes[2].set_ylabel("Amp")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.25)

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
    cfg["decoder_type"] = decoder_type
    if decoder_type in {"pt_vocos", "tf_vocos"}:
        cfg.setdefault("vocos", {})
        cfg["vocos"]["checkpoint_path"] = str(vocos_checkpoint)

    model = KModel(config=cfg, model=str(kokoro_checkpoint), repo_id="hexgrad/Kokoro-82M").eval()
    if decoder_type != "tf_vocos":
        model = model.to(device)
    return model


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
    for decoder_type in ("istft", "pt_vocos", "tf_vocos"):
        km = _build_model(
            base_config=base_config,
            decoder_type=decoder_type,
            kokoro_checkpoint=args.kokoro_checkpoint,
            vocos_checkpoint=args.vocos_checkpoint,
            device=device,
        )
        if decoder_type == "pt_vocos" and not isinstance(km.decoder, PTVocosDecoder):
            raise RuntimeError("decoder_type='pt_vocos' did not build a PTVocosDecoder")
        if decoder_type == "tf_vocos" and not isinstance(km.decoder, TFVocosDecoder):
            raise RuntimeError("decoder_type='tf_vocos' did not build a TFVocosDecoder")
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
            tf_audio = _to_numpy_audio(decoders["tf_vocos"](asr, f0, noise, style), expected_samples=expected_samples)

        stem = f"{i:03d}_{wav_path.stem}"
        _save_wav_16bit(output_dir / "audio" / "istft" / f"{stem}.wav", istft_audio, int(args.sample_rate))
        _save_wav_16bit(output_dir / "audio" / "pt_vocos" / f"{stem}.wav", pt_audio, int(args.sample_rate))
        _save_wav_16bit(output_dir / "audio" / "tf_vocos" / f"{stem}.wav", tf_audio, int(args.sample_rate))
        _plot_waveforms(output_dir / "plots" / f"{stem}.png", int(args.sample_rate), istft_audio, pt_audio, tf_audio)

        mfcc_istft = _embed_mfcc(istft_audio, mfcc_tf)
        mfcc_pt = _embed_mfcc(pt_audio, mfcc_tf)
        mfcc_tf_emb = _embed_mfcc(tf_audio, mfcc_tf)
        mel_istft = _embed_logmel(istft_audio, mel_tf)
        mel_pt = _embed_logmel(pt_audio, mel_tf)
        mel_tf_emb = _embed_logmel(tf_audio, mel_tf)

        row = {
            "index": i,
            "wav_path": str(wav_path),
            "pair_path": str(pair_path),
            "mfcc_cos_pt_vs_istft": _cosine_similarity(mfcc_pt, mfcc_istft),
            "mfcc_cos_tf_vs_istft": _cosine_similarity(mfcc_tf_emb, mfcc_istft),
            "logmel_cos_pt_vs_istft": _cosine_similarity(mel_pt, mel_istft),
            "logmel_cos_tf_vs_istft": _cosine_similarity(mel_tf_emb, mel_istft),
        }
        rows.append(row)
        logger.info(
            f"[{i}/{len(pair_paths)}] {wav_path.name} "
            f"mfcc(pt/istft)={row['mfcc_cos_pt_vs_istft']:.4f} "
            f"mfcc(tf/istft)={row['mfcc_cos_tf_vs_istft']:.4f}"
        )

    metric_keys = [
        "mfcc_cos_pt_vs_istft",
        "mfcc_cos_tf_vs_istft",
        "logmel_cos_pt_vs_istft",
        "logmel_cos_tf_vs_istft",
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
        f"MFCC tf-vs-istft={summary['mfcc_cos_tf_vs_istft_mean']:.4f} "
        f"logmel pt-vs-istft={summary['logmel_cos_pt_vs_istft_mean']:.4f} "
        f"logmel tf-vs-istft={summary['logmel_cos_tf_vs_istft_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
