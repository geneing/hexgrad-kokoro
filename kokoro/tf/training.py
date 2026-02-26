"""TensorFlow Vocos decoder training entrypoint.

This module contains the full TensorFlow training loop for the Vocos decoder
distillation task used in this repository. It expects paired Kokoro vocoder
inputs (`asr`, `f0`, `noise`, `style`) and waveform targets, and trains the
generator/discriminator stack with losses aligned to the PyTorch flow.

Scope:
- Build and train TF generator + TF discriminators.
- Handle mixed precision policy and adaptive frame-cap backoff for OOM control.
- Log training/validation metrics and sample audio for model monitoring.
- Optionally initialize TF generator from a PyTorch checkpoint.

This file is intentionally focused on training orchestration; model components
live in `kokoro.tf.model`, and checkpoint conversion helpers live in
`kokoro.tf.checkpoint_utils`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf
import torch
from loguru import logger

try:
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    plt = None

from .model import (
    MultiPeriodDiscriminatorTF,
    MultiResolutionComplexSTFTDiscriminatorTF,
    MultiResolutionDiscriminatorTF,
    MultiResolutionGroupDelayLossTF,
    MultiResolutionSTFTLossTF,
    PairedVocosGeneratorTF,
    align_audio,
    compute_dynamic_weights,
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_hinge_loss,
)
from .checkpoint_utils import load_pytorch_checkpoint_into_tf_generator


@dataclass
class TrainItem:
    wav_path: Path
    pair_path: Path
    frames: int


@dataclass
class PreviewSample:
    tag: str
    features: np.ndarray
    audio: np.ndarray


@dataclass
class AdaptiveBatchState:
    frame_cap: int
    min_frame_cap: int
    hop_length: int

    def apply_oom_backoff(self, ratio: float = 0.8) -> int:
        new_cap = max(self.min_frame_cap, int(self.frame_cap * ratio))
        changed = int(new_cap != self.frame_cap)
        self.frame_cap = int(new_cap)
        return changed


class NonFiniteLossError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train TensorFlow Vocos on saved Kokoro vocoder pairs")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--train-filelist", type=Path, default=None)
    parser.add_argument("--val-filelist", type=Path, default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)

    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=300)
    parser.add_argument("--n-fft", type=int, default=1200)

    parser.add_argument("--model-input-channels", type=int, default=192)
    parser.add_argument("--backbone-dim", type=int, default=384)
    parser.add_argument("--backbone-intermediate-dim", type=int, default=1152)
    parser.add_argument("--backbone-layers", type=int, default=8)

    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frame-budget", type=int, default=10240)
    parser.add_argument("--max-batch-size", type=int, default=12)
    parser.add_argument("--frame-cap", type=int, default=520)
    parser.add_argument("--min-frame-cap", type=int, default=192)
    parser.add_argument("--oom-backoff", type=float, default=0.8)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-steps", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=1000)

    parser.add_argument("--gen-lr", type=float, default=3e-4)
    parser.add_argument("--disc-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)

    parser.add_argument("--pretrain-mel-steps", type=int, default=5000)
    parser.add_argument("--adv-ramp-ratio", type=float, default=0.02)
    parser.add_argument("--disc-update-interval", type=int, default=2)
    parser.add_argument("--adv-loss-interval", type=int, default=2)

    parser.add_argument("--gan-loss-coeff", type=float, default=1.0)
    parser.add_argument("--fm-loss-coeff", type=float, default=2.0)
    parser.add_argument("--mrstft-loss-coeff", type=float, default=45.0)
    parser.add_argument("--group-delay-loss-coeff", type=float, default=2.0)
    parser.add_argument("--mrstft-final-ratio", type=float, default=0.25)
    parser.add_argument("--mrd-loss-coeff", type=float, default=1.0)
    parser.add_argument("--cstft-disc-loss-coeff", type=float, default=1.0)

    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--train-frame-policy", type=str, default="max", choices=["max", "random"])
    parser.add_argument("--train-frame-min-ratio", type=float, default=0.6)
    parser.add_argument("--fixed-shapes", dest="fixed_shapes", action="store_true")
    parser.add_argument("--no-fixed-shapes", dest="fixed_shapes", action="store_false")
    parser.set_defaults(fixed_shapes=True)

    parser.add_argument("--sample-voices", type=str, default="af_bella,af_nicole,af_heart")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--sample-max-frames", type=int, default=480)

    parser.add_argument("--output-dir", type=Path, default=Path("runs/vocos_kokoro_paired_tf"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--init-from-pytorch-checkpoint",
        type=Path,
        default=None,
        help="Optional PyTorch generator checkpoint to initialize TensorFlow generator weights before training",
    )

    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--synthetic-frames", type=int, default=256)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)


def configure_precision(precision: str) -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if precision == "fp32":
        policy = "float32"
    elif precision == "fp16":
        policy = "mixed_float16"
    elif precision == "bf16":
        policy = "mixed_bfloat16"
    else:
        policy = "mixed_float16" if gpus else "float32"
    tf.keras.mixed_precision.set_global_policy(policy)
    return policy


def derive_pair_path(data_root: Path, wav_path: Path) -> Path:
    wav_path = wav_path.resolve()
    audio_root = (data_root / "audio").resolve()
    pair_root = (data_root / "pairs").resolve()
    rel = wav_path.relative_to(audio_root)
    return pair_root / rel.with_suffix(".pt")


def load_wav_list(path: Path) -> List[Path]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [Path(x).resolve() for x in lines]


def ensure_filelists(
    data_root: Path,
    train_filelist: Path,
    val_filelist: Path,
    seed: int,
    val_ratio: float = 0.02,
) -> tuple[Path, Path]:
    if train_filelist.exists() and val_filelist.exists():
        return train_filelist, val_filelist

    audio_root = data_root / "audio"
    wavs = sorted(p.resolve() for p in audio_root.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No wav files found under {audio_root}")

    rng = random.Random(seed)
    rng.shuffle(wavs)
    val_count = max(1, int(len(wavs) * val_ratio))
    if val_count >= len(wavs):
        val_count = max(1, len(wavs) - 1)

    val_wavs = wavs[:val_count]
    train_wavs = wavs[val_count:]

    train_filelist.parent.mkdir(parents=True, exist_ok=True)
    val_filelist.parent.mkdir(parents=True, exist_ok=True)
    train_filelist.write_text("\n".join(str(p) for p in train_wavs) + "\n", encoding="utf-8")
    val_filelist.write_text("\n".join(str(p) for p in val_wavs) + "\n", encoding="utf-8")
    logger.warning(
        f"Filelists were missing. Auto-generated train/val lists at "
        f"{train_filelist} ({len(train_wavs)} files) and {val_filelist} ({len(val_wavs)} files)"
    )
    return train_filelist, val_filelist


def build_metadata_index(manifest_root: Path, hop_length: int) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    if not manifest_root.exists():
        return idx
    for f in sorted(manifest_root.glob("*.jsonl")):
        with f.open("r", encoding="utf-8") as r:
            for line in r:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                wav_path = str(Path(rec["wav_path"]).resolve())
                frames = int(rec.get("audio_num_samples", 0)) // hop_length
                if frames <= 0:
                    frames = int(rec.get("asr_frames", 0)) * 2
                idx[wav_path] = max(1, frames)
    return idx


def build_items(
    data_root: Path,
    wav_paths: Sequence[Path],
    metadata_index: Dict[str, int],
) -> List[TrainItem]:
    items: List[TrainItem] = []
    for wav in wav_paths:
        pair = derive_pair_path(data_root, wav)
        if not pair.exists():
            logger.warning(f"Missing pair file for wav, skipping: {wav}")
            continue
        frames = metadata_index.get(str(wav.resolve()))
        if frames is None:
            pair_obj = torch.load(pair, map_location="cpu", weights_only=False)
            frames = int(pair_obj["f0"].shape[-1])
        items.append(TrainItem(wav_path=wav, pair_path=pair, frames=max(1, int(frames))))
    return items


def _load_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sr = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        raw = wav_file.readframes(n_frames)

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483647.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sampwidth} for {path}")
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio, sr


def _repeat_pad_1d(x: np.ndarray, target_length: int) -> np.ndarray:
    if len(x) >= target_length:
        return x[:target_length]
    if len(x) == 0:
        return np.zeros(target_length, dtype=np.float32)
    repeat = 1 + target_length // len(x)
    return np.tile(x, repeat)[:target_length]


def _repeat_pad_2d(x: np.ndarray, target_length: int) -> np.ndarray:
    if x.shape[1] >= target_length:
        return x[:, :target_length]
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], target_length), dtype=np.float32)
    repeat = 1 + target_length // x.shape[1]
    return np.tile(x, (1, repeat))[:, :target_length]


def _interp_asr_to_frames(asr: np.ndarray, target_frames: int) -> np.ndarray:
    if asr.shape[-1] == target_frames:
        return asr
    src_x = np.linspace(0.0, 1.0, asr.shape[-1], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    out = np.stack([np.interp(dst_x, src_x, ch).astype(np.float32) for ch in asr], axis=0)
    return out


class PairedBatchLoader:
    def __init__(
        self,
        items: Sequence[TrainItem],
        sample_rate: int,
        hop_length: int,
        frame_cap: int,
        batch_size: int,
        train: bool,
        seed: int,
        frame_budget: int,
        max_batch_size: int,
        train_frame_policy: str = "max",
        train_frame_min_ratio: float = 0.6,
        fixed_shapes: bool = True,
    ):
        self.items = list(items)
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.frame_cap = int(frame_cap)
        self.batch_size = max(1, int(batch_size))
        self.frame_budget = int(frame_budget)
        self.max_batch_size = max(1, int(max_batch_size))
        self.train = bool(train)
        self.train_frame_policy = str(train_frame_policy)
        self.train_frame_min_ratio = float(train_frame_min_ratio)
        self.fixed_shapes = bool(fixed_shapes)
        self.rng = random.Random(seed + (0 if train else 1))
        self.seed = int(seed)
        self.index = 0
        self.order: List[int] = []
        self._carry_indices: List[int] = []
        self._reset_order()

    def _reset_order(self) -> None:
        self.order = list(range(len(self.items)))
        if not self.order:
            return
        if self.train:
            self.rng.shuffle(self.order)
            chunk = max(1, self.max_batch_size * 50)
            sorted_indices: List[int] = []
            for i in range(0, len(self.order), chunk):
                sub = self.order[i : i + chunk]
                sub.sort(key=lambda idx: self.items[idx].frames, reverse=True)
                sorted_indices.extend(sub)
            self.order = sorted_indices
        else:
            self.order.sort(key=lambda idx: self.items[idx].frames, reverse=True)
        self.index = 0

    def _next_index(self) -> int:
        if not self.order:
            raise RuntimeError("No items available")
        if self._carry_indices:
            return self._carry_indices.pop(0)
        if self.index >= len(self.order):
            self._reset_order()
        idx = self.order[self.index % len(self.order)]
        self.index += 1
        return idx

    def _batch_size_from_budget(self, target_frames: int) -> int:
        if self.frame_budget <= 0:
            return self.batch_size
        est = max(1, self.frame_budget // max(1, int(target_frames)))
        return max(1, min(self.max_batch_size, est))

    def _choose_target_frames(self, item_indices: Sequence[int]) -> int:
        if self.fixed_shapes:
            return max(1, int(self.frame_cap))
        cap = max(1, int(self.frame_cap))
        if item_indices:
            cap = min(cap, min(max(1, int(self.items[i].frames)) for i in item_indices))
        if not self.train:
            return cap
        if self.train_frame_policy == "max":
            return cap
        min_cap = max(32, int(cap * max(0.0, min(1.0, self.train_frame_min_ratio))))
        min_cap = min(min_cap, cap)
        return self.rng.randint(min_cap, cap)

    def _select_item_indices(self) -> List[int]:
        if not self.items:
            raise RuntimeError("No items available")

        if self.frame_budget <= 0:
            return [self._next_index() for _ in range(self.batch_size)]

        selected: List[int] = []
        used = 0
        budget_cap = max(1, int(self.frame_cap))
        while len(selected) < self.max_batch_size:
            idx = self._next_index()
            item_frames = min(max(1, int(self.items[idx].frames)), budget_cap)
            if selected and (used + item_frames > self.frame_budget):
                self._carry_indices.insert(0, idx)
                break
            selected.append(idx)
            used += item_frames
            if used >= self.frame_budget:
                break

        if not selected:
            selected = [self._next_index()]
        return selected

    def _load_row(self, item: TrainItem) -> Dict[str, np.ndarray]:
        pair = torch.load(item.pair_path, map_location="cpu", weights_only=False)
        wav, sr = _load_mono_wav(item.wav_path)
        if sr != self.sample_rate:
            raise ValueError(
                f"Expected sample_rate={self.sample_rate}, got {sr} for {item.wav_path}. "
                "Resample your dataset to keep TensorFlow path deterministic."
            )
        return {
            "asr": pair["asr"].float().cpu().numpy().astype(np.float32),
            "f0": pair["f0"].float().cpu().numpy().astype(np.float32),
            "noise": pair["noise"].float().cpu().numpy().astype(np.float32),
            "style": pair["style"].float().cpu().numpy().astype(np.float32),
            "audio": wav.astype(np.float32),
        }

    def next_batch(self) -> Dict[str, np.ndarray]:
        feats: List[np.ndarray] = []
        audio: List[np.ndarray] = []
        item_indices = self._select_item_indices()
        target_frames = self._choose_target_frames(item_indices)
        max_rows = self._batch_size_from_budget(target_frames)
        if len(item_indices) > max_rows:
            overflow = item_indices[max_rows:]
            self._carry_indices = overflow + self._carry_indices
            item_indices = item_indices[:max_rows]
        target_audio_samples = target_frames * self.hop_length
        for idx in item_indices:
            row = self._load_row(self.items[idx])
            total_frames = int(row["f0"].shape[-1])
            if total_frames <= 0:
                continue
            start = 0
            if self.train and total_frames > target_frames:
                start = self.rng.randint(0, total_frames - target_frames)
            end = min(total_frames, start + target_frames)

            asr = _interp_asr_to_frames(row["asr"], total_frames)
            asr_s = _repeat_pad_2d(asr[:, start:end], target_frames)
            f0_s = _repeat_pad_1d(row["f0"][start:end], target_frames)[None, :]
            noise_s = _repeat_pad_1d(row["noise"][start:end], target_frames)[None, :]
            style = row["style"][:, None]
            style_s = np.repeat(style, target_frames, axis=1)
            feat = np.concatenate([asr_s, f0_s, noise_s, style_s], axis=0)

            start_sample = start * self.hop_length
            end_sample = start_sample + target_audio_samples
            wav_s = _repeat_pad_1d(row["audio"][start_sample:end_sample], target_audio_samples)
            feats.append(feat)
            audio.append(wav_s)
        if not feats:
            raise RuntimeError("Failed to construct a non-empty batch")
        return {
            "features": np.stack(feats, axis=0).astype(np.float32),
            "audio": np.stack(audio, axis=0).astype(np.float32),
            "target_frames": target_frames,
        }


class SyntheticBatchLoader:
    def __init__(
        self,
        batch_size: int,
        frame_cap: int,
        hop_length: int,
        in_channels: int,
        seed: int,
        frame_budget: int,
        max_batch_size: int,
        train: bool,
        train_frame_policy: str = "max",
        train_frame_min_ratio: float = 0.6,
        fixed_shapes: bool = True,
    ):
        self.batch_size = max(1, int(batch_size))
        self.frame_cap = int(frame_cap)
        self.hop_length = int(hop_length)
        self.in_channels = int(in_channels)
        self.frame_budget = int(frame_budget)
        self.max_batch_size = max(1, int(max_batch_size))
        self.train = bool(train)
        self.train_frame_policy = str(train_frame_policy)
        self.train_frame_min_ratio = float(train_frame_min_ratio)
        self.fixed_shapes = bool(fixed_shapes)
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

    def _batch_size_from_budget(self, target_frames: int) -> int:
        if self.frame_budget <= 0:
            return self.batch_size
        est = max(1, self.frame_budget // max(1, int(target_frames)))
        return max(1, min(self.max_batch_size, est))

    def _choose_target_frames(self) -> int:
        cap = max(1, int(self.frame_cap))
        if self.fixed_shapes or not self.train or self.train_frame_policy == "max":
            return cap
        min_cap = max(32, int(cap * max(0.0, min(1.0, self.train_frame_min_ratio))))
        min_cap = min(min_cap, cap)
        return self.py_rng.randint(min_cap, cap)

    def next_batch(self) -> Dict[str, np.ndarray]:
        target_frames = self._choose_target_frames()
        bsz = self._batch_size_from_budget(target_frames)
        features = self.rng.standard_normal((bsz, self.in_channels, target_frames), dtype=np.float32)
        audio = self.rng.standard_normal((bsz, target_frames * self.hop_length), dtype=np.float32) * 0.05
        return {"features": features.astype(np.float32), "audio": audio.astype(np.float32), "target_frames": target_frames}


def maybe_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "resource exhausted" in msg or "oom" in msg


def assert_finite(tag: str, x: tf.Tensor) -> None:
    if not bool(tf.reduce_all(tf.math.is_finite(tf.cast(x, tf.float32))).numpy()):
        raise NonFiniteLossError(f"Non-finite detected at {tag}")


def crop_batch_for_retry(batch: Dict[str, np.ndarray], new_frame_cap: int, hop_length: int, seed: int) -> Dict[str, np.ndarray]:
    feats = np.asarray(batch["features"], dtype=np.float32)
    audio = np.asarray(batch["audio"], dtype=np.float32)
    bsz, _, current_frames = feats.shape
    if new_frame_cap >= current_frames:
        return batch

    rng = random.Random(seed)
    new_audio_len = int(new_frame_cap) * int(hop_length)
    out_feats: list[np.ndarray] = []
    out_audio: list[np.ndarray] = []
    for i in range(bsz):
        start = rng.randint(0, current_frames - int(new_frame_cap))
        out_feats.append(feats[i, :, start : start + int(new_frame_cap)])
        s = start * int(hop_length)
        out_audio.append(audio[i, s : s + new_audio_len])
    return {
        "features": np.stack(out_feats, axis=0).astype(np.float32),
        "audio": np.stack(out_audio, axis=0).astype(np.float32),
        "target_frames": int(new_frame_cap),
    }


def voice_from_wav_path(wav_path: str) -> str:
    return Path(wav_path).parent.name


def select_preview_items(
    items: Sequence[TrainItem],
    voices: Sequence[str],
    sample_count: int,
    seed: int,
) -> List[TrainItem]:
    by_voice: Dict[str, List[TrainItem]] = {v: [] for v in voices}
    for item in items:
        voice = item.wav_path.parent.name
        if voice in by_voice:
            by_voice[voice].append(item)

    rng = random.Random(seed)
    selected: List[TrainItem] = []
    voice_order = list(voices)
    while len(selected) < sample_count and voice_order:
        next_voice_order: List[str] = []
        for voice in voice_order:
            pool = by_voice.get(voice, [])
            if not pool:
                continue
            idx = rng.randrange(len(pool))
            selected.append(pool.pop(idx))
            if pool:
                next_voice_order.append(voice)
            if len(selected) >= sample_count:
                break
        voice_order = next_voice_order

    if len(selected) < sample_count:
        fallback = [x for x in items if x not in selected]
        rng.shuffle(fallback)
        selected.extend(fallback[: max(0, sample_count - len(selected))])

    return selected[:sample_count]


def build_preview_cache(
    preview_items: Sequence[TrainItem],
    sample_rate: int,
    hop_length: int,
    max_frames: int,
) -> List[PreviewSample]:
    if not preview_items:
        return []

    out: List[PreviewSample] = []
    for i, item in enumerate(preview_items):
        pair = torch.load(item.pair_path, map_location="cpu", weights_only=False)
        wav, sr = _load_mono_wav(item.wav_path)
        if sr != sample_rate:
            continue
        asr = pair["asr"].float().cpu().numpy().astype(np.float32)
        f0 = pair["f0"].float().cpu().numpy().astype(np.float32)
        noise = pair["noise"].float().cpu().numpy().astype(np.float32)
        style = pair["style"].float().cpu().numpy().astype(np.float32)

        total_frames = int(f0.shape[-1])
        if total_frames <= 0:
            continue
        target_frames = max(1, min(int(max_frames), total_frames))
        asr = _interp_asr_to_frames(asr, total_frames)
        asr_s = _repeat_pad_2d(asr[:, :target_frames], target_frames)
        f0_s = _repeat_pad_1d(f0[:target_frames], target_frames)[None, :]
        noise_s = _repeat_pad_1d(noise[:target_frames], target_frames)[None, :]
        style_s = np.repeat(style[:, None], target_frames, axis=1)
        feat = np.concatenate([asr_s, f0_s, noise_s, style_s], axis=0).astype(np.float32)
        audio = _repeat_pad_1d(wav[: target_frames * hop_length], target_frames * hop_length).astype(np.float32)
        voice = voice_from_wav_path(str(item.wav_path))
        out.append(PreviewSample(tag=f"samples/{i+1:02d}_{voice}", features=feat, audio=audio))
    return out


def log_preview_samples(
    generator: PairedVocosGeneratorTF,
    preview_cache: Sequence[PreviewSample],
    writer: tf.summary.SummaryWriter,
    step: int,
    sample_rate: int,
) -> None:
    if not preview_cache:
        return
    for sample in preview_cache:
        feat = tf.convert_to_tensor(sample.features[None, ...], dtype=tf.float32)
        real = tf.convert_to_tensor(sample.audio[None, ...], dtype=tf.float32)
        pred = generator(feat, training=False)
        pred, real = align_audio(pred, real)
        real_audio = tf.expand_dims(tf.expand_dims(real[0], axis=-1), axis=0)
        pred_audio = tf.expand_dims(tf.expand_dims(pred[0], axis=-1), axis=0)
        with writer.as_default():
            tf.summary.audio(f"{sample.tag}/target", real_audio, sample_rate=sample_rate, step=step)
            tf.summary.audio(f"{sample.tag}/pred", pred_audio, sample_rate=sample_rate, step=step)


def waveform_to_image(target_wav: tf.Tensor, pred_wav: Optional[tf.Tensor] = None, width: int = 1024, height: int = 256) -> tf.Tensor:
    if plt is None:
        return tf.ones([height, width, 3], dtype=tf.float32)
    target = tf.reshape(tf.cast(target_wav, tf.float32), [-1]).numpy()
    pred = None if pred_wav is None else tf.reshape(tf.cast(pred_wav, tf.float32), [-1]).numpy()
    if target.size == 0:
        return tf.ones([height, width, 3], dtype=tf.float32)

    dpi = 100
    fig_w = max(1.0, width / dpi)
    fig_h = max(1.0, height / dpi)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    try:
        ax.plot(target, label="target")
        if pred is not None and pred.size > 0:
            ax.plot(pred, label="pred")
            ax.legend(loc="upper right")
        ax.set_xlim(0, max(1, len(target) - 1))
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgb = rgba[..., :3]
    finally:
        plt.close(fig)
    return tf.cast(rgb, tf.float32) / 255.0


def get_gpu_peak_memory_gb() -> float:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return 0.0
    try:
        mem = tf.config.experimental.get_memory_info("GPU:0")
        peak = float(mem.get("peak", 0.0))
        return peak / float(1024**3)
    except Exception:
        return 0.0


def reset_gpu_peak_memory() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


def run_validation(
    generator: PairedVocosGeneratorTF,
    val_loader,
    mrstft_loss_fn: MultiResolutionSTFTLossTF,
    group_delay_loss_fn: MultiResolutionGroupDelayLossTF,
    writer: tf.summary.SummaryWriter,
    step: int,
    val_steps: int,
    sample_rate: int,
    loss_weights,
    log_waveform_images: bool,
) -> None:
    mrstft_vals: List[float] = []
    gd_vals: List[float] = []
    l1_vals: List[float] = []
    total_vals: List[float] = []

    for i in range(max(1, val_steps)):
        batch = val_loader.next_batch()
        features = tf.convert_to_tensor(batch["features"], dtype=tf.float32)
        real = tf.convert_to_tensor(batch["audio"], dtype=tf.float32)
        pred = generator(features, training=False)
        pred, real = align_audio(pred, real)
        mrstft = mrstft_loss_fn(real, pred)
        gd = group_delay_loss_fn(real, pred)
        l1 = tf.reduce_mean(tf.abs(pred - real))
        total = loss_weights.mrstft * mrstft + loss_weights.group_delay * gd
        mrstft_vals.append(float(mrstft.numpy()))
        gd_vals.append(float(gd.numpy()))
        l1_vals.append(float(l1.numpy()))
        total_vals.append(float(total.numpy()))

        if i == 0:
            real_audio = tf.expand_dims(tf.expand_dims(real[0], axis=-1), axis=0)
            pred_audio = tf.expand_dims(tf.expand_dims(pred[0], axis=-1), axis=0)
            with writer.as_default():
                tf.summary.audio("val/audio_target", real_audio, sample_rate=sample_rate, step=step)
                tf.summary.audio("val/audio_pred", pred_audio, sample_rate=sample_rate, step=step)
                if log_waveform_images:
                    tf.summary.image(
                        "val/waveform_overlay",
                        tf.expand_dims(waveform_to_image(real[0], pred[0]), axis=0),
                        step=step,
                    )

    with writer.as_default():
        tf.summary.scalar("val/generator_total_estimate", float(np.mean(total_vals)), step=step)
        tf.summary.scalar("val/gen_mrstft_raw", float(np.mean(mrstft_vals)), step=step)
        tf.summary.scalar("val/gen_mrstft_weighted", float(loss_weights.mrstft * np.mean(mrstft_vals)), step=step)
        tf.summary.scalar("val/gen_group_delay_raw", float(np.mean(gd_vals)), step=step)
        tf.summary.scalar("val/gen_group_delay_weighted", float(loss_weights.group_delay * np.mean(gd_vals)), step=step)
        tf.summary.scalar("val/l1_wave_loss", float(np.mean(l1_vals)), step=step)


def safe_apply_gradients(optimizer: tf.keras.optimizers.Optimizer, grads, vars_) -> None:
    pairs = [(g, v) for g, v in zip(grads, vars_) if g is not None]
    if pairs:
        optimizer.apply_gradients(pairs)


def main() -> None:
    logger.enable("kokoro.tf.training")
    args = parse_args()
    set_seed(args.seed)
    policy = configure_precision(args.precision)

    out_dir = args.output_dir.resolve()
    tb_dir = out_dir / "tensorboard"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(str(tb_dir))
    with writer.as_default():
        tf.summary.text("run/args", json.dumps(vars(args), indent=2, default=str), step=0)
    log_waveform_images = plt is not None
    if not log_waveform_images:
        logger.warning("matplotlib is not available; waveform image logging is disabled.")

    in_channels = 512 + 1 + 1 + 128
    generator = PairedVocosGeneratorTF(
        in_channels=in_channels,
        model_input_channels=args.model_input_channels,
        backbone_dim=args.backbone_dim,
        backbone_intermediate_dim=args.backbone_intermediate_dim,
        backbone_layers=args.backbone_layers,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        padding="same",
    )
    mpd = MultiPeriodDiscriminatorTF()
    mrd = MultiResolutionDiscriminatorTF()
    cstft_disc = MultiResolutionComplexSTFTDiscriminatorTF()

    dummy_feat = tf.zeros([1, in_channels, max(16, args.frame_cap)], dtype=tf.float32)
    dummy_audio = generator(dummy_feat, training=False)
    if args.init_from_pytorch_checkpoint is not None:
        if not args.init_from_pytorch_checkpoint.exists():
            raise FileNotFoundError(f"PyTorch init checkpoint not found: {args.init_from_pytorch_checkpoint}")
        init_report = load_pytorch_checkpoint_into_tf_generator(generator, args.init_from_pytorch_checkpoint)
        logger.info(
            "Initialized TensorFlow generator from PyTorch checkpoint "
            f"{args.init_from_pytorch_checkpoint} "
            f"(loaded={init_report['num_loaded_keys']}, ignored={init_report['ignored_keys']})"
        )
        if init_report.get("metadata"):
            logger.info(f"PyTorch init checkpoint metadata: {init_report['metadata']}")
    mpd(dummy_audio, dummy_audio, training=False)
    mrd(dummy_audio, dummy_audio, training=False)
    cstft_disc(dummy_audio, dummy_audio, training=False)

    gen_lr_sched = tf.keras.optimizers.schedules.CosineDecay(args.gen_lr, decay_steps=max(1, args.max_steps))
    disc_lr_sched = tf.keras.optimizers.schedules.CosineDecay(args.disc_lr, decay_steps=max(1, args.max_steps))
    gen_opt = tf.keras.optimizers.AdamW(
        learning_rate=gen_lr_sched,
        beta_1=0.8,
        beta_2=0.9,
        weight_decay=args.weight_decay,
    )
    disc_opt = tf.keras.optimizers.AdamW(
        learning_rate=disc_lr_sched,
        beta_1=0.8,
        beta_2=0.9,
        weight_decay=args.weight_decay,
    )

    mrstft_loss_fn = MultiResolutionSTFTLossTF(sample_rate=args.sample_rate)
    group_delay_loss_fn = MultiResolutionGroupDelayLossTF()

    state = AdaptiveBatchState(
        frame_cap=int(args.frame_cap),
        min_frame_cap=int(args.min_frame_cap),
        hop_length=int(args.hop_length),
    )

    preview_cache: List[PreviewSample] = []
    if args.synthetic_data:
        train_loader = SyntheticBatchLoader(
            batch_size=args.batch_size,
            frame_cap=args.synthetic_frames,
            hop_length=args.hop_length,
            in_channels=in_channels,
            seed=args.seed,
            frame_budget=args.frame_budget,
            max_batch_size=args.max_batch_size,
            train=True,
            train_frame_policy=args.train_frame_policy,
            train_frame_min_ratio=args.train_frame_min_ratio,
            fixed_shapes=args.fixed_shapes,
        )
        val_loader = SyntheticBatchLoader(
            batch_size=max(1, min(2, args.batch_size)),
            frame_cap=args.synthetic_frames,
            hop_length=args.hop_length,
            in_channels=in_channels,
            seed=args.seed + 1,
            frame_budget=args.frame_budget,
            max_batch_size=max(1, args.max_batch_size // 2),
            train=False,
            train_frame_policy=args.train_frame_policy,
            train_frame_min_ratio=args.train_frame_min_ratio,
            fixed_shapes=args.fixed_shapes,
        )
        state.frame_cap = int(args.synthetic_frames)
        logger.warning("Using synthetic random data; this is a smoke-test mode only.")
    else:
        data_root = args.data_root.resolve()
        train_filelist = args.train_filelist or data_root / "filelists" / "vocos.train.txt"
        val_filelist = args.val_filelist or data_root / "filelists" / "vocos.val.txt"
        manifest_root = args.manifest_root or data_root / "manifests"
        train_filelist, val_filelist = ensure_filelists(
            data_root=data_root,
            train_filelist=train_filelist,
            val_filelist=val_filelist,
            seed=args.seed,
        )
        metadata_index = build_metadata_index(manifest_root, args.hop_length)
        train_items = build_items(data_root, load_wav_list(train_filelist), metadata_index)
        val_items = build_items(data_root, load_wav_list(val_filelist), metadata_index)
        if not train_items:
            raise RuntimeError("No train items found")
        if not val_items:
            raise RuntimeError("No val items found")
        logger.info(f"Train items: {len(train_items)} | Val items: {len(val_items)}")
        sample_voices = [x.strip() for x in args.sample_voices.split(",") if x.strip()]
        preview_items = select_preview_items(
            items=val_items,
            voices=sample_voices,
            sample_count=max(1, args.sample_count),
            seed=args.seed,
        )
        preview_voices_found = sorted({item.wav_path.parent.name for item in preview_items})
        logger.info(
            f"Preview samples: {len(preview_items)} items from voices={sample_voices} "
            f"(requested={args.sample_count})"
        )
        if len(preview_items) < args.sample_count or not all(v in preview_voices_found for v in sample_voices):
            logger.warning(
                f"Could not fully satisfy requested preview voice mix. Found voices in preview: {preview_voices_found}"
            )
        preview_cache = build_preview_cache(
            preview_items=preview_items,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            max_frames=args.sample_max_frames,
        )
        logger.info(f"Prepared {len(preview_cache)} cached preview samples for TensorBoard audio logging")
        train_loader = PairedBatchLoader(
            items=train_items,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            frame_cap=state.frame_cap,
            batch_size=args.batch_size,
            train=True,
            seed=args.seed,
            frame_budget=args.frame_budget,
            max_batch_size=args.max_batch_size,
            train_frame_policy=args.train_frame_policy,
            train_frame_min_ratio=args.train_frame_min_ratio,
            fixed_shapes=args.fixed_shapes,
        )
        val_loader = PairedBatchLoader(
            items=val_items,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            frame_cap=state.frame_cap,
            batch_size=max(1, args.batch_size // 2),
            train=False,
            seed=args.seed,
            frame_budget=args.frame_budget,
            max_batch_size=max(1, args.max_batch_size // 2),
            train_frame_policy=args.train_frame_policy,
            train_frame_min_ratio=args.train_frame_min_ratio,
            fixed_shapes=args.fixed_shapes,
        )

    ckpt_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="step")
    ckpt_frame_cap = tf.Variable(int(state.frame_cap), dtype=tf.int64, trainable=False, name="frame_cap")
    ckpt = tf.train.Checkpoint(
        step=ckpt_step,
        frame_cap=ckpt_frame_cap,
        generator=generator,
        mpd=mpd,
        mrd=mrd,
        cstft_disc=cstft_disc,
        gen_opt=gen_opt,
        disc_opt=disc_opt,
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=str(ckpt_dir), max_to_keep=5)

    resume_path: Optional[str] = args.resume if args.resume else ckpt_manager.latest_checkpoint
    if resume_path:
        ckpt.restore(resume_path)
        state.frame_cap = int(ckpt_frame_cap.numpy())
        if isinstance(train_loader, (PairedBatchLoader, SyntheticBatchLoader)):
            train_loader.frame_cap = int(state.frame_cap)
        if isinstance(val_loader, (PairedBatchLoader, SyntheticBatchLoader)):
            val_loader.frame_cap = int(state.frame_cap)
        logger.info(
            f"Resumed from checkpoint: {resume_path} "
            f"at step={int(ckpt_step.numpy())}, frame_cap={state.frame_cap}"
        )

    logger.info(
        f"Training TensorFlow Vocos with policy={policy}, "
        f"device_count_gpu={len(tf.config.list_physical_devices('GPU'))}, logs={tb_dir}"
    )

    disc_vars = mpd.trainable_variables + mrd.trainable_variables + cstft_disc.trainable_variables
    gen_vars = generator.trainable_variables

    last_log_t = time.time()
    throughput_t = time.time()
    throughput_step = int(ckpt_step.numpy())
    running: Dict[str, float] = {}
    reset_gpu_peak_memory()

    while int(ckpt_step.numpy()) < args.max_steps:
        step = int(ckpt_step.numpy()) + 1
        iter_t0 = time.perf_counter()
        batch = train_loader.next_batch()
        batch_local: Dict[str, np.ndarray] = {
            "features": np.asarray(batch["features"], dtype=np.float32),
            "audio": np.asarray(batch["audio"], dtype=np.float32),
            "target_frames": int(batch["target_frames"]),
        }
        retried = 0

        while True:
            try:
                timing: Dict[str, float] = {}
                t0 = time.perf_counter()
                features = tf.convert_to_tensor(batch_local["features"], dtype=tf.float32)
                real = tf.convert_to_tensor(batch_local["audio"], dtype=tf.float32)
                timing["time_h2d_ms"] = (time.perf_counter() - t0) * 1000.0
                assert_finite("batch/features", features)
                assert_finite("batch/audio", real)

                weights = compute_dynamic_weights(
                    step=step,
                    max_steps=args.max_steps,
                    pretrain_steps=args.pretrain_mel_steps,
                    adv_ramp_ratio=args.adv_ramp_ratio,
                    gan_base=args.gan_loss_coeff,
                    fm_base=args.fm_loss_coeff,
                    mrstft_base=args.mrstft_loss_coeff,
                    group_delay_base=args.group_delay_loss_coeff,
                    mrstft_final_ratio=args.mrstft_final_ratio,
                )
                adv_started = step >= args.pretrain_mel_steps
                run_disc_step = adv_started and (step % max(1, args.disc_update_interval) == 0)
                run_adv_loss = adv_started and (step % max(1, args.adv_loss_interval) == 0)

                d_total = tf.constant(0.0, dtype=tf.float32)
                d_mp = tf.constant(0.0, dtype=tf.float32)
                d_mrd = tf.constant(0.0, dtype=tf.float32)
                d_cstft = tf.constant(0.0, dtype=tf.float32)

                if run_disc_step:
                    t0 = time.perf_counter()
                    fake_disc = generator(features, training=True)
                    fake_disc, real_aligned_disc = align_audio(fake_disc, real)
                    fake_detached = tf.stop_gradient(fake_disc)
                    timing["time_gen_forward_for_disc_ms"] = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    with tf.GradientTape() as disc_tape:
                        r_mp, g_mp, _, _ = mpd(real_aligned_disc, fake_detached, training=True)
                        r_mrd, g_mrd, _, _ = mrd(real_aligned_disc, fake_detached, training=True)
                        r_cstft, g_cstft, _, _ = cstft_disc(real_aligned_disc, fake_detached, training=True)
                        d_mp_raw, d_mp_real_terms, _ = discriminator_hinge_loss(r_mp, g_mp)
                        d_mrd_raw, d_mrd_real_terms, _ = discriminator_hinge_loss(r_mrd, g_mrd)
                        d_cstft_raw, d_cstft_real_terms, _ = discriminator_hinge_loss(r_cstft, g_cstft)
                        d_mp = d_mp_raw / float(max(1, len(d_mp_real_terms)))
                        d_mrd = d_mrd_raw / float(max(1, len(d_mrd_real_terms)))
                        d_cstft = d_cstft_raw / float(max(1, len(d_cstft_real_terms)))
                        d_total = d_mp + args.mrd_loss_coeff * d_mrd + args.cstft_disc_loss_coeff * d_cstft
                        assert_finite("disc/d_mp", d_mp)
                        assert_finite("disc/d_mrd", d_mrd)
                        assert_finite("disc/d_cstft", d_cstft)
                        assert_finite("disc/d_total", d_total)
                    timing["time_disc_loss_compute_ms"] = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    disc_grads = disc_tape.gradient(d_total, disc_vars)
                    safe_apply_gradients(disc_opt, disc_grads, disc_vars)
                    timing["time_disc_backward_step_ms"] = (time.perf_counter() - t0) * 1000.0
                else:
                    timing["time_gen_forward_for_disc_ms"] = 0.0
                    timing["time_disc_loss_compute_ms"] = 0.0
                    timing["time_disc_backward_step_ms"] = 0.0

                mpd.trainable = False
                mrd.trainable = False
                cstft_disc.trainable = False
                try:
                    with tf.GradientTape() as gen_tape:
                        t_forward = time.perf_counter()
                        fake = generator(features, training=True)
                        fake, real_aligned = align_audio(fake, real)
                        timing["time_gen_forward_ms"] = (time.perf_counter() - t_forward) * 1000.0
                        g_gan_raw = tf.constant(0.0, dtype=tf.float32)
                        g_fm_raw = tf.constant(0.0, dtype=tf.float32)
                        t_adv = time.perf_counter()
                        if run_adv_loss and (weights.gan > 0.0 or weights.fm > 0.0):
                            _, g_mp_outs, fmap_r_mp, fmap_g_mp = mpd(real_aligned, fake, training=True)
                            _, g_mrd_outs, fmap_r_mrd, fmap_g_mrd = mrd(real_aligned, fake, training=True)
                            _, g_cstft_outs, fmap_r_cstft, fmap_g_cstft = cstft_disc(real_aligned, fake, training=True)
                            g_mp_adv, g_mp_terms = generator_hinge_loss(g_mp_outs)
                            g_mrd_adv, g_mrd_terms = generator_hinge_loss(g_mrd_outs)
                            g_cstft_adv, g_cstft_terms = generator_hinge_loss(g_cstft_outs)
                            g_mp_adv = g_mp_adv / float(max(1, len(g_mp_terms)))
                            g_mrd_adv = g_mrd_adv / float(max(1, len(g_mrd_terms)))
                            g_cstft_adv = g_cstft_adv / float(max(1, len(g_cstft_terms)))

                            fm_mp = feature_matching_loss(fmap_r_mp, fmap_g_mp) / float(max(1, len(fmap_r_mp)))
                            fm_mrd = feature_matching_loss(fmap_r_mrd, fmap_g_mrd) / float(max(1, len(fmap_r_mrd)))
                            fm_cstft = feature_matching_loss(fmap_r_cstft, fmap_g_cstft) / float(max(1, len(fmap_r_cstft)))
                            g_gan_raw = g_mp_adv + args.mrd_loss_coeff * g_mrd_adv + args.cstft_disc_loss_coeff * g_cstft_adv
                            g_fm_raw = fm_mp + args.mrd_loss_coeff * fm_mrd + args.cstft_disc_loss_coeff * fm_cstft
                        timing["time_gen_adv_fm_loss_ms"] = (time.perf_counter() - t_adv) * 1000.0

                        t_mrstft = time.perf_counter()
                        g_mrstft_raw = mrstft_loss_fn(real_aligned, fake)
                        timing["time_gen_mrstft_loss_ms"] = (time.perf_counter() - t_mrstft) * 1000.0

                        t_group = time.perf_counter()
                        g_group_delay_raw = group_delay_loss_fn(real_aligned, fake)
                        timing["time_gen_group_delay_loss_ms"] = (time.perf_counter() - t_group) * 1000.0

                        g_gan_weighted = tf.cast(weights.gan, tf.float32) * g_gan_raw
                        g_fm_weighted = tf.cast(weights.fm, tf.float32) * g_fm_raw
                        g_mrstft_weighted = tf.cast(weights.mrstft, tf.float32) * g_mrstft_raw
                        g_group_weighted = tf.cast(weights.group_delay, tf.float32) * g_group_delay_raw
                        g_total = g_gan_weighted + g_fm_weighted + g_mrstft_weighted + g_group_weighted
                        assert_finite("gen/g_gan_raw", g_gan_raw)
                        assert_finite("gen/g_fm_raw", g_fm_raw)
                        assert_finite("gen/g_mrstft_raw", g_mrstft_raw)
                        assert_finite("gen/g_group_delay_raw", g_group_delay_raw)
                        assert_finite("gen/g_total", g_total)
                finally:
                    mpd.trainable = True
                    mrd.trainable = True
                    cstft_disc.trainable = True

                t0 = time.perf_counter()
                gen_grads = gen_tape.gradient(g_total, gen_vars)
                safe_apply_gradients(gen_opt, gen_grads, gen_vars)
                timing["time_gen_backward_step_ms"] = (time.perf_counter() - t0) * 1000.0

                ckpt_step.assign(step)
                ckpt_frame_cap.assign(int(state.frame_cap))

                running["gen_total"] = float(g_total.numpy())
                running["gen_gan_raw"] = float(g_gan_raw.numpy())
                running["gen_feat_match_raw"] = float(g_fm_raw.numpy())
                running["gen_mrstft_raw"] = float(g_mrstft_raw.numpy())
                running["gen_group_delay_raw"] = float(g_group_delay_raw.numpy())
                running["gen_gan_weighted"] = float(g_gan_weighted.numpy())
                running["gen_feat_match_weighted"] = float(g_fm_weighted.numpy())
                running["gen_mrstft_weighted"] = float(g_mrstft_weighted.numpy())
                running["gen_group_delay_weighted"] = float(g_group_weighted.numpy())
                running["weight_gan"] = float(weights.gan)
                running["weight_feat_match"] = float(weights.fm)
                running["weight_mrstft"] = float(weights.mrstft)
                running["weight_group_delay"] = float(weights.group_delay)
                running["disc_total"] = float(d_total.numpy())
                running["disc_mp"] = float(d_mp.numpy())
                running["disc_mrd"] = float(d_mrd.numpy())
                running["disc_cstft"] = float(d_cstft.numpy())
                running["ran_disc_step"] = float(1.0 if run_disc_step else 0.0)
                running["ran_adv_loss"] = float(1.0 if run_adv_loss else 0.0)
                running["batch_size"] = float(features.shape[0] or batch_local["features"].shape[0])
                running["frame_cap"] = float(state.frame_cap)
                running["target_frames"] = float(batch_local["target_frames"])
                running["lr_gen"] = float(gen_lr_sched(step).numpy())
                running["lr_disc"] = float(disc_lr_sched(step).numpy())
                running.update(timing)
                running["time_step_total_ms"] = (time.perf_counter() - iter_t0) * 1000.0

                if step % max(1, args.log_every) == 0:
                    now = time.time()
                    if step > throughput_step and now > throughput_t:
                        running["steps_per_sec"] = (step - throughput_step) / (now - throughput_t)
                    throughput_step = step
                    throughput_t = now
                    with writer.as_default():
                        for k, v in running.items():
                            tf.summary.scalar(f"train/{k}", float(v), step=step)
                        tf.summary.scalar("train/cuda_mem_gb", float(get_gpu_peak_memory_gb()), step=step)
                    reset_gpu_peak_memory()

                if step % 200 == 0:
                    real_audio = tf.expand_dims(tf.expand_dims(real_aligned[0], axis=-1), axis=0)
                    pred_audio = tf.expand_dims(tf.expand_dims(fake[0], axis=-1), axis=0)
                    with writer.as_default():
                        tf.summary.audio("train/audio_target", real_audio, sample_rate=args.sample_rate, step=step)
                        tf.summary.audio("train/audio_pred", pred_audio, sample_rate=args.sample_rate, step=step)
                        if log_waveform_images:
                            tf.summary.image(
                                "train/waveform_overlay",
                                tf.expand_dims(waveform_to_image(real_aligned[0], fake[0]), axis=0),
                                step=step,
                            )
                    log_preview_samples(
                        generator=generator,
                        preview_cache=preview_cache,
                        writer=writer,
                        step=step,
                        sample_rate=args.sample_rate,
                    )

                if time.time() - last_log_t > 10:
                    sps = running.get("steps_per_sec", 0.0)
                    logger.info(
                        f"step={step} gen={running['gen_total']:.4f} disc={running['disc_total']:.4f} "
                        f"frames={batch_local['target_frames']} cap={state.frame_cap} sps={sps:.2f}"
                    )
                    last_log_t = time.time()

                if step % max(1, args.val_every) == 0:
                    run_validation(
                        generator=generator,
                        val_loader=val_loader,
                        mrstft_loss_fn=mrstft_loss_fn,
                        group_delay_loss_fn=group_delay_loss_fn,
                        writer=writer,
                        step=step,
                        val_steps=args.val_steps,
                        sample_rate=args.sample_rate,
                        loss_weights=weights,
                        log_waveform_images=log_waveform_images,
                    )

                if step % max(1, args.save_every) == 0:
                    ckpt_manager.save(checkpoint_number=step)

                break
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, NonFiniteLossError):
                    with writer.as_default():
                        tf.summary.scalar("train/non_finite_events", 1.0, step=step)
                        tf.summary.scalar("train/non_finite_target_frames", float(batch_local["target_frames"]), step=step)
                    logger.error(
                        f"Non-finite loss at step={step}, retry={retried}, "
                        f"frames={batch_local['target_frames']}, cap={state.frame_cap}: {exc}. "
                        "Skipping batch."
                    )
                    break
                if not maybe_oom(exc):
                    raise
                retried += 1
                changed = state.apply_oom_backoff(args.oom_backoff)
                if isinstance(train_loader, (PairedBatchLoader, SyntheticBatchLoader)):
                    train_loader.frame_cap = int(state.frame_cap)
                if isinstance(val_loader, (PairedBatchLoader, SyntheticBatchLoader)):
                    val_loader.frame_cap = int(state.frame_cap)
                if changed == 0 or batch_local["target_frames"] <= state.min_frame_cap:
                    logger.error(
                        f"OOM at step={step} and cannot reduce further (cap={state.frame_cap}). Skipping batch."
                    )
                    break
                batch_local = crop_batch_for_retry(
                    batch_local,
                    new_frame_cap=state.frame_cap,
                    hop_length=args.hop_length,
                    seed=args.seed + step + retried,
                )
                with writer.as_default():
                    tf.summary.scalar("train/oom_events", 1.0, step=step)
                    tf.summary.scalar("train/frame_cap_after_oom", float(state.frame_cap), step=step)
                logger.warning(
                    f"OOM at step={step}, retry={retried}, reduced frame_cap to {state.frame_cap} and recropped batch"
                )
                continue

    ckpt_frame_cap.assign(int(state.frame_cap))
    ckpt_manager.save(checkpoint_number=int(ckpt_step.numpy()))
    writer.flush()
    logger.info(
        f"TensorFlow training complete at step={int(ckpt_step.numpy())}, "
        f"frame_cap={state.frame_cap}. Logs at {tb_dir}"
    )


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
