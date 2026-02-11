from __future__ import annotations

import argparse
import json
import math
import random
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter

from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.heads import ISTFTHead
from vocos.loss import (
    DiscriminatorLoss as VocosDiscriminatorLoss,
    FeatureMatchingLoss,
    GeneratorLoss as VocosGeneratorLoss,
)
from vocos.models import VocosBackbone

from .styletts2_losses import StyleTTS2MultiResolutionSTFTLoss


@dataclass
class TrainItem:
    wav_path: Path
    pair_path: Path
    frames: int


@dataclass
class AdaptiveBatchState:
    frame_cap: int
    min_frame_cap: int
    hop_length: int

    def apply_oom_backoff(self, ratio: float = 0.8) -> int:
        new_cap = max(self.min_frame_cap, int(self.frame_cap * ratio))
        changed = new_cap != self.frame_cap
        self.frame_cap = new_cap
        return 1 if changed else 0


class PairedVocoderDataset(Dataset):
    """Loads saved vocoder conditioning tensors + waveform targets."""

    def __init__(
        self,
        items: Sequence[TrainItem],
        sample_rate: int,
    ):
        self.items = list(items)
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _load_mono_wav(path: Path) -> tuple[torch.Tensor, int]:
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
        return torch.from_numpy(audio), sr

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.items[index]
        pair = torch.load(item.pair_path, map_location="cpu", weights_only=False)
        wav, sr = self._load_mono_wav(item.wav_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.sample_rate).squeeze(0)

        return {
            "asr": pair["asr"].float(),
            "f0": pair["f0"].float(),
            "noise": pair["noise"].float(),
            "style": pair["style"].float(),
            "audio": wav.float(),
            "wav_path": str(item.wav_path),
        }


class DynamicFrameBatchSampler(Sampler[List[int]]):
    """Dynamic batching using a per-batch frame budget."""

    def __init__(
        self,
        lengths: Sequence[int],
        frame_budget: int,
        max_batch_size: int,
        state: AdaptiveBatchState,
        shuffle: bool,
        seed: int,
    ):
        self.lengths = list(lengths)
        self.frame_budget = frame_budget
        self.max_batch_size = max_batch_size
        self.state = state
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        # approximate; exact length depends on current frame cap.
        avg = max(1, int(sum(min(l, self.state.frame_cap) for l in self.lengths) / max(1, len(self.lengths))))
        est_bs = max(1, min(self.max_batch_size, self.frame_budget // max(1, avg)))
        return max(1, math.ceil(len(self.lengths) / est_bs))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))

        if self.shuffle:
            rng.shuffle(indices)
            # sortish sampling: sort short windows inside shuffled chunks.
            chunk = max(1, self.max_batch_size * 50)
            sorted_indices: List[int] = []
            for i in range(0, len(indices), chunk):
                sub = indices[i : i + chunk]
                sub.sort(key=lambda idx: self.lengths[idx], reverse=True)
                sorted_indices.extend(sub)
            indices = sorted_indices
        else:
            indices.sort(key=lambda idx: self.lengths[idx], reverse=True)

        batch: List[int] = []
        used = 0
        cap = self.state.frame_cap
        for idx in indices:
            item_frames = min(self.lengths[idx], cap)
            if batch and (used + item_frames > self.frame_budget or len(batch) >= self.max_batch_size):
                yield batch
                batch = []
                used = 0
            batch.append(idx)
            used += item_frames

        if batch:
            yield batch


class SlicedPairCollator:
    """Random frame slicing and feature assembly.

    Features are assembled at the F0/noise temporal resolution:
    [asr(up/downsampled), f0, noise, style(broadcast)]
    """

    def __init__(
        self,
        state: AdaptiveBatchState,
        train: bool,
    ):
        self.state = state
        self.train = train

    @staticmethod
    def _repeat_pad_1d(x: torch.Tensor, target_length: int) -> torch.Tensor:
        if x.numel() >= target_length:
            return x[:target_length]
        if x.numel() == 0:
            return torch.zeros(target_length, dtype=x.dtype)
        repeat = 1 + target_length // x.numel()
        x_rep = x.repeat(repeat)
        return x_rep[:target_length]

    def __call__(self, rows: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not rows:
            raise ValueError("Empty batch")

        f0_lengths = [int(r["f0"].shape[-1]) for r in rows]
        cap = min(self.state.frame_cap, min(f0_lengths))
        min_cap = min(32, cap)

        if self.train:
            target_frames = random.randint(min_cap, cap)
        else:
            target_frames = cap

        hop = self.state.hop_length
        target_audio_samples = target_frames * hop

        feats: List[torch.Tensor] = []
        audio: List[torch.Tensor] = []
        wav_paths: List[str] = []

        for row in rows:
            f0 = row["f0"]
            noise = row["noise"]
            asr = row["asr"]
            style = row["style"]
            wav = row["audio"]

            total_frames = int(f0.shape[-1])
            if total_frames < target_frames:
                # fallback, should be rare due cap=min(f0_lengths)
                start_frame = 0
                target_local = total_frames
            else:
                if self.train and total_frames > target_frames:
                    start_frame = random.randint(0, total_frames - target_frames)
                else:
                    start_frame = 0
                target_local = target_frames

            # match ASR temporal dimension to F0 length.
            if asr.shape[-1] != total_frames:
                asr = torch.nn.functional.interpolate(
                    asr.unsqueeze(0),
                    size=total_frames,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)

            end_frame = start_frame + target_local
            asr_s = asr[:, start_frame:end_frame]
            f0_s = f0[start_frame:end_frame].unsqueeze(0)
            noise_s = noise[start_frame:end_frame].unsqueeze(0)
            style_s = style.unsqueeze(-1).expand(style.shape[0], target_local)
            feat = torch.cat([asr_s, f0_s, noise_s, style_s], dim=0)

            start_sample = start_frame * hop
            end_sample = start_sample + target_audio_samples
            wav_seg = self._repeat_pad_1d(wav[start_sample:end_sample], target_audio_samples)

            feats.append(feat)
            audio.append(wav_seg)
            wav_paths.append(row["wav_path"])

        return {
            "features": torch.stack(feats, dim=0),
            "audio": torch.stack(audio, dim=0),
            "target_frames": target_frames,
            "wav_paths": wav_paths,
        }


class PairedVocosGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_input_channels: int,
        backbone_dim: int,
        backbone_intermediate_dim: int,
        backbone_layers: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
    ):
        super().__init__()
        self.conditioner = nn.Sequential(
            nn.Conv1d(in_channels, model_input_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(model_input_channels, model_input_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(model_input_channels, model_input_channels, kernel_size=1),
        )
        self.backbone = VocosBackbone(
            input_channels=model_input_channels,
            dim=backbone_dim,
            intermediate_dim=backbone_intermediate_dim,
            num_layers=backbone_layers,
        )
        self.head = ISTFTHead(dim=backbone_dim, n_fft=n_fft, hop_length=hop_length, padding=padding)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.conditioner(features)
        x = self.backbone(x)
        return self.head(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train Vocos on saved Kokoro vocoder pairs")
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

    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-steps", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=1000)

    parser.add_argument("--gen-lr", type=float, default=3e-4)
    parser.add_argument("--disc-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)

    parser.add_argument("--frame-budget", type=int, default=10240)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--frame-cap", type=int, default=640)
    parser.add_argument("--min-frame-cap", type=int, default=96)
    parser.add_argument("--oom-backoff", type=float, default=0.8)

    parser.add_argument("--pretrain-mel-steps", type=int, default=3000)

    parser.add_argument("--gan-loss-coeff", type=float, default=1.0)
    parser.add_argument("--fm-loss-coeff", type=float, default=2.0)
    parser.add_argument("--mrstft-loss-coeff", type=float, default=45.0)
    parser.add_argument("--mrstft-final-ratio", type=float, default=0.25)
    parser.add_argument("--mrd-loss-coeff", type=float, default=1.0)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--sample-voices", type=str, default="af_bella,af_nicole,af_heart")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--sample-max-frames", type=int, default=640)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vocos_kokoro_paired"))
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    hop_length: int,
) -> List[TrainItem]:
    items: List[TrainItem] = []
    for wav in wav_paths:
        pair = derive_pair_path(data_root, wav)
        if not pair.exists():
            logger.warning(f"Missing pair file for wav, skipping: {wav}")
            continue
        frames = metadata_index.get(str(wav.resolve()))
        if frames is None:
            # fallback if manifest is unavailable
            pair_obj = torch.load(pair, map_location="cpu", weights_only=False)
            frames = int(pair_obj["f0"].shape[-1])
        items.append(TrainItem(wav_path=wav, pair_path=pair, frames=max(1, frames)))
    return items


def crop_batch_for_retry(batch: Dict[str, torch.Tensor], new_frame_cap: int, hop_length: int) -> Dict[str, torch.Tensor]:
    features = batch["features"]
    audio = batch["audio"]
    current_frames = features.shape[-1]
    if new_frame_cap >= current_frames:
        return batch

    bsz = features.shape[0]
    new_audio = new_frame_cap * hop_length

    f_list = []
    a_list = []
    for i in range(bsz):
        start = random.randint(0, current_frames - new_frame_cap)
        f = features[i, :, start : start + new_frame_cap]
        a = audio[i, start * hop_length : start * hop_length + new_audio]
        f_list.append(f)
        a_list.append(a)

    return {
        "features": torch.stack(f_list, dim=0),
        "audio": torch.stack(a_list, dim=0),
        "target_frames": new_frame_cap,
        "wav_paths": batch["wav_paths"],
    }


def align_audio(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


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


def compute_dynamic_weights(
    step: int,
    max_steps: int,
    pretrain_steps: int,
    gan_base: float,
    fm_base: float,
    mrstft_base: float,
    mrstft_final_ratio: float,
) -> Dict[str, float]:
    post_warmup = max(1, max_steps - pretrain_steps)
    adv_ramp_window = max(1, int(0.1 * post_warmup))
    if step < pretrain_steps:
        adv_scale = 0.0
    else:
        adv_scale = min(1.0, (step - pretrain_steps) / float(adv_ramp_window))
    progress = min(1.0, step / float(max(1, max_steps)))
    final_ratio = max(0.0, min(1.0, mrstft_final_ratio))
    mrstft_scale = 1.0 - (1.0 - final_ratio) * progress
    return {
        "gan": gan_base * adv_scale,
        "fm": fm_base * adv_scale,
        "mrstft": mrstft_base * mrstft_scale,
    }


def log_preview_samples(
    generator: nn.Module,
    dataset: PairedVocoderDataset,
    preview_items: Sequence[TrainItem],
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    sample_rate: int,
    hop_length: int,
    max_frames: int,
) -> None:
    if not preview_items:
        return
    index_by_wav = {str(it.wav_path): i for i, it in enumerate(dataset.items)}
    generator.eval()
    with torch.no_grad():
        for i, item in enumerate(preview_items):
            row_idx = index_by_wav.get(str(item.wav_path))
            if row_idx is None:
                continue
            row = dataset[row_idx]
            total_frames = int(row["f0"].shape[-1])
            target_frames = min(total_frames, max_frames)
            if target_frames <= 0:
                continue

            asr = row["asr"]
            f0 = row["f0"]
            noise = row["noise"]
            style = row["style"]
            wav = row["audio"]

            if asr.shape[-1] != total_frames:
                asr = torch.nn.functional.interpolate(
                    asr.unsqueeze(0),
                    size=total_frames,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)

            asr_s = asr[:, :target_frames]
            f0_s = f0[:target_frames].unsqueeze(0)
            noise_s = noise[:target_frames].unsqueeze(0)
            style_s = style.unsqueeze(-1).expand(style.shape[0], target_frames)
            feat = torch.cat([asr_s, f0_s, noise_s, style_s], dim=0).unsqueeze(0).to(device)

            target_audio = target_frames * hop_length
            real = wav[:target_audio].unsqueeze(0).to(device)
            pred = generator(feat)
            pred, real = align_audio(pred, real)

            voice = voice_from_wav_path(str(item.wav_path))
            tag = f"samples/{i+1:02d}_{voice}"
            writer.add_audio(f"{tag}/target", real[0].detach().cpu(), step, sample_rate)
            writer.add_audio(f"{tag}/pred", pred[0].detach().cpu(), step, sample_rate)
    generator.train()


def waveform_to_image(
    target_wav: torch.Tensor,
    pred_wav: torch.Tensor | None = None,
    width: int = 1024,
    height: int = 256,
) -> torch.Tensor:
    """Render one simple waveform overlay plot for TensorBoard."""
    target = target_wav.detach().float().cpu().flatten().numpy()
    pred = None if pred_wav is None else pred_wav.detach().float().cpu().flatten().numpy()
    if target.size == 0:
        return torch.ones(3, height, width, dtype=torch.float32)

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

    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def maybe_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def save_checkpoint(
    path: Path,
    step: int,
    epoch: int,
    generator: nn.Module,
    mpd: nn.Module,
    mrd: nn.Module,
    gen_opt: torch.optim.Optimizer,
    disc_opt: torch.optim.Optimizer,
    gen_sched: torch.optim.lr_scheduler._LRScheduler,
    disc_sched: torch.optim.lr_scheduler._LRScheduler,
    state: AdaptiveBatchState,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "generator": generator.state_dict(),
            "mpd": mpd.state_dict(),
            "mrd": mrd.state_dict(),
            "gen_opt": gen_opt.state_dict(),
            "disc_opt": disc_opt.state_dict(),
            "gen_sched": gen_sched.state_dict(),
            "disc_sched": disc_sched.state_dict(),
            "frame_cap": state.frame_cap,
        },
        path,
    )


def discover_resume_checkpoint(ckpt_dir: Path) -> Path | None:
    """Find best-effort checkpoint to resume from in a prior partial run."""
    last_ckpt = ckpt_dir / "last.pt"
    if last_ckpt.exists():
        return last_ckpt

    best: tuple[int, Path] | None = None
    for path in ckpt_dir.glob("step_*.pt"):
        stem = path.stem
        try:
            step_str = stem.split("_", maxsplit=1)[1]
            step = int(step_str)
        except (IndexError, ValueError):
            continue
        if best is None or step > best[0]:
            best = (step, path)

    if best is not None:
        return best[1]
    return None


def run_validation(
    generator: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    mrstft_loss: StyleTTS2MultiResolutionSTFTLoss,
    writer: SummaryWriter,
    log_waveform_images: bool,
    step: int,
    max_batches: int,
    sample_rate: int,
    loss_weights: Dict[str, float],
) -> None:
    generator.eval()
    mrstft_values: List[float] = []
    l1_values: List[float] = []
    total_values: List[float] = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            features = batch["features"].to(device, non_blocking=True)
            real = batch["audio"].to(device, non_blocking=True)
            pred = generator(features)
            pred, real = align_audio(pred, real)

            s = mrstft_loss(pred, real)
            l1 = torch.mean(torch.abs(pred - real))
            total_values.append(float((loss_weights["mrstft"] * s).item()))
            mrstft_values.append(float(s.item()))
            l1_values.append(float(l1.item()))

            if i == 0:
                writer.add_audio("val/audio_target", real[0].detach().cpu(), step, sample_rate)
                writer.add_audio("val/audio_pred", pred[0].detach().cpu(), step, sample_rate)
                if log_waveform_images:
                    writer.add_image(
                        "val/waveform_overlay",
                        waveform_to_image(real[0], pred[0]),
                        step,
                    )

    if mrstft_values:
        writer.add_scalar("val/generator_total_estimate", sum(total_values) / len(total_values), step)
        writer.add_scalar("val/gen_mrstft_raw", sum(mrstft_values) / len(mrstft_values), step)
        writer.add_scalar("val/gen_mrstft_weighted", sum(total_values) / len(total_values), step)
        writer.add_scalar("val/l1_wave_loss", sum(l1_values) / len(l1_values), step)

    generator.train()


def main() -> None:
    logger.enable("kokoro.train_vocos")
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

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

    if not train_filelist.exists():
        raise FileNotFoundError(f"Train filelist not found: {train_filelist}")
    if not val_filelist.exists():
        raise FileNotFoundError(f"Val filelist not found: {val_filelist}")

    metadata_index = build_metadata_index(manifest_root, args.hop_length)
    train_wavs = load_wav_list(train_filelist)
    val_wavs = load_wav_list(val_filelist)

    train_items = build_items(data_root, train_wavs, metadata_index, args.hop_length)
    val_items = build_items(data_root, val_wavs, metadata_index, args.hop_length)

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

    state = AdaptiveBatchState(
        frame_cap=args.frame_cap,
        min_frame_cap=args.min_frame_cap,
        hop_length=args.hop_length,
    )

    train_dataset = PairedVocoderDataset(train_items, sample_rate=args.sample_rate)
    val_dataset = PairedVocoderDataset(val_items, sample_rate=args.sample_rate)

    train_sampler = DynamicFrameBatchSampler(
        lengths=[it.frames for it in train_items],
        frame_budget=args.frame_budget,
        max_batch_size=args.max_batch_size,
        state=state,
        shuffle=True,
        seed=args.seed,
    )
    val_sampler = DynamicFrameBatchSampler(
        lengths=[it.frames for it in val_items],
        frame_budget=args.frame_budget,
        max_batch_size=max(1, args.max_batch_size // 2),
        state=state,
        shuffle=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=SlicedPairCollator(state=state, train=True),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=SlicedPairCollator(state=state, train=False),
    )

    in_channels = 512 + 1 + 1 + 128
    generator = PairedVocosGenerator(
        in_channels=in_channels,
        model_input_channels=args.model_input_channels,
        backbone_dim=args.backbone_dim,
        backbone_intermediate_dim=args.backbone_intermediate_dim,
        backbone_layers=args.backbone_layers,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        padding="same",
    ).to(device)

    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)

    vocos_disc_loss = VocosDiscriminatorLoss()
    vocos_gen_loss = VocosGeneratorLoss()
    feat_match_loss = FeatureMatchingLoss()
    mrstft_loss = StyleTTS2MultiResolutionSTFTLoss(sample_rate=args.sample_rate).to(device)

    gen_opt = torch.optim.AdamW(generator.parameters(), lr=args.gen_lr, betas=(0.8, 0.9), weight_decay=args.weight_decay)
    disc_params = list(mpd.parameters()) + list(mrd.parameters())
    disc_opt = torch.optim.AdamW(disc_params, lr=args.disc_lr, betas=(0.8, 0.9), weight_decay=args.weight_decay)

    gen_sched = torch.optim.lr_scheduler.CosineAnnealingLR(gen_opt, T_max=max(1, args.max_steps))
    disc_sched = torch.optim.lr_scheduler.CosineAnnealingLR(disc_opt, T_max=max(1, args.max_steps))

    step = 0
    epoch = 0

    out_dir = args.output_dir.resolve()
    tb_dir = out_dir / "tensorboard"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("run/args", json.dumps(vars(args), indent=2, default=str), 0)
    log_waveform_images = True
    try:
        import PIL  # noqa: F401
    except ImportError:
        log_waveform_images = False
        logger.warning("Pillow is not installed; waveform image logging is disabled. Install `pillow` to enable it.")

    resume_path = args.resume
    if resume_path is None:
        resume_path = discover_resume_checkpoint(ckpt_dir)
        if resume_path is not None:
            logger.info(f"Auto-detected checkpoint in output dir: {resume_path}")

    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        mpd.load_state_dict(ckpt["mpd"])
        mrd.load_state_dict(ckpt["mrd"])
        gen_opt.load_state_dict(ckpt["gen_opt"])
        disc_opt.load_state_dict(ckpt["disc_opt"])
        gen_sched.load_state_dict(ckpt["gen_sched"])
        disc_sched.load_state_dict(ckpt["disc_sched"])
        step = int(ckpt.get("step", 0))
        epoch = int(ckpt.get("epoch", 0))
        state.frame_cap = int(ckpt.get("frame_cap", state.frame_cap))
        logger.info(f"Resumed from {resume_path} at step={step}, epoch={epoch}, frame_cap={state.frame_cap}")

    logger.info(f"Training on device={device} | tensorboard={tb_dir}")

    generator.train()
    mpd.train()
    mrd.train()

    last_log = time.time()
    running: Dict[str, float] = {}

    while step < args.max_steps:
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= args.max_steps:
                break

            # OOM-aware retry loop: shrink frame cap and recrop in-memory batch when needed.
            batch_local = {
                "features": batch["features"],
                "audio": batch["audio"],
                "target_frames": batch["target_frames"],
                "wav_paths": batch["wav_paths"],
            }

            retried = 0
            while True:
                try:
                    features = batch_local["features"].to(device, non_blocking=True)
                    real = batch_local["audio"].to(device, non_blocking=True)

                    with torch.no_grad():
                        fake_detached = generator(features)
                        fake_detached, real_for_disc = align_audio(fake_detached, real)

                    loss_weights = compute_dynamic_weights(
                        step=step,
                        max_steps=args.max_steps,
                        pretrain_steps=args.pretrain_mel_steps,
                        gan_base=args.gan_loss_coeff,
                        fm_base=args.fm_loss_coeff,
                        mrstft_base=args.mrstft_loss_coeff,
                        mrstft_final_ratio=args.mrstft_final_ratio,
                    )
                    train_discriminator = step >= args.pretrain_mel_steps
                    d_total = torch.zeros(1, device=device)
                    d_mp = torch.zeros(1, device=device)
                    d_mrd = torch.zeros(1, device=device)

                    if train_discriminator:
                        disc_opt.zero_grad(set_to_none=True)
                        r_mp, g_mp, _, _ = mpd(real_for_disc, fake_detached)
                        r_mrd, g_mrd, _, _ = mrd(real_for_disc, fake_detached)

                        d_mp_raw, d_mp_real, _ = vocos_disc_loss(r_mp, g_mp)
                        d_mrd_raw, d_mrd_real, _ = vocos_disc_loss(r_mrd, g_mrd)
                        d_mp = d_mp_raw / max(1, len(d_mp_real))
                        d_mrd = d_mrd_raw / max(1, len(d_mrd_real))
                        d_total = d_mp + args.mrd_loss_coeff * d_mrd
                        d_total.backward()
                        disc_opt.step()

                    # Generator
                    gen_opt.zero_grad(set_to_none=True)
                    fake = generator(features)
                    fake, real = align_audio(fake, real)

                    g_gan_raw = torch.zeros(1, device=device)
                    g_fm_raw = torch.zeros(1, device=device)
                    if train_discriminator and loss_weights["gan"] > 0.0:
                        _, g_mp_outs, fmap_r_mp, fmap_g_mp = mpd(real, fake)
                        _, g_mrd_outs, fmap_r_mrd, fmap_g_mrd = mrd(real, fake)

                        g_mp_adv, g_mp_terms = vocos_gen_loss(g_mp_outs)
                        g_mrd_adv, g_mrd_terms = vocos_gen_loss(g_mrd_outs)
                        g_mp_adv = g_mp_adv / max(1, len(g_mp_terms))
                        g_mrd_adv = g_mrd_adv / max(1, len(g_mrd_terms))

                        fm_mp = feat_match_loss(fmap_r_mp, fmap_g_mp) / max(1, len(fmap_r_mp))
                        fm_mrd = feat_match_loss(fmap_r_mrd, fmap_g_mrd) / max(1, len(fmap_r_mrd))

                        g_gan_raw = g_mp_adv + args.mrd_loss_coeff * g_mrd_adv
                        g_fm_raw = fm_mp + args.mrd_loss_coeff * fm_mrd
                    g_mrstft_raw = mrstft_loss(fake, real)

                    g_gan_weighted = loss_weights["gan"] * g_gan_raw
                    g_fm_weighted = loss_weights["fm"] * g_fm_raw
                    g_mrstft_weighted = loss_weights["mrstft"] * g_mrstft_raw
                    g_total = g_gan_weighted + g_fm_weighted + g_mrstft_weighted
                    g_total.backward()
                    gen_opt.step()

                    gen_sched.step()
                    if train_discriminator:
                        disc_sched.step()

                    # Logging
                    step += 1
                    running["gen_total"] = float(g_total.item())
                    running["gen_gan_raw"] = float(g_gan_raw.item())
                    running["gen_feat_match_raw"] = float(g_fm_raw.item())
                    running["gen_mrstft_raw"] = float(g_mrstft_raw.item())
                    running["gen_gan_weighted"] = float(g_gan_weighted.item())
                    running["gen_feat_match_weighted"] = float(g_fm_weighted.item())
                    running["gen_mrstft_weighted"] = float(g_mrstft_weighted.item())
                    running["weight_gan"] = float(loss_weights["gan"])
                    running["weight_feat_match"] = float(loss_weights["fm"])
                    running["weight_mrstft"] = float(loss_weights["mrstft"])
                    running["disc_total"] = float(d_total.item())
                    running["disc_mp"] = float(d_mp.item())
                    running["disc_mrd"] = float(d_mrd.item())
                    running["frame_cap"] = float(state.frame_cap)
                    running["target_frames"] = float(batch_local["target_frames"])
                    running["lr_gen"] = float(gen_opt.param_groups[0]["lr"])
                    running["lr_disc"] = float(disc_opt.param_groups[0]["lr"])

                    for k, v in running.items():
                        writer.add_scalar(f"train/{k}", v, step)

                    if device.type == "cuda":
                        writer.add_scalar(
                            "train/cuda_mem_gb",
                            torch.cuda.max_memory_allocated() / (1024**3),
                            step,
                        )
                        torch.cuda.reset_peak_memory_stats()

                    if step % 200 == 0:
                        writer.add_audio("train/audio_target", real[0].detach().cpu(), step, args.sample_rate)
                        writer.add_audio("train/audio_pred", fake[0].detach().cpu(), step, args.sample_rate)
                        if log_waveform_images:
                            writer.add_image(
                                "train/waveform_overlay",
                                waveform_to_image(real[0], fake[0]),
                                step,
                            )
                        log_preview_samples(
                            generator=generator,
                            dataset=val_dataset,
                            preview_items=preview_items,
                            device=device,
                            step=step,
                            writer=writer,
                            sample_rate=args.sample_rate,
                            hop_length=args.hop_length,
                            max_frames=args.sample_max_frames,
                        )

                    if time.time() - last_log > 10:
                        logger.info(
                            f"step={step} gen={running['gen_total']:.4f} disc={running['disc_total']:.4f} "
                            f"frames={batch_local['target_frames']} cap={state.frame_cap}"
                        )
                        last_log = time.time()

                    if step % args.val_every == 0:
                        run_validation(
                            generator=generator,
                            val_loader=val_loader,
                            device=device,
                            mrstft_loss=mrstft_loss,
                            writer=writer,
                            log_waveform_images=log_waveform_images,
                            step=step,
                            max_batches=args.val_steps,
                            sample_rate=args.sample_rate,
                            loss_weights=loss_weights,
                        )

                    if step % args.save_every == 0:
                        ckpt_path = ckpt_dir / f"step_{step:08d}.pt"
                        save_checkpoint(
                            path=ckpt_path,
                            step=step,
                            epoch=epoch,
                            generator=generator,
                            mpd=mpd,
                            mrd=mrd,
                            gen_opt=gen_opt,
                            disc_opt=disc_opt,
                            gen_sched=gen_sched,
                            disc_sched=disc_sched,
                            state=state,
                        )
                        save_checkpoint(
                            path=ckpt_dir / "last.pt",
                            step=step,
                            epoch=epoch,
                            generator=generator,
                            mpd=mpd,
                            mrd=mrd,
                            gen_opt=gen_opt,
                            disc_opt=disc_opt,
                            gen_sched=gen_sched,
                            disc_sched=disc_sched,
                            state=state,
                        )

                    break

                except RuntimeError as exc:
                    if device.type != "cuda" or not maybe_oom(exc):
                        raise
                    retried += 1
                    torch.cuda.empty_cache()
                    changed = state.apply_oom_backoff(args.oom_backoff)
                    if changed == 0 or batch_local["target_frames"] <= state.min_frame_cap:
                        logger.error(
                            f"OOM at step={step} and cannot reduce further (cap={state.frame_cap}). Skipping batch."
                        )
                        break
                    batch_local = crop_batch_for_retry(batch_local, state.frame_cap, args.hop_length)
                    writer.add_scalar("train/oom_events", 1.0, step)
                    writer.add_scalar("train/frame_cap_after_oom", float(state.frame_cap), step)
                    logger.warning(
                        f"CUDA OOM at step={step}, retry={retried}, reduced frame_cap to {state.frame_cap} and recropped batch"
                    )
                    continue

        epoch += 1

    # Final save
    save_checkpoint(
        path=ckpt_dir / "final.pt",
        step=step,
        epoch=epoch,
        generator=generator,
        mpd=mpd,
        mrd=mrd,
        gen_opt=gen_opt,
        disc_opt=disc_opt,
        gen_sched=gen_sched,
        disc_sched=disc_sched,
        state=state,
    )
    save_checkpoint(
        path=ckpt_dir / "last.pt",
        step=step,
        epoch=epoch,
        generator=generator,
        mpd=mpd,
        mrd=mrd,
        gen_opt=gen_opt,
        disc_opt=disc_opt,
        gen_sched=gen_sched,
        disc_sched=disc_sched,
        state=state,
    )

    writer.flush()
    writer.close()
    logger.info(f"Training complete. Step={step}. Logs at {tb_dir}")


if __name__ == "__main__":
    main()
