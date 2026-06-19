from __future__ import annotations

import argparse
import json
import math
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
import torchaudio
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset

from kokoro.styletts2_losses import StyleTTS2MultiResolutionGroupDelayLoss, StyleTTS2MultiResolutionSTFTLoss
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.loss import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss


@dataclass
class TrainItem:
    wav_path: Path
    pair_path: Path
    frames: int


class KokoroFeatureConditioner(nn.Module):
    """Project Kokoro ASR/F0/noise/style tensors into vocoder conditioning."""

    def __init__(
        self,
        asr_channels: int = 512,
        style_channels: int = 128,
        out_channels: int = 192,
        control_channels: int = 32,
        control_layers: int = 2,
    ):
        super().__init__()
        self.asr_channels = int(asr_channels)
        self.style_channels = int(style_channels)
        self.asr_proj = nn.Conv1d(self.asr_channels, out_channels, 1)
        self.style_proj = nn.Conv1d(self.style_channels, out_channels, 1)
        self.f0_proj = self._control_branch(out_channels, control_channels, control_layers)
        self.noise_proj = self._control_branch(out_channels, control_channels, control_layers)
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels * 4, out_channels, 1),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 1),
        )

    @staticmethod
    def _control_branch(out_channels: int, control_channels: int, control_layers: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_ch = 1
        hidden = max(1, int(control_channels))
        for _ in range(max(1, int(control_layers))):
            layers.extend([nn.Conv1d(in_ch, hidden, 5, padding=2), nn.GELU()])
            in_ch = hidden
        layers.append(nn.Conv1d(hidden, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        asr_end = self.asr_channels
        f0_end = asr_end + 1
        noise_end = f0_end + 1
        expected = self.asr_channels + self.style_channels + 2
        if features.ndim != 3 or features.shape[1] != expected:
            raise ValueError(f"Expected Kokoro features [B,{expected},T], got {tuple(features.shape)}")
        return self.fuse(
            torch.cat(
                [
                    self.asr_proj(features[:, :asr_end]),
                    self.f0_proj(features[:, asr_end:f0_end]),
                    self.noise_proj(features[:, f0_end:noise_end]),
                    self.style_proj(features[:, noise_end:]),
                ],
                dim=1,
            )
        )


class PairedKokoroDataset(Dataset):
    def __init__(self, items: Sequence[TrainItem], sample_rate: int):
        self.items = list(items)
        self.sample_rate = int(sample_rate)

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _load_wav(path: Path) -> tuple[torch.Tensor, int]:
        with wave.open(str(path), "rb") as wav_file:
            sr = wav_file.getframerate()
            channels = wav_file.getnchannels()
            width = wav_file.getsampwidth()
            raw = wav_file.readframes(wav_file.getnframes())
        if width == 2:
            wav = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        elif width == 4:
            wav = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483647.0
        else:
            raise ValueError(f"Unsupported sample width {width}: {path}")
        if channels > 1:
            wav = wav.reshape(-1, channels).mean(axis=1)
        return torch.from_numpy(wav), sr

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        item = self.items[idx]
        pair = torch.load(item.pair_path, map_location="cpu", weights_only=False)
        wav, sr = self._load_wav(item.wav_path)
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


class SliceCollator:
    def __init__(self, frame_cap: int, hop_length: int, train: bool):
        self.frame_cap = int(frame_cap)
        self.hop_length = int(hop_length)
        self.train = bool(train)

    @staticmethod
    def _pad_1d(x: torch.Tensor, target: int) -> torch.Tensor:
        if x.numel() >= target:
            return x[:target]
        return torch.nn.functional.pad(x, (0, target - x.numel()))

    @staticmethod
    def _pad_2d(x: torch.Tensor, target: int) -> torch.Tensor:
        if x.shape[-1] >= target:
            return x[:, :target]
        return torch.nn.functional.pad(x, (0, target - x.shape[-1]))

    def __call__(self, rows: Sequence[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor]:
        target_frames = min(self.frame_cap, min(int(r["f0"].shape[-1]) for r in rows))  # type: ignore[index]
        features: list[torch.Tensor] = []
        audio: list[torch.Tensor] = []
        for row in rows:
            asr = row["asr"]  # type: ignore[assignment]
            f0 = row["f0"]  # type: ignore[assignment]
            noise = row["noise"]  # type: ignore[assignment]
            style = row["style"]  # type: ignore[assignment]
            wav = row["audio"]  # type: ignore[assignment]
            assert isinstance(asr, torch.Tensor)
            assert isinstance(f0, torch.Tensor)
            assert isinstance(noise, torch.Tensor)
            assert isinstance(style, torch.Tensor)
            assert isinstance(wav, torch.Tensor)
            total = int(f0.shape[-1])
            if asr.shape[-1] != total:
                asr = torch.nn.functional.interpolate(asr.unsqueeze(0), size=total, mode="linear", align_corners=False).squeeze(0)
            start = random.randint(0, total - target_frames) if self.train and total > target_frames else 0
            end = start + target_frames
            feat = torch.cat(
                [
                    self._pad_2d(asr[:, start:end], target_frames),
                    self._pad_1d(f0[start:end], target_frames).unsqueeze(0),
                    self._pad_1d(noise[start:end], target_frames).unsqueeze(0),
                    style.unsqueeze(-1).expand(style.shape[0], target_frames),
                ],
                dim=0,
            )
            wav_start = start * self.hop_length
            wav_end = wav_start + target_frames * self.hop_length
            features.append(feat)
            audio.append(self._pad_1d(wav[wav_start:wav_end], target_frames * self.hop_length))
        return {"features": torch.stack(features), "audio": torch.stack(audio)}


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-root", type=Path, default=Path("data/outputs"))
    parser.add_argument("--train-filelist", type=Path, default=None)
    parser.add_argument("--val-filelist", type=Path, default=None)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=300)
    parser.add_argument("--n-fft", type=int, default=1200)
    parser.add_argument("--model-input-channels", type=int, default=192)
    parser.add_argument("--control-channels", type=int, default=32)
    parser.add_argument("--control-layers", type=int, default=2)
    parser.add_argument("--frame-cap", type=int, default=520)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--pretrain-steps", type=int, default=5000)
    parser.add_argument("--gen-lr", type=float, default=3e-4)
    parser.add_argument("--disc-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gan-loss-coeff", type=float, default=1.0)
    parser.add_argument("--fm-loss-coeff", type=float, default=2.0)
    parser.add_argument("--mrstft-loss-coeff", type=float, default=45.0)
    parser.add_argument("--group-delay-loss-coeff", type=float, default=2.0)
    parser.add_argument("--mrd-loss-coeff", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def ensure_filelists(data_root: Path, train_filelist: Path, val_filelist: Path, seed: int) -> tuple[Path, Path]:
    if train_filelist.exists() and val_filelist.exists():
        return train_filelist, val_filelist
    wavs = sorted((data_root / "audio").rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No wav files found under {data_root / 'audio'}")
    rng = random.Random(seed)
    rng.shuffle(wavs)
    val_count = max(1, int(len(wavs) * 0.02))
    train_filelist.parent.mkdir(parents=True, exist_ok=True)
    train_filelist.write_text("\n".join(str(p.resolve()) for p in wavs[val_count:]) + "\n", encoding="utf-8")
    val_filelist.parent.mkdir(parents=True, exist_ok=True)
    val_filelist.write_text("\n".join(str(p.resolve()) for p in wavs[:val_count]) + "\n", encoding="utf-8")
    return train_filelist, val_filelist


def build_items(data_root: Path, filelist: Path) -> List[TrainItem]:
    items: list[TrainItem] = []
    audio_root = (data_root / "audio").resolve()
    pair_root = (data_root / "pairs").resolve()
    for line in filelist.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        wav_path = Path(line.strip()).resolve()
        pair_path = pair_root / wav_path.relative_to(audio_root).with_suffix(".pt")
        if not pair_path.exists():
            logger.warning(f"Missing pair for {wav_path}")
            continue
        pair = torch.load(pair_path, map_location="cpu", weights_only=False)
        items.append(TrainItem(wav_path=wav_path, pair_path=pair_path, frames=int(pair["f0"].shape[-1])))
    return items


def align_audio(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_decoder(
    args: argparse.Namespace,
    backend_name: str,
    build_generator: Callable[[argparse.Namespace], nn.Module],
    backend_config: Dict[str, object],
) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    data_root = args.data_root.resolve()
    train_filelist = args.train_filelist or data_root / "filelists" / "vocos.train.txt"
    val_filelist = args.val_filelist or data_root / "filelists" / "vocos.val.txt"
    train_filelist, val_filelist = ensure_filelists(data_root, train_filelist, val_filelist, args.seed)
    train_items = build_items(data_root, train_filelist)
    val_items = build_items(data_root, val_filelist)
    if not train_items or not val_items:
        raise RuntimeError(f"Need non-empty train and val sets, got train={len(train_items)} val={len(val_items)}")

    train_loader = DataLoader(
        PairedKokoroDataset(train_items, args.sample_rate),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=SliceCollator(args.frame_cap, args.hop_length, train=True),
        drop_last=True,
    )
    val_loader = DataLoader(
        PairedKokoroDataset(val_items, args.sample_rate),
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=device.type == "cuda",
        collate_fn=SliceCollator(args.frame_cap, args.hop_length, train=False),
    )

    generator = build_generator(args).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)
    gen_opt = torch.optim.AdamW(generator.parameters(), lr=args.gen_lr, betas=(0.8, 0.9), weight_decay=args.weight_decay)
    disc_opt = torch.optim.AdamW(
        list(mpd.parameters()) + list(mrd.parameters()),
        lr=args.disc_lr,
        betas=(0.8, 0.9),
        weight_decay=args.weight_decay,
    )
    disc_loss_fn = DiscriminatorLoss()
    gen_loss_fn = GeneratorLoss()
    fm_loss_fn = FeatureMatchingLoss()
    mrstft = StyleTTS2MultiResolutionSTFTLoss(sample_rate=args.sample_rate).to(device)
    group_delay = StyleTTS2MultiResolutionGroupDelayLoss().to(device)

    out_dir = args.output_dir.resolve()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"backend": backend_name, "backend_config": backend_config, "args": vars(args)}, indent=2, default=str),
        encoding="utf-8",
    )

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        mpd.load_state_dict(ckpt["mpd"])
        mrd.load_state_dict(ckpt["mrd"])
        gen_opt.load_state_dict(ckpt["gen_opt"])
        disc_opt.load_state_dict(ckpt["disc_opt"])
        step = int(ckpt.get("step", 0))

    generator.train()
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            features = batch["features"].to(device, non_blocking=True)
            real = batch["audio"].to(device, non_blocking=True)
            fake = generator(features)
            fake, real = align_audio(fake, real)

            adv = step >= args.pretrain_steps
            if adv:
                disc_opt.zero_grad(set_to_none=True)
                real_mp, fake_mp, _, _ = mpd(real, fake.detach())
                real_mrd, fake_mrd, _, _ = mrd(real, fake.detach())
                d_mp, _, _ = disc_loss_fn(real_mp, fake_mp)
                d_mrd, _, _ = disc_loss_fn(real_mrd, fake_mrd)
                d_loss = d_mp + args.mrd_loss_coeff * d_mrd
                d_loss.backward()
                disc_opt.step()
            else:
                d_loss = torch.zeros((), device=device)

            gen_opt.zero_grad(set_to_none=True)
            stft_loss = mrstft(fake, real)
            gd_loss = group_delay(fake, real)
            g_adv = torch.zeros((), device=device)
            g_fm = torch.zeros((), device=device)
            if adv:
                _, fake_mp, fmap_real_mp, fmap_fake_mp = mpd(real, fake)
                _, fake_mrd, fmap_real_mrd, fmap_fake_mrd = mrd(real, fake)
                g_mp, _ = gen_loss_fn(fake_mp)
                g_mrd, _ = gen_loss_fn(fake_mrd)
                g_adv = g_mp + args.mrd_loss_coeff * g_mrd
                g_fm = fm_loss_fn(fmap_real_mp, fmap_fake_mp) + args.mrd_loss_coeff * fm_loss_fn(fmap_real_mrd, fmap_fake_mrd)
            g_loss = (
                args.mrstft_loss_coeff * stft_loss
                + args.group_delay_loss_coeff * gd_loss
                + args.gan_loss_coeff * g_adv
                + args.fm_loss_coeff * g_fm
            )
            g_loss.backward()
            gen_opt.step()
            step += 1

            if step % max(1, args.log_every) == 0:
                logger.info(
                    f"{backend_name} step={step} gen={float(g_loss.item()):.4f} disc={float(d_loss.item()):.4f} "
                    f"mrstft={float(stft_loss.item()):.4f} gd={float(gd_loss.item()):.4f}"
                )

            if step % max(1, args.save_every) == 0:
                save_path = ckpt_dir / f"step_{step:08d}.pt"
                save_checkpoint(save_path, step, generator, mpd, mrd, gen_opt, disc_opt, backend_name, backend_config)
                save_checkpoint(ckpt_dir / "last.pt", step, generator, mpd, mrd, gen_opt, disc_opt, backend_name, backend_config)

        validate_once(generator, val_loader, device, mrstft, group_delay, backend_name, step)

    save_checkpoint(ckpt_dir / "final.pt", step, generator, mpd, mrd, gen_opt, disc_opt, backend_name, backend_config)
    save_checkpoint(ckpt_dir / "last.pt", step, generator, mpd, mrd, gen_opt, disc_opt, backend_name, backend_config)


@torch.no_grad()
def validate_once(
    generator: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mrstft: nn.Module,
    group_delay: nn.Module,
    backend_name: str,
    step: int,
) -> None:
    generator.eval()
    vals = []
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        features = batch["features"].to(device, non_blocking=True)
        real = batch["audio"].to(device, non_blocking=True)
        fake, real = align_audio(generator(features), real)
        vals.append(float((mrstft(fake, real) + group_delay(fake, real)).item()))
    if vals:
        logger.info(f"{backend_name} validation step={step} loss={sum(vals) / len(vals):.4f}")
    generator.train()


def save_checkpoint(
    path: Path,
    step: int,
    generator: nn.Module,
    mpd: nn.Module,
    mrd: nn.Module,
    gen_opt: torch.optim.Optimizer,
    disc_opt: torch.optim.Optimizer,
    backend_name: str,
    backend_config: Dict[str, object],
) -> None:
    torch.save(
        {
            "step": step,
            "backend": backend_name,
            "backend_config": backend_config,
            "generator": generator.state_dict(),
            "mpd": mpd.state_dict(),
            "mrd": mrd.state_dict(),
            "gen_opt": gen_opt.state_dict(),
            "disc_opt": disc_opt.state_dict(),
        },
        path,
    )
