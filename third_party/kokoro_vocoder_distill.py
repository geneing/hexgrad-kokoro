from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torchaudio
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional runtime dependency
    plt = None

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
        return {
            "features": torch.stack(features),
            "audio": torch.stack(audio),
            "target_frames": torch.tensor(target_frames, dtype=torch.long),
            "wav_paths": [str(r["wav_path"]) for r in rows],
        }


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
    parser.add_argument("--min-frame-cap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--val-steps", type=int, default=4)
    parser.add_argument("--sample-every", type=int, default=2000)
    parser.add_argument("--mel-plot-every", type=int, default=2000)
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--pretrain-steps", type=int, default=5000)
    parser.add_argument("--gen-lr", type=float, default=1e-4)
    parser.add_argument("--disc-lr", type=float, default=.5e-4)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--lr-schedule", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--lr-min-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
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


def maybe_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg or "cublas_status_alloc_failed" in msg


def slice_batch(batch: Dict[str, object], batch_size: int) -> Dict[str, object]:
    sliced: Dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] >= batch_size:
            sliced[key] = value[:batch_size]
        elif isinstance(value, list):
            sliced[key] = value[:batch_size]
        else:
            sliced[key] = value
    return sliced


def crop_batch_frames(batch: Dict[str, object], frame_cap: int, hop_length: int) -> Dict[str, object]:
    cropped = dict(batch)
    features = batch.get("features")
    audio = batch.get("audio")
    if not isinstance(features, torch.Tensor) or not isinstance(audio, torch.Tensor):
        return cropped
    target_frames = min(int(frame_cap), int(features.shape[-1]))
    target_samples = target_frames * int(hop_length)
    cropped["features"] = features[..., :target_frames]
    cropped["audio"] = audio[..., :target_samples]
    cropped["target_frames"] = torch.tensor(target_frames, dtype=torch.long)
    return cropped


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    norms = []
    for param in parameters:
        if param.grad is not None:
            norms.append(param.grad.detach().float().norm(2))
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def audio_metrics(fake: torch.Tensor, real: torch.Tensor) -> Dict[str, float]:
    diff = fake - real
    l1 = torch.mean(torch.abs(diff))
    mse = torch.mean(diff.square())
    rmse = torch.sqrt(mse.clamp_min(1e-12))
    signal = torch.mean(real.square()).clamp_min(1e-12)
    snr = 10.0 * torch.log10(signal / mse.clamp_min(1e-12))
    return {
        "l1": float(l1.item()),
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "snr_db": float(snr.item()),
        "real_peak": float(real.detach().abs().max().item()),
        "fake_peak": float(fake.detach().abs().max().item()),
        "real_rms": float(torch.sqrt(torch.mean(real.detach().square()).clamp_min(1e-12)).item()),
        "fake_rms": float(torch.sqrt(torch.mean(fake.detach().square()).clamp_min(1e-12)).item()),
    }


def mel_figure(
    mel_transform: torchaudio.transforms.MelSpectrogram,
    real_wav: torch.Tensor,
    fake_wav: torch.Tensor,
):
    if plt is None:
        return None
    with torch.no_grad():
        real_mel = torch.log(mel_transform(real_wav.detach().float().cpu()).clamp_min(1e-5)).numpy()
        fake_mel = torch.log(mel_transform(fake_wav.detach().float().cpu()).clamp_min(1e-5)).numpy()
        diff_mel = np.abs(fake_mel - real_mel)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    for ax, data, title in zip(
        axes,
        (real_mel, fake_mel, diff_mel),
        ("baseline log-mel", "generated log-mel", "absolute log-mel error"),
    ):
        im = ax.imshow(data, origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title(title)
        ax.set_ylabel("mel")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axes[-1].set_xlabel("frame")
    return fig


def log_samples(
    writer: SummaryWriter,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    tag: str,
    real: torch.Tensor,
    fake: torch.Tensor,
    step: int,
    sample_rate: int,
    sample_count: int,
    include_mels: bool,
) -> None:
    n = min(max(1, int(sample_count)), real.shape[0], fake.shape[0])
    for i in range(n):
        writer.add_audio(f"{tag}/sample_{i}/baseline", real[i].detach().cpu(), step, sample_rate)
        writer.add_audio(f"{tag}/sample_{i}/generated", fake[i].detach().cpu(), step, sample_rate)
        if include_mels:
            fig = mel_figure(mel_transform, real[i], fake[i])
            if fig is not None:
                writer.add_figure(f"{tag}/sample_{i}/mel", fig, step, close=True)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # pragma: no cover - compatibility with older torch signatures
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_lr_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    if args.lr_schedule == "none":
        return None
    min_ratio = float(args.lr_min_ratio)
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError(f"--lr-min-ratio must be between 0 and 1, got {min_ratio}")
    max_steps = max(1, int(args.max_steps))

    def lr_lambda(step: int) -> float:
        progress = min(max(0, int(step)), max_steps) / float(max_steps)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def sync_scheduler_to_step(scheduler, step: int) -> None:
    if scheduler is None:
        return
    step = max(0, int(step))
    scheduler.last_epoch = step
    lrs = [base_lr * fn(step) for base_lr, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
    for group, lr in zip(scheduler.optimizer.param_groups, lrs):
        group["lr"] = lr
    scheduler._last_lr = lrs


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

    train_dataset = PairedKokoroDataset(train_items, args.sample_rate)
    val_dataset = PairedKokoroDataset(val_items, args.sample_rate)

    current_frame_cap = max(1, int(args.frame_cap))
    min_frame_cap = max(1, min(int(args.min_frame_cap), current_frame_cap))

    def make_train_loader(batch_size: int, frame_cap: int) -> DataLoader:
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=SliceCollator(frame_cap, args.hop_length, train=True),
            drop_last=batch_size > 1,
        )

    def make_val_loader(frame_cap: int) -> DataLoader:
        return DataLoader(
            val_dataset,
            batch_size=max(1, current_batch_size // 2),
            shuffle=False,
            num_workers=max(0, args.num_workers // 2),
            pin_memory=device.type == "cuda",
            collate_fn=SliceCollator(frame_cap, args.hop_length, train=False),
        )

    current_batch_size = max(1, int(args.batch_size))
    min_batch_size = max(1, int(args.min_batch_size))
    train_loader = make_train_loader(current_batch_size, current_frame_cap)
    val_loader = make_val_loader(current_frame_cap)

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
    gen_sched = make_lr_scheduler(gen_opt, args)
    disc_sched = make_lr_scheduler(disc_opt, args)
    scaler = make_grad_scaler(device, args.precision)
    disc_loss_fn = DiscriminatorLoss()
    gen_loss_fn = GeneratorLoss()
    fm_loss_fn = FeatureMatchingLoss()
    mrstft = StyleTTS2MultiResolutionSTFTLoss(sample_rate=args.sample_rate).to(device)
    group_delay = StyleTTS2MultiResolutionGroupDelayLoss().to(device)

    out_dir = args.output_dir.resolve()
    tb_dir = out_dir / "tensorboard"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"backend": backend_name, "backend_config": backend_config, "args": vars(args)}, indent=2, default=str),
        encoding="utf-8",
    )
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("run/config", json.dumps({"backend": backend_name, "backend_config": backend_config, "args": vars(args)}, indent=2, default=str), 0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=max(int(args.n_fft), 1200),
        hop_length=args.hop_length,
        n_mels=80,
        center=True,
        power=1.0,
    )
    if plt is None:
        logger.warning("matplotlib is not installed; TensorBoard mel plot logging is disabled.")
    if device.type != "cuda" and args.precision != "fp32":
        logger.warning(f"precision={args.precision} requested on device={device}; autocast is disabled outside CUDA.")

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        mpd.load_state_dict(ckpt["mpd"])
        mrd.load_state_dict(ckpt["mrd"])
        gen_opt.load_state_dict(ckpt["gen_opt"])
        disc_opt.load_state_dict(ckpt["disc_opt"])
        if gen_sched is not None and "gen_sched" in ckpt:
            gen_sched.load_state_dict(ckpt["gen_sched"])
        if disc_sched is not None and "disc_sched" in ckpt:
            disc_sched.load_state_dict(ckpt["disc_sched"])
        if scaler.is_enabled() and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        step = int(ckpt.get("step", 0))
        if gen_sched is not None and "gen_sched" not in ckpt:
            sync_scheduler_to_step(gen_sched, step)
        if disc_sched is not None and "disc_sched" not in ckpt:
            sync_scheduler_to_step(disc_sched, step)

    logger.info(f"Training {backend_name} on device={device}; tensorboard={tb_dir}")
    logger.info(
        f"Precision={args.precision} scaler_enabled={scaler.is_enabled()} "
        f"lr_schedule={args.lr_schedule} lr_min_ratio={args.lr_min_ratio}"
    )
    logger.info(
        f"Memory budget: batch_size={current_batch_size} min_batch_size={min_batch_size} "
        f"frame_cap={current_frame_cap} min_frame_cap={min_frame_cap}"
    )
    generator.train()
    last_log = time.time()
    throughput_step = step
    throughput_time = time.time()
    try:
        while step < args.max_steps:
            rebuild_loader = False
            for batch in train_loader:
                if step >= args.max_steps:
                    break
                batch_local: Dict[str, object] = dict(batch)
                retried = 0
                iter_start = time.perf_counter()
                while True:
                    try:
                        features = batch_local["features"].to(device, non_blocking=True)  # type: ignore[union-attr]
                        real = batch_local["audio"].to(device, non_blocking=True)  # type: ignore[union-attr]
                        with autocast_context(device, args.precision):
                            fake = generator(features)
                            fake, real = align_audio(fake, real)

                        adv = step >= args.pretrain_steps
                        if adv:
                            disc_opt.zero_grad(set_to_none=True)
                            with autocast_context(device, args.precision):
                                real_mp, fake_mp, _, _ = mpd(real, fake.detach())
                                real_mrd, fake_mrd, _, _ = mrd(real, fake.detach())
                                d_mp, _, _ = disc_loss_fn(real_mp, fake_mp)
                                d_mrd, _, _ = disc_loss_fn(real_mrd, fake_mrd)
                                d_loss = d_mp + args.mrd_loss_coeff * d_mrd
                            if scaler.is_enabled():
                                scaler.scale(d_loss).backward()
                                scaler.unscale_(disc_opt)
                            else:
                                d_loss.backward()
                            d_grad_norm = grad_norm(list(mpd.parameters()) + list(mrd.parameters()))
                            if scaler.is_enabled():
                                scaler.step(disc_opt)
                            else:
                                disc_opt.step()
                        else:
                            d_mp = torch.zeros((), device=device)
                            d_mrd = torch.zeros((), device=device)
                            d_loss = torch.zeros((), device=device)
                            d_grad_norm = 0.0

                        gen_opt.zero_grad(set_to_none=True)
                        with autocast_context(device, args.precision):
                            stft_loss = mrstft(fake, real)
                            gd_loss = group_delay(fake, real)
                            g_adv = torch.zeros((), device=device)
                            g_fm = torch.zeros((), device=device)
                            if adv:
                                set_requires_grad(mpd, False)
                                set_requires_grad(mrd, False)
                                try:
                                    _, fake_mp, fmap_real_mp, fmap_fake_mp = mpd(real, fake)
                                    _, fake_mrd, fmap_real_mrd, fmap_fake_mrd = mrd(real, fake)
                                    g_mp, _ = gen_loss_fn(fake_mp)
                                    g_mrd, _ = gen_loss_fn(fake_mrd)
                                    g_adv = g_mp + args.mrd_loss_coeff * g_mrd
                                    g_fm = fm_loss_fn(fmap_real_mp, fmap_fake_mp) + args.mrd_loss_coeff * fm_loss_fn(
                                        fmap_real_mrd, fmap_fake_mrd
                                    )
                                finally:
                                    set_requires_grad(mpd, True)
                                    set_requires_grad(mrd, True)
                            g_mrstft_weighted = args.mrstft_loss_coeff * stft_loss
                            g_gd_weighted = args.group_delay_loss_coeff * gd_loss
                            g_adv_weighted = args.gan_loss_coeff * g_adv
                            g_fm_weighted = args.fm_loss_coeff * g_fm
                            g_loss = g_mrstft_weighted + g_gd_weighted + g_adv_weighted + g_fm_weighted
                        if scaler.is_enabled():
                            scaler.scale(g_loss).backward()
                            scaler.unscale_(gen_opt)
                        else:
                            g_loss.backward()
                        g_grad_norm = grad_norm(generator.parameters())
                        if scaler.is_enabled():
                            scaler.step(gen_opt)
                            scaler.update()
                        else:
                            gen_opt.step()
                        step += 1
                        if gen_sched is not None:
                            gen_sched.step()
                        if disc_sched is not None:
                            disc_sched.step()

                        fake_log = fake.detach().float()
                        real_log = real.detach().float()
                        metrics = audio_metrics(fake_log, real_log)
                        running = {
                            "gen_total": float(g_loss.item()),
                            "gen_mrstft_raw": float(stft_loss.item()),
                            "gen_group_delay_raw": float(gd_loss.item()),
                            "gen_gan_raw": float(g_adv.item()),
                            "gen_feat_match_raw": float(g_fm.item()),
                            "gen_mrstft_weighted": float(g_mrstft_weighted.item()),
                            "gen_group_delay_weighted": float(g_gd_weighted.item()),
                            "gen_gan_weighted": float(g_adv_weighted.item()),
                            "gen_feat_match_weighted": float(g_fm_weighted.item()),
                            "disc_total": float(d_loss.item()),
                            "disc_mp_raw": float(d_mp.item()),
                            "disc_mrd_raw": float(d_mrd.item()),
                            "disc_grad_norm": d_grad_norm,
                            "gen_grad_norm": g_grad_norm,
                            "batch_size_effective": float(features.shape[0]),
                            "batch_size_configured": float(current_batch_size),
                            "target_frames": float(batch_local["target_frames"].item()),  # type: ignore[union-attr]
                            "adv_enabled": float(1.0 if adv else 0.0),
                            "precision_fp16_scaler_scale": float(scaler.get_scale()) if scaler.is_enabled() else 1.0,
                            "lr_gen": float(gen_opt.param_groups[0]["lr"]),
                            "lr_disc": float(disc_opt.param_groups[0]["lr"]),
                            "time_step_ms": (time.perf_counter() - iter_start) * 1000.0,
                            **metrics,
                        }

                        if step % max(1, args.log_every) == 0:
                            now = time.time()
                            if now > throughput_time and step > throughput_step:
                                running["steps_per_sec"] = (step - throughput_step) / (now - throughput_time)
                            throughput_step = step
                            throughput_time = now
                            for key, value in running.items():
                                writer.add_scalar(f"train/{key}", value, step)
                            if device.type == "cuda":
                                writer.add_scalar("train/cuda_memory_allocated_gb", torch.cuda.memory_allocated() / (1024**3), step)
                                writer.add_scalar(
                                    "train/cuda_max_memory_allocated_gb",
                                    torch.cuda.max_memory_allocated() / (1024**3),
                                    step,
                                )
                                torch.cuda.reset_peak_memory_stats()

                        log_audio = step % max(1, args.sample_every) == 0
                        log_mels = step % max(1, args.mel_plot_every) == 0
                        if log_audio or log_mels:
                            log_samples(
                                writer=writer,
                                mel_transform=mel_transform,
                                tag="train",
                                real=real_log,
                                fake=fake_log,
                                step=step,
                                sample_rate=args.sample_rate,
                                sample_count=args.sample_count,
                                include_mels=log_mels,
                            )

                        if step % max(1, args.val_every) == 0:
                            validate_once(
                                generator=generator,
                                loader=val_loader,
                                device=device,
                                precision=args.precision,
                                mrstft=mrstft,
                                group_delay=group_delay,
                                backend_name=backend_name,
                                step=step,
                                writer=writer,
                                mel_transform=mel_transform,
                                sample_rate=args.sample_rate,
                                max_batches=args.val_steps,
                                sample_count=args.sample_count,
                                log_mels=log_mels,
                                loss_weights={
                                    "mrstft": args.mrstft_loss_coeff,
                                    "group_delay": args.group_delay_loss_coeff,
                                },
                            )

                        if time.time() - last_log > 10 or step % max(1, args.log_every) == 0:
                            logger.info(
                                f"{backend_name} step={step} gen={running['gen_total']:.4f} "
                                f"disc={running['disc_total']:.4f} mrstft={running['gen_mrstft_raw']:.4f} "
                                f"gd={running['gen_group_delay_raw']:.4f} bs={features.shape[0]}"
                            )
                            last_log = time.time()

                        if step % max(1, args.save_every) == 0:
                            save_path = ckpt_dir / f"step_{step:08d}.pt"
                            save_checkpoint(
                                save_path,
                                step,
                                generator,
                                mpd,
                                mrd,
                                gen_opt,
                                disc_opt,
                                gen_sched,
                                disc_sched,
                                scaler,
                                backend_name,
                                backend_config,
                            )
                            save_checkpoint(
                                ckpt_dir / "last.pt",
                                step,
                                generator,
                                mpd,
                                mrd,
                                gen_opt,
                                disc_opt,
                                gen_sched,
                                disc_sched,
                                scaler,
                                backend_name,
                                backend_config,
                            )
                        break
                    except RuntimeError as exc:
                        if device.type != "cuda" or not maybe_oom(exc):
                            raise
                        gen_opt.zero_grad(set_to_none=True)
                        disc_opt.zero_grad(set_to_none=True)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        writer.add_scalar("train/oom_events", 1.0, step)
                        old_batch_size = current_batch_size
                        old_frame_cap = current_frame_cap
                        if current_batch_size > min_batch_size:
                            current_batch_size = max(min_batch_size, current_batch_size // 2)
                            writer.add_scalar("train/batch_size_after_oom", float(current_batch_size), step)
                            batch_local = slice_batch(batch_local, current_batch_size)
                            reason = f"batch size {old_batch_size} -> {current_batch_size}"
                        elif current_frame_cap > min_frame_cap:
                            current_frame_cap = max(min_frame_cap, int(current_frame_cap * 0.75))
                            writer.add_scalar("train/frame_cap_after_oom", float(current_frame_cap), step)
                            batch_local = crop_batch_frames(batch_local, current_frame_cap, args.hop_length)
                            reason = f"frame cap {old_frame_cap} -> {current_frame_cap}"
                        else:
                            logger.error(
                                f"CUDA OOM at step={step}; already at minimum batch size {current_batch_size} "
                                f"and minimum frame cap {current_frame_cap}. Skipping batch."
                            )
                            break
                        retried += 1
                        rebuild_loader = True
                        val_loader = make_val_loader(current_frame_cap)
                        logger.warning(
                            f"CUDA OOM at step={step}, retry={retried}; reducing {reason} and retrying current batch"
                        )
                        continue
                if rebuild_loader:
                    break
            if rebuild_loader and step < args.max_steps:
                train_loader = make_train_loader(current_batch_size, current_frame_cap)
                val_loader = make_val_loader(current_frame_cap)
    finally:
        writer.flush()
        writer.close()

    save_checkpoint(
        ckpt_dir / "final.pt",
        step,
        generator,
        mpd,
        mrd,
        gen_opt,
        disc_opt,
        gen_sched,
        disc_sched,
        scaler,
        backend_name,
        backend_config,
    )
    save_checkpoint(
        ckpt_dir / "last.pt",
        step,
        generator,
        mpd,
        mrd,
        gen_opt,
        disc_opt,
        gen_sched,
        disc_sched,
        scaler,
        backend_name,
        backend_config,
    )


@torch.no_grad()
def validate_once(
    generator: nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    mrstft: nn.Module,
    group_delay: nn.Module,
    backend_name: str,
    step: int,
    writer: SummaryWriter,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    sample_rate: int,
    max_batches: int,
    sample_count: int,
    log_mels: bool,
    loss_weights: Dict[str, float],
) -> None:
    generator.eval()
    totals: list[float] = []
    stfts: list[float] = []
    gds: list[float] = []
    l1s: list[float] = []
    mses: list[float] = []
    rmses: list[float] = []
    snrs: list[float] = []
    real_rms: list[float] = []
    fake_rms: list[float] = []
    for i, batch in enumerate(loader):
        if i >= max(1, int(max_batches)):
            break
        features = batch["features"].to(device, non_blocking=True)
        real = batch["audio"].to(device, non_blocking=True)
        with autocast_context(device, precision):
            fake, real = align_audio(generator(features), real)
            stft_loss = mrstft(fake, real)
            gd_loss = group_delay(fake, real)
            total = loss_weights["mrstft"] * stft_loss + loss_weights["group_delay"] * gd_loss
        fake_log = fake.detach().float()
        real_log = real.detach().float()
        metric = audio_metrics(fake_log, real_log)
        totals.append(float(total.item()))
        stfts.append(float(stft_loss.item()))
        gds.append(float(gd_loss.item()))
        l1s.append(metric["l1"])
        mses.append(metric["mse"])
        rmses.append(metric["rmse"])
        snrs.append(metric["snr_db"])
        real_rms.append(metric["real_rms"])
        fake_rms.append(metric["fake_rms"])
        if i == 0:
            log_samples(
                writer=writer,
                mel_transform=mel_transform,
                tag="val",
                real=real_log,
                fake=fake_log,
                step=step,
                sample_rate=sample_rate,
                sample_count=sample_count,
                include_mels=log_mels,
            )
    if totals:
        writer.add_scalar("val/gen_total_estimate", sum(totals) / len(totals), step)
        writer.add_scalar("val/gen_mrstft_raw", sum(stfts) / len(stfts), step)
        writer.add_scalar("val/gen_group_delay_raw", sum(gds) / len(gds), step)
        writer.add_scalar("val/gen_mrstft_weighted", loss_weights["mrstft"] * (sum(stfts) / len(stfts)), step)
        writer.add_scalar("val/gen_group_delay_weighted", loss_weights["group_delay"] * (sum(gds) / len(gds)), step)
        writer.add_scalar("val/l1", sum(l1s) / len(l1s), step)
        writer.add_scalar("val/mse", sum(mses) / len(mses), step)
        writer.add_scalar("val/rmse", sum(rmses) / len(rmses), step)
        writer.add_scalar("val/snr_db", sum(snrs) / len(snrs), step)
        writer.add_scalar("val/real_rms", sum(real_rms) / len(real_rms), step)
        writer.add_scalar("val/fake_rms", sum(fake_rms) / len(fake_rms), step)
        logger.info(f"{backend_name} validation step={step} loss={sum(totals) / len(totals):.4f}")
    generator.train()


def save_checkpoint(
    path: Path,
    step: int,
    generator: nn.Module,
    mpd: nn.Module,
    mrd: nn.Module,
    gen_opt: torch.optim.Optimizer,
    disc_opt: torch.optim.Optimizer,
    gen_sched,
    disc_sched,
    scaler,
    backend_name: str,
    backend_config: Dict[str, object],
) -> None:
    payload = {
        "step": step,
        "backend": backend_name,
        "backend_config": backend_config,
        "generator": generator.state_dict(),
        "mpd": mpd.state_dict(),
        "mrd": mrd.state_dict(),
        "gen_opt": gen_opt.state_dict(),
        "disc_opt": disc_opt.state_dict(),
    }
    if gen_sched is not None:
        payload["gen_sched"] = gen_sched.state_dict()
    if disc_sched is not None:
        payload["disc_sched"] = disc_sched.state_dict()
    if scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)
