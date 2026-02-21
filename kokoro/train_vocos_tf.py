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

from .tf_vocos import (
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
from .tf_checkpoint_utils import load_pytorch_checkpoint_into_tf_generator


@dataclass
class TrainItem:
    wav_path: Path
    pair_path: Path
    frames: int


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
    parser.add_argument("--frame-cap", type=int, default=520)
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
    parser.add_argument("--log-every", type=int, default=50)

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
    ):
        self.items = list(items)
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.frame_cap = int(frame_cap)
        self.batch_size = int(batch_size)
        self.train = bool(train)
        self.rng = random.Random(seed + (0 if train else 1))
        self.index = 0
        self.order = list(range(len(self.items)))
        if self.train:
            self.rng.shuffle(self.order)

    def _next_index(self) -> int:
        if self.train:
            idx = self.rng.choice(self.order)
            return idx
        if not self.order:
            raise RuntimeError("No items available")
        idx = self.order[self.index % len(self.order)]
        self.index += 1
        return idx

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
        target_frames = self.frame_cap
        target_audio_samples = target_frames * self.hop_length
        for _ in range(self.batch_size):
            row = self._load_row(self.items[self._next_index()])
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
    def __init__(self, batch_size: int, frame_cap: int, hop_length: int, in_channels: int, seed: int):
        self.batch_size = int(batch_size)
        self.frame_cap = int(frame_cap)
        self.hop_length = int(hop_length)
        self.in_channels = int(in_channels)
        self.rng = np.random.default_rng(seed)

    def next_batch(self) -> Dict[str, np.ndarray]:
        features = self.rng.standard_normal((self.batch_size, self.in_channels, self.frame_cap), dtype=np.float32)
        audio = self.rng.standard_normal((self.batch_size, self.frame_cap * self.hop_length), dtype=np.float32) * 0.05
        return {"features": features.astype(np.float32), "audio": audio.astype(np.float32), "target_frames": self.frame_cap}


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

    with writer.as_default():
        tf.summary.scalar("val/generator_total_estimate", float(np.mean(total_vals)), step=step)
        tf.summary.scalar("val/gen_mrstft_raw", float(np.mean(mrstft_vals)), step=step)
        tf.summary.scalar("val/gen_group_delay_raw", float(np.mean(gd_vals)), step=step)
        tf.summary.scalar("val/l1_wave_loss", float(np.mean(l1_vals)), step=step)


def safe_apply_gradients(optimizer: tf.keras.optimizers.Optimizer, grads, vars_) -> None:
    pairs = [(g, v) for g, v in zip(grads, vars_) if g is not None]
    if pairs:
        optimizer.apply_gradients(pairs)


def main() -> None:
    logger.enable("kokoro.train_vocos_tf")
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

    if args.synthetic_data:
        train_loader = SyntheticBatchLoader(
            batch_size=args.batch_size,
            frame_cap=args.synthetic_frames,
            hop_length=args.hop_length,
            in_channels=in_channels,
            seed=args.seed,
        )
        val_loader = SyntheticBatchLoader(
            batch_size=max(1, min(2, args.batch_size)),
            frame_cap=args.synthetic_frames,
            hop_length=args.hop_length,
            in_channels=in_channels,
            seed=args.seed + 1,
        )
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
        train_loader = PairedBatchLoader(
            items=train_items,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            frame_cap=args.frame_cap,
            batch_size=args.batch_size,
            train=True,
            seed=args.seed,
        )
        val_loader = PairedBatchLoader(
            items=val_items,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            frame_cap=args.frame_cap,
            batch_size=max(1, args.batch_size // 2),
            train=False,
            seed=args.seed,
        )

    ckpt_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="step")
    ckpt = tf.train.Checkpoint(
        step=ckpt_step,
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
        logger.info(f"Resumed from checkpoint: {resume_path} at step={int(ckpt_step.numpy())}")

    logger.info(
        f"Training TensorFlow Vocos with policy={policy}, "
        f"device_count_gpu={len(tf.config.list_physical_devices('GPU'))}, logs={tb_dir}"
    )

    disc_vars = mpd.trainable_variables + mrd.trainable_variables + cstft_disc.trainable_variables
    gen_vars = generator.trainable_variables

    last_log_t = time.time()
    throughput_t = time.time()
    throughput_step = int(ckpt_step.numpy())

    while int(ckpt_step.numpy()) < args.max_steps:
        step = int(ckpt_step.numpy()) + 1
        iter_t0 = time.perf_counter()
        batch = train_loader.next_batch()
        features = tf.convert_to_tensor(batch["features"], dtype=tf.float32)
        real = tf.convert_to_tensor(batch["audio"], dtype=tf.float32)

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
            with tf.GradientTape() as disc_tape:
                fake = generator(features, training=True)
                fake_detached = tf.stop_gradient(fake)
                fake_detached, real_d = align_audio(fake_detached, real)
                r_mp, g_mp, _, _ = mpd(real_d, fake_detached, training=True)
                r_mrd, g_mrd, _, _ = mrd(real_d, fake_detached, training=True)
                r_cstft, g_cstft, _, _ = cstft_disc(real_d, fake_detached, training=True)

                d_mp_raw, d_mp_real_terms, _ = discriminator_hinge_loss(r_mp, g_mp)
                d_mrd_raw, d_mrd_real_terms, _ = discriminator_hinge_loss(r_mrd, g_mrd)
                d_cstft_raw, d_cstft_real_terms, _ = discriminator_hinge_loss(r_cstft, g_cstft)

                d_mp = d_mp_raw / float(max(1, len(d_mp_real_terms)))
                d_mrd = d_mrd_raw / float(max(1, len(d_mrd_real_terms)))
                d_cstft = d_cstft_raw / float(max(1, len(d_cstft_real_terms)))
                d_total = d_mp + args.mrd_loss_coeff * d_mrd + args.cstft_disc_loss_coeff * d_cstft
            disc_grads = disc_tape.gradient(d_total, disc_vars)
            safe_apply_gradients(disc_opt, disc_grads, disc_vars)

        with tf.GradientTape() as gen_tape:
            fake = generator(features, training=True)
            fake, real_g = align_audio(fake, real)

            g_gan_raw = tf.constant(0.0, dtype=tf.float32)
            g_fm_raw = tf.constant(0.0, dtype=tf.float32)

            if run_adv_loss and (weights.gan > 0.0 or weights.fm > 0.0):
                _, g_mp_outs, fmap_r_mp, fmap_g_mp = mpd(real_g, fake, training=True)
                _, g_mrd_outs, fmap_r_mrd, fmap_g_mrd = mrd(real_g, fake, training=True)
                _, g_cstft_outs, fmap_r_cstft, fmap_g_cstft = cstft_disc(real_g, fake, training=True)

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

            g_mrstft_raw = mrstft_loss_fn(real_g, fake)
            g_group_delay_raw = group_delay_loss_fn(real_g, fake)
            g_gan_weighted = tf.cast(weights.gan, tf.float32) * g_gan_raw
            g_fm_weighted = tf.cast(weights.fm, tf.float32) * g_fm_raw
            g_mrstft_weighted = tf.cast(weights.mrstft, tf.float32) * g_mrstft_raw
            g_group_weighted = tf.cast(weights.group_delay, tf.float32) * g_group_delay_raw
            g_total = g_gan_weighted + g_fm_weighted + g_mrstft_weighted + g_group_weighted

        if not bool(tf.math.is_finite(g_total).numpy()):
            raise RuntimeError(f"Non-finite generator loss at step={step}")
        gen_grads = gen_tape.gradient(g_total, gen_vars)
        safe_apply_gradients(gen_opt, gen_grads, gen_vars)

        ckpt_step.assign(step)
        step_time_ms = (time.perf_counter() - iter_t0) * 1000.0

        if step % max(1, args.log_every) == 0:
            now = time.time()
            steps_per_sec = 0.0
            if step > throughput_step and now > throughput_t:
                steps_per_sec = (step - throughput_step) / (now - throughput_t)
            throughput_step = step
            throughput_t = now
            with writer.as_default():
                tf.summary.scalar("train/gen_total", float(g_total.numpy()), step=step)
                tf.summary.scalar("train/gen_gan_raw", float(g_gan_raw.numpy()), step=step)
                tf.summary.scalar("train/gen_feat_match_raw", float(g_fm_raw.numpy()), step=step)
                tf.summary.scalar("train/gen_mrstft_raw", float(g_mrstft_raw.numpy()), step=step)
                tf.summary.scalar("train/gen_group_delay_raw", float(g_group_delay_raw.numpy()), step=step)
                tf.summary.scalar("train/disc_total", float(d_total.numpy()), step=step)
                tf.summary.scalar("train/disc_mp", float(d_mp.numpy()), step=step)
                tf.summary.scalar("train/disc_mrd", float(d_mrd.numpy()), step=step)
                tf.summary.scalar("train/disc_cstft", float(d_cstft.numpy()), step=step)
                tf.summary.scalar("train/weight_gan", float(weights.gan), step=step)
                tf.summary.scalar("train/weight_feat_match", float(weights.fm), step=step)
                tf.summary.scalar("train/weight_mrstft", float(weights.mrstft), step=step)
                tf.summary.scalar("train/weight_group_delay", float(weights.group_delay), step=step)
                tf.summary.scalar("train/target_frames", float(batch["target_frames"]), step=step)
                tf.summary.scalar("train/steps_per_sec", float(steps_per_sec), step=step)
                tf.summary.scalar("train/time_step_total_ms", float(step_time_ms), step=step)
                tf.summary.scalar("train/lr_gen", float(gen_lr_sched(step).numpy()), step=step)
                tf.summary.scalar("train/lr_disc", float(disc_lr_sched(step).numpy()), step=step)

        if step % 200 == 0:
            real_audio = tf.expand_dims(tf.expand_dims(real_g[0], axis=-1), axis=0)
            pred_audio = tf.expand_dims(tf.expand_dims(fake[0], axis=-1), axis=0)
            with writer.as_default():
                tf.summary.audio("train/audio_target", real_audio, sample_rate=args.sample_rate, step=step)
                tf.summary.audio("train/audio_pred", pred_audio, sample_rate=args.sample_rate, step=step)

        if time.time() - last_log_t > 10:
            logger.info(
                f"step={step} gen={float(g_total.numpy()):.4f} disc={float(d_total.numpy()):.4f} "
                f"frames={batch['target_frames']}"
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
            )

        if step % max(1, args.save_every) == 0:
            ckpt_manager.save(checkpoint_number=step)

    ckpt_manager.save(checkpoint_number=int(ckpt_step.numpy()))
    writer.flush()
    logger.info(f"TensorFlow training complete at step={int(ckpt_step.numpy())}. Logs at {tb_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
