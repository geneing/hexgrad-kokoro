"""
Train Kokoro TCN sequence mixers from frozen LSTM-teacher tensors.

Example:
  uv run python export/train_tcn_distill.py \
    --data-dir /export/eingerman/audio/tcl_distil/teacher/$(git rev-parse --short HEAD) \
    --output-dir /export/eingerman/audio/tcl_distil/checkpoints/$(git rev-parse --short HEAD) \
    --device cuda \
    --batch-size 8 \
    --epochs 20

Monitor:
  uv run tensorboard --logdir /export/eingerman/audio/tcl_distil/checkpoints
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from kokoro import KModel


DEFAULT_DISTILL_ROOT = Path("/export/eingerman/audio/tcl_distil")


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


def load_config(path: Path, sequence_mixer: str = "tcn") -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["sequence_mixer"] = {
        "type": sequence_mixer,
        **config.get("sequence_mixer", {}),
    }
    config["sequence_mixer"]["type"] = sequence_mixer
    return config


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_tensor(array: np.ndarray) -> torch.Tensor:
    if array.dtype == np.bool_:
        return torch.from_numpy(array.astype(np.bool_))
    if np.issubdtype(array.dtype, np.integer):
        return torch.from_numpy(array.astype(np.int64))
    return torch.from_numpy(array.astype(np.float32))


class DistillTensorDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        val_fraction: float,
        seed: int,
        max_samples: int | None = None,
    ):
        self.data_dir = data_dir
        samples = read_jsonl(data_dir / "samples.jsonl")
        if max_samples is not None:
            samples = samples[:max_samples]
        rng = random.Random(seed)
        rng.shuffle(samples)
        val_count = int(round(len(samples) * val_fraction))
        if split == "val":
            self.samples = samples[:val_count]
        else:
            self.samples = samples[val_count:]
        if not self.samples:
            raise RuntimeError(f"No samples for split={split}. total={len(samples)} val_fraction={val_fraction}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        meta = self.samples[index]
        with np.load(self.data_dir / meta["tensor_path"]) as npz:
            item = {key: to_tensor(npz[key]) for key in npz.files}
        item["meta"] = meta
        return item


def pad_1d(values: list[torch.Tensor], pad_value: int | float | bool = 0) -> torch.Tensor:
    max_len = max(v.shape[-1] for v in values)
    out = values[0].new_full((len(values), max_len), pad_value)
    for idx, value in enumerate(values):
        value = value.reshape(-1)
        out[idx, : value.shape[0]] = value
    return out


def pad_ct(values: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    channels = values[0].shape[-2]
    max_len = max(v.shape[-1] for v in values)
    out = values[0].new_full((len(values), channels, max_len), pad_value)
    for idx, value in enumerate(values):
        value = value.squeeze(0) if value.dim() == 3 and value.shape[0] == 1 else value
        out[idx, :, : value.shape[-1]] = value
    return out


def pad_tc(values: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    channels = values[0].shape[-1]
    max_len = max(v.shape[-2] for v in values)
    out = values[0].new_full((len(values), max_len, channels), pad_value)
    for idx, value in enumerate(values):
        value = value.squeeze(0) if value.dim() == 3 and value.shape[0] == 1 else value
        out[idx, : value.shape[-2], :] = value
    return out


def collate_distill(batch: list[dict[str, Any]]) -> dict[str, Any]:
    token_lengths = [int(item["input_ids"].shape[-1]) for item in batch]
    aligned_lengths = [int(item["predictor_aligned_en"].shape[-1]) for item in batch]
    f0_lengths = [int(item["f0"].shape[-1]) for item in batch]

    max_t = max(token_lengths)
    input_ids = pad_1d([item["input_ids"].reshape(-1) for item in batch], 0).long()
    text_mask = torch.ones(len(batch), max_t, dtype=torch.bool)
    for idx, length in enumerate(token_lengths):
        text_mask[idx, :length] = False
    input_lengths = torch.tensor(token_lengths, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "input_lengths": input_lengths,
        "text_mask": text_mask,
        "d_en": pad_ct([item["d_en"] for item in batch]),
        "style_predictor": torch.cat([item["style_predictor"].reshape(1, -1) for item in batch], dim=0),
        "target_text_encoder": pad_ct([item["text_encoder"] for item in batch]),
        "target_duration_encoded": pad_tc([item["duration_encoded"] for item in batch]),
        "target_duration_mixer": pad_tc([item["duration_mixer"] for item in batch]),
        "target_duration_logits": pad_tc([item["duration_logits"] for item in batch]),
        "predictor_aligned_en": pad_ct([item["predictor_aligned_en"] for item in batch]),
        "target_f0n_shared": pad_tc([item["f0n_shared"] for item in batch]),
        "target_f0": pad_1d([item["f0"].reshape(-1) for item in batch]),
        "target_n": pad_1d([item["n"].reshape(-1) for item in batch]),
        "token_lengths": torch.tensor(token_lengths, dtype=torch.long),
        "aligned_lengths": torch.tensor(aligned_lengths, dtype=torch.long),
        "f0_lengths": torch.tensor(f0_lengths, dtype=torch.long),
        "meta": [item["meta"] for item in batch],
    }


def length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def masked_mse(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() < student.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.to(student.dtype)
    denom = mask.sum().clamp_min(1.0) * student.shape[-1] if student.dim() == 3 else mask.sum().clamp_min(1.0)
    return ((student - teacher).pow(2) * mask).sum() / denom


def masked_cosine(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(student.float(), teacher.float(), dim=-1)
    mask = mask.to(cosine.dtype)
    return ((1.0 - cosine) * mask).sum() / mask.sum().clamp_min(1.0)


def run_f0n_heads(model: KModel, shared: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = shared.transpose(-1, -2)

    f0 = x
    for block in model.predictor.F0:
        f0 = block(f0, style)
    f0 = model.predictor.F0_proj(f0).squeeze(1)

    n = x
    for block in model.predictor.N:
        n = block(n, style)
    n = model.predictor.N_proj(n).squeeze(1)
    return f0, n


@dataclass
class LossWeights:
    text: float
    text_cosine: float
    duration_encoder: float
    duration_mixer: float
    duration_logits: float
    f0n_shared: float
    f0: float
    n: float


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def compute_losses(
    model: KModel,
    batch: dict[str, Any],
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    input_ids = batch["input_ids"]
    input_lengths = batch["input_lengths"]
    text_mask = batch["text_mask"]
    token_mask = ~text_mask
    style = batch["style_predictor"]

    text_out = model.text_encoder(input_ids, input_lengths, text_mask)
    text_loss = masked_mse(
        text_out.transpose(1, 2),
        batch["target_text_encoder"].transpose(1, 2),
        token_mask,
    )
    text_cos = masked_cosine(
        text_out.transpose(1, 2),
        batch["target_text_encoder"].transpose(1, 2),
        token_mask,
    )

    duration_encoded = model.predictor.text_encoder(batch["d_en"], style, input_lengths, text_mask)
    duration_encoder_loss = masked_mse(
        duration_encoded,
        batch["target_duration_encoded"],
        token_mask,
    )

    duration_mixer = model.predictor.run_duration_mixer(duration_encoded)
    duration_mixer_loss = masked_mse(
        duration_mixer,
        batch["target_duration_mixer"],
        token_mask,
    )
    duration_logits = model.predictor.duration_proj(duration_mixer)
    duration_logits_loss = masked_mse(
        duration_logits,
        batch["target_duration_logits"],
        token_mask,
    )

    aligned_mask = length_mask(batch["aligned_lengths"], batch["predictor_aligned_en"].shape[-1])
    f0n_shared = model.predictor.run_shared_mixer(batch["predictor_aligned_en"])
    f0n_shared_loss = masked_mse(
        f0n_shared,
        batch["target_f0n_shared"],
        aligned_mask,
    )

    f0, n = run_f0n_heads(model, f0n_shared, style)
    f0_mask = length_mask(batch["f0_lengths"], batch["target_f0"].shape[-1])
    f0_loss = masked_mse(f0, batch["target_f0"], f0_mask)
    n_loss = masked_mse(n, batch["target_n"], f0_mask)

    loss_parts = {
        "text": text_loss,
        "text_cosine": text_cos,
        "duration_encoder": duration_encoder_loss,
        "duration_mixer": duration_mixer_loss,
        "duration_logits": duration_logits_loss,
        "f0n_shared": f0n_shared_loss,
        "f0": f0_loss,
        "n": n_loss,
    }
    total = (
        weights.text * text_loss
        + weights.text_cosine * text_cos
        + weights.duration_encoder * duration_encoder_loss
        + weights.duration_mixer * duration_mixer_loss
        + weights.duration_logits * duration_logits_loss
        + weights.f0n_shared * f0n_shared_loss
        + weights.f0 * f0_loss
        + weights.n * n_loss
    )
    return total, loss_parts


def set_trainable(model: KModel, unfreeze_heads: bool) -> None:
    for param in model.parameters():
        param.requires_grad = False

    modules: list[nn.Module] = [
        model.text_encoder.sequence_mixer,
        model.predictor.text_encoder.tcn_layers,
        model.predictor.duration_mixer,
        model.predictor.shared_mixer,
    ]
    if unfreeze_heads:
        modules.extend([
            model.predictor.duration_proj,
            model.predictor.F0,
            model.predictor.N,
            model.predictor.F0_proj,
            model.predictor.N_proj,
        ])

    for module in modules:
        for param in module.parameters():
            param.requires_grad = True


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def split_batch(batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    batch_size = batch["input_ids"].shape[0]
    midpoint = batch_size // 2
    left: dict[str, Any] = {}
    right: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[0] == batch_size:
            left[key] = value[:midpoint]
            right[key] = value[midpoint:]
        elif isinstance(value, list) and len(value) == batch_size:
            left[key] = value[:midpoint]
            right[key] = value[midpoint:]
        else:
            left[key] = value
            right[key] = value
    return left, right


def clear_oom() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_batch_adaptive(
    model: KModel,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    weights: LossWeights,
    device: torch.device,
    grad_accum_steps: int,
    use_amp: bool,
    max_grad_norm: float,
    writer: SummaryWriter,
    global_step: int,
    train: bool,
) -> tuple[int, dict[str, float]]:
    if batch["input_ids"].shape[0] == 0:
        return 0, {}

    try:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                loss, parts = compute_losses(model, batch, weights)
                scaled_loss = loss / grad_accum_steps

            if train:
                scaler.scale(scaled_loss).backward()

        metrics = {"loss": float(loss.detach().cpu())}
        metrics.update({key: float(value.detach().cpu()) for key, value in parts.items()})
        return batch["input_ids"].shape[0], metrics
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or batch["input_ids"].shape[0] <= 1:
            raise
        clear_oom()
        left, right = split_batch(batch)
        writer.add_scalar("runtime/oom_split_batch_size", batch["input_ids"].shape[0], global_step)
        left_count, left_metrics = process_batch_adaptive(
            model, left, optimizer, scaler, weights, device, grad_accum_steps,
            use_amp, max_grad_norm, writer, global_step, train
        )
        right_count, right_metrics = process_batch_adaptive(
            model, right, optimizer, scaler, weights, device, grad_accum_steps,
            use_amp, max_grad_norm, writer, global_step, train
        )
        total_count = left_count + right_count
        metrics = {}
        for key in set(left_metrics) | set(right_metrics):
            metrics[key] = (
                left_metrics.get(key, 0.0) * left_count
                + right_metrics.get(key, 0.0) * right_count
            ) / max(total_count, 1)
        return total_count, metrics


def optimizer_step(
    model: KModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    max_grad_norm: float,
) -> None:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def save_checkpoint(
    path: Path,
    model: KModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_val: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "git_hash": git_hash(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: KModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"]), float(checkpoint["best_val"])


def log_metrics(writer: SummaryWriter, prefix: str, metrics: dict[str, float], step: int) -> None:
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


def average_metric(total: dict[str, float], metrics: dict[str, float], count: int) -> None:
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + value * count


def normalize_metrics(total: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in total.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TCN student mixers from Kokoro LSTM-teacher tensors.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/kokoro-v1_0.pth"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--unfreeze-heads", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--text-weight", type=float, default=1.0)
    parser.add_argument("--text-cosine-weight", type=float, default=0.2)
    parser.add_argument("--duration-encoder-weight", type=float, default=1.0)
    parser.add_argument("--duration-mixer-weight", type=float, default=1.0)
    parser.add_argument("--duration-logits-weight", type=float, default=0.5)
    parser.add_argument("--f0n-shared-weight", type=float, default=1.0)
    parser.add_argument("--f0-weight", type=float, default=0.5)
    parser.add_argument("--n-weight", type=float, default=0.5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_hash = git_hash()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or DEFAULT_DISTILL_ROOT / "checkpoints" / f"{run_hash}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    device = torch.device(args.device)
    config = load_config(args.config, sequence_mixer="tcn")
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=config,
        model=str(args.checkpoint),
        disable_complex=True,
    ).to(device)
    set_trainable(model, unfreeze_heads=args.unfreeze_heads)
    model.train()

    params = trainable_parameters(model)
    if not params:
        raise RuntimeError("No trainable parameters selected.")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(not args.no_amp and device.type == "cuda"))
    weights = LossWeights(
        text=args.text_weight,
        text_cosine=args.text_cosine_weight,
        duration_encoder=args.duration_encoder_weight,
        duration_mixer=args.duration_mixer_weight,
        duration_logits=args.duration_logits_weight,
        f0n_shared=args.f0n_shared_weight,
        f0=args.f0_weight,
        n=args.n_weight,
    )

    train_ds = DistillTensorDataset(args.data_dir, "train", args.val_fraction, args.seed, args.max_samples)
    val_ds = DistillTensorDataset(args.data_dir, "val", args.val_fraction, args.seed, args.max_samples)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_distill,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_distill,
    )

    metadata = {
        "git_hash": run_hash,
        "data_dir": str(args.data_dir),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "trainable_parameters": sum(param.numel() for param in params),
        "args": jsonable(vars(args)),
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    writer.add_text("run/manifest", json.dumps(metadata, indent=2))

    start_epoch = 0
    global_step = 0
    best_val = math.inf
    if args.resume:
        start_epoch, global_step, best_val = load_checkpoint(args.resume, model, optimizer, scaler, device)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total: dict[str, float] = {}
        train_count = 0

        for step, batch in enumerate(train_loader, start=1):
            count, metrics = process_batch_adaptive(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scaler=scaler,
                weights=weights,
                device=device,
                grad_accum_steps=args.grad_accum_steps,
                use_amp=not args.no_amp,
                max_grad_norm=args.max_grad_norm,
                writer=writer,
                global_step=global_step,
                train=True,
            )
            average_metric(train_total, metrics, count)
            train_count += count

            if step % args.grad_accum_steps == 0:
                optimizer_step(model, optimizer, scaler, args.max_grad_norm)
                global_step += 1

            if global_step > 0 and global_step % args.log_every == 0:
                log_metrics(writer, "train_step", metrics, global_step)
                writer.add_scalar("runtime/effective_batch_count", count, global_step)

        if len(train_loader) % args.grad_accum_steps != 0:
            optimizer_step(model, optimizer, scaler, args.max_grad_norm)
            global_step += 1

        train_avg = normalize_metrics(train_total, train_count)
        log_metrics(writer, "train_epoch", train_avg, epoch)

        model.eval()
        val_total: dict[str, float] = {}
        val_count = 0
        with torch.inference_mode():
            for batch in val_loader:
                count, metrics = process_batch_adaptive(
                    model=model,
                    batch=batch,
                    optimizer=None,
                    scaler=scaler,
                    weights=weights,
                    device=device,
                    grad_accum_steps=1,
                    use_amp=not args.no_amp,
                    max_grad_norm=args.max_grad_norm,
                    writer=writer,
                    global_step=global_step,
                    train=False,
                )
                average_metric(val_total, metrics, count)
                val_count += count

        val_avg = normalize_metrics(val_total, val_count)
        log_metrics(writer, "val", val_avg, epoch)
        val_loss = val_avg.get("loss", math.inf)
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_avg.get('loss', math.nan):.6f} "
            f"val_loss={val_loss:.6f} "
            f"global_step={global_step}"
        )

        latest_path = output_dir / "latest.pt"
        save_checkpoint(latest_path, model, optimizer, scaler, epoch, global_step, best_val, args)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, scaler, epoch, global_step, best_val, args)
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch + 1:04d}.pt", model, optimizer, scaler, epoch, global_step, best_val, args)

    writer.close()
    print(f"Done. output_dir={output_dir} best_val={best_val:.6f}")


if __name__ == "__main__":
    main()
