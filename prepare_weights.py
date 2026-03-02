"""Prepare deployable Vocos weights (fp32/fp16/int8) from train_vocos checkpoints.

This utility converts a `kokoro.train_vocos` training checkpoint into inference
artifacts and optional quantized artifacts for downstream deployment experiments.

Primary responsibilities:
1) Load a training checkpoint and extract the generator weights.
2) Infer generator architecture from checkpoint keys (streaming or legacy).
3) Save:
   - fp32 generator state: `vocos.pt`
   - fp16 generator state: `vocos_fp16.pt`
4) Run QAT/PT2E conversion on the quantizable core and save:
   - slim int8-ish deployment state: `vocos_q8.pt`
   - optional full state: `vocos_q8_full.pt` (with `--save-full-q8`)
5) Optionally write sample audio for side-by-side qualitative checks.
6) Print a summary table of parameter dtypes and rough op-type estimates.

Streaming Vocos support:
- `--vocos-impl auto` (default) infers backend from checkpoint keys.
- For streaming checkpoints, the script builds `PairedVocosGenerator` with
  streaming settings so exported fp32/fp16/int8 artifacts match training-time
  architecture.

Typical usage:

1) Default conversion from a streaming train checkpoint (auto backend detect)
   uv run python prepare_weights.py \
     --input output/checkpoints/last.pt \
     --output-dir output/weights

2) Force streaming backend and explicit streaming repo path
   uv run python prepare_weights.py \
     --input output/checkpoints/last.pt \
     --output-dir output/weights_streaming \
     --vocos-impl streaming \
     --streaming-vocos-repo third_party/vocos_streaming \
     --backbone-causal \
     --backbone-pad-mode constant \
     --backbone-norm weight_norm

3) Faster smoke run (small data + few QAT steps)
   uv run python prepare_weights.py \
     --input output/checkpoints/last.pt \
     --output-dir output/weights_smoke \
     --max-train-items 256 \
     --batch-size 2 \
     --qat-steps 20 \
     --sample-count 2

4) Disable adversarial losses during QAT fallback
   uv run python prepare_weights.py \
     --input output/checkpoints/last.pt \
     --output-dir output/weights_no_adv \
     --disable-adversarial

5) Save both slim and full q8 outputs
   uv run python prepare_weights.py \
     --input output/checkpoints/last.pt \
     --output-dir output/weights_q8_full \
     --save-full-q8

Important notes:
- This script expects checkpoints produced by `kokoro/train_vocos.py`.
- Quantization path uses PT2E/XNNPACK-style flow and may vary across torch/
  executorch versions.
- Generated sample audio is for qualitative sanity checking, not MOS scoring.
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import random
import re
import wave
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch

logging.getLogger("torchao").setLevel(logging.ERROR)

from loguru import logger
from torch import nn
from torchao.quantization.pt2e import allow_exported_model_train_eval, disable_observer
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
try:
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    USING_EXECUTORCH_XNNPACK = True
except ImportError:
    from torchao.testing.pt2e._xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    USING_EXECUTORCH_XNNPACK = False
from torch.utils.data import DataLoader
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.loss import FeatureMatchingLoss, GeneratorLoss as VocosGeneratorLoss

from kokoro.styletts2_losses import StyleTTS2MultiResolutionGroupDelayLoss, StyleTTS2MultiResolutionSTFTLoss
from kokoro.train_vocos import (
    AdaptiveBatchState,
    MultiResolutionComplexSTFTDiscriminator,
    PairedVocoderDataset,
    PairedVocosGenerator,
    SlicedPairCollator,
    align_audio,
    build_items,
    build_metadata_index,
    ensure_filelists,
    load_wav_list,
    set_requires_grad,
)

@dataclass
class GeneratorConfig:
    in_channels: int
    model_input_channels: int
    backbone_dim: int
    backbone_intermediate_dim: int
    backbone_layers: int
    backbone_kernel_size: int
    vocos_impl: str
    backbone_causal: bool
    backbone_pad_mode: str
    backbone_norm: str
    streaming_vocos_repo: Optional[Path]
    n_fft: int
    hop_length: int
    padding: str


@dataclass
class LossBundle:
    mrstft: StyleTTS2MultiResolutionSTFTLoss
    group_delay: StyleTTS2MultiResolutionGroupDelayLoss
    mpd: Optional[MultiPeriodDiscriminator]
    mrd: Optional[MultiResolutionDiscriminator]
    cstft: Optional[MultiResolutionComplexSTFTDiscriminator]
    vocos_gen_loss: Optional[VocosGeneratorLoss]
    feat_match_loss: Optional[FeatureMatchingLoss]


@dataclass
class InferenceSample:
    tag: str
    features: torch.Tensor


class QuantizableVocosCore(nn.Module):
    def __init__(self, generator: PairedVocosGenerator):
        super().__init__()
        self.convnext = generator.backbone.convnext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.convnext:
            x = block(x)
        return x


class FloatPreBlocks(nn.Module):
    def __init__(self, generator: PairedVocosGenerator):
        super().__init__()
        self.conditioner = generator.conditioner
        self.embed = generator.backbone.embed
        self.norm = generator.backbone.norm

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Keep conditioner (including last conv) and early backbone blocks in float.
        x = self.conditioner(features)
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x


class FloatPostBlocks(nn.Module):
    def __init__(self, generator: PairedVocosGenerator):
        super().__init__()
        self.final_layer_norm = generator.backbone.final_layer_norm
        self.head = generator.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep final_layer_norm and ISTFT head in float.
        x = self.final_layer_norm(x.transpose(1, 2))
        return self.head(x)


class QuantizedVocosInference(nn.Module):
    def __init__(self, pre: nn.Module, core: nn.Module, post: nn.Module):
        super().__init__()
        self.pre = pre
        self.core = core
        self.post = post

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.pre(features)
        x = self.core(x)
        return self.post(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare inference and int8-quantized Vocos weights from train_vocos checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="Path to checkpoint produced by kokoro/train_vocos.py")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory for vocos.pt and vocos_q8.pt")

    parser.add_argument("--data-root", type=Path, default=Path("inputs/"))
    parser.add_argument("--train-filelist", type=Path, default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--max-train-items", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--frame-cap", type=int, default=520)
    parser.add_argument("--seed", type=int, default=4444)

    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=300)
    parser.add_argument("--padding", type=str, default="same")
    parser.add_argument(
        "--vocos-impl",
        type=str,
        choices=("auto", "streaming", "legacy"),
        default="auto",
        help="Generator backend to build for export. 'auto' infers from checkpoint keys.",
    )
    parser.add_argument(
        "--streaming-vocos-repo",
        type=Path,
        default=Path("third_party/vocos_streaming"),
        help="Path to local streaming-vocos repo root (contains src/components).",
    )
    parser.add_argument("--backbone-causal", dest="backbone_causal", action="store_true")
    parser.add_argument("--no-backbone-causal", dest="backbone_causal", action="store_false")
    parser.set_defaults(backbone_causal=True)
    parser.add_argument("--backbone-pad-mode", type=str, default="constant")
    parser.add_argument("--backbone-norm", type=str, default="weight_norm")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--sample-max-frames", type=int, default=480)
    parser.add_argument("--samples-dir", type=Path, default=None)

    parser.add_argument("--qat-steps", type=int, default=400)
    parser.add_argument("--qat-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument("--gan-loss-coeff", type=float, default=1.0)
    parser.add_argument("--fm-loss-coeff", type=float, default=2.0)
    parser.add_argument("--mrstft-loss-coeff", type=float, default=45.0)
    parser.add_argument("--group-delay-loss-coeff", type=float, default=2.0)
    parser.add_argument("--mrd-loss-coeff", type=float, default=1.0)
    parser.add_argument("--cstft-disc-loss-coeff", type=float, default=1.0)
    parser.add_argument("--disable-adversarial", action="store_true")

    parser.add_argument("--quant-backend", type=str, default="qnnpack")
    parser.add_argument("--save-full-q8", action="store_true")
    return parser.parse_args()


def _strip_parallel_prefixes(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    fixed: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        fixed[new_key] = value
    return fixed


def _is_tensor_state_dict(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return all(torch.is_tensor(v) for v in value.values())


def load_checkpoint(path: Path) -> tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if _is_tensor_state_dict(raw):
        return {}, _strip_parallel_prefixes(raw)  # already a generator state_dict
    if not isinstance(raw, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(raw)}")
    generator = raw.get("generator")
    if not _is_tensor_state_dict(generator):
        raise KeyError(f"Checkpoint {path} does not contain a valid 'generator' state_dict")
    return dict(raw), _strip_parallel_prefixes(generator)


def _infer_vocos_impl(state_dict: Mapping[str, torch.Tensor], requested: str) -> str:
    if requested in {"streaming", "legacy"}:
        return requested
    if "backbone.embed.conv.conv.weight_v" in state_dict or "backbone.convnext.0.dwconv.conv.conv.weight_v" in state_dict:
        return "streaming"
    return "legacy"


def infer_generator_config(state_dict: Mapping[str, torch.Tensor], args: argparse.Namespace) -> GeneratorConfig:
    vocos_impl = _infer_vocos_impl(state_dict, str(args.vocos_impl).lower())
    try:
        model_input_channels = int(state_dict["conditioner.0.weight"].shape[0])
        in_channels = int(state_dict["conditioner.0.weight"].shape[1])
        if vocos_impl == "streaming":
            if "backbone.embed.conv.conv.weight_v" in state_dict:
                embed_w = state_dict["backbone.embed.conv.conv.weight_v"]
            elif "backbone.embed.conv.conv.weight" in state_dict:
                embed_w = state_dict["backbone.embed.conv.conv.weight"]
            else:
                raise KeyError("backbone.embed.conv.conv.weight_v")
            backbone_dim = int(embed_w.shape[0])
            backbone_kernel_size = int(embed_w.shape[2])
        else:
            backbone_dim = int(state_dict["backbone.embed.weight"].shape[0])
            backbone_kernel_size = int(state_dict["backbone.embed.weight"].shape[2])
        backbone_intermediate_dim = int(state_dict["backbone.convnext.0.pwconv1.weight"].shape[0])
        n_fft = int(state_dict["head.out.weight"].shape[0]) - 2
    except KeyError as exc:
        raise KeyError(f"Could not infer generator config, missing key: {exc}") from exc

    layer_ids = set()
    pattern = re.compile(r"^backbone\.convnext\.(\d+)\.dwconv(\.conv\.conv)?\.weight(_v)?$")
    for key in state_dict.keys():
        m = pattern.match(key)
        if m:
            layer_ids.add(int(m.group(1)))
    if not layer_ids:
        raise RuntimeError("Could not infer backbone layer count from state_dict keys")
    backbone_layers = max(layer_ids) + 1

    return GeneratorConfig(
        in_channels=in_channels,
        model_input_channels=model_input_channels,
        backbone_dim=backbone_dim,
        backbone_intermediate_dim=backbone_intermediate_dim,
        backbone_layers=backbone_layers,
        backbone_kernel_size=backbone_kernel_size,
        vocos_impl=vocos_impl,
        backbone_causal=bool(args.backbone_causal),
        backbone_pad_mode=str(args.backbone_pad_mode),
        backbone_norm=str(args.backbone_norm),
        streaming_vocos_repo=(Path(args.streaming_vocos_repo).resolve() if args.streaming_vocos_repo else None),
        n_fft=n_fft,
        hop_length=int(args.hop_length),
        padding=str(args.padding),
    )


def build_generator(cfg: GeneratorConfig, generator_state: Mapping[str, torch.Tensor]) -> PairedVocosGenerator:
    model = PairedVocosGenerator(
        in_channels=cfg.in_channels,
        model_input_channels=cfg.model_input_channels,
        backbone_dim=cfg.backbone_dim,
        backbone_intermediate_dim=cfg.backbone_intermediate_dim,
        backbone_layers=cfg.backbone_layers,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        padding=cfg.padding,
        vocos_impl=cfg.vocos_impl,
        streaming_vocos_repo=cfg.streaming_vocos_repo,
        backbone_causal=cfg.backbone_causal,
        backbone_pad_mode=cfg.backbone_pad_mode,
        backbone_norm=cfg.backbone_norm,
    )
    missing, unexpected = model.load_state_dict(generator_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Generator state_dict mismatch. "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    return model


def build_train_loader(args: argparse.Namespace) -> DataLoader:
    data_root = args.data_root.resolve()
    train_filelist = args.train_filelist or data_root / "filelists" / "vocos.train.txt"
    val_filelist = data_root / "filelists" / "vocos.val.txt"
    manifest_root = args.manifest_root or data_root / "manifests"
    train_filelist, _ = ensure_filelists(
        data_root=data_root,
        train_filelist=train_filelist,
        val_filelist=val_filelist,
        seed=args.seed,
    )

    metadata_index = build_metadata_index(manifest_root, args.hop_length)
    train_wavs = load_wav_list(train_filelist)
    train_items = build_items(data_root, train_wavs, metadata_index, args.hop_length)
    if not train_items:
        raise RuntimeError(f"No training items found under {data_root}")

    if args.max_train_items > 0 and len(train_items) > args.max_train_items:
        rng = random.Random(args.seed)
        rng.shuffle(train_items)
        train_items = train_items[: args.max_train_items]

    dataset = PairedVocoderDataset(train_items, sample_rate=args.sample_rate)
    state = AdaptiveBatchState(
        frame_cap=args.frame_cap,
        min_frame_cap=max(32, min(args.frame_cap, 192)),
        hop_length=args.hop_length,
    )
    collator = SlicedPairCollator(state=state, train=True, fixed_shapes=True)
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=True,
        num_workers=max(0, args.num_workers),
        pin_memory=False,
        drop_last=False,
        collate_fn=collator,
        persistent_workers=(args.num_workers > 0),
    )
    return loader


def build_inference_samples(
    dataset: PairedVocoderDataset,
    count: int,
    seed: int,
    max_frames: int,
) -> list[InferenceSample]:
    if count <= 0:
        return []

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    if len(indices) > count:
        indices = rng.sample(indices, count)

    samples: list[InferenceSample] = []
    for sample_idx, row_idx in enumerate(indices, start=1):
        row = dataset[row_idx]
        total_frames = int(row["f0"].shape[-1])
        target_frames = total_frames if max_frames <= 0 else min(total_frames, max_frames)
        if target_frames <= 0:
            continue

        asr = row["asr"]
        f0 = row["f0"]
        noise = row["noise"]
        style = row["style"]
        wav_path = Path(row["wav_path"])
        voice = wav_path.parent.name
        stem = wav_path.stem

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
        features = torch.cat([asr_s, f0_s, noise_s, style_s], dim=0)
        samples.append(InferenceSample(tag=f"{sample_idx:02d}_{voice}_{stem}", features=features))
    return samples


def _load_disc_state(ckpt: Mapping[str, object], key: str) -> Optional[Mapping[str, torch.Tensor]]:
    state = ckpt.get(key)
    if not _is_tensor_state_dict(state):
        return None
    return _strip_parallel_prefixes(state)


def build_losses(args: argparse.Namespace, ckpt: Mapping[str, object]) -> LossBundle:
    mrstft = StyleTTS2MultiResolutionSTFTLoss(sample_rate=args.sample_rate)
    group_delay = StyleTTS2MultiResolutionGroupDelayLoss()

    if args.disable_adversarial:
        logger.warning("Adversarial branches disabled; QAT loss will use MR-STFT + GroupDelay only")
        return LossBundle(
            mrstft=mrstft,
            group_delay=group_delay,
            mpd=None,
            mrd=None,
            cstft=None,
            vocos_gen_loss=None,
            feat_match_loss=None,
        )

    mpd_state = _load_disc_state(ckpt, "mpd")
    mrd_state = _load_disc_state(ckpt, "mrd")
    cstft_state = _load_disc_state(ckpt, "cstft_disc")
    if not (mpd_state and mrd_state and cstft_state):
        logger.warning(
            "Checkpoint does not contain full discriminator states; using MR-STFT + GroupDelay only for QAT"
        )
        return LossBundle(
            mrstft=mrstft,
            group_delay=group_delay,
            mpd=None,
            mrd=None,
            cstft=None,
            vocos_gen_loss=None,
            feat_match_loss=None,
        )

    mpd = MultiPeriodDiscriminator().cpu()
    mrd = MultiResolutionDiscriminator().cpu()
    cstft = MultiResolutionComplexSTFTDiscriminator().cpu()
    mpd.load_state_dict(mpd_state, strict=True)
    mrd.load_state_dict(mrd_state, strict=True)
    cstft.load_state_dict(cstft_state, strict=True)
    set_requires_grad(mpd, False)
    set_requires_grad(mrd, False)
    set_requires_grad(cstft, False)
    mpd.eval()
    mrd.eval()
    cstft.eval()
    return LossBundle(
        mrstft=mrstft,
        group_delay=group_delay,
        mpd=mpd,
        mrd=mrd,
        cstft=cstft,
        vocos_gen_loss=VocosGeneratorLoss(),
        feat_match_loss=FeatureMatchingLoss(),
    )


def _next_batch(it: Iterable[Dict[str, torch.Tensor]], fallback_loader: DataLoader) -> tuple[Dict[str, torch.Tensor], Iterable[Dict[str, torch.Tensor]]]:
    iterator = iter(it)
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(fallback_loader)
        return next(iterator), iterator


def sanitize_batch_tensors(
    features: torch.Tensor,
    real: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    clean_features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)
    clean_real = torch.nan_to_num(real, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    changed = bool((not torch.equal(clean_features, features)) or (not torch.equal(clean_real, real)))
    return clean_features, clean_real, changed


def has_non_finite_gradients(module: nn.Module) -> bool:
    for p in module.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return True
    return False


def stabilize_non_finite_parameters(module: nn.Module) -> int:
    fixed = 0
    with torch.no_grad():
        for p in module.parameters():
            if not torch.is_floating_point(p):
                continue
            if not torch.isfinite(p).all():
                p.copy_(torch.nan_to_num(p, nan=0.0, posinf=1e3, neginf=-1e3))
                fixed += 1
    return fixed


def compute_generator_loss(
    generator: nn.Module,
    features: torch.Tensor,
    real: torch.Tensor,
    losses: LossBundle,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float]]:
    fake = generator(features)
    fake, real = align_audio(fake, real)

    g_gan_raw = fake.new_zeros(1)
    g_fm_raw = fake.new_zeros(1)

    if losses.mpd is not None:
        assert losses.mrd is not None
        assert losses.cstft is not None
        assert losses.vocos_gen_loss is not None
        assert losses.feat_match_loss is not None

        _, g_mp_outs, fmap_r_mp, fmap_g_mp = losses.mpd(real, fake)
        _, g_mrd_outs, fmap_r_mrd, fmap_g_mrd = losses.mrd(real, fake)
        _, g_cstft_outs, fmap_r_cstft, fmap_g_cstft = losses.cstft(real, fake)

        g_mp_adv, g_mp_terms = losses.vocos_gen_loss(g_mp_outs)
        g_mrd_adv, g_mrd_terms = losses.vocos_gen_loss(g_mrd_outs)
        g_cstft_adv, g_cstft_terms = losses.vocos_gen_loss(g_cstft_outs)
        g_mp_adv = g_mp_adv / max(1, len(g_mp_terms))
        g_mrd_adv = g_mrd_adv / max(1, len(g_mrd_terms))
        g_cstft_adv = g_cstft_adv / max(1, len(g_cstft_terms))

        fm_mp = losses.feat_match_loss(fmap_r_mp, fmap_g_mp) / max(1, len(fmap_r_mp))
        fm_mrd = losses.feat_match_loss(fmap_r_mrd, fmap_g_mrd) / max(1, len(fmap_r_mrd))
        fm_cstft = losses.feat_match_loss(fmap_r_cstft, fmap_g_cstft) / max(1, len(fmap_r_cstft))

        g_gan_raw = g_mp_adv + args.mrd_loss_coeff * g_mrd_adv + args.cstft_disc_loss_coeff * g_cstft_adv
        g_fm_raw = fm_mp + args.mrd_loss_coeff * fm_mrd + args.cstft_disc_loss_coeff * fm_cstft

    g_mrstft_raw = losses.mrstft(fake, real)
    g_group_delay_raw = losses.group_delay(fake, real)

    total = (
        args.gan_loss_coeff * g_gan_raw
        + args.fm_loss_coeff * g_fm_raw
        + args.mrstft_loss_coeff * g_mrstft_raw
        + args.group_delay_loss_coeff * g_group_delay_raw
    )
    stats = {
        "total": float(total.item()),
        "gan": float(g_gan_raw.item()),
        "fm": float(g_fm_raw.item()),
        "mrstft": float(g_mrstft_raw.item()),
        "group_delay": float(g_group_delay_raw.item()),
    }
    return total, stats


def resolve_quant_backend(requested: str) -> str:
    requested_norm = requested.lower()
    if requested_norm not in {"qnnpack", "xnnpack"}:
        logger.warning(
            f"PT2E path uses XNNPACK quantizer; ignoring requested backend='{requested}'. "
            "Use '--quant-backend qnnpack' for compatibility."
        )
    return "qnnpack"


def run_qat(
    generator: PairedVocosGenerator,
    train_loader: DataLoader,
    losses: LossBundle,
    args: argparse.Namespace,
) -> nn.Module:
    resolve_quant_backend(args.quant_backend)
    pre_fp = FloatPreBlocks(generator).train()
    core_fp = QuantizableVocosCore(generator).train()
    post_fp = FloatPostBlocks(generator).train()

    example_batch = next(iter(train_loader))
    example_features = example_batch["features"].float()
    example_core_input = pre_fp(example_features)
    example_input = (example_core_input,)
    dynamic_shapes = (
        {
            0: torch.export.Dim("batch", min=1, max=max(1, args.batch_size)),
            2: torch.export.Dim("frames", min=16, max=max(16, args.frame_cap)),
        },
    )
    exported_core = torch.export.export(
        core_fp,
        example_input,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    ).module()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*XNNPACKQuantizer.*deprecated.*",
            category=Warning,
        )
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(
                is_per_channel=True,
                is_qat=False,
                act_qmin=-127,
                act_qmax=127,
            )
        )
    qat_core = prepare_pt2e(exported_core, quantizer)
    allow_exported_model_train_eval(qat_core)
    qat_core.train()
    pre_fp.train()
    post_fp.train()

    qat_model = QuantizedVocosInference(pre=pre_fp, core=qat_core, post=post_fp)

    optimizer = torch.optim.AdamW(
        qat_model.parameters(),
        lr=args.qat_lr,
        betas=(0.8, 0.9),
        weight_decay=args.weight_decay,
    )

    batch_iter: Iterable[Dict[str, torch.Tensor]] = iter(train_loader)
    for step in range(1, args.qat_steps + 1):
        batch, batch_iter = _next_batch(batch_iter, train_loader)
        features = batch["features"].float()
        real = batch["audio"].float()
        features, real, batch_sanitized = sanitize_batch_tensors(features, real)
        if batch_sanitized:
            logger.warning(f"QAT step={step}: sanitized non-finite batch values before forward")

        optimizer.zero_grad(set_to_none=True)
        try:
            loss, stats = compute_generator_loss(qat_model, features, real, losses, args)
        except RuntimeError as exc:
            if "torch.histc: range of [nan, nan] is not finite" in str(exc):
                logger.warning(
                    f"QAT step={step}: non-finite observer histogram input; skipping batch and continuing"
                )
                fixed = stabilize_non_finite_parameters(qat_model)
                if fixed > 0:
                    logger.warning(f"QAT step={step}: repaired {fixed} non-finite parameter tensors")
                continue
            raise
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite QAT loss at step={step}: {stats}")
        loss.backward()
        if has_non_finite_gradients(qat_model):
            logger.warning(f"QAT step={step}: non-finite gradients detected; skipping optimizer step")
            optimizer.zero_grad(set_to_none=True)
            fixed = stabilize_non_finite_parameters(qat_model)
            if fixed > 0:
                logger.warning(f"QAT step={step}: repaired {fixed} non-finite parameter tensors")
            continue
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(qat_model.parameters(), max_norm=args.grad_clip_norm)
        optimizer.step()
        fixed = stabilize_non_finite_parameters(qat_model)
        if fixed > 0:
            logger.warning(f"QAT step={step}: repaired {fixed} non-finite parameter tensors after optimizer step")

        if step % max(1, args.log_every) == 0 or step == 1 or step == args.qat_steps:
            logger.info(
                "QAT step={step}/{max_steps} total={total:.4f} mrstft={mrstft:.4f} gd={gd:.4f} "
                "gan={gan:.4f} fm={fm:.4f}".format(
                    step=step,
                    max_steps=args.qat_steps,
                    total=stats["total"],
                    mrstft=stats["mrstft"],
                    gd=stats["group_delay"],
                    gan=stats["gan"],
                    fm=stats["fm"],
                )
            )

    qat_core.apply(disable_observer)
    qat_core.eval()
    converted_core = convert_pt2e(qat_core)
    allow_exported_model_train_eval(converted_core)
    converted_core.eval()
    pre_fp.eval()
    post_fp.eval()
    converted = QuantizedVocosInference(pre=pre_fp, core=converted_core, post=post_fp)
    return converted


def save_wav_16bit(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = audio.detach().cpu().float().flatten().numpy()
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm16 = (waveform * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def cast_floating_state_dict_to_fp16(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.is_floating_point():
            out[key] = value.half()
        else:
            out[key] = value
    return out


def build_slim_q8_state_dict(
    full_q8_state: Mapping[str, torch.Tensor],
    int8_model: nn.Module,
) -> tuple[Dict[str, torch.Tensor], set[str]]:
    keep_keys: set[str] = set()
    for key in full_q8_state.keys():
        if not key.startswith("core."):
            keep_keys.add(key)

    core = getattr(int8_model, "core", None)
    if core is None or not hasattr(core, "graph"):
        # Fallback: keep full state if core graph is unavailable.
        return dict(full_q8_state), set()

    core_keep_local: set[str] = set()
    for node in core.graph.nodes:
        if node.op == "get_attr":
            core_keep_local.add(str(node.target))

    # Keep only graph-referenced core attrs.
    for target in core_keep_local:
        full_key = f"core.{target}"
        if full_key in full_q8_state:
            keep_keys.add(full_key)

    slim_state: Dict[str, torch.Tensor] = {k: v for k, v in full_q8_state.items() if k in keep_keys}
    removed = set(full_q8_state.keys()) - set(slim_state.keys())
    return slim_state, removed


def verify_slim_q8_state_dict(
    int8_model: nn.Module,
    full_q8_state: Mapping[str, torch.Tensor],
    slim_state: Mapping[str, torch.Tensor],
    removed_keys: set[str],
    sample_features: torch.Tensor,
) -> None:
    ref_model = copy.deepcopy(int8_model).eval()
    ref_load = ref_model.load_state_dict(full_q8_state, strict=False)
    if ref_load.unexpected_keys:
        raise RuntimeError(f"Full q8 state_dict has unexpected keys: {sorted(ref_load.unexpected_keys)[:8]}")

    slim_model = copy.deepcopy(int8_model).eval()
    load_result = slim_model.load_state_dict(slim_state, strict=False)
    missing = set(load_result.missing_keys)
    unexpected = set(load_result.unexpected_keys)

    # Any missing key must be an intentionally removed key.
    if unexpected:
        raise RuntimeError(f"Slim q8 state_dict has unexpected keys: {sorted(unexpected)[:8]}")
    if not missing.issubset(removed_keys):
        extra_missing = sorted(missing - removed_keys)[:8]
        raise RuntimeError(f"Slim q8 verification failed; unexpected missing keys: {extra_missing}")

    with torch.inference_mode():
        ref = ref_model(sample_features.unsqueeze(0).float()).squeeze(0).float()
        got = slim_model(sample_features.unsqueeze(0).float()).squeeze(0).float()
    max_abs_err = torch.max(torch.abs(ref - got)).item()
    if max_abs_err > 1e-5:
        raise RuntimeError(f"Slim q8 verification mismatch: max_abs_err={max_abs_err:.6g}")


def count_weight_dtypes(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    counts = {"fp32": 0, "fp16": 0, "int8": 0}
    for value in state_dict.values():
        if not torch.is_tensor(value):
            continue
        n = int(value.numel())
        dtype = value.dtype
        if value.is_quantized:
            if dtype in (torch.qint8, torch.quint8):
                counts["int8"] += n
            elif dtype == torch.float16:
                counts["fp16"] += n
            elif dtype == torch.float32:
                counts["fp32"] += n
            continue

        if dtype == torch.float32:
            counts["fp32"] += n
        elif dtype in (torch.float16, torch.bfloat16):
            counts["fp16"] += n
        elif dtype in (torch.int8, torch.uint8):
            counts["int8"] += n
    return counts


def estimate_ops_for_50_frames(cfg: GeneratorConfig, frames: int = 50) -> Dict[str, Dict[str, int]]:
    # MAC estimate using Conv1d/Linear style terms over learned layers.
    t = int(frames)
    c_in = cfg.in_channels
    c_mid = cfg.model_input_channels
    d = cfg.backbone_dim
    d_int = cfg.backbone_intermediate_dim
    layers = cfg.backbone_layers
    head_out = cfg.n_fft + 2

    conditioner_ops = t * (c_mid * c_in + c_mid * c_mid * 3 + c_mid * c_mid)
    embed_ops = t * (d * c_mid * 7)
    convnext_per_layer = t * (d * 7 + d_int * d + d * d_int)
    convnext_ops = layers * convnext_per_layer
    head_ops = t * (head_out * d)

    total = conditioner_ops + embed_ops + convnext_ops + head_ops
    pre_ops = conditioner_ops + embed_ops
    post_ops = head_ops

    # Quantized pipeline currently: pre(float) -> core(int8) -> post(float).
    return {
        "original": {"fp32": total, "fp16": 0, "int8": 0},
        "fp16": {"fp32": 0, "fp16": total, "int8": 0},
        "int8": {"fp32": 0, "fp16": pre_ops + post_ops, "int8": convnext_ops},
    }


def format_int(n: int) -> str:
    return f"{int(n):,}"


def print_quant_summary_table(
    original_state: Mapping[str, torch.Tensor],
    fp16_state: Mapping[str, torch.Tensor],
    int8_state: Mapping[str, torch.Tensor],
    cfg: GeneratorConfig,
) -> None:
    weight_rows = {
        "original": count_weight_dtypes(original_state),
        "fp16": count_weight_dtypes(fp16_state),
        "int8": count_weight_dtypes(int8_state),
    }
    op_rows = estimate_ops_for_50_frames(cfg=cfg, frames=50)

    headers = [
        "model",
        "weights_fp32",
        "weights_fp16",
        "weights_int8",
        "ops_fp32@50f",
        "ops_fp16@50f",
        "ops_int8@50f",
    ]
    print()
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for model_name in ("original", "fp16", "int8"):
        w = weight_rows[model_name]
        o = op_rows[model_name]
        row = [
            model_name,
            format_int(w["fp32"]),
            format_int(w["fp16"]),
            format_int(w["int8"]),
            format_int(o["fp32"]),
            format_int(o["fp16"]),
            format_int(o["int8"]),
        ]
        print("| " + " | ".join(row) + " |")
    print()


def run_fp16_inference(
    cfg: GeneratorConfig,
    fp16_state: Mapping[str, torch.Tensor],
    features: torch.Tensor,
) -> torch.Tensor:
    half_model = PairedVocosGenerator(
        in_channels=cfg.in_channels,
        model_input_channels=cfg.model_input_channels,
        backbone_dim=cfg.backbone_dim,
        backbone_intermediate_dim=cfg.backbone_intermediate_dim,
        backbone_layers=cfg.backbone_layers,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        padding=cfg.padding,
        vocos_impl=cfg.vocos_impl,
        streaming_vocos_repo=cfg.streaming_vocos_repo,
        backbone_causal=cfg.backbone_causal,
        backbone_pad_mode=cfg.backbone_pad_mode,
        backbone_norm=cfg.backbone_norm,
    ).half().eval()
    half_model.load_state_dict(fp16_state, strict=True)
    with torch.inference_mode():
        try:
            return half_model(features.unsqueeze(0).half()).squeeze(0).float()
        except RuntimeError as exc:
            logger.warning(f"FP16 CPU inference failed ({exc}); falling back to fp16-rounded fp32 inference")

    rounded_state = {k: v.float() for k, v in fp16_state.items()}
    rounded_model = build_generator(cfg, rounded_state).eval()
    with torch.inference_mode():
        return rounded_model(features.unsqueeze(0).float()).squeeze(0)


def save_inference_samples(
    samples: list[InferenceSample],
    generator: nn.Module,
    cfg: GeneratorConfig,
    fp16_state: Mapping[str, torch.Tensor],
    int8_model: nn.Module,
    output_dir: Path,
    sample_rate: int,
) -> None:
    if not samples:
        logger.warning("No inference samples were selected, skipping sample-audio export")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    int8_model.eval()
    with torch.inference_mode():
        for sample in samples:
            features = sample.features
            pred_orig = generator(features.unsqueeze(0).float()).squeeze(0)
            pred_fp16 = run_fp16_inference(cfg=cfg, fp16_state=fp16_state, features=features)
            pred_int8 = int8_model(features.unsqueeze(0).float()).squeeze(0).float()
            save_wav_16bit(output_dir / f"{sample.tag}_orig.wav", pred_orig, sample_rate)
            save_wav_16bit(output_dir / f"{sample.tag}_fp16.wav", pred_fp16, sample_rate)
            save_wav_16bit(output_dir / f"{sample.tag}_int8.wav", pred_int8, sample_rate)


def main() -> None:
    logger.enable("prepare_weights")
    args = parse_args()
    if not USING_EXECUTORCH_XNNPACK:
        logger.warning(
            "ExecuTorch XNNPACK quantizer not found; using torchao fallback. "
            "Install optional deps: `uv sync --extra executorch-quant`"
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    ckpt, generator_state = load_checkpoint(input_path)
    config = infer_generator_config(generator_state, args=args)
    logger.info(
        "Inferred generator config: "
        f"vocos_impl={config.vocos_impl}, in_channels={config.in_channels}, "
        f"model_input_channels={config.model_input_channels}, backbone_dim={config.backbone_dim}, "
        f"layers={config.backbone_layers}, n_fft={config.n_fft}, hop_length={config.hop_length}"
    )
    generator = build_generator(config, generator_state).cpu().eval()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = out_dir / "vocos.pt"
    torch.save(generator.state_dict(), fp32_path)
    logger.info(f"Saved inference generator state_dict: {fp32_path}")

    fp16_state = {k: v.half() for k, v in generator.state_dict().items()}
    fp16_path = out_dir / "vocos_fp16.pt"
    torch.save(fp16_state, fp16_path)
    logger.info(f"Saved fp16 inference generator state_dict: {fp16_path}")

    train_loader = build_train_loader(args)
    inference_samples = build_inference_samples(
        dataset=train_loader.dataset,
        count=args.sample_count,
        seed=args.seed,
        max_frames=args.sample_max_frames,
    )
    losses = build_losses(args, ckpt)
    int8_model = run_qat(generator, train_loader, losses, args)

    q8_state = cast_floating_state_dict_to_fp16(int8_model.state_dict())

    sample_features_for_verify: torch.Tensor
    if inference_samples:
        sample_features_for_verify = inference_samples[0].features
    else:
        verify_batch = next(iter(train_loader))
        sample_features_for_verify = verify_batch["features"][0]

    q8_slim_state, removed_q8_keys = build_slim_q8_state_dict(q8_state, int8_model)
    verify_slim_q8_state_dict(
        int8_model=int8_model,
        full_q8_state=q8_state,
        slim_state=q8_slim_state,
        removed_keys=removed_q8_keys,
        sample_features=sample_features_for_verify,
    )
    q8_path = out_dir / "vocos_q8.pt"
    torch.save(q8_slim_state, q8_path)
    logger.info(
        "Saved quantized int8 inference state_dict (slim default): "
        f"{q8_path} (removed {len(removed_q8_keys)} keys)"
    )
    if args.save_full_q8:
        q8_full_path = out_dir / "vocos_q8_full.pt"
        torch.save(q8_state, q8_full_path)
        logger.info(f"Saved full quantized int8 inference state_dict: {q8_full_path}")
    # Release redundant full-q8 tensors before sample generation.
    del q8_state
    gc.collect()

    samples_dir = args.samples_dir.resolve() if args.samples_dir else (out_dir / "sample_audio")
    save_inference_samples(
        samples=inference_samples,
        generator=generator,
        cfg=config,
        fp16_state=fp16_state,
        int8_model=int8_model,
        output_dir=samples_dir,
        sample_rate=args.sample_rate,
    )
    logger.info(f"Saved inference comparison audio files to: {samples_dir}")
    print_quant_summary_table(
        original_state=generator.state_dict(),
        fp16_state=fp16_state,
        int8_state=q8_slim_state,
        cfg=config,
    )


if __name__ == "__main__":
    main()
