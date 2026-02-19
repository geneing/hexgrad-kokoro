from __future__ import annotations

import argparse
import random
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.ao.quantization import allow_exported_model_train_eval
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
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

    parser.add_argument("--data-root", type=Path, default=Path("/export/eingerman/audio/vocoder"))
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


def infer_generator_config(state_dict: Mapping[str, torch.Tensor], hop_length: int, padding: str) -> GeneratorConfig:
    try:
        model_input_channels = int(state_dict["conditioner.0.weight"].shape[0])
        in_channels = int(state_dict["conditioner.0.weight"].shape[1])
        backbone_dim = int(state_dict["backbone.embed.weight"].shape[0])
        backbone_intermediate_dim = int(state_dict["backbone.convnext.0.pwconv1.weight"].shape[0])
        n_fft = int(state_dict["head.out.weight"].shape[0]) - 2
    except KeyError as exc:
        raise KeyError(f"Could not infer generator config, missing key: {exc}") from exc

    layer_ids = set()
    pattern = re.compile(r"^backbone\.convnext\.(\d+)\.dwconv\.weight$")
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
        n_fft=n_fft,
        hop_length=hop_length,
        padding=padding,
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
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(
            is_per_channel=True,
            is_qat=True,
        )
    )
    qat_core = prepare_qat_pt2e(exported_core, quantizer)
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

        optimizer.zero_grad(set_to_none=True)
        loss, stats = compute_generator_loss(qat_model, features, real, losses, args)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite QAT loss at step={step}: {stats}")
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(qat_model.parameters(), max_norm=args.grad_clip_norm)
        optimizer.step()

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

    qat_core.apply(torch.ao.quantization.disable_observer)
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    ckpt, generator_state = load_checkpoint(input_path)
    config = infer_generator_config(generator_state, hop_length=args.hop_length, padding=args.padding)
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

    q8_path = out_dir / "vocos_q8.pt"
    q8_state = cast_floating_state_dict_to_fp16(int8_model.state_dict())
    torch.save(q8_state, q8_path)
    logger.info(f"Saved quantized int8 inference state_dict: {q8_path}")

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


if __name__ == "__main__":
    main()
