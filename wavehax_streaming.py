#!/usr/bin/env python3
"""Generate Wavehax PyTorch WAVs from saved Kokoro feature tensors."""

from __future__ import annotations

import argparse
import glob
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parent
WAVEHAX_ROOT = ROOT / "third_party" / "wavehax"
for path in (ROOT, WAVEHAX_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wavehax.generators.wavehax import MultiScaleWavehaxGenerator


class KokoroFeatureConditioner(nn.Module):
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

    def forward(self, features: Tensor) -> Tensor:
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


class KokoroMultiScaleWavehaxGenerator(nn.Module):
    def __init__(self, config: Mapping[str, object], sample_rate: int):
        super().__init__()
        self.conditioner = KokoroFeatureConditioner(
            out_channels=int(config["model_input_channels"]),
            control_channels=int(config["control_channels"]),
            control_layers=int(config["control_layers"]),
        )
        self.generator = MultiScaleWavehaxGenerator(
            in_channels=int(config["model_input_channels"]),
            channels=int(config["channels"]),
            mult_channels=int(config["mult_channels"]),
            kernel_size=int(config["kernel_size"]),
            num_blocks=int(config["num_blocks"]),
            decomposer=str(config["decomposer"]),
            num_splits=int(config["num_splits"]),
            n_fft=int(config["n_fft"]),
            hop_length=int(config["hop_length"]),
            sample_rate=int(sample_rate),
            prior_type=str(config["prior_type"]),
            drop_prob=float(config["drop_prob"]),
            framewise_norm=bool(config.get("framewise_norm", True)),
            use_gradient_checkpointing=False,
            norm_type=str(config.get("norm_type", "layer")),
            padding_mode=str(config.get("padding_mode", "reflect")),
            export_safe_ops=bool(config.get("export_safe_ops", False)),
        )

    def forward(
        self,
        features: Tensor,
        prior_phase: Tensor | None = None,
        return_prior_phase: bool = False,
    ) -> Tensor:
        cond = self.conditioner(features)
        f0 = features[:, 512:513, :]
        result = self.generator(cond, f0, prior_phase=prior_phase, return_prior_phase=return_prior_phase)
        if return_prior_phase:
            audio, _prior, next_phase = result
            audio = audio[:, 0, :] if audio.ndim == 3 and audio.shape[1] == 1 else audio
            return audio, next_phase
        audio, _prior = result
        return audio[:, 0, :] if audio.ndim == 3 and audio.shape[1] == 1 else audio

    def phase_advance(self, features: Tensor) -> Tensor:
        return self.generator.phase_advance(features[:, 512:513, :])


class StreamingWavehaxChunk(nn.Module):
    """Fixed-shape center-window chunk wrapper with explicit state tensors."""

    def __init__(
        self,
        model: KokoroMultiScaleWavehaxGenerator,
        chunk_frames: int = 24,
        feature_channels: int = 642,
        sample_rate: int = 24000,
        hop_length: int = 300,
    ):
        super().__init__()
        self.model = model.eval()
        self.chunk_frames = int(chunk_frames)
        self.feature_channels = int(feature_channels)
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.chunk_samples = self.chunk_frames * self.hop_length

    def initial_state(self, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        feature_state = torch.zeros(1, self.feature_channels, self.chunk_frames, device=device, dtype=dtype)
        prior_phase = torch.zeros(1, 1, 1, device=device, dtype=dtype)
        return feature_state, prior_phase

    def forward(
        self,
        features_chunk: Tensor,
        next_features_chunk: Tensor,
        feature_state: Tensor,
        prior_phase: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if features_chunk.shape != (1, self.feature_channels, self.chunk_frames):
            raise ValueError(
                f"Expected features_chunk [1,{self.feature_channels},{self.chunk_frames}], got {tuple(features_chunk.shape)}"
            )
        if next_features_chunk.shape != (1, self.feature_channels, self.chunk_frames):
            raise ValueError(
                f"Expected next_features_chunk [1,{self.feature_channels},{self.chunk_frames}], got {tuple(next_features_chunk.shape)}"
            )
        full_features = torch.cat([feature_state, features_chunk, next_features_chunk], dim=-1)
        full_audio, _next_window_phase = self.model(full_features, prior_phase=prior_phase, return_prior_phase=True)
        audio_chunk = full_audio[..., self.chunk_samples : 2 * self.chunk_samples]
        next_feature_state = features_chunk.contiguous()
        next_prior_phase = torch.fmod(prior_phase + self.model.phase_advance(features_chunk), 2.0 * torch.pi)
        return audio_chunk, next_feature_state, next_prior_phase


class StreamingKokoroWavehax:
    def __init__(self, model: KokoroMultiScaleWavehaxGenerator, chunk_frames: int, sample_rate: int, hop_length: int):
        self.chunk = StreamingWavehaxChunk(
            model=model,
            chunk_frames=chunk_frames,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )

    @torch.inference_mode()
    def synthesize(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError(f"Expected features [1,642,T], got {tuple(features.shape)}")
        device = features.device
        dtype = features.dtype
        feature_state, prior_phase = self.chunk.initial_state(device=device, dtype=dtype)
        outputs: list[Tensor] = []
        frames = int(features.shape[-1])
        pos = 0
        while pos < frames:
            valid = min(self.chunk.chunk_frames, frames - pos)
            chunk = features[..., pos : pos + valid]
            if valid < self.chunk.chunk_frames:
                chunk = F.pad(chunk, (0, self.chunk.chunk_frames - valid))
            next_start = pos + valid
            next_valid = min(self.chunk.chunk_frames, max(0, frames - next_start))
            next_chunk = features[..., next_start : next_start + next_valid]
            if next_valid < self.chunk.chunk_frames:
                next_chunk = F.pad(next_chunk, (0, self.chunk.chunk_frames - next_valid))
            audio, feature_state, prior_phase = self.chunk(
                chunk,
                next_chunk,
                feature_state,
                prior_phase,
            )
            outputs.append(audio[..., : valid * self.chunk.hop_length])
            pos += valid
        return torch.cat(outputs, dim=-1)[..., : frames * self.chunk.hop_length] if outputs else torch.empty(1, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate Wavehax WAVs from Kokoro feature .pt files using PyTorch")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/wavehax/last.pt"))
    parser.add_argument("--input-feature-glob", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("test_output"))
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=300)
    parser.add_argument("--chunk-frames", type=int, default=24)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def save_wav_16bit(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    waveform = np.nan_to_num(waveform)
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm16 = (waveform * 32767.0).astype(np.int16)
    wavfile.write(path, int(sample_rate), pcm16)


def audio_stats(audio: np.ndarray) -> tuple[float, float]:
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    waveform = np.nan_to_num(waveform)
    rms = float(np.sqrt(np.mean(waveform * waveform))) if waveform.size else 0.0
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    return rms, peak


def comparison_metrics(reference: np.ndarray, candidate: np.ndarray, chunk_samples: int) -> dict[str, float]:
    ref = np.nan_to_num(np.asarray(reference, dtype=np.float32).reshape(-1))
    pred = np.nan_to_num(np.asarray(candidate, dtype=np.float32).reshape(-1))
    n = min(ref.size, pred.size)
    ref = ref[:n]
    pred = pred[:n]
    diff = pred - ref
    corr = float(np.corrcoef(ref, pred)[0, 1]) if n > 1 and np.std(ref) > 0 and np.std(pred) > 0 else 0.0
    boundary = 0.0
    if chunk_samples > 0 and pred.size > chunk_samples:
        jumps = []
        for idx in range(chunk_samples, pred.size, chunk_samples):
            jumps.append(abs(float(pred[idx] - pred[idx - 1])))
        boundary = float(np.mean(jumps)) if jumps else 0.0
    return {
        "mae": float(np.mean(np.abs(diff))) if n else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if n else 0.0,
        "corr": corr,
        "boundary_click": boundary,
    }


def feature_paths_from_globs(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(pattern))
    return sorted(dict.fromkeys(path.resolve() for path in paths))


def compose_features_from_pt(path: Path) -> torch.Tensor:
    row = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(row, Mapping):
        raise TypeError(f"Expected mapping in {path}, got {type(row)}")

    asr = row["asr"].float()
    f0 = row["f0"].float()
    noise = row["noise"].float()
    style = row["style"].float()
    total_frames = int(f0.shape[-1])
    if asr.shape[-1] != total_frames:
        asr = F.interpolate(asr.unsqueeze(0), size=total_frames, mode="linear", align_corners=False).squeeze(0)

    return torch.cat(
        [
            asr[:, :total_frames],
            f0[:total_frames].unsqueeze(0),
            noise[:total_frames].unsqueeze(0),
            style.unsqueeze(-1).expand(style.shape[0], total_frames),
        ],
        dim=0,
    ).contiguous()


def load_wavehax_model(checkpoint_path: Path, sample_rate: int, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, Mapping):
        raise TypeError(f"Expected checkpoint mapping in {checkpoint_path}, got {type(ckpt)}")
    if "backend_config" not in ckpt or "generator" not in ckpt:
        raise KeyError(f"Expected backend_config and generator entries in {checkpoint_path}")

    model = KokoroMultiScaleWavehaxGenerator(dict(ckpt["backend_config"]), sample_rate=sample_rate).eval()
    model.load_state_dict(ckpt["generator"], strict=True)
    return model.to(device)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    if args.input_feature_glob is None:
        args.input_feature_glob = ["data/af*.pt"]
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = feature_paths_from_globs(args.input_feature_glob)
    if not paths:
        raise RuntimeError(f"No feature files matched: {args.input_feature_glob}")

    torch.set_grad_enabled(False)
    model = load_wavehax_model(args.checkpoint.resolve(), args.sample_rate, device)

    with torch.inference_mode():
        for path in paths:
            features = compose_features_from_pt(path).unsqueeze(0).to(device)
            torch_audio = model(features).detach().cpu().numpy()
            torch_path = args.output_dir / f"{path.stem}_pytorch.wav"
            save_wav_16bit(torch_path, torch_audio, args.sample_rate)
            torch_rms, torch_peak = audio_stats(torch_audio)
            print(f"Wrote {torch_path} shape={tuple(torch_audio.shape)} rms={torch_rms:.6f} peak={torch_peak:.6f}")

            streamer = StreamingKokoroWavehax(
                model=model,
                chunk_frames=args.chunk_frames,
                sample_rate=args.sample_rate,
                hop_length=args.hop_length,
            )
            stream_audio = streamer.synthesize(features).detach().cpu().numpy()
            stream_path = args.output_dir / f"{path.stem}_streaming_{args.chunk_frames}f.wav"
            save_wav_16bit(stream_path, stream_audio, args.sample_rate)
            stream_rms, stream_peak = audio_stats(stream_audio)
            metrics = comparison_metrics(torch_audio, stream_audio, args.chunk_frames * args.hop_length)
            print(
                f"Wrote {stream_path} shape={tuple(stream_audio.shape)} "
                f"rms={stream_rms:.6f} peak={stream_peak:.6f} "
                f"mae={metrics['mae']:.6f} rmse={metrics['rmse']:.6f} "
                f"corr={metrics['corr']:.6f} boundary={metrics['boundary_click']:.6f}"
            )


if __name__ == "__main__":
    main()
