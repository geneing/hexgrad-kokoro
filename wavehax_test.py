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
            framewise_norm=bool(config["framewise_norm"]),
            use_gradient_checkpointing=False,
        )

    def forward(self, features: Tensor) -> Tensor:
        cond = self.conditioner(features)
        f0 = features[:, 512:513, :]
        audio, _prior = self.generator(cond, f0)
        return audio[:, 0, :] if audio.ndim == 3 and audio.shape[1] == 1 else audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate Wavehax WAVs from Kokoro feature .pt files")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/wavehax/last.pt"))
    parser.add_argument("--tflite-model", type=Path, default=Path("runs/wavehax/wavehax_fp16_multisig_static_litert.tflite"))
    parser.add_argument("--input-feature-glob", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/wavehax_test"))
    parser.add_argument("--sample-rate", type=int, default=24000)
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


def run_tflite_inference(model_path: Path, features: np.ndarray) -> np.ndarray:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model_path))
    signatures = interpreter.get_signature_list()
    if signatures:
        frames = int(features.shape[-1])
        signature = f"frames_{frames}"
        if signature not in signatures:
            signature = "serving_default" if "serving_default" in signatures else next(iter(signatures.keys()))
        runner = interpreter.get_signature_runner(signature)
        input_details = runner.get_input_details()
        input_name = next(iter(input_details.keys()))
        input_dtype = np.dtype(input_details[input_name]["dtype"])
        output_map = runner(**{input_name: np.asarray(features, dtype=input_dtype)})
        return np.asarray(next(iter(output_map.values())), dtype=np.float32)

    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    input_shape = tuple(int(dim) for dim in input_detail["shape"])
    if input_shape != tuple(features.shape):
        interpreter.resize_tensor_input(input_detail["index"], features.shape, strict=False)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_detail["index"], np.asarray(features, dtype=input_detail["dtype"]))
    interpreter.invoke()
    output_detail = interpreter.get_output_details()[0]
    return np.asarray(interpreter.get_tensor(output_detail["index"]), dtype=np.float32)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    if args.input_feature_glob is None:
        args.input_feature_glob = ["data/af_*.pt"]
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = feature_paths_from_globs(args.input_feature_glob)
    if not paths:
        raise RuntimeError(f"No feature files matched: {args.input_feature_glob}")

    torch.set_grad_enabled(False)
    model = load_wavehax_model(args.checkpoint.resolve(), args.sample_rate, device)
    tflite_model = args.tflite_model.resolve()
    if not tflite_model.exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_model}")
    print(f"Using TFLite model: {tflite_model}")

    with torch.inference_mode():
        for path in paths:
            features = compose_features_from_pt(path).unsqueeze(0).to(device)
            torch_audio = model(features).detach().cpu().numpy()
            torch_path = args.output_dir / f"{path.stem}_pytorch.wav"
            save_wav_16bit(torch_path, torch_audio, args.sample_rate)
            torch_rms, torch_peak = audio_stats(torch_audio)
            print(f"Wrote {torch_path} shape={tuple(torch_audio.shape)} rms={torch_rms:.6f} peak={torch_peak:.6f}")

            input_np = features.detach().cpu().numpy().astype(np.float32)
            tflite_audio = run_tflite_inference(tflite_model, input_np)
            tflite_path = args.output_dir / f"{path.stem}_tflite.wav"
            save_wav_16bit(tflite_path, tflite_audio, args.sample_rate)
            tflite_rms, tflite_peak = audio_stats(tflite_audio)
            rms_ratio = tflite_rms / torch_rms if torch_rms > 0.0 else 0.0
            diff_rms = float(np.sqrt(np.mean((tflite_audio.reshape(-1) - torch_audio.reshape(-1)) ** 2)))
            print(
                f"Wrote {tflite_path} shape={tuple(tflite_audio.shape)} "
                f"rms={tflite_rms:.6f} peak={tflite_peak:.6f} "
                f"rms_ratio={rms_ratio:.3f} diff_rms={diff_rms:.6f}"
            )


if __name__ == "__main__":
    main()
