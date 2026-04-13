import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import iree.runtime as ireert
import matplotlib
import numpy as np
import scipy.io.wavfile as wavfile

from export_parts import (
    AUDIO_SAMPLES_PER_FRAME,
    KModel,
    KModelForONNX,
    prepare_part_inputs,
)


DEFAULT_VMFB_SUFFIX = ".fp16.vmfb"
DEFAULT_ONNX_SUFFIX = ".fp16.onnx"
DEFAULT_MLIR_SUFFIX = ".fp16.mlir"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_OPSET_VERSION = 17
DEFAULT_VULKAN_TARGET = "valhall"
MODEL_PARTS = ("bert", "duration_predictor", "text_encoder")


def to_numpy(value):
    if isinstance(value, tuple):
        return tuple(to_numpy(item) for item in value)
    return np.asarray(value)


def normalize_outputs(outputs):
    if isinstance(outputs, tuple):
        return tuple(to_numpy(item) for item in outputs)
    return (to_numpy(outputs),)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def prepare_iree_artifacts(
    onnx_dir: Path,
    onnx_suffix: str = DEFAULT_ONNX_SUFFIX,
    mlir_suffix: str = DEFAULT_MLIR_SUFFIX,
    vmfb_suffix: str = DEFAULT_VMFB_SUFFIX,
    opset_version: int = DEFAULT_OPSET_VERSION,
    vulkan_target: str = DEFAULT_VULKAN_TARGET,
    force: bool = False,
) -> None:
    if shutil.which("uv") is None:
        raise RuntimeError("`uv` is required to prepare IREE artifacts but was not found on PATH")

    onnx_dir.mkdir(parents=True, exist_ok=True)

    for part in MODEL_PARTS:
        onnx_path = require_file(onnx_dir / f"{part}{onnx_suffix}")
        mlir_path = onnx_dir / f"{part}{mlir_suffix}"
        vmfb_path = onnx_dir / f"{part}{vmfb_suffix}"

        if force or not mlir_path.exists():
            run_command(
                [
                    "uv",
                    "run",
                    "iree-import-onnx",
                    str(onnx_path),
                    "--opset-version",
                    str(opset_version),
                    "-o",
                    str(mlir_path),
                ]
            )
        else:
            print(f"Skipping existing MLIR: {mlir_path}")

        if force or not vmfb_path.exists():
            run_command(
                [
                    "uv",
                    "run",
                    "iree-compile",
                    "--iree-hal-target-device=vulkan",
                    "--iree-hal-target-backends=vulkan-spirv",
                    f"--iree-vulkan-target={vulkan_target}",
                    "-o",
                    str(vmfb_path),
                    str(require_file(mlir_path)),
                ]
            )
        else:
            print(f"Skipping existing VMFB: {vmfb_path}")


def load_iree_module(config: ireert.Config, vmfb_path: Path):
    vm_module = ireert.VmModule.mmap(config.vm_instance, str(require_file(vmfb_path)))
    return ireert.load_vm_modules(vm_module, config=config)[0]


def save_audio_plot(audio: np.ndarray, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(audio)
    plt.savefig(output_dir / f"{stem}.png")
    plt.close()

    wavfile.write(output_dir / f"{stem}.wav", DEFAULT_SAMPLE_RATE, (audio * 32767).astype(np.int16))


def inference_iree_parts(
    model,
    vmfb_dir: Path,
    device_uri: str = "vulkan",
    text: str | None = None,
    vmfb_suffix: str = DEFAULT_VMFB_SUFFIX,
):
    config = ireert.Config(device_uri)

    bert_module = load_iree_module(config, vmfb_dir / f"bert{vmfb_suffix}")
    duration_module = load_iree_module(config, vmfb_dir / f"duration_predictor{vmfb_suffix}")
    text_encoder_module = load_iree_module(config, vmfb_dir / f"text_encoder{vmfb_suffix}")

    input_ids, style, speed, input_length, text_mask = prepare_part_inputs(model, text=text)

    input_ids_np = input_ids.numpy().astype(np.int64, copy=False)
    style_np = style.numpy().astype(np.float32, copy=False)
    speed_np = speed.numpy().astype(np.int32, copy=False)
    input_length_np = input_length.numpy().astype(np.int32, copy=False)
    text_mask_np = text_mask.numpy().astype(np.float32, copy=False)

    d_en = normalize_outputs(bert_module.main_graph(input_ids_np, text_mask_np))[0]

    pred_dur, d, expanded_indices, en, t_en, expanded_length = normalize_outputs(
        duration_module.main_graph(input_ids_np, d_en, style_np, input_length_np, speed_np)
    )

    audio, asr, f0_pred, n_pred = normalize_outputs(
        text_encoder_module.main_graph(en, style_np, expanded_indices, t_en, expanded_length)
    )
    expanded_length_value = int(np.asarray(expanded_length).reshape(()))
    audio = audio[: expanded_length_value * AUDIO_SAMPLES_PER_FRAME]

    print(
        "inference_iree_parts:",
        f"device={device_uri}",
        f"pred_dur.shape={pred_dur.shape}",
        f"d.shape={d.shape}",
        f"expanded_indices.shape={expanded_indices.shape}",
        f"en.shape={en.shape}",
        f"t_en.shape={t_en.shape}",
        f"expanded_length={expanded_length_value}",
        f"audio.shape={audio.shape}",
        f"asr.shape={asr.shape}",
        f"f0_pred.shape={f0_pred.shape}",
        f"n_pred.shape={n_pred.shape}",
    )

    return audio


def main():
    parser = argparse.ArgumentParser("Run Kokoro part-wise inference with IREE Vulkan modules")
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        default="checkpoints/config.json",
        help="Path to Kokoro config file",
    )
    parser.add_argument(
        "--checkpoint_path",
        "-p",
        type=str,
        default="checkpoints/kokoro-v1_0.pth",
        help="Path to Kokoro checkpoint",
    )
    parser.add_argument(
        "--vmfb_dir",
        "-v",
        type=str,
        default="onnx",
        help="Directory containing compiled IREE part modules",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="onnx",
        help="Directory to write waveform and plot outputs",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="vulkan",
        help="IREE device URI, for example vulkan or vulkan://0",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Optional input text override",
    )
    parser.add_argument(
        "--vmfb_suffix",
        type=str,
        default=DEFAULT_VMFB_SUFFIX,
        help="Suffix appended to bert/duration_predictor/text_encoder module names",
    )
    parser.add_argument(
        "--onnx_suffix",
        type=str,
        default=DEFAULT_ONNX_SUFFIX,
        help="Suffix appended to ONNX part names during IREE import",
    )
    parser.add_argument(
        "--mlir_suffix",
        type=str,
        default=DEFAULT_MLIR_SUFFIX,
        help="Suffix appended to MLIR part names during IREE import/compile",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=DEFAULT_OPSET_VERSION,
        help="Opset version passed to iree-import-onnx",
    )
    parser.add_argument(
        "--vulkan_target",
        type=str,
        default=DEFAULT_VULKAN_TARGET,
        help="Value passed to --iree-vulkan-target during compile",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Convert FP16 ONNX parts to MLIR and compile them to Vulkan VMFBs before inference",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only build MLIR and VMFB artifacts, then exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild MLIR and VMFB artifacts even if they already exist",
    )

    args = parser.parse_args()

    if args.prepare or args.prepare_only:
        prepare_iree_artifacts(
            onnx_dir=Path(args.vmfb_dir),
            onnx_suffix=args.onnx_suffix,
            mlir_suffix=args.mlir_suffix,
            vmfb_suffix=args.vmfb_suffix,
            opset_version=args.opset_version,
            vulkan_target=args.vulkan_target,
            force=args.force,
        )

    if args.prepare_only:
        return

    kmodel = KModel(config=args.config_file, model=args.checkpoint_path, disable_complex=True)
    model = KModelForONNX(kmodel).eval()

    audio = inference_iree_parts(
        model=model,
        vmfb_dir=Path(args.vmfb_dir),
        device_uri=args.device,
        text=args.text,
        vmfb_suffix=args.vmfb_suffix,
    )
    save_audio_plot(np.asarray(audio).squeeze(), Path(args.output_dir), "iree_parts_test")


if __name__ == "__main__":
    main()
