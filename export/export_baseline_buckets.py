"""
Export baseline-compatible Kokoro TFLite submodule signatures.

The original Step 2/3 exports intentionally only guaranteed exact parity for
full-bucket LSTM inputs. This script adds signatures for the actual
export/test.txt baseline chunk lengths so real baseline parity does not depend
on padded bidirectional LSTM behavior.

Produces:
  outputs/<git_hash>/kokoro_text_encoder_baseline_fp32.tflite
  outputs/<git_hash>/kokoro_predictor_dur_baseline_fp32.tflite
  outputs/<git_hash>/kokoro_predictor_f0n_baseline_fp32.tflite

Run with:
  uv run python export/export_baseline_buckets.py
"""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import numpy as np
import torch
import litert_torch

from kokoro import KModel
from export.export_text_encoder import TextEncoderWrapper
from export.export_predictor_dur import PredictorDurWrapper
from export.export_predictor_f0n import PredictorF0NWrapper, _replace_pool_layers


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


def baseline_lengths(root: Path) -> tuple[list[int], list[int]]:
    token_lengths: set[int] = set()
    aligned_lengths: set[int] = set()
    for inputs_path in root.glob("line_*/chunk_*/inputs.npz"):
        chunk_dir = inputs_path.parent
        token_lengths.add(int(np.load(inputs_path)["input_ids"].shape[1]))
        aligned_lengths.add(int(np.load(chunk_dir / "decoder.npz")["asr"].shape[-1]))
    if not token_lengths or not aligned_lengths:
        raise FileNotFoundError(f"No baseline tensors found under {root}")
    return sorted(token_lengths), sorted(aligned_lengths)


def signature_name(prefix: str, length: int) -> str:
    return f"{prefix}_T{length}"


def add_signatures(first_name: str, module: torch.nn.Module, first_inputs, rest):
    builder = litert_torch.signature(first_name, module, first_inputs)
    for name, inputs in rest:
        builder = builder.signature(name, module, inputs)
    return builder.convert()


def export_text_encoder(model: KModel, token_lengths: list[int], out_dir: Path) -> Path:
    lengths = sorted(set([32, 128, 256, *token_lengths]))
    wrapper = TextEncoderWrapper(copy.deepcopy(model.text_encoder).eval()).eval()

    def make_inputs(length: int):
        ids = torch.randint(1, 178, (1, length), dtype=torch.long)
        mask = torch.zeros(1, length, dtype=torch.bool)
        return ids, mask

    names = [signature_name("text_encoder", length) for length in lengths]
    edge_model = add_signatures(
        names[0],
        wrapper,
        make_inputs(lengths[0]),
        [(name, make_inputs(length)) for name, length in zip(names[1:], lengths[1:])],
    )
    path = out_dir / "kokoro_text_encoder_baseline_fp32.tflite"
    edge_model.export(str(path))
    return path


def export_predictor_dur(model: KModel, token_lengths: list[int], out_dir: Path) -> Path:
    lengths = sorted(set([32, 128, 256, *token_lengths]))
    wrapper = PredictorDurWrapper(copy.deepcopy(model.predictor).eval()).eval()

    def make_inputs(length: int):
        return torch.randn(1, 512, length), torch.randn(1, 128)

    names = [signature_name("predictor_dur", length) for length in lengths]
    edge_model = add_signatures(
        names[0],
        wrapper,
        make_inputs(lengths[0]),
        [(name, make_inputs(length)) for name, length in zip(names[1:], lengths[1:])],
    )
    path = out_dir / "kokoro_predictor_dur_baseline_fp32.tflite"
    edge_model.export(str(path))
    return path


def export_predictor_f0n(model: KModel, aligned_lengths: list[int], out_dir: Path) -> Path:
    lengths = sorted(set([200, 800, *aligned_lengths]))
    predictor = copy.deepcopy(model.predictor).eval()
    _replace_pool_layers(predictor)
    wrapper = PredictorF0NWrapper(predictor).eval()

    def make_inputs(length: int):
        return torch.randn(1, 640, length), torch.randn(1, 128)

    names = [signature_name("predictor_f0n", length) for length in lengths]
    edge_model = add_signatures(
        names[0],
        wrapper,
        make_inputs(lengths[0]),
        [(name, make_inputs(length)) for name, length in zip(names[1:], lengths[1:])],
    )
    path = out_dir / "kokoro_predictor_f0n_baseline_fp32.tflite"
    edge_model.export(str(path))
    return path


def main() -> None:
    baseline_root = Path("test_output/baseline/tensors")
    token_lengths, aligned_lengths = baseline_lengths(baseline_root)
    out_dir = Path("outputs") / git_hash()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Baseline token lengths: {token_lengths}")
    print(f"Baseline aligned lengths: {aligned_lengths}")
    print("Loading KModel...")
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config="checkpoints/config.json",
        model="checkpoints/kokoro-v1_0.pth",
        disable_complex=True,
    ).eval()

    print("Exporting TextEncoder baseline signatures...")
    print(f"Saved: {export_text_encoder(model, token_lengths, out_dir)}")
    print("Exporting PredictorDur baseline signatures...")
    print(f"Saved: {export_predictor_dur(model, token_lengths, out_dir)}")
    print("Exporting PredictorF0N baseline signatures...")
    print(f"Saved: {export_predictor_f0n(model, aligned_lengths, out_dir)}")


if __name__ == "__main__":
    main()
