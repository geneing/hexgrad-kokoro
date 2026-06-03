"""
Export and run an end-to-end Kokoro TFLite pipeline with separate recurrent
TFLite LSTM submodels.

The recurrent submodels are produced via export.export_fused_lstm_tf, which
currently lowers Keras Bidirectional(LSTM) to compact TFLite WHILE recurrent
subgraphs rather than static unrolled litert_torch graphs.

Run:
  uv run python export/export_hybrid_fused_lstm_pipeline.py

Outputs:
  outputs/<git_hash>/hybrid_fused_lstm/*.tflite
  test_output/<git_hash>/hybrid_fused_lstm/summary.tsv
  test_output/<git_hash>/hybrid_fused_lstm/wavs/*.wav
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile

# TensorFlow conversion/inference does not need GPU. The local driver/runtime
# combination can fail if TensorFlow tries to initialize CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import litert_torch
from ai_edge_litert import interpreter

from kokoro import KModel
from kokoro.modules import AdaLayerNorm
from export.export_fused_lstm_tf import build_keras_lstm_model, convert_to_tflite
from export.export_predictor_f0n import _replace_pool_layers


SAMPLE_RATE = 24000


@dataclass(frozen=True)
class Bucket:
    name: str
    size: int


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


def max_abs_diff(reference: np.ndarray, actual: np.ndarray) -> float:
    return float(np.max(np.abs(reference.astype(np.float32) - actual.astype(np.float32))))


def rms(array: np.ndarray) -> float:
    array = array.astype(np.float32)
    return float(np.sqrt(np.mean(array * array))) if array.size else 0.0


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    samples = np.clip(samples, -1.0, 1.0)
    wavfile.write(path, SAMPLE_RATE, (samples * 32767.0).astype(np.int16))


def pad_axis(array: np.ndarray, axis: int, target: int, value=0) -> np.ndarray:
    if array.shape[axis] > target:
        raise ValueError(f"Cannot pad axis {axis} from {array.shape[axis]} to {target}")
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, target - array.shape[axis])
    return np.pad(array, pad_width, mode="constant", constant_values=value)


class SignatureTFLite:
    def __init__(self, path: Path):
        self.path = path
        self.interpreter = interpreter.Interpreter(model_path=str(path))
        self.signatures = self.interpreter.get_signature_list()

    def run(self, signature_name: str, *args: np.ndarray):
        runner = self.interpreter.get_signature_runner(signature_name)
        outputs = runner(**{f"args_{index}": arg for index, arg in enumerate(args)})
        values = [outputs[name] for name in sorted(outputs)]
        return values[0] if len(values) == 1 else tuple(values)


class SingleTFLite:
    def __init__(self, path: Path):
        self.path = path
        self.interpreter = interpreter.Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.ops = [op["op_name"] for op in self.interpreter._get_ops_details()]

    def run(self, x: np.ndarray) -> np.ndarray:
        detail = self.input_details[0]
        self.interpreter.set_tensor(detail["index"], x.astype(detail["dtype"]))
        self.interpreter.invoke()
        out = self.output_details[0]
        return self.interpreter.get_tensor(out["index"])


class TextEncoderFrontend(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.embedding = text_encoder.embedding
        self.cnn = text_encoder.cnn

    def forward(self, input_ids: torch.LongTensor, mask: torch.BoolTensor):
        x = self.embedding(input_ids).transpose(1, 2)
        m = mask.unsqueeze(1)
        x = x.masked_fill(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x = x.masked_fill(m, 0.0)
        return x.transpose(1, 2)


class TextEncoderPost(torch.nn.Module):
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor):
        m = mask.unsqueeze(1)
        return x.transpose(1, 2).masked_fill(m, 0.0)


class DurationAdaConcatCT(torch.nn.Module):
    def __init__(self, block: AdaLayerNorm):
        super().__init__()
        self.block = block

    def forward(self, x: torch.FloatTensor, style: torch.FloatTensor):
        # Mirror DurationEncoder exactly at the AdaLayerNorm boundary:
        # input/output are channel-first temporal tensors [B, C, T].
        y = self.block(x.transpose(-1, -2), style).transpose(-1, -2)
        s = style.unsqueeze(-1).expand(x.shape[0], -1, x.shape[-1])
        return torch.cat([y, s], dim=1)


class DurationProj(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.duration_proj = predictor.duration_proj

    def forward(self, x: torch.FloatTensor):
        return self.duration_proj(x)


class F0NHeads(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.F0_blocks = predictor.F0
        self.N_blocks = predictor.N
        self.F0_proj = predictor.F0_proj
        self.N_proj = predictor.N_proj

    def forward(self, x: torch.FloatTensor, style: torch.FloatTensor):
        h = x.transpose(1, 2)
        f0 = h
        for block in self.F0_blocks:
            f0 = block(f0, style)
        f0 = self.F0_proj(f0).squeeze(1)

        n = h
        for block in self.N_blocks:
            n = block(n, style)
        n = self.N_proj(n).squeeze(1)
        return f0, n


def add_signatures(first_name: str, module: torch.nn.Module, first_inputs, rest):
    builder = litert_torch.signature(first_name, module, first_inputs)
    for name, inputs in rest:
        builder = builder.signature(name, module, inputs)
    return builder.convert()


def export_signature_module(path: Path, names: list[str], module: torch.nn.Module, make_inputs) -> None:
    if path.exists():
        print(f"Reusing: {path}")
        return
    edge_model = add_signatures(
        names[0],
        module.eval(),
        make_inputs(names[0]),
        [(name, make_inputs(name)) for name in names[1:]],
    )
    edge_model.export(str(path))
    print(f"Saved: {path}")


def export_lstm_models(model: KModel, token_lengths: list[int], aligned_lengths: list[int], out_dir: Path) -> dict[str, dict[int, Path]]:
    specs = {
        "text_encoder_lstm": (model.text_encoder.lstm, token_lengths),
        "predictor_lstm": (model.predictor.lstm, token_lengths),
        "shared_f0n_lstm": (model.predictor.shared, aligned_lengths),
    }
    duration_lstm_index = 0
    for block in model.predictor.text_encoder.lstms:
        if not isinstance(block, AdaLayerNorm):
            specs[f"duration_lstm_{duration_lstm_index}"] = (block, token_lengths)
            duration_lstm_index += 1
    result: dict[str, dict[int, Path]] = {}
    for name, (lstm, lengths) in specs.items():
        result[name] = {}
        for length in lengths:
            path = out_dir / f"kokoro_hybrid_{name}_T{length}_fp32.tflite"
            if not path.exists():
                keras_model = build_keras_lstm_model(lstm, length)
                convert_to_tflite(keras_model, str(path))
                print(f"Saved: {path}")
            else:
                print(f"Reusing: {path}")
            result[name][length] = path
    return result


def export_non_lstm_modules(model: KModel, token_lengths: list[int], aligned_lengths: list[int], out_dir: Path) -> dict[str, Path]:
    names_by_t = [f"T{length}" for length in token_lengths]
    names_by_a = [f"T{length}" for length in aligned_lengths]

    paths = {
        "text_frontend": out_dir / "kokoro_hybrid_text_frontend_fp32.tflite",
        "text_post": out_dir / "kokoro_hybrid_text_post_fp32.tflite",
        "duration_proj": out_dir / "kokoro_hybrid_duration_proj_fp32.tflite",
        "f0n_heads": out_dir / "kokoro_hybrid_f0n_heads_fp32.tflite",
    }
    duration_ada_blocks = [
        block for block in model.predictor.text_encoder.lstms if isinstance(block, AdaLayerNorm)
    ]
    for index in range(len(duration_ada_blocks)):
        paths[f"duration_ada_{index}"] = (
            out_dir / f"kokoro_hybrid_duration_ada_{index}_ct_fp32.tflite"
        )

    export_signature_module(
        paths["text_frontend"],
        names_by_t,
        TextEncoderFrontend(model.text_encoder),
        lambda name: (
            torch.randint(1, 178, (1, int(name[1:])), dtype=torch.long),
            torch.zeros(1, int(name[1:]), dtype=torch.bool),
        ),
    )
    export_signature_module(
        paths["text_post"],
        names_by_t,
        TextEncoderPost(),
        lambda name: (
            torch.randn(1, int(name[1:]), 512),
            torch.zeros(1, int(name[1:]), dtype=torch.bool),
        ),
    )
    for index, block in enumerate(duration_ada_blocks):
        export_signature_module(
            paths[f"duration_ada_{index}"],
            names_by_t,
            DurationAdaConcatCT(block),
            lambda name: (torch.randn(1, 512, int(name[1:])), torch.randn(1, 128)),
        )
    export_signature_module(
        paths["duration_proj"],
        names_by_t,
        DurationProj(model.predictor),
        lambda name: (torch.randn(1, int(name[1:]), 512),),
    )

    predictor_copy = model.predictor
    _replace_pool_layers(predictor_copy)
    export_signature_module(
        paths["f0n_heads"],
        names_by_a,
        F0NHeads(predictor_copy),
        lambda name: (torch.randn(1, int(name[1:]), 512), torch.randn(1, 128)),
    )
    return paths


def build_alignment(pred_dur: np.ndarray) -> np.ndarray:
    pred_dur = pred_dur.astype(np.int64).reshape(-1)
    indices = np.repeat(np.arange(pred_dur.shape[0]), pred_dur)
    aln = np.zeros((1, pred_dur.shape[0], indices.shape[0]), dtype=np.float32)
    aln[0, indices, np.arange(indices.shape[0])] = 1.0
    return aln


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("test_output/baseline/tensors"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs"))
    parser.add_argument("--test-root", type=Path, default=Path("test_output"))
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    hash_ = git_hash()
    out_dir = args.out_root / hash_ / "hybrid_fused_lstm"
    test_dir = args.test_root / hash_ / "hybrid_fused_lstm"
    wav_dir = test_dir / "wavs"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    token_lengths, aligned_lengths = baseline_lengths(args.baseline)
    print(f"Token lengths: {token_lengths}")
    print(f"Aligned lengths: {aligned_lengths}")

    print("Loading KModel...")
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config="checkpoints/config.json",
        model="checkpoints/kokoro-v1_0.pth",
        disable_complex=True,
    ).eval()

    non_lstm_paths = export_non_lstm_modules(model, token_lengths, aligned_lengths, out_dir)
    lstm_paths = export_lstm_models(model, token_lengths, aligned_lengths, out_dir)

    bert = SignatureTFLite(Path("outputs/e7de808/kokoro_bert_multisig_fp32.tflite"))
    decoder = SignatureTFLite(Path("outputs/c46f2e1/kokoro_decoder_multisig_fp32.tflite"))
    non_lstm = {name: SignatureTFLite(path) for name, path in non_lstm_paths.items()}
    lstms = {
        name: {length: SingleTFLite(path) for length, path in by_len.items()}
        for name, by_len in lstm_paths.items()
    }

    rows: list[dict[str, str]] = []
    for chunk_dir in sorted(args.baseline.glob("line_*/chunk_*")):
        case = f"{chunk_dir.parent.name}_{chunk_dir.name}"
        inputs = np.load(chunk_dir / "inputs.npz")
        bert_ref = np.load(chunk_dir / "bert.npz")
        text_ref = np.load(chunk_dir / "text_encoder.npz")
        dur_ref = np.load(chunk_dir / "predictor_dur.npz")
        f0n_ref = np.load(chunk_dir / "predictor_f0n.npz")
        dec_ref = np.load(chunk_dir / "decoder.npz")

        ids = inputs["input_ids"].astype(np.int64)
        mask = inputs["text_mask"].astype(bool)
        attn = inputs["attention_mask"].astype(np.int64)
        style_pred = inputs["style_predictor"].astype(np.float32)
        style_dec = inputs["style_decoder"].astype(np.float32)
        token_len = ids.shape[1]

        # BERT uses existing TFLite bucket and crops back to exact token length.
        bert_bucket = "bert_short" if token_len <= 32 else "bert_medium" if token_len <= 128 else "bert_long" if token_len <= 256 else "bert_max"
        bert_size = {"bert_short": 32, "bert_medium": 128, "bert_long": 256, "bert_max": 510}[bert_bucket]
        d_en = bert.run(
            bert_bucket,
            pad_axis(ids, 1, bert_size).astype(np.int64),
            pad_axis(attn, 1, bert_size).astype(np.int64),
        )[:, :, :token_len]

        # TextEncoder: frontend TFLite -> compact recurrent LSTM TFLite -> post TFLite.
        sig = f"T{token_len}"
        text_front = non_lstm["text_frontend"].run(sig, ids, mask)
        text_lstm = lstms["text_encoder_lstm"][token_len].run(text_front)
        text_out = non_lstm["text_post"].run(sig, text_lstm, mask)

        # DurationEncoder: compact LSTM / AdaConcat / compact LSTM / AdaConcat.
        x0 = np.concatenate(
            [np.transpose(d_en, (0, 2, 1)), np.broadcast_to(style_pred[:, None, :], (1, token_len, 128))],
            axis=-1,
        ).astype(np.float32)
        duration_input = x0
        duration_layer = 0
        while f"duration_lstm_{duration_layer}" in lstms:
            lstm_out = lstms[f"duration_lstm_{duration_layer}"][token_len].run(duration_input)
            ada_ct = non_lstm[f"duration_ada_{duration_layer}"].run(
                sig, np.transpose(lstm_out, (0, 2, 1)), style_pred
            )
            duration_input = np.transpose(ada_ct, (0, 2, 1))
            duration_layer += 1
        duration_encoded = duration_input
        predictor_lstm = lstms["predictor_lstm"][token_len].run(duration_encoded)
        duration_logits = non_lstm["duration_proj"].run(sig, predictor_lstm)
        duration_scores = 1.0 / (1.0 + np.exp(-duration_logits))
        duration_scores = np.sum(duration_scores, axis=-1) / args.speed
        pred_dur = np.rint(duration_scores).clip(min=1).astype(np.int64).reshape(-1)
        pred_aln = build_alignment(pred_dur)
        aligned_len = pred_aln.shape[-1]

        predictor_aligned = np.matmul(np.transpose(duration_encoded, (0, 2, 1)), pred_aln)
        asr = np.matmul(text_out, pred_aln)

        # F0/N: compact shared LSTM TFLite -> heads TFLite.
        f0n_lstm_input = np.transpose(predictor_aligned, (0, 2, 1)).astype(np.float32)
        shared = lstms["shared_f0n_lstm"][aligned_len].run(f0n_lstm_input)
        f0, n = non_lstm["f0n_heads"].run(f"T{aligned_len}", shared, style_pred)

        # Decoder: existing TFLite decoder bucket.
        dec_bucket = "decoder_short" if aligned_len <= 200 else "decoder_medium" if aligned_len <= 800 else "decoder_long"
        dec_size = {"decoder_short": 200, "decoder_medium": 800, "decoder_long": 2000}[dec_bucket]
        audio = decoder.run(
            dec_bucket,
            pad_axis(asr, 2, dec_size).astype(np.float32),
            pad_axis(f0, 1, dec_size * 2).astype(np.float32),
            pad_axis(n, 1, dec_size * 2).astype(np.float32),
            style_dec,
        ).reshape(-1)
        audio = audio[: dec_ref["audio"].shape[0]]

        case_dir = test_dir / case
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / "bert_tflite.npy", d_en)
        np.save(case_dir / "text_encoder_tflite.npy", text_out)
        np.save(case_dir / "duration_encoded_tflite.npy", duration_encoded)
        np.save(case_dir / "duration_logits_tflite.npy", duration_logits)
        np.save(case_dir / "pred_dur_tflite.npy", pred_dur)
        np.save(case_dir / "predictor_aligned_tflite.npy", predictor_aligned)
        np.save(case_dir / "asr_tflite.npy", asr)
        np.save(case_dir / "f0_tflite.npy", f0)
        np.save(case_dir / "n_tflite.npy", n)
        np.save(case_dir / "audio_tflite.npy", audio)

        write_wav(wav_dir / f"{case}_hybrid_fused_lstm_tflite.wav", audio)

        checks = [
            ("bert", "output", bert_ref["output"], d_en, 2e-3),
            ("text_encoder", "output", text_ref["output"], text_out, 2e-3),
            ("predictor_dur", "duration_encoded", dur_ref["duration_encoded"], duration_encoded, 2e-3),
            ("predictor_dur", "logits", dur_ref["logits"], duration_logits, 2e-3),
            ("predictor_f0n", "f0", f0n_ref["f0"], f0[:, : f0n_ref["f0"].shape[1]], 2e-3),
            ("predictor_f0n", "n", f0n_ref["n"], n[:, : f0n_ref["n"].shape[1]], 2e-3),
        ]
        for module, tensor, ref, actual, atol in checks:
            diff = max_abs_diff(ref, actual[:, : ref.shape[1], ...] if actual.ndim == 3 and ref.ndim == 3 else actual)
            rows.append(
                {
                    "case": case,
                    "module": module,
                    "tensor": tensor,
                    "max_abs_diff": f"{diff:.8g}",
                    "status": "PASS" if diff < atol else "FAIL",
                    "note": "",
                }
            )
        ref_audio = dec_ref["audio"].reshape(-1)
        ratio = max(rms(ref_audio), rms(audio)) / (min(rms(ref_audio), rms(audio)) + 1e-9)
        rows.append(
            {
                "case": case,
                "module": "decoder",
                "tensor": "audio",
                "max_abs_diff": f"{max_abs_diff(ref_audio, audio):.8g}",
                "status": "PASS" if np.isfinite(audio).all() and ratio < 10.0 else "FAIL",
                "note": f"rms_ratio={ratio:.4g}; aligned_len={aligned_len}",
            }
        )

    summary_path = test_dir / "summary.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "module", "tensor", "max_abs_diff", "status", "note"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {summary_path}")
    for row in rows:
        print(
            f"{row['status']:>4} {row['case']:<16} {row['module']:<15} "
            f"{row['tensor']:<18} {row['max_abs_diff']:<12} {row['note']}"
        )

    failed = [row for row in rows if row["status"] == "FAIL"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
