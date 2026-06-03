"""
Prototype fused TFLite LSTM export for Kokoro's PyTorch nn.LSTM layers.

This script bypasses litert_torch for bare LSTM layers:
  1. Build an equivalent tf.keras Bidirectional(LSTM) model.
  2. Copy PyTorch LSTM weights into Keras gate layout.
  3. Convert with tf.lite.TFLiteConverter.
  4. Verify PyTorch vs Keras vs TFLite parity.
  5. Print the FlatBuffer op list so compact recurrent lowering is visible.

Run:
  uv run python export/export_fused_lstm_tf.py

Outputs:
  outputs/<git_hash>/fused_lstm/kokoro_fused_lstm_<name>_T<seq>_fp32.tflite
  test_output/<git_hash>/fused_lstm/<name>_T<seq>_*.npy
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass

import numpy as np
import torch

# TensorFlow conversion does not need GPU, and the local CUDA driver/runtime
# combination can fail during Keras variable initialization.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class LstmTarget:
    name: str
    module_path: str


TARGETS = [
    LstmTarget("text_encoder", "text_encoder.lstm"),
    LstmTarget("predictor_duration_encoder_0", "predictor.text_encoder.lstms.0"),
    LstmTarget("predictor_duration_encoder_1", "predictor.text_encoder.lstms.2"),
    LstmTarget("predictor_duration_encoder_2", "predictor.text_encoder.lstms.4"),
    LstmTarget("predictor_predictor_lstm", "predictor.lstm"),
    LstmTarget("predictor_shared_f0n", "predictor.shared"),
]


def get_module(root: torch.nn.Module, path: str) -> torch.nn.Module:
    mod = root
    for part in path.split("."):
        if part.isdigit():
            mod = mod[int(part)]  # type: ignore[index]
        else:
            mod = getattr(mod, part)
    return mod


def _torch_lstm_direction_weights(
    lstm: torch.nn.LSTM,
    suffix: str,
) -> list[np.ndarray]:
    """Return Keras [kernel, recurrent_kernel, bias] for one LSTM direction."""
    weight_ih = getattr(lstm, f"weight_ih_l0{suffix}").detach().cpu().numpy()
    weight_hh = getattr(lstm, f"weight_hh_l0{suffix}").detach().cpu().numpy()
    bias_ih = getattr(lstm, f"bias_ih_l0{suffix}").detach().cpu().numpy()
    bias_hh = getattr(lstm, f"bias_hh_l0{suffix}").detach().cpu().numpy()

    # PyTorch and Keras both use gate order i, f, cell, o for LSTM weights.
    kernel = weight_ih.T.astype(np.float32)
    recurrent_kernel = weight_hh.T.astype(np.float32)
    bias = (bias_ih + bias_hh).astype(np.float32)
    return [kernel, recurrent_kernel, bias]


def build_keras_lstm_model(lstm: torch.nn.LSTM, seq_len: int):
    import tensorflow as tf

    if lstm.num_layers != 1:
        raise ValueError(f"Only single-layer LSTMs are supported, got {lstm.num_layers}")
    if not lstm.batch_first:
        raise ValueError("This prototype expects batch_first=True")
    if lstm.proj_size != 0:
        raise ValueError("Projected LSTM is not supported in this prototype")

    input_size = int(lstm.input_size)
    hidden_size = int(lstm.hidden_size)

    x = tf.keras.Input(shape=(seq_len, input_size), batch_size=1, name="x")

    def make_layer(name: str, go_backwards: bool):
        return tf.keras.layers.LSTM(
            hidden_size,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            unit_forget_bias=False,
            return_sequences=True,
            go_backwards=go_backwards,
            dropout=0.0,
            recurrent_dropout=0.0,
            unroll=False,
            name=name,
        )

    if lstm.bidirectional:
        fw = make_layer("forward_lstm", go_backwards=False)
        bw = make_layer("backward_lstm", go_backwards=True)
        y = tf.keras.layers.Bidirectional(
            fw,
            backward_layer=bw,
            merge_mode="concat",
            name="bidirectional_lstm",
        )(x)
        model = tf.keras.Model(x, y, name="kokoro_fused_bidir_lstm")
        model.get_layer("bidirectional_lstm").forward_layer.set_weights(
            _torch_lstm_direction_weights(lstm, "")
        )
        model.get_layer("bidirectional_lstm").backward_layer.set_weights(
            _torch_lstm_direction_weights(lstm, "_reverse")
        )
    else:
        y = make_layer("lstm", go_backwards=False)(x)
        model = tf.keras.Model(x, y, name="kokoro_fused_lstm")
        model.get_layer("lstm").set_weights(_torch_lstm_direction_weights(lstm, ""))

    return model


def convert_to_tflite(keras_model, out_path: str) -> bytes:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations = []
    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    return tflite_model


def run_tflite(tflite_model: bytes, x_np: np.ndarray) -> tuple[np.ndarray, list[str]]:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_detail["index"], x_np.astype(input_detail["dtype"]))
    interpreter.invoke()
    y = interpreter.get_tensor(output_detail["index"])
    ops = [op["op_name"] for op in interpreter._get_ops_details()]
    return y, ops


def assert_close(name: str, ref: np.ndarray, actual: np.ndarray, atol: float) -> float:
    diff = float(np.max(np.abs(ref.astype(np.float32) - actual.astype(np.float32))))
    status = "PASS" if diff < atol else "FAIL"
    print(f"  {name}: max_abs_diff={diff:.6f}  {status}")
    if diff >= atol:
        raise AssertionError(f"{name} parity FAILED: {diff:.6f} >= {atol}")
    return diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--target",
        action="append",
        choices=[target.name for target in TARGETS],
        help="Export only the named target. May be passed multiple times.",
    )
    parser.add_argument(
        "--require-sequence-lstm",
        action="store_true",
        help="Fail unless the FlatBuffer op list contains a TFLite LSTM op.",
    )
    args = parser.parse_args()

    # Import after argument parsing so --help remains lightweight.
    import tensorflow as tf
    from kokoro import KModel

    np.random.seed(42)
    torch.manual_seed(42)
    tf.random.set_seed(42)

    hash_ = git_hash()
    out_dir = os.path.join("outputs", hash_, "fused_lstm")
    test_dir = os.path.join("test_output", hash_, "fused_lstm")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Loading KModel...")
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config="checkpoints/config.json",
        model="checkpoints/kokoro-v1_0.pth",
    ).eval()

    selected = {name for name in args.target} if args.target else None
    targets = [target for target in TARGETS if selected is None or target.name in selected]

    all_ok = True
    for target in targets:
        print(f"\n--- {target.name} ({target.module_path}) ---")
        lstm = get_module(model, target.module_path).eval()
        if not isinstance(lstm, torch.nn.LSTM):
            raise TypeError(f"{target.module_path} is not nn.LSTM: {type(lstm)}")

        x = torch.randn(1, args.seq_len, lstm.input_size)
        with torch.no_grad():
            torch_out, _ = lstm(x)
        torch_np = torch_out.detach().cpu().numpy().astype(np.float32)

        keras_model = build_keras_lstm_model(lstm, args.seq_len)
        keras_np = keras_model(x.detach().cpu().numpy(), training=False).numpy()

        tflite_path = os.path.join(
            out_dir, f"kokoro_fused_lstm_{target.name}_T{args.seq_len}_fp32.tflite"
        )
        tflite_model = convert_to_tflite(keras_model, tflite_path)
        tflite_np, ops = run_tflite(tflite_model, x.detach().cpu().numpy())

        prefix = os.path.join(test_dir, f"{target.name}_T{args.seq_len}")
        np.save(prefix + "_input.npy", x.detach().cpu().numpy())
        np.save(prefix + "_torch.npy", torch_np)
        np.save(prefix + "_keras.npy", keras_np)
        np.save(prefix + "_tflite.npy", tflite_np)

        try:
            assert_close(f"{target.name} torch vs keras", torch_np, keras_np, args.atol)
            assert_close(f"{target.name} torch vs tflite", torch_np, tflite_np, args.atol)
        except AssertionError as exc:
            print(f"  ERROR: {exc}")
            all_ok = False

        op_summary = ", ".join(ops)
        fused = any("LSTM" in op for op in ops)
        compact_recurrent = fused or any(op == "WHILE" for op in ops)
        print(f"  saved: {tflite_path}")
        print(f"  ops: {op_summary}")
        print(f"  fused_lstm_detected={fused}")
        print(f"  compact_recurrent_detected={compact_recurrent}")
        if args.require_sequence_lstm and not fused:
            all_ok = False
        elif not compact_recurrent:
            all_ok = False

    if not all_ok:
        raise SystemExit(1)

    print("\nAll fused LSTM prototype exports passed.")


if __name__ == "__main__":
    main()
