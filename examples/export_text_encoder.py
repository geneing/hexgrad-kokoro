"""
Export Kokoro TextEncoder sub-module to TFLite.

Produces:
  outputs/<git_hash>/kokoro_text_encoder_multisig_fp32.tflite
    Signatures: text_encoder_short (T=32), text_encoder_medium (T=128),
                text_encoder_long (T=256)
  outputs/<git_hash>/kokoro_text_encoder_Google_Tensor_G5.tflite  (AOT NPU)

Parity tests saved to:
  test_output/<git_hash>/text_encoder/<sig>_T<len>_pt.npy
  test_output/<git_hash>/text_encoder/<sig>_T<len>_tflite.npy

Run with:
  uv run python examples/export_text_encoder.py

Design decisions for export compatibility:
  1. Two tensor inputs: input_ids [1,T] (long) and mask [1,T] (bool).
     mask[i]=True marks position i as padding (zeroed out).
  2. LSTM: called directly — pack_padded_sequence / pad_packed_sequence are
     not torch.export-friendly. Bidirectional LSTM backward direction sees
     zero-padded positions; contamination decays via forget gate. Exact
     parity only guaranteed when mask is all-False (T_actual == T_bucket).
  3. masked_fill_ -> masked_fill (out-of-place, required for torch.export).
  4. weight_norm (new parametrizations API) left in place — fully traceable.
  5. flatten_parameters() omitted — CPU no-op.

AOT compilation requires:
  pip install ai-edge-litert-sdk-google-tensor==2.1.5
  (SDK plugin tar.gz at litert_npu/litert_plugin_compiler.tar.gz)
"""

import copy
import numpy as np
import os
import subprocess
import torch
import litert_torch

from kokoro import KModel

# Path to the Google Tensor SDK plugin (relative to project root)
_SDK_TAR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "litert_npu", "litert_plugin_compiler.tar.gz",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def assert_close(pt_out: torch.Tensor, tflite_out: np.ndarray,
                 name: str, atol: float = 2e-3) -> None:
    pt = pt_out.detach().float().numpy()
    diff = float(np.abs(pt - tflite_out).max())
    status = "PASS" if diff < atol else "FAIL"
    print(f"  {name}: max_abs_diff={diff:.6f}  {status}")
    assert diff < atol, f"{name} parity FAILED: max diff {diff:.6f} >= {atol}"


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class TextEncoderWrapper(torch.nn.Module):
    """Wraps TextEncoder for litert_torch export.

    forward(input_ids [1,T], mask [1,T]) -> [1, H, T]

    mask[i] = True means position i is padding and will be zeroed.
    For full-bucket inputs (no padding), pass mask = torch.zeros(1, T, dtype=torch.bool).
    """

    def __init__(self, text_encoder):
        super().__init__()
        self.embedding = text_encoder.embedding
        self.cnn = text_encoder.cnn
        self.lstm = text_encoder.lstm

    def forward(
        self,
        input_ids: torch.LongTensor,   # [1, T]
        mask: torch.BoolTensor,        # [1, T] — True = padding
    ) -> torch.FloatTensor:            # [1, H, T]
        x = self.embedding(input_ids)  # [1, T, C]
        x = x.transpose(1, 2)          # [1, C, T]
        m = mask.unsqueeze(1)          # [1, 1, T]

        # Zero out padding positions before CNN (out-of-place for torch.export)
        x = x.masked_fill(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x = x.masked_fill(m, 0.0)

        x = x.transpose(1, 2)  # [1, T, C]

        # Direct LSTM call — no pack_padded_sequence (not torch.export-friendly)
        # flatten_parameters() omitted (CPU no-op)
        x, _ = self.lstm(x)    # [1, T, H]

        x = x.transpose(-1, -2)  # [1, H, T]
        x = x.masked_fill(m, 0.0)
        return x


# ---------------------------------------------------------------------------
# AOT compile helper (shared pattern with export_bert.py)
# ---------------------------------------------------------------------------

def aot_compile_tensor_g5(tflite_path: str, out_dir: str, model_name: str) -> None:
    """AOT-compile a .tflite for the Pixel 10 (Google Tensor G5) NPU."""
    print("\n--- AOT compile for Google Tensor G5 ---")

    if os.path.isfile(_SDK_TAR):
        os.environ["GOOGLE_TENSOR_SDK_BETA"] = _SDK_TAR
        print(f"Using SDK plugin: {_SDK_TAR}")
    else:
        print(f"WARNING: SDK plugin not found at {_SDK_TAR} — skipping AOT.")
        return

    try:
        from ai_edge_litert.aot import aot_compile as aot_lib
        from ai_edge_litert.aot.vendors.google_tensor import target as gt_target
    except ImportError:
        print(
            "WARNING: ai-edge-litert-sdk-google-tensor not installed — skipping AOT.\n"
            "  Install with: GOOGLE_TENSOR_SDK_BETA=<path> pip install "
            "ai-edge-litert-sdk-google-tensor==2.1.5"
        )
        return

    tensor_g5_target = gt_target.Target(gt_target.SocModel.TENSOR_G5)

    print(f"Compiling {tflite_path} ...")
    compiled_models = aot_lib.aot_compile(
        tflite_path,
        target=[tensor_g5_target],
        keep_going=True,
        google_tensor_truncation_type="half",
        google_tensor_int64_to_int32=True,
        google_tensor_sharding_intensity="extensive",
    )

    print("\nCompilation report:")
    print(compiled_models.compilation_report())

    compiled_models.export(out_dir, model_name=model_name)
    print(f"AOT outputs written to: {out_dir}/")
    for fname in sorted(os.listdir(out_dir)):
        if (fname.startswith(model_name) and fname.endswith(".tflite")
                and "_multisig_" not in fname):
            size_mb = os.path.getsize(os.path.join(out_dir, fname)) / 1e6
            print(f"  {fname}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    HASH = git_hash()
    OUT_DIR = os.path.join("outputs", HASH)
    TEST_DIR = os.path.join("test_output", HASH, "text_encoder")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    TFLITE_PATH = os.path.join(OUT_DIR, "kokoro_text_encoder_multisig_fp32.tflite")

    # Bucket sequence lengths
    BUCKETS = [32, 128, 256]
    SIG_NAMES = ["text_encoder_short", "text_encoder_medium", "text_encoder_long"]

    # Parity tests:
    # EXACT (T == bucket) — LSTM direct call == pack/unpad, strict pass/fail
    # INFO  (T <  bucket) — bidirectional LSTM backward direction is contaminated
    #   by zero-padded positions (pack/unpad would start backward at T_actual-1;
    #   direct call starts at T_bucket-1 seeing zeros). Large diffs expected;
    #   printed for documentation but NOT counted as failures.
    #   Deployment constraint: call with T_actual == T_bucket (full bucket) for
    #   correct bidirectional output. Accept approximate quality otherwise.
    TEST_EXACT = [
        # (T_actual, sig, T_bucket)
        (32,  "text_encoder_short",  32),
        (128, "text_encoder_medium", 128),
        (256, "text_encoder_long",   256),
    ]
    TEST_INFO = [
        (10,  "text_encoder_short",  32),
        (80,  "text_encoder_medium", 128),
        (200, "text_encoder_long",   256),
    ]
    PARITY_ATOL = 2e-3
    PARITY_SEED = 42

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("Loading KModel...")
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config="checkpoints/config.json",
        model="checkpoints/kokoro-v1_0.pth",
    ).eval()

    # Deep-copy to avoid modifying the live model (weight_norm stays in place)
    te_copy = copy.deepcopy(model.text_encoder).eval()
    wrapper = TextEncoderWrapper(te_copy).eval()

    # Reference: original TextEncoder (for parity comparison)
    te_ref = model.text_encoder.eval()

    # -----------------------------------------------------------------------
    # Build multi-signature TFLite model
    # -----------------------------------------------------------------------
    print("Building multi-signature TFLite model...")

    def make_inputs(T: int):
        ids  = torch.randint(1, 178, (1, T), dtype=torch.long)
        mask = torch.zeros(1, T, dtype=torch.bool)
        return (ids, mask)

    first_inputs = make_inputs(BUCKETS[0])
    builder = litert_torch.signature(SIG_NAMES[0], wrapper, first_inputs)
    for name, T in zip(SIG_NAMES[1:], BUCKETS[1:]):
        builder = builder.signature(name, wrapper, make_inputs(T))

    edge_model = builder.convert()
    edge_model.export(TFLITE_PATH)
    print(f"Saved: {TFLITE_PATH}")

    # -----------------------------------------------------------------------
    # Parity tests
    # -----------------------------------------------------------------------
    print("\nRunning parity tests...")
    torch.manual_seed(PARITY_SEED)

    all_passed = True
    for T_actual, sig, T_bucket in TEST_EXACT + TEST_INFO:
        strict = (T_actual == T_bucket)
        padded = not strict
        label = f"{sig} T={T_actual}" + (" [APPROX]" if padded else "")

        # Padded inputs to bucket size
        ids_raw = torch.randint(1, 178, (1, T_actual), dtype=torch.long)
        ids  = torch.zeros(1, T_bucket, dtype=torch.long)
        mask = torch.zeros(1, T_bucket, dtype=torch.bool)
        ids[:, :T_actual] = ids_raw
        mask[:, T_actual:] = True  # mark padding

        # PyTorch reference — original TextEncoder with pack/unpad
        with torch.no_grad():
            ref_mask_1d = torch.zeros(1, T_bucket, dtype=torch.bool)
            ref_mask_1d[:, T_actual:] = True
            lengths = torch.tensor([T_actual])
            pt_out = te_ref(ids, lengths, ref_mask_1d)  # [1, H, T_bucket]
        pt_out_T = pt_out[:, :, :T_actual]              # [1, H, T_actual]

        # TFLite (wrapper — direct LSTM call)
        tflite_out = edge_model(ids, mask, signature_name=sig)  # [1, H, T_bucket]
        tflite_out_T = tflite_out[:, :, :T_actual]

        # Save tensors
        prefix = os.path.join(TEST_DIR, f"{sig}_T{T_actual}")
        np.save(prefix + "_pt.npy",     pt_out_T.float().numpy())
        np.save(prefix + "_tflite.npy", tflite_out_T)

        try:
            assert_close(pt_out_T, tflite_out_T, label, atol=PARITY_ATOL)
        except AssertionError as e:
            if strict:
                print(f"  ERROR: {e}")
                all_passed = False
            # padded cases: diff is expected and informational only

    if all_passed:
        print("\nAll parity tests PASSED.")
    else:
        print("\nSome parity tests FAILED — check test_output/ for tensors.")
        raise SystemExit(1)

    # -----------------------------------------------------------------------
    # AOT compile for Google Tensor G5
    # -----------------------------------------------------------------------
    aot_compile_tensor_g5(TFLITE_PATH, OUT_DIR, model_name="kokoro_text_encoder")


if __name__ == "__main__":
    main()
