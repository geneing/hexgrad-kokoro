"""
Export Kokoro BERT sub-module (CustomAlbert + bert_encoder linear) to TFLite.

Produces:
  outputs/<git_hash>/kokoro_bert_multisig_fp32.tflite
    Signatures: bert_short (T=32), bert_medium (T=128),
                bert_long (T=256), bert_max (T=510)
  outputs/<git_hash>/kokoro_bert_fallback.tflite      (AOT CPU/GPU fallback)
  outputs/<git_hash>/kokoro_bert_Google_Tensor_G5.tflite  (AOT NPU)

Parity tests saved to:
  test_output/<git_hash>/bert/<sig>_pt.npy   - PyTorch reference output
  test_output/<git_hash>/bert/<sig>_tflite.npy - TFLite output

Run with:
  uv run python export/export_bert.py

AOT compilation requires:
  pip install ai-edge-litert-sdk-google-tensor==2.1.5
  (SDK plugin tar.gz at litert_npu/litert_plugin_compiler.tar.gz)
"""

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
                 name: str, atol: float = 1e-4) -> None:
    pt = pt_out.detach().float().numpy()
    diff = float(np.abs(pt - tflite_out).max())
    print(f"  {name}: max_abs_diff={diff:.6f}  {'PASS' if diff < atol else 'FAIL'}")
    assert diff < atol, f"{name} parity FAILED: max diff {diff:.6f} >= {atol}"


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class BertWrapper(torch.nn.Module):
    """
    Wraps CustomAlbert + bert_encoder into a single positional-arg module.

    CustomAlbert.forward already returns last_hidden_state directly
    (not the full HuggingFace BaseModelOutput).

    forward(input_ids [B,T], attention_mask [B,T]) -> [B, H, T]
    """
    def __init__(self, bert, bert_encoder):
        super().__init__()
        self.bert = bert
        self.bert_encoder = bert_encoder

    def forward(
        self,
        input_ids: torch.LongTensor,       # [B, T]
        attention_mask: torch.LongTensor,  # [B, T]
    ) -> torch.FloatTensor:                # [B, H, T]
        hidden = self.bert(input_ids, attention_mask=attention_mask)  # [B, T, H_bert]
        return self.bert_encoder(hidden).transpose(-1, -2)            # [B, H, T]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    HASH = git_hash()
    OUT_DIR = os.path.join("outputs", HASH)
    TEST_DIR = os.path.join("test_output", HASH, "bert")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    TFLITE_PATH = os.path.join(OUT_DIR, "kokoro_bert_multisig_fp32.tflite")

    # Bucket sequence lengths
    BUCKETS = [32, 128, 256, 510]
    SIG_NAMES = ["bert_short", "bert_medium", "bert_long", "bert_max"]

    # Test lengths — use one per bucket plus an intermediate value
    TEST_LENS = [10, 32, 100, 128, 200, 256, 400, 510]
    PARITY_ATOL = 2e-3   # TFLite/XNNPACK vs PyTorch fp32 tolerance
    PARITY_SEED = 42     # fixed seed for reproducible parity checks

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("Loading KModel...")
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config="checkpoints/config.json",
        model="checkpoints/kokoro-v1_0.pth",
    ).eval()

    wrapper = BertWrapper(model.bert, model.bert_encoder).eval()

    # -----------------------------------------------------------------------
    # Build multi-signature edge model
    # -----------------------------------------------------------------------
    print("Building multi-signature TFLite model...")

    def make_inputs(seq_len: int):
        return (
            torch.randint(0, 178, (1, seq_len), dtype=torch.long),
            torch.ones(1, seq_len, dtype=torch.long),
        )

    first_sig_name = SIG_NAMES[0]
    first_inputs = make_inputs(BUCKETS[0])
    builder = litert_torch.signature(first_sig_name, wrapper, first_inputs)

    for name, seq_len in zip(SIG_NAMES[1:], BUCKETS[1:]):
        builder = builder.signature(name, wrapper, make_inputs(seq_len))

    edge_model = builder.convert()
    edge_model.export(TFLITE_PATH)
    print(f"Saved: {TFLITE_PATH}")

    # -----------------------------------------------------------------------
    # Parity tests
    # -----------------------------------------------------------------------
    print("\nRunning parity tests...")
    torch.manual_seed(PARITY_SEED)

    def pick_sig(T: int) -> tuple[str, int]:
        """Return (sig_name, bucket_len) for input length T."""
        for sig, bucket in zip(SIG_NAMES, BUCKETS):
            if T <= bucket:
                return sig, bucket
        return SIG_NAMES[-1], BUCKETS[-1]

    all_passed = True
    for T in TEST_LENS:
        sig, bucket = pick_sig(T)

        # Pad to bucket length
        ids_raw  = torch.randint(1, 178, (1, T), dtype=torch.long)
        mask_raw = torch.ones(1, T, dtype=torch.long)

        ids  = torch.zeros(1, bucket, dtype=torch.long)
        mask = torch.zeros(1, bucket, dtype=torch.long)
        ids[:, :T]  = ids_raw
        mask[:, :T] = mask_raw

        # PyTorch reference (padded inputs, slice output back to T)
        with torch.no_grad():
            pt_out = wrapper(ids, mask)  # [1, H, bucket]
        pt_out_T = pt_out[:, :, :T]     # [1, H, T] — valid tokens only

        # TFLite
        tflite_out = edge_model(ids, mask, signature_name=sig)  # [1, H, bucket]
        tflite_out_T = tflite_out[:, :, :T]

        # Save tensors
        prefix = os.path.join(TEST_DIR, f"{sig}_T{T}")
        np.save(prefix + "_pt.npy", pt_out_T.float().numpy())
        np.save(prefix + "_tflite.npy", tflite_out_T)

        try:
            assert_close(pt_out_T, tflite_out_T, f"{sig} T={T}", atol=PARITY_ATOL)
        except AssertionError as e:
            print(f"  ERROR: {e}")
            all_passed = False

    if all_passed:
        print("\nAll parity tests PASSED.")
    else:
        print("\nSome parity tests FAILED — check test_output/ for tensors.")
        raise SystemExit(1)

    # -----------------------------------------------------------------------
    # AOT compile for Google Tensor G5
    # -----------------------------------------------------------------------
    aot_compile_tensor_g5(TFLITE_PATH, OUT_DIR, model_name="kokoro_bert")


def aot_compile_tensor_g5(tflite_path: str, out_dir: str, model_name: str) -> None:
    """AOT-compile a .tflite for the Pixel 10 (Google Tensor G5) NPU.

    Requires ai-edge-litert-sdk-google-tensor==2.1.5.  If the package or the
    SDK plugin tar.gz is missing the step is skipped with a warning so that
    the rest of the export pipeline is not blocked.
    """
    print("\n--- AOT compile for Google Tensor G5 ---")

    # The SDK plugin must be registered via env-var BEFORE the aot module is
    # imported, so we set it here and import lazily.
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
            "  Install with: pip install ai-edge-litert-sdk-google-tensor==2.1.5"
        )
        return

    tensor_g5_target = gt_target.Target(gt_target.SocModel.TENSOR_G5)

    print(f"Compiling {tflite_path} ...")
    compiled_models = aot_lib.aot_compile(
        tflite_path,
        target=[tensor_g5_target],
        keep_going=True,
        # fp16 weights — best throughput on GPU/NPU
        google_tensor_truncation_type="half",
        # BERT uses int64 token indices; cast to int32 for NPU
        google_tensor_int64_to_int32=True,
        # Maximise parallelism across EdgeTPU cores
        google_tensor_sharding_intensity="extensive",
    )

    print("\nCompilation report:")
    print(compiled_models.compilation_report())

    compiled_models.export(out_dir, model_name=model_name)
    print(f"AOT outputs written to: {out_dir}/")
    for fname in sorted(os.listdir(out_dir)):
        if fname.startswith(model_name) and fname.endswith(".tflite") and "_multisig_" not in fname:
            size_mb = os.path.getsize(os.path.join(out_dir, fname)) / 1e6
            print(f"  {fname}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
