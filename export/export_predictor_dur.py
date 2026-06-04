"""
Export Kokoro ProsodyPredictor duration head to TFLite (Step 3a).

Covers:
  predictor.text_encoder (DurationEncoder)  +
  predictor.lstm                            +
  predictor.duration_proj

Produces:
  outputs/<git_hash>/kokoro_predictor_dur_multisig_fp32.tflite
    Signatures: predictor_dur_short (T=32), predictor_dur_medium (T=128),
                predictor_dur_long (T=256)

Note: AOT NPU compilation is intentionally skipped for this module.
  The duration head contains LSTM layers (DurationEncoder + predictor.lstm)
  which take excessive time (>20 min) to compile on the Tensor G5 plugin.
  This module will run on CPU/GPU fallback at inference time.

Wrapper inputs:
  d_en  [1, H=512, T]   — BERT + bert_encoder output (same T as phoneme seq)
  style [1, style=128]  — ref_s[:, 128:] from voice style vector

Wrapper outputs:
  duration [1, T, max_dur=50]  — duration logits (sigmoid + sum on CPU)
  d        [1, T, H+style=640] — DurationEncoder output; CPU computes
                                   en = d.transpose(-1,-2) @ pred_aln_trg
                                   for the F0Ntrain step (3b)

On-device pipeline after this step:
  1. (NPU) run predictor_dur → (duration, d)
  2. (CPU) duration → sigmoid → sum → round → pred_dur
  3. (CPU) pred_aln_trg = build_alignment(pred_dur)        [dynamic, must stay on CPU]
  4. (CPU) en = d.transpose(-1,-2) @ pred_aln_trg          [cheap matmul]
  5. (CPU) t_en @ pred_aln_trg  → asr                      [from TextEncoder output]
  6. (NPU) run predictor_f0n(en, style)  → F0_pred, N_pred  (Step 3b)
  7. (NPU) run decoder(asr, F0_pred, N_pred, ref_s[:,:128]) (Step 4)

Export compatibility changes vs original:
  - pack_padded_sequence / pad_packed_sequence → direct LSTM calls
  - masked_fill_  → masked_fill  (out-of-place; required by torch.export)
  - flatten_parameters() omitted (CPU no-op)
  - isinstance(block, AdaLayerNorm) evaluated statically during trace (fine)
  - Full-bucket inputs only: exact parity. Padded inputs: bidir LSTM backward
    direction contaminated (same limitation as TextEncoder, Step 2).

Run with:
  uv run python export/export_predictor_dur.py
"""

import copy
import numpy as np
import os
import subprocess
import torch
import torch.nn.functional as F
import litert_torch

from kokoro import KModel
from kokoro.modules import AdaLayerNorm

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


def assert_close(pt: torch.Tensor, tflite: np.ndarray,
                 name: str, atol: float = 2e-3) -> None:
    pt_np = pt.detach().float().numpy()
    diff = float(np.abs(pt_np - tflite).max())
    status = "PASS" if diff < atol else "FAIL"
    print(f"  {name}: max_abs_diff={diff:.6f}  {status}")
    assert diff < atol, f"{name} parity FAILED: max diff {diff:.6f} >= {atol}"


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class PredictorDurWrapper(torch.nn.Module):
    """Wraps ProsodyPredictor duration path for litert_torch export.

    forward(d_en [1, H, T], style [1, style_dim])
        -> duration [1, T, max_dur], d [1, T, H+style_dim]

    Duration path: DurationEncoder → predictor.lstm → duration_proj.
    Returns d (DurationEncoder output) so the CPU can project it with the
    alignment matrix to produce en for the F0Ntrain step.

    Changes vs. original:
      - pack_padded_sequence / pad_packed_sequence → direct LSTM call
      - masked_fill_ → masked_fill (out-of-place)
      - flatten_parameters() removed (CPU no-op)
      - isinstance(block, AdaLayerNorm) statically evaluated during trace
    """

    def __init__(self, predictor):
        super().__init__()
        self.sequence_mixer_type = predictor.sequence_mixer_type
        if self.sequence_mixer_type == "tcn":
            self.predictor = predictor
            return
        # DurationEncoder layers (alternating LSTM / AdaLayerNorm, nlayers each)
        self.dur_lstms    = predictor.text_encoder.lstms
        self.dur_dropout  = predictor.text_encoder.dropout
        # Predictor duration layers
        self.lstm          = predictor.lstm
        self.duration_proj = predictor.duration_proj

    def _duration_encoder(
        self,
        x: torch.FloatTensor,     # [B, H, T]
        style: torch.FloatTensor, # [B, style_dim]
    ) -> torch.FloatTensor:       # [B, T, H+style_dim]
        T = x.shape[2]
        B = x.shape[0]

        x = x.permute(2, 0, 1)              # [T, B, H]
        s = style.expand(T, B, -1)          # [T, B, style_dim]
        x = torch.cat([x, s], dim=-1)       # [T, B, H+style_dim]
        # mask all-False for full-bucket inputs; masked_fill noop but kept
        # x = x.masked_fill(mask_3d, 0.0)  — omitted: no padding case only
        x = x.transpose(0, 1)              # [B, T, H+style_dim]
        x = x.transpose(-1, -2)            # [B, H+style_dim, T]

        for block in self.dur_lstms:
            if isinstance(block, AdaLayerNorm):
                # AdaLayerNorm block: normalise + re-append style
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)  # [B, H, T]
                x = torch.cat([x, s.permute(1, 2, 0)], dim=1)            # [B, H+style_dim, T]
            else:
                # LSTM block: direct call replaces pack/unpad
                x = x.transpose(-1, -2)          # [B, T, H+style_dim]
                x, _ = block(x)                  # [B, T, H]
                x = F.dropout(x, p=self.dur_dropout, training=False)
                x = x.transpose(-1, -2)          # [B, H, T]
                # x_pad logic from original is noop for full-bucket inputs

        return x.transpose(-1, -2)              # [B, T, H+style_dim]

    def forward(
        self,
        d_en:  torch.FloatTensor,  # [1, H=512, T]
        style: torch.FloatTensor,  # [1, style_dim=128]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if self.sequence_mixer_type == "tcn":
            lengths = torch.ones(d_en.shape[0], dtype=torch.long, device=d_en.device) * d_en.shape[2]
            mask = torch.zeros(d_en.shape[0], d_en.shape[2], dtype=torch.bool, device=d_en.device)
            d = self.predictor.text_encoder(d_en, style, lengths, mask)
            x = self.predictor.run_duration_mixer(d)
            duration = self.predictor.duration_proj(x)
            return duration, d

        d = self._duration_encoder(d_en, style)    # [1, T, 640]
        x, _ = self.lstm(d)                         # [1, T, 512]
        duration = self.duration_proj(x)            # [1, T, 50]
        return duration, d


# ---------------------------------------------------------------------------
# AOT compile helper
# ---------------------------------------------------------------------------

def aot_compile_tensor_g5(tflite_path: str, out_dir: str, model_name: str) -> None:
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
        print("WARNING: ai-edge-litert-sdk-google-tensor not installed — skipping AOT.")
        return

    tensor_g5 = gt_target.Target(gt_target.SocModel.TENSOR_G5)
    print(f"Compiling {tflite_path} ...")
    compiled = aot_lib.aot_compile(
        tflite_path,
        target=[tensor_g5],
        keep_going=True,
        google_tensor_truncation_type="half",
        google_tensor_int64_to_int32=True,
        google_tensor_sharding_intensity="extensive",
    )
    print("\nCompilation report:")
    print(compiled.compilation_report())
    compiled.export(out_dir, model_name=model_name)
    print(f"AOT outputs written to: {out_dir}/")
    for fname in sorted(os.listdir(out_dir)):
        if fname.startswith(model_name) and fname.endswith(".tflite") and "_multisig_" not in fname:
            size_mb = os.path.getsize(os.path.join(out_dir, fname)) / 1e6
            print(f"  {fname}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    HASH     = git_hash()
    OUT_DIR  = os.path.join("outputs", HASH)
    TEST_DIR = os.path.join("test_output", HASH, "predictor_dur")
    os.makedirs(OUT_DIR,  exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    TFLITE_PATH = os.path.join(OUT_DIR, "kokoro_predictor_dur_multisig_fp32.tflite")

    BUCKETS   = [32, 128, 256]
    SIG_NAMES = ["predictor_dur_short", "predictor_dur_medium", "predictor_dur_long"]

    # Only full-bucket tests for exact parity (bidir LSTM limitation at T<bucket)
    TEST_LENS  = BUCKETS
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

    pred_copy = copy.deepcopy(model.predictor).eval()
    wrapper   = PredictorDurWrapper(pred_copy).eval()
    pred_ref  = model.predictor.eval()

    # -----------------------------------------------------------------------
    # Build multi-signature TFLite model
    # -----------------------------------------------------------------------
    print("Building multi-signature TFLite model...")

    def make_inputs(T: int):
        d_en  = torch.randn(1, 512, T)
        style = torch.randn(1, 128)
        return (d_en, style)

    builder = litert_torch.signature(SIG_NAMES[0], wrapper, make_inputs(BUCKETS[0]))
    for name, T in zip(SIG_NAMES[1:], BUCKETS[1:]):
        builder = builder.signature(name, wrapper, make_inputs(T))

    edge_model = builder.convert()
    edge_model.export(TFLITE_PATH)
    print(f"Saved: {TFLITE_PATH}")

    # -----------------------------------------------------------------------
    # Parity tests (full-bucket only — exact parity with direct LSTM call)
    # -----------------------------------------------------------------------
    print("\nRunning parity tests...")
    torch.manual_seed(PARITY_SEED)

    all_passed = True
    for T, sig in zip(TEST_LENS, SIG_NAMES):
        d_en  = torch.randn(1, 512, T)
        style = torch.randn(1, 128)
        lengths = torch.tensor([T])
        mask    = torch.zeros(1, T, dtype=torch.bool)

        # PyTorch reference — original forward with pack/unpad
        with torch.no_grad():
            d_ref    = pred_ref.text_encoder(d_en, style, lengths, mask)  # [1, T, 640]
            x_ref    = pred_ref.run_duration_mixer(d_ref)
            dur_ref  = pred_ref.duration_proj(x_ref)                      # [1, T, 50]

        # TFLite (wrapper — direct LSTM)
        tflite_dur, tflite_d = edge_model(d_en, style, signature_name=sig)

        # Save tensors
        prefix = os.path.join(TEST_DIR, f"{sig}_T{T}")
        np.save(prefix + "_dur_pt.npy",     dur_ref.float().numpy())
        np.save(prefix + "_dur_tflite.npy", tflite_dur)
        np.save(prefix + "_d_pt.npy",       d_ref.float().numpy())
        np.save(prefix + "_d_tflite.npy",   tflite_d)

        try:
            assert_close(dur_ref,  tflite_dur, f"{sig} T={T} duration", atol=PARITY_ATOL)
            assert_close(d_ref,    tflite_d,   f"{sig} T={T} d_out",    atol=PARITY_ATOL)
        except AssertionError as e:
            print(f"  ERROR: {e}")
            all_passed = False

    if all_passed:
        print("\nAll parity tests PASSED.")
    else:
        print("\nSome parity tests FAILED — check test_output/ for tensors.")
        raise SystemExit(1)

    print("\nNote: AOT NPU compilation skipped — LSTM-containing modules are not compiled for Tensor G5.")


if __name__ == "__main__":
    main()
