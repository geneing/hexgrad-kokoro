"""
Export Kokoro ProsodyPredictor F0Ntrain path to TFLite (Step 3b).

Covers:
  predictor.shared (LSTM)  +
  predictor.F0 (AdainResBlk1d × 3)  +  predictor.F0_proj  +
  predictor.N  (AdainResBlk1d × 3)  +  predictor.N_proj

Produces:
  outputs/<git_hash>/kokoro_predictor_f0n_multisig_fp32.tflite
    Signatures: predictor_f0n (T_aligned=200), predictor_f0n_long (T_aligned=800)

Note: AOT NPU compilation is intentionally skipped for this module.
  The F0Ntrain path contains the shared LSTM which takes excessive time
  to compile on the Tensor G5 plugin. This module runs on CPU/GPU fallback.

Wrapper inputs:
  x  [1, 640, T_aligned]  — en from CPU: d.transpose(-1,-2) @ pred_aln_trg
                             where d is DurationEncoder output [1, T, 640]
  s  [1, 128]             — ref_s[:, 128:] style slice

Wrapper outputs:
  F0 [1, T_aligned*2]     — upsampled F0 curve (AdainResBlk1d upsample×2)
  N  [1, T_aligned*2]     — upsampled aperiodicity

On-device pipeline after this step:
  1. (NPU) predictor_dur → (duration [1,T,50], d [1,T,640])
  2. (CPU) duration → sigmoid → sum → round → pred_dur
  3. (CPU) pred_aln_trg = build_alignment(pred_dur)
  4. (CPU) en = d.transpose(-1,-2) @ pred_aln_trg         [1, 640, T_aligned]
  5. (CPU) asr = t_en @ pred_aln_trg                      [from TextEncoder]
  6. (CPU/GPU) predictor_f0n(en, style) → F0, N           ← this step
  7. (NPU) decoder(asr, F0, N, ref_s[:,:128])             (Step 4)

Export compatibility notes:
  - shared LSTM uses direct call (no pack/unpad) — already export-compatible.
  - AdainResBlk1d uses weight_norm parametrizations on conv layers.
    torch.export handles these correctly (they are constant-folded).
  - F.interpolate(scale_factor=2, mode='nearest') in UpSample1d is supported.
  - Output shape: [1, T_aligned*2] due to the upsample block in F0/N heads.

Run with:
  uv run python export/export_predictor_f0n.py
"""

import copy
import numpy as np
import os
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import litert_torch

from kokoro import KModel
from kokoro.istftnet import AdainResBlk1d


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
# ConvTranspose1d output_padding workaround
# ---------------------------------------------------------------------------

class PoolEquiv(nn.Module):
    """Equivalent to weight_norm(ConvTranspose1d(k=3, s=2, p=1, op=1, groups=C)).

    litert_torch does not support output_padding in ConvTranspose1d.
    This replaces it with a mathematically equivalent formulation:
      1. Zero-interleave input:  [B, C, T] → [B, C, 2T-1]
      2. Asymmetric pad (1, 2):  [B, C, 2T-1] → [B, C, 2T+2]
      3. Depthwise Conv1d with flipped kernel: → [B, C, 2T]

    For ConvTranspose1d(k=3, s=2, p=1, op=1, groups=C), the equivalent
    conv1d uses the flipped weight tensor (standard transposed-conv identity).
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, groups: int):
        super().__init__()
        self.groups = groups
        # Pre-flip the weight once; register as buffer so it's a constant.
        self.register_buffer("weight", weight.flip(-1))  # [C, 1, 3]
        self.register_buffer("bias",   bias)             # [C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Zero-interleave: insert 1 zero between each element
        # [B, C, T] → stack with zeros → [B, C, T, 2] → reshape [B, C, 2T] → drop last → [B, C, 2T-1]
        zeros  = torch.zeros_like(x)
        x_zi   = torch.stack([x, zeros], dim=3).reshape(
            x.shape[0], x.shape[1], 2 * x.shape[2]
        )[:, :, :-1]
        # Asymmetric padding: (kernel-1-padding)=1 left, (kernel-1-padding+output_padding)=2 right
        x_pad  = F.pad(x_zi, (1, 2))
        return F.conv1d(x_pad, self.weight, self.bias,
                        stride=1, padding=0, groups=self.groups)


def _replace_pool_layers(module: nn.Module) -> None:
    """Recursively replace AdainResBlk1d.pool ConvTranspose1d with PoolEquiv."""
    for child_name, child in list(module.named_children()):
        if isinstance(child, AdainResBlk1d) and child.upsample_type != 'none':
            orig_pool = child.pool  # weight_norm(ConvTranspose1d)
            w = orig_pool.weight.detach().clone()   # [C, 1, 3]
            b = orig_pool.bias.detach().clone()     # [C]
            g = orig_pool.groups
            child.pool = PoolEquiv(w, b, g)
        else:
            _replace_pool_layers(child)



class PredictorF0NWrapper(torch.nn.Module):
    """Wraps ProsodyPredictor.F0Ntrain for litert_torch export.

    forward(x [1, 640, T_aligned], s [1, 128])
        -> F0 [1, T_aligned*2], N [1, T_aligned*2]

    Inputs:
      x : en tensor = d.transpose(-1,-2) @ pred_aln_trg
          where d is DurationEncoder output [1, T, 640].
          Input size matches predictor.shared LSTM input_size (d_hid+sty_dim=640).
      s : style slice ref_s[:, 128:]

    The F0 and N AdainResBlk1d heads each include one upsample block
    (stride-2 ConvTranspose1d + nearest interpolation) so T doubles.
    """

    def __init__(self, predictor):
        super().__init__()
        self.sequence_mixer_type = predictor.sequence_mixer_type
        if self.sequence_mixer_type == "lstm":
            self.shared = predictor.shared
        else:
            self.shared_mixer = predictor.shared_mixer
        self.F0_blocks = predictor.F0
        self.N_blocks  = predictor.N
        self.F0_proj   = predictor.F0_proj
        self.N_proj    = predictor.N_proj

    def forward(
        self,
        x: torch.FloatTensor,  # [1, 640, T_aligned]
        s: torch.FloatTensor,  # [1, 128]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # x.transpose(-1,-2) : [1, T_aligned, 640]
        if self.sequence_mixer_type == "lstm":
            h, _ = self.shared(x.transpose(-1, -2))  # [1, T_aligned, 512]
        else:
            h = self.shared_mixer(x.transpose(-1, -2))
        h = h.transpose(-1, -2)                   # [1, 512, T_aligned]

        F0 = h
        for block in self.F0_blocks:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0).squeeze(1)          # [1, T_aligned*2]

        N = h
        for block in self.N_blocks:
            N = block(N, s)
        N = self.N_proj(N).squeeze(1)             # [1, T_aligned*2]

        return F0, N


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    HASH     = git_hash()
    OUT_DIR  = os.path.join("outputs", HASH)
    TEST_DIR = os.path.join("test_output", HASH, "predictor_f0n")
    os.makedirs(OUT_DIR,  exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    TFLITE_PATH = os.path.join(OUT_DIR, "kokoro_predictor_f0n_multisig_fp32.tflite")

    # T_aligned bucket sizes.  Output shape will be [1, T*2] due to upsample.
    BUCKETS   = [200, 800]
    SIG_NAMES = ["predictor_f0n", "predictor_f0n_long"]

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
    _replace_pool_layers(pred_copy)   # replace ConvTranspose1d(output_padding=1) before export
    wrapper   = PredictorF0NWrapper(pred_copy).eval()
    pred_ref  = model.predictor.eval()

    # -----------------------------------------------------------------------
    # Sanity-check output shapes
    # -----------------------------------------------------------------------
    with torch.no_grad():
        _x = torch.randn(1, 640, BUCKETS[0])
        _s = torch.randn(1, 128)
        _F0, _N = wrapper(_x, _s)
        assert _F0.shape == (1, BUCKETS[0] * 2), f"Unexpected F0 shape: {_F0.shape}"
        assert _N.shape  == (1, BUCKETS[0] * 2), f"Unexpected N shape: {_N.shape}"
        print(f"Shape check OK: F0={tuple(_F0.shape)}, N={tuple(_N.shape)}")

    # -----------------------------------------------------------------------
    # Build multi-signature TFLite model
    # -----------------------------------------------------------------------
    print("Building multi-signature TFLite model...")

    def make_inputs(T: int):
        x = torch.randn(1, 640, T)
        s = torch.randn(1, 128)
        return (x, s)

    builder = litert_torch.signature(SIG_NAMES[0], wrapper, make_inputs(BUCKETS[0]))
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
    for T, sig in zip(BUCKETS, SIG_NAMES):
        x = torch.randn(1, 640, T)
        s = torch.randn(1, 128)

        # PyTorch reference — original F0Ntrain
        with torch.no_grad():
            F0_ref, N_ref = pred_ref.F0Ntrain(x, s)

        # TFLite
        F0_tflite, N_tflite = edge_model(x, s, signature_name=sig)

        # Save tensors
        prefix = os.path.join(TEST_DIR, f"{sig}_T{T}")
        np.save(prefix + "_F0_pt.npy",     F0_ref.float().numpy())
        np.save(prefix + "_F0_tflite.npy", F0_tflite)
        np.save(prefix + "_N_pt.npy",      N_ref.float().numpy())
        np.save(prefix + "_N_tflite.npy",  N_tflite)

        try:
            assert_close(F0_ref, F0_tflite, f"{sig} T={T} F0", atol=PARITY_ATOL)
            assert_close(N_ref,  N_tflite,  f"{sig} T={T} N",  atol=PARITY_ATOL)
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
