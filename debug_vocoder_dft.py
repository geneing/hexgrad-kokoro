"""debug_vocoder_dft.py

Debug script: export backbone up to the point just before irfft (i.e. the
complex spectrum x_real/x_imag), compare those pre-DFT values between PyTorch
and TFLite, then apply numpy/tf.signal-equivalent IRFFT in Python and verify
the full audio matches.

Run with:
    uv run python debug_vocoder_dft.py
"""

from __future__ import annotations

import copy
import math as _math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from streaming_vocos import StreamingVocos
from export_real_vocoder import VocosStreamChunk, overlap_add_numpy

VOCOS_CKPT   = "vocos_fp16.pt"
CHUNK_FRAMES = 16
OPSET        = 20
DEBUG_DIR    = Path("debug_dft")
DEBUG_ONNX   = DEBUG_DIR / "vocoder_pre_irfft.onnx"
DEBUG_TFLITE = DEBUG_DIR / "vocoder_pre_irfft_float32.tflite"


# ── Model: backbone + head up to just before irfft ───────────────────────────

class VocosPreIRFFT(nn.Module):
    """Identical to VocosStreamChunkNoOLA backbone, but outputs the
    complex spectrum (x_real, x_imag) instead of time_frames.

    Outputs:
        x_real  [1, F, K]   — real part of spectrum  (K = n_fft//2 + 1)
        x_imag  [1, F, K]   — imaginary part of spectrum
        embed_prev_new, block_0..7_prev_new  — backbone state
    """
    EMBED_IN  = 192
    BLOCK_DIM = 384
    KERNEL_M1 = 6
    N_LAYERS  = 8

    def __init__(self, vocos: StreamingVocos):
        super().__init__()
        bb = vocos.model.backbone
        hd = vocos.model.head
        self.embed_norm_conv = bb.embed.conv
        self.backbone_norm   = bb.norm
        self.convnext        = bb.convnext
        self.final_ln        = bb.final_layer_norm
        self.head_out        = hd.out
        self.n_fft   = hd.istft.n_fft
        self.hop     = hd.istft.hop
        self.win_len = hd.istft.win_length
        self.tail    = hd.istft.tail
        self.register_buffer("window", hd.istft.window)

    @staticmethod
    def initial_state() -> Dict[str, Tensor]:
        state: Dict[str, Tensor] = {
            "embed_prev": torch.zeros(1, VocosPreIRFFT.EMBED_IN, VocosPreIRFFT.KERNEL_M1),
        }
        for i in range(VocosPreIRFFT.N_LAYERS):
            state[f"block_{i}_prev"] = torch.zeros(
                1, VocosPreIRFFT.BLOCK_DIM, VocosPreIRFFT.KERNEL_M1
            )
        return state

    def state_as_tuple(self, state: Dict[str, Tensor]) -> Tuple[Tensor, ...]:
        keys = ["embed_prev"] + [f"block_{i}_prev" for i in range(self.N_LAYERS)]
        return tuple(state[k] for k in keys)

    def _causal_conv(self, norm_conv1d, x, prev):
        xp = torch.cat([prev, x], dim=-1)
        y  = norm_conv1d(xp)
        return y, xp[..., -self.KERNEL_M1:]

    def forward(
        self,
        conditioned_chunk: Tensor,
        embed_prev: Tensor,
        *block_prevs_tuple,
    ) -> Tuple[Tensor, ...]:
        block_prevs = list(block_prevs_tuple[: self.N_LAYERS])
        x, new_embed_prev = self._causal_conv(self.embed_norm_conv, conditioned_chunk, embed_prev)
        x = self.backbone_norm(x.transpose(1, 2)).transpose(1, 2)
        new_block_prevs: List[Tensor] = []
        for i, block in enumerate(self.convnext):
            identity = x
            xd, new_bp = self._causal_conv(block.dwconv.conv, x, block_prevs[i])
            new_block_prevs.append(new_bp)
            xd = xd.permute(0, 2, 1)
            xd = block.norm(xd)
            xd = block.pwconv1(xd)
            xd = block.act(xd)
            xd = block.pwconv2(xd)
            xd = block.gamma * xd
            xd = xd.permute(0, 2, 1)
            x = identity + xd
        x = self.final_ln(x.transpose(1, 2)).transpose(1, 2)

        h = self.head_out(x.transpose(1, 2)).transpose(1, 2)
        mag, phase = h.chunk(2, dim=1)
        mag    = torch.exp(mag).clamp(max=1e2)
        x_real = mag * torch.cos(phase)          # [1, K, F]
        x_imag = mag * torch.sin(phase)          # [1, K, F]

        # Transpose to [1, F, K] for export (NTK layout)
        x_real_ntk = x_real.transpose(1, 2)      # [1, F, K]
        x_imag_ntk = x_imag.transpose(1, 2)      # [1, F, K]

        return (x_real_ntk, x_imag_ntk, new_embed_prev) + tuple(new_block_prevs)


def irfft_numpy(x_real_ntk: np.ndarray, x_imag_ntk: np.ndarray, n_fft: int) -> np.ndarray:
    """numpy IRFFT — identical to tf.signal.irfft / torch.fft.irfft.

    Inputs:  x_real, x_imag  [1, F, K]  (NTK)
    Returns: time_frames      [1, F, win_len=n_fft]
    """
    # Reconstruct complex spectrum [1, F, K]
    spec_np = x_real_ntk + 1j * x_imag_ntk
    # numpy irfft along last axis → [1, F, n_fft]
    return np.fft.irfft(spec_np, n=n_fft, axis=-1).astype(np.float32)


def main():
    print("Loading StreamingVocos …")
    vocos = StreamingVocos.from_checkpoint(VOCOS_CKPT, chunk_frames=CHUNK_FRAMES, device="cpu", use_fp16=False)

    model      = VocosPreIRFFT(vocos).eval()
    full_model = VocosStreamChunk(vocos).eval()

    torch.manual_seed(0)
    sample_chunk = torch.randn(1, model.EMBED_IN, CHUNK_FRAMES)
    state        = model.initial_state()
    state_tuple  = model.state_as_tuple(state)

    # ── Step 1: PyTorch reference spectrum ────────────────────────────────────
    print("\n─── Step 1: Reference PyTorch outputs ───")
    with torch.no_grad():
        pt_out = model(sample_chunk, *state_tuple)
    pt_xreal = pt_out[0].numpy()   # [1, F, K]
    pt_ximag = pt_out[1].numpy()   # [1, F, K]

    # Verify IRFFT via numpy matches full PyTorch model audio
    full_state_tuple = full_model.state_as_tuple(full_model.initial_state())
    with torch.no_grad():
        full_out = full_model(sample_chunk, *full_state_tuple)
    pt_audio_ref = full_out[0].numpy().flatten()   # [F*hop]

    win_np = model.window.numpy()                  # [win_len]
    tf_numpy = irfft_numpy(pt_xreal, pt_ximag, model.n_fft)[..., : model.win_len]
    tf_numpy = tf_numpy * win_np                   # apply Hann window [1, F, win_len]
    prev = np.zeros((1, model.tail), dtype=np.float32)
    audio_numpy, _ = overlap_add_numpy(tf_numpy, prev, model.hop, model.tail)

    diff_ref = np.abs(pt_audio_ref - audio_numpy.flatten()).max()
    print(f"  PT vs numpy-IRFFT+OLA audio diff : {diff_ref:.6e}  (should be < 1e-4)")

    # ── Step 2: Export to ONNX ────────────────────────────────────────────────
    print("\n─── Step 2: Export VocosPreIRFFT to ONNX ───")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    export_model = copy.deepcopy(model).eval()
    for m in export_model.modules():
        if isinstance(m, nn.Conv1d):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                pass

    state_in_names  = ["embed_prev"] + [f"block_{i}_prev" for i in range(8)]
    state_out_names = ["embed_prev_new"] + [f"block_{i}_prev_new" for i in range(8)]

    with torch.no_grad():
        torch.onnx.export(
            export_model,
            args=(sample_chunk,) + state_tuple,
            f=str(DEBUG_ONNX),
            export_params=True,
            input_names=["conditioned_chunk"] + state_in_names,
            output_names=["x_real", "x_imag"] + state_out_names,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=True,
            external_data=False,
        )

    import onnx, onnxsim
    proto = onnx.load(str(DEBUG_ONNX))
    proto = onnx.shape_inference.infer_shapes(proto)
    proto_sim, ok = onnxsim.simplify(proto)
    if ok:
        proto = proto_sim
    onnx.save(proto, str(DEBUG_ONNX))
    ops = sorted({n.op_type for n in proto.graph.node})
    print(f"  ONNX ops: {ops}")
    for forbidden in ("DFT", "STFT", "ConvTranspose"):
        if forbidden in ops:
            print(f"  ERROR: {forbidden} op found — should not be present!")

    # ── Step 3: Convert to TFLite ─────────────────────────────────────────────
    print("\n─── Step 3: Convert to TFLite (standard onnx2tf) ───")
    work_dir = DEBUG_DIR / "tflite_onnx2tf"
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = ["uv", "run", "onnx2tf"] if shutil.which("uv") else ["onnx2tf"]
    cmd = runner + ["-i", str(DEBUG_ONNX), "-o", str(work_dir), "-osd"]
    subprocess.run(cmd, check=True, capture_output=True)

    produced = sorted(p for p in work_dir.rglob("*.tflite") if "_float32" in p.name)
    if not produced:
        produced = sorted(work_dir.rglob("*.tflite"))
    assert produced, f"No .tflite in {work_dir}"
    shutil.copy2(produced[0], DEBUG_TFLITE)
    print(f"  TFLite saved: {DEBUG_TFLITE}")

    # ── Step 4: Compare pre-IRFFT spectrum PyTorch vs TFLite ──────────────────
    print("\n─── Step 4: Compare pre-IRFFT spectrum (PyTorch vs TFLite) ───")
    import ai_edge_litert.interpreter as litert
    interp = litert.Interpreter(model_path=str(DEBUG_TFLITE), num_threads=4)
    interp.allocate_tensors()
    in_d  = interp.get_input_details()
    out_d = interp.get_output_details()

    print("  TFLite inputs:")
    for d in in_d:
        print(f"    [{d['index']:3d}] {d['name']:<35s}  shape={list(d['shape'])}")
    print("  TFLite outputs:")
    for d in out_d:
        print(f"    [{d['index']:3d}] {d['name']:<35s}  shape={list(d['shape'])}")

    # Feed inputs (auto-detect NTC layout for conditioned_chunk)
    chunk_np = sample_chunk.numpy()
    chunk_in_shape = in_d[0]["shape"]
    ntc = (len(chunk_in_shape) == 3 and chunk_in_shape[-1] == model.EMBED_IN)
    chunk_feed = np.transpose(chunk_np, (0, 2, 1)) if ntc else chunk_np

    lt_state = [np.zeros(d["shape"], dtype=np.float32) for d in in_d[1:]]
    interp.set_tensor(in_d[0]["index"], chunk_feed.astype(np.float32))
    for i, s in enumerate(lt_state):
        interp.set_tensor(in_d[1 + i]["index"], s)
    interp.invoke()

    lt_xreal_raw = interp.get_tensor(out_d[0]["index"])   # x_real
    lt_ximag_raw = interp.get_tensor(out_d[1]["index"])   # x_imag
    print(f"  TFLite x_real shape: {list(lt_xreal_raw.shape)}")
    print(f"  TFLite x_imag shape: {list(lt_ximag_raw.shape)}")

    # Detect layout: NTK = [1, F, K] or NKT = [1, K, F]
    K = model.n_fft // 2 + 1
    F = CHUNK_FRAMES
    def ensure_ntk(arr):
        if arr.ndim == 3 and arr.shape[1] == K:   # [1, K, F] → [1, F, K]
            return np.transpose(arr, (0, 2, 1))
        return arr

    lt_xreal = ensure_ntk(lt_xreal_raw)
    lt_ximag = ensure_ntk(lt_ximag_raw)

    diff_real = np.abs(pt_xreal - lt_xreal).max()
    diff_imag = np.abs(pt_ximag - lt_ximag).max()
    print(f"\n  Max |PT − LT| x_real : {diff_real:.6e}")
    print(f"  Max |PT − LT| x_imag : {diff_imag:.6e}")

    spectrum_ok = diff_real < 1e-3 and diff_imag < 1e-3
    if spectrum_ok:
        print("  ✓ Pre-IRFFT spectrum matches — backbone is correct in TFLite")
    else:
        print("  ⚠ Pre-IRFFT spectrum DIFFERS — backbone has a TFLite conversion error")
        print("    (fix backbone conversion before worrying about IRFFT implementation)")
        return

    # ── Step 5: Apply numpy IRFFT on TFLite spectrum → verify audio matches ───
    print("\n─── Step 5: Apply numpy IRFFT (= tf.signal.irfft) on TFLite spectrum ───")
    tf_lt = irfft_numpy(lt_xreal, lt_ximag, model.n_fft)[..., : model.win_len]
    tf_lt  = tf_lt * win_np                            # [1, F, win_len]
    audio_lt, _ = overlap_add_numpy(tf_lt, prev, model.hop, model.tail)

    diff_audio = np.abs(pt_audio_ref - audio_lt.flatten()).max()
    print(f"  PT vs TFLite-spectrum + numpy-IRFFT + OLA audio diff: {diff_audio:.6e}")
    if diff_audio < 1e-2:
        print("  ✓ numpy IRFFT on TFLite spectrum gives correct audio")
        print("\n  → Conclusion: VocosPreIRFFT TFLite backbone is correct.")
        print("    The fix is to use numpy/tf.signal IRFFT at inference time")
        print("    instead of the cos/sin matmul in VocosStreamChunkNoOLA.")
    else:
        print("  ⚠ Still differ after IRFFT — check further")

    print("\nDone.")


if __name__ == "__main__":
    main()
