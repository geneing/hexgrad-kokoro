"""export_real_vocoder.py

Exports VocosPreIRFFT to ONNX and converts to TFLite.
VocosPreIRFFT outputs the complex spectrum (x_real, x_imag) just before
irfft, so the problematic ONNX DFT op never appears.  At inference time
numpy.fft.irfft (numerically identical to tf.signal.irfft) is applied in
Python, followed by windowing and overlap-add.

Run with:
    uv run python export_real_vocoder.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from streaming_vocos import StreamingVocos

# ── Paths ─────────────────────────────────────────────────────────────────────
VOCOS_CKPT     = "vocos_fp16.pt"
ONNX_DIR       = Path("onnx_streaming_vocos")
ONNX_PATH      = ONNX_DIR / "vocoder_stream_chunk.onnx"
TFLITE_DEST    = Path("onnx2tf_conversion/saved_model")
ONNX2TF_DIR    = Path("onnx2tf_conversion")
OPSET          = 20
CHUNK_FRAMES   = 16


# ── VocosStreamChunkReal (copy from streaming_vocos_export.ipynb) ─────────────
class VocosStreamChunk(nn.Module):
    EMBED_IN   = 192
    BLOCK_DIM  = 384
    KERNEL_M1  = 6
    ISTFT_TAIL = 900
    N_LAYERS   = 8

    def __init__(self, vocos: StreamingVocos):
        super().__init__()
        bb = vocos.model.backbone
        hd = vocos.model.head
        self.embed_norm_conv = bb.embed.conv
        self.backbone_norm   = bb.norm
        self.convnext        = bb.convnext
        self.final_ln        = bb.final_layer_norm
        self.head_out        = hd.out
        self.overlap_add     = hd.istft.overlap_add.deconv
        self.register_buffer("window", hd.istft.window)
        self.n_fft   = hd.istft.n_fft
        self.hop     = hd.istft.hop
        self.win_len = hd.istft.win_length
        self.tail    = hd.istft.tail

    @staticmethod
    def initial_state() -> Dict[str, Tensor]:
        state: Dict[str, Tensor] = {
            "embed_prev": torch.zeros(1, VocosStreamChunk.EMBED_IN, VocosStreamChunk.KERNEL_M1),
            "istft_prev": torch.zeros(1, VocosStreamChunk.ISTFT_TAIL),
        }
        for i in range(VocosStreamChunk.N_LAYERS):
            state[f"block_{i}_prev"] = torch.zeros(1, VocosStreamChunk.BLOCK_DIM, VocosStreamChunk.KERNEL_M1)
        return state

    def state_as_tuple(self, state: Dict[str, Tensor]) -> Tuple[Tensor, ...]:
        keys = ["embed_prev"] + [f"block_{i}_prev" for i in range(self.N_LAYERS)] + ["istft_prev"]
        return tuple(state[k] for k in keys)

    def _causal_conv(self, norm_conv1d, x, prev):
        xp = torch.cat([prev, x], dim=-1)
        y  = norm_conv1d(xp)
        return y, xp[..., -self.KERNEL_M1:]

    def forward(self, conditioned_chunk, embed_prev, *block_and_istft):
        block_prevs = list(block_and_istft[:self.N_LAYERS])
        istft_prev  = block_and_istft[self.N_LAYERS]
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
            x  = identity + xd
        x = self.final_ln(x.transpose(1, 2)).transpose(1, 2)
        frames = x.shape[-1]
        h    = self.head_out(x.transpose(1, 2)).transpose(1, 2)
        mag, phase = h.chunk(2, dim=1)
        mag  = torch.exp(mag).clamp(max=1e2)
        spec = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        time_frames = torch.fft.irfft(spec.transpose(1, 2), n=self.n_fft)[..., : self.win_len]
        time_frames = time_frames * self.window
        ola_in  = time_frames.transpose(1, 2)
        ola_out = self.overlap_add(ola_in)[:, 0, :]
        audio_overlap  = ola_out[:, :self.tail] + istft_prev
        audio_rest     = ola_out[:, self.tail : frames * self.hop]
        audio          = torch.cat([audio_overlap, audio_rest], dim=-1)
        new_istft_prev = ola_out[:, frames * self.hop:]
        return (audio, new_embed_prev) + tuple(new_block_prevs) + (new_istft_prev,)


class VocosPreIRFFT(VocosStreamChunk):
    """TFLite export variant: backbone up to just before irfft.

    Outputs the complex spectrum ``(x_real [1,F,K], x_imag [1,F,K])`` and
    backbone state.  No DFT/irfft op appears in the ONNX graph.

    At inference time, apply ``numpy.fft.irfft`` (= ``tf.signal.irfft``) in
    Python, then window + overlap-add with :func:`irfft_numpy` and
    :func:`overlap_add_numpy`.

    State: ``embed_prev`` + ``block_0..7_prev`` (no ``istft_prev``).
    """

    @staticmethod
    def initial_state() -> Dict[str, Tensor]:
        state: Dict[str, Tensor] = {
            "embed_prev": torch.zeros(1, VocosStreamChunk.EMBED_IN, VocosStreamChunk.KERNEL_M1),
        }
        for i in range(VocosStreamChunk.N_LAYERS):
            state[f"block_{i}_prev"] = torch.zeros(
                1, VocosStreamChunk.BLOCK_DIM, VocosStreamChunk.KERNEL_M1
            )
        return state

    def state_as_tuple(self, state: Dict[str, Tensor]) -> Tuple[Tensor, ...]:
        keys = ["embed_prev"] + [f"block_{i}_prev" for i in range(self.N_LAYERS)]
        return tuple(state[k] for k in keys)

    def forward(
        self,
        conditioned_chunk: Tensor,
        embed_prev: Tensor,
        *block_prevs_tuple,
    ) -> Tuple[Tensor, ...]:
        """Returns ``(x_real [1,F,K], x_imag [1,F,K], embed_prev_new, block_0..7_prev_new)``."""
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
        x_real = (mag * torch.cos(phase)).transpose(1, 2)  # [1, F, K]
        x_imag = (mag * torch.sin(phase)).transpose(1, 2)  # [1, F, K]

        return (x_real, x_imag, new_embed_prev) + tuple(new_block_prevs)


def irfft_numpy(
    x_real: np.ndarray,   # [1, F, K]
    x_imag: np.ndarray,   # [1, F, K]
    n_fft: int,
) -> np.ndarray:
    """numpy IRFFT — identical to ``tf.signal.irfft`` / ``torch.fft.irfft``.

    Returns time_frames ``[1, F, n_fft]`` (before windowing).
    """
    spec = x_real + 1j * x_imag          # [1, F, K] complex
    return np.fft.irfft(spec, n=n_fft, axis=-1).astype(np.float32)


def overlap_add_numpy(
    time_frames: np.ndarray,   # [1, F, win_len]  (NTC)
    istft_prev:  np.ndarray,   # [1, tail]
    hop: int,
    tail: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlap-add one chunk of windowed time frames.

    Returns ``(audio [1, F*hop], new_istft_prev [1, tail])``.
    """
    B, F, win_len = time_frames.shape
    # Buffer extends to end of last frame: (F-1)*hop + win_len = F*hop + tail
    buf = np.zeros((B, F * hop + tail), dtype=np.float32)
    for f in range(F):
        buf[:, f * hop : f * hop + win_len] += time_frames[:, f, :]
    buf[:, :tail] += istft_prev
    audio          = buf[:, :F * hop]
    new_istft_prev = buf[:, F * hop:]
    return audio, new_istft_prev


def main():
    print("Loading StreamingVocos …")
    vocos = StreamingVocos.from_checkpoint(VOCOS_CKPT, chunk_frames=CHUNK_FRAMES, device="cpu", use_fp16=False)

    stream_chunk   = VocosStreamChunk(vocos).eval()
    stream_pre_dft = VocosPreIRFFT(vocos).eval()

    # ── Build sample inputs ────────────────────────────────────────────────────
    cond_channels = vocos.model.backbone.embed.conv.conv.in_channels
    print(f"Conditioner output channels: {cond_channels}")
    sample_chunk = torch.randn(1, cond_channels, CHUNK_FRAMES)
    init_state   = VocosPreIRFFT.initial_state()
    state_tuple  = stream_pre_dft.state_as_tuple(init_state)

    # ── Verify numerical match vs full model ──────────────────────────────────
    print("Verifying VocosPreIRFFT + numpy IRFFT vs VocosStreamChunk …")
    win_np = stream_pre_dft.window.numpy()
    with torch.no_grad():
        out_full    = stream_chunk(sample_chunk, *stream_chunk.state_as_tuple(VocosStreamChunk.initial_state()))
        out_pre_dft = stream_pre_dft(sample_chunk, *state_tuple)

    x_real_pt = out_pre_dft[0].numpy()   # [1, F, K]
    x_imag_pt = out_pre_dft[1].numpy()
    tf_pt = irfft_numpy(x_real_pt, x_imag_pt, stream_pre_dft.n_fft)[..., : stream_pre_dft.win_len]
    tf_pt = tf_pt * win_np
    prev  = np.zeros((1, stream_pre_dft.tail), dtype=np.float32)
    audio_np, _ = overlap_add_numpy(tf_pt, prev, stream_pre_dft.hop, stream_pre_dft.tail)
    audio_pt = out_full[0].detach().numpy().flatten()
    diff = np.abs(audio_pt - audio_np.flatten()).max()
    print(f"  VocosPreIRFFT + numpy-IRFFT + OLA diff: {diff:.6e}  (should be < 1e-4)")
    if diff > 1e-3:
        print("  WARNING: Large difference — check implementation")

    # ── Remove weight_norm ────────────────────────────────────────────────────
    import copy
    stream_export = copy.deepcopy(stream_pre_dft).eval()
    for module in stream_export.modules():
        if isinstance(module, torch.nn.Conv1d):
            try:
                torch.nn.utils.remove_weight_norm(module)
            except ValueError:
                pass

    with torch.no_grad():
        out_no_wn = stream_export(sample_chunk, *state_tuple)
    diff_wn = np.abs(out_pre_dft[0].detach().numpy() - out_no_wn[0].detach().numpy()).max()
    print(f"  After weight_norm removal diff: {diff_wn:.2e}  (should be ≈0)")

    # ── Export to ONNX ────────────────────────────────────────────────────────
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    state_in_names  = ["embed_prev"] + [f"block_{i}_prev"     for i in range(8)]
    state_out_names = ["embed_prev_new"] + [f"block_{i}_prev_new" for i in range(8)]

    print(f"\nExporting VocosPreIRFFT to {ONNX_PATH} …")
    with torch.no_grad():
        torch.onnx.export(
            stream_export,
            args=(sample_chunk,) + state_tuple,
            f=str(ONNX_PATH),
            export_params=True,
            input_names=["conditioned_chunk"] + state_in_names,
            output_names=["x_real", "x_imag"] + state_out_names,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=True,
            external_data=False,
        )

    # ── Sort + simplify the ONNX graph ────────────────────────────────────────
    import onnx
    import onnxsim

    def topo_sort_onnx(proto):
        graph = proto.graph
        available = set(x.name for x in graph.input) | set(x.name for x in graph.initializer)
        nodes = list(graph.node)
        sorted_out: list = []
        remaining = list(range(len(nodes)))
        while remaining:
            progress = False
            for i in list(remaining):
                if all(inp == "" or inp in available for inp in nodes[i].input):
                    sorted_out.append(nodes[i])
                    available.update(nodes[i].output)
                    remaining.remove(i)
                    progress = True
                    break
            if not progress:
                for i in remaining:
                    sorted_out.append(nodes[i])
                break
        del graph.node[:]
        graph.node.extend(sorted_out)
        return proto

    proto = onnx.load(str(ONNX_PATH))
    proto = onnx.shape_inference.infer_shapes(proto)
    proto = topo_sort_onnx(proto)

    proto_sim, ok = onnxsim.simplify(proto)
    if ok:
        proto = proto_sim
        print("✓ onnxsim simplification successful")
    else:
        print("⚠ onnxsim simplification skipped (using topo-sorted model)")
    onnx.save(proto, str(ONNX_PATH))

    ops = sorted({n.op_type for n in proto.graph.node})
    print(f"ONNX ops: {ops}")
    for forbidden in ("DFT", "STFT", "ConvTranspose"):
        if forbidden in ops:
            print(f"ERROR: {forbidden} op still present — model will fail in TFLite!")
            import sys; sys.exit(1)
    print("✓ No DFT/ConvTranspose ops in ONNX model")

    # ── Copy ONNX to onnx2tf_conversion/ ──────────────────────────────────────
    ONNX2TF_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ONNX_PATH, ONNX2TF_DIR / ONNX_PATH.name)
    print(f"Copied ONNX to {ONNX2TF_DIR / ONNX_PATH.name}")

    # ── Convert to TFLite ─────────────────────────────────────────────────────
    TFLITE_DEST.mkdir(parents=True, exist_ok=True)
    work_dir = ONNX_DIR / "tflite" / "vocoder_stream_chunk_onnx2tf"
    work_dir.mkdir(parents=True, exist_ok=True)

    runner = ["uv", "run", "onnx2tf"] if shutil.which("uv") else ["onnx2tf"]
    cmd = runner + [
        "-i", str(ONNX_PATH),
        "-o", str(work_dir),
        "-osd",
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    produced = sorted(p for p in work_dir.rglob("*.tflite") if "_float32" in p.name)
    if not produced:
        produced = sorted(work_dir.rglob("*.tflite"))
    if not produced:
        print(f"ERROR: No .tflite produced in {work_dir}")
        import sys; sys.exit(1)

    final_tflite = TFLITE_DEST / "vocoder_stream_chunk_float32.tflite"
    shutil.copy2(produced[0], final_tflite)
    print(f"\n✓ TFLite saved to: {final_tflite}")

    # ── Inspect TFLite tensor details ─────────────────────────────────────────
    import ai_edge_litert.interpreter as litert
    interp = litert.Interpreter(model_path=str(final_tflite), num_threads=4)
    interp.allocate_tensors()
    print("\nTFLite inputs:")
    for d in interp.get_input_details():
        print(f"  [{d['index']:2d}] {d['name']:<40s}  shape={list(d['shape'])}")
    print("\nTFLite outputs:")
    for d in interp.get_output_details():
        print(f"  [{d['index']:2d}] {d['name']:<40s}  shape={list(d['shape'])}")

    # ── Numerical check: PyTorch vs TFLite ────────────────────────────────────
    print("\nRunning numerical PyTorch vs TFLite comparison …")
    in_d  = interp.get_input_details()
    out_d = interp.get_output_details()

    chunk_in_shape = in_d[0]["shape"]
    chunk_is_ntc   = (len(chunk_in_shape) == 3 and chunk_in_shape[-1] == cond_channels)
    chunk_np = sample_chunk.numpy()
    chunk_tflite = np.transpose(chunk_np, (0, 2, 1)) if chunk_is_ntc else chunk_np

    lt_state = [np.zeros(d["shape"], dtype=np.float32) for d in in_d[1:]]
    interp.set_tensor(in_d[0]["index"], chunk_tflite.astype(np.float32))
    for i, s in enumerate(lt_state):
        interp.set_tensor(in_d[1 + i]["index"], s)
    interp.invoke()

    K = stream_pre_dft.n_fft // 2 + 1
    lt_xreal_raw = interp.get_tensor(out_d[0]["index"])   # x_real
    lt_ximag_raw = interp.get_tensor(out_d[1]["index"])   # x_imag
    print(f"Chunk input layout : {'NTC' if chunk_is_ntc else 'NCT'}")
    print(f"TFLite x_real shape: {list(lt_xreal_raw.shape)}")
    print(f"TFLite x_imag shape: {list(lt_ximag_raw.shape)}")

    # Detect layout: NTK=[1,F,K] vs NKT=[1,K,F]
    def ensure_ntk(arr):
        return np.transpose(arr, (0, 2, 1)) if arr.ndim == 3 and arr.shape[1] == K else arr
    lt_xreal = ensure_ntk(lt_xreal_raw)
    lt_ximag = ensure_ntk(lt_ximag_raw)

    # Apply numpy IRFFT (= tf.signal.irfft) + window + OLA
    win_np = stream_pre_dft.window.numpy()
    lt_tf  = irfft_numpy(lt_xreal, lt_ximag, stream_pre_dft.n_fft)[..., : stream_pre_dft.win_len]
    lt_tf  = lt_tf * win_np
    istft_prev_np = np.zeros((1, stream_pre_dft.tail), dtype=np.float32)
    lt_audio, _ = overlap_add_numpy(lt_tf, istft_prev_np, stream_pre_dft.hop, stream_pre_dft.tail)
    pt_audio = audio_np   # PT audio computed above

    diff = np.abs(pt_audio.flatten() - lt_audio.flatten()).max()
    mean_diff = np.abs(pt_audio.flatten() - lt_audio.flatten()).mean()
    print(f"PT audio range: [{pt_audio.min():.4f}, {pt_audio.max():.4f}]")
    print(f"LT audio range: [{lt_audio.min():.4f}, {lt_audio.max():.4f}]")
    print(f"Max |PT−LT|   : {diff:.6e}")
    print(f"Mean|PT−LT|   : {mean_diff:.6e}")
    if diff < 1e-2:
        print("✓ PyTorch and TFLite outputs match closely")
    else:
        print("⚠ Large difference between PyTorch and TFLite")

    print("\nDone.")


if __name__ == "__main__":
    main()
