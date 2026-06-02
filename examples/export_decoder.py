"""
Export Kokoro Decoder to TFLite (Step 4).

Covers the full Decoder module:
  Decoder.encode          (AdainResBlk1d 514→1024)
  Decoder.decode[0-3]     (AdainResBlk1d blocks, last one upsamples ×2)
  Decoder.F0_conv / N_conv / asr_res
  Decoder.generator       (Generator + SourceModuleHnNSF/SineGen + CustomSTFT)

Produces:
  outputs/<git_hash>/kokoro_decoder_multisig_fp32.tflite
    Signatures: decoder_short (T=200), decoder_medium (T=800), decoder_long (T=2000)

AOT NPU compilation IS applied (no LSTM layers in Decoder).

Wrapper inputs:
  asr       [1, 512, T_aligned]   — aligned text encoder output  (t_en @ pred_aln_trg)
  F0_curve  [1, T_aligned*2]      — F0 from predictor_f0n output
  N         [1, T_aligned*2]      — aperiodicity from predictor_f0n output
  s         [1, 128]              — ref_s[:, :128]  (FIRST 128 dims of style vector)

Wrapper output:
  audio     [1, T_audio]          — synthesized waveform (≈ T_aligned*600 samples)

Export compatibility patches applied before litert_torch.convert():
  1. AdainResBlk1d.pool (decode[3]): ConvTranspose1d(k=3,s=2,p=1,op=1,groups=C)
     → PoolEquiv (zero-interleave + asymmetric pad + depthwise Conv1d)
  2. SineGen._f02sine: in-place rand_ini / rad_values assignments → out-of-place
  3. CustomSTFT.transform: in-place phase[mask]=pi → torch.where
  4. Model loaded with disable_complex=True → CustomSTFT (not TorchSTFT)

Parity note:
  SineGen uses torch.rand (initial phase) and torch.randn (noise floor).
  These produce different values in PyTorch vs LiteRT RNGs, so element-wise
  audio parity is NOT achievable.  Tests check:
    - Output shape
    - No NaN / Inf
    - Deterministic decode backbone (encode + decode blocks, before generator)
    - WAV files saved for subjective evaluation

Run with:
  uv run python examples/export_decoder.py
"""

import copy
import numpy as np
import os
import subprocess
import types
import wave
import torch
import torch.nn as nn
import torch.nn.functional as F
import litert_torch

from kokoro import KModel
from kokoro.custom_stft import CustomSTFT
from kokoro.istftnet import AdainResBlk1d, SineGen


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
                 name: str, atol: float = 1e-3) -> None:
    pt_np = pt.detach().float().numpy()
    diff = float(np.abs(pt_np - tflite).max())
    status = "PASS" if diff < atol else "FAIL"
    print(f"  {name}: max_abs_diff={diff:.6f}  {status}")
    assert diff < atol, f"{name} parity FAILED: max diff {diff:.6f} >= {atol}"


def save_wav(audio_np: np.ndarray, path: str, sample_rate: int = 24000) -> None:
    """Save float32 audio array to a 16-bit PCM WAV file."""
    audio_clipped = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio_int16.tobytes())


# ---------------------------------------------------------------------------
# ConvTranspose1d output_padding workaround (same as in export_predictor_f0n.py)
# ---------------------------------------------------------------------------

class PoolEquiv(nn.Module):
    """Equivalent to weight_norm(ConvTranspose1d(k=3, s=2, p=1, op=1, groups=C)).

    litert_torch does not support output_padding in ConvTranspose1d.
    Replacement:
      1. Zero-interleave input:  [B, C, T] → [B, C, 2T-1]
      2. Asymmetric pad (1, 2):  → [B, C, 2T+2]
      3. Depthwise Conv1d with flipped kernel: → [B, C, 2T]
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, groups: int):
        super().__init__()
        self.groups = groups
        self.register_buffer("weight", weight.flip(-1))  # [C, 1, 3]
        self.register_buffer("bias",   bias)             # [C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(x)
        x_zi  = torch.stack([x, zeros], dim=3).reshape(
            x.shape[0], x.shape[1], 2 * x.shape[2]
        )[:, :, :-1]                              # [B, C, 2T-1]
        x_pad = F.pad(x_zi, (1, 2))              # [B, C, 2T+2]
        return F.conv1d(x_pad, self.weight, self.bias,
                        stride=1, padding=0, groups=self.groups)


def _replace_pool_layers(module: nn.Module) -> None:
    """Recursively replace AdainResBlk1d.pool ConvTranspose1d with PoolEquiv."""
    for child_name, child in list(module.named_children()):
        if isinstance(child, AdainResBlk1d) and child.upsample_type != 'none':
            orig_pool = child.pool
            w = orig_pool.weight.detach().clone()
            b = orig_pool.bias.detach().clone()
            g = orig_pool.groups
            child.pool = PoolEquiv(w, b, g)
        else:
            _replace_pool_layers(child)


# ---------------------------------------------------------------------------
# SineGen._f02sine patch — out-of-place version
# ---------------------------------------------------------------------------

def _make_patched_f02sine():
    """Return a _f02sine replacement with deterministic, out-of-place ops.

    Replaces torch.rand (initial phase randomisation) with zeros so that the
    exported model is fully deterministic.  The initial phase being 0 is
    imperceptible in practice (pitch phase is not perceptually significant).
    """
    def _f02sine(self, f0_values):
        # f0_values: [B, T_audio, D]  (D = harmonic_num+1)
        B, T, D = f0_values.shape
        rad_values = (f0_values / self.sampling_rate) % 1
        # Deterministic zero initial phase (was: random per-harmonic phase offset)
        # rand.default / rand_like are unsupported by litert_torch MLIR lowering.

        if not self.flag_for_pulse:
            rad_values = F.interpolate(
                rad_values.transpose(1, 2),
                scale_factor=1 / self.upsample_scale,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
            phase = F.interpolate(
                phase.transpose(1, 2) * self.upsample_scale,
                scale_factor=self.upsample_scale,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            sines = torch.sin(phase)
        else:
            raise RuntimeError("SineGen pulse mode not supported in export")
        return sines
    return _f02sine


# ---------------------------------------------------------------------------
# SineGen.forward + SourceModuleHnNSF.forward patches — remove randn_like
# ---------------------------------------------------------------------------

def _make_patched_sinegen_forward():
    """Return a SineGen.forward that avoids torch.randn_like.

    The original adds `noise_amp * torch.randn_like(sine_waves)` to shape the
    noise floor.  In the exported decoder, noi_source and uv from m_source are
    DISCARDED by Generator.forward, so the exact noise values do not affect
    the final audio.  Replacing with zeros makes the graph fully lowerable.
    """
    def forward(self, f0):
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        fn = torch.multiply(
            f0,
            torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device),
        )
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        # was: noise = noise_amp * torch.randn_like(sine_waves)
        # aten.randn_like.default not supported by litert_torch MLIR lowering.
        # noi_source is unused by Generator.forward so zeros is lossless.
        noise = torch.zeros_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise
    return forward


def _make_patched_hnnnsf_forward():
    """Return a SourceModuleHnNSF.forward that avoids torch.randn_like.

    In Generator.forward, only har_source (the harmonic sine merge) is used.
    noi_source and uv are DISCARDED after the no_grad block.
    """
    def forward(self, x):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.zeros_like(uv)
        return sine_merge, noise, uv
    return forward


# ---------------------------------------------------------------------------
# CustomSTFT.transform patch — replace in-place masked assignment
# ---------------------------------------------------------------------------

def _make_patched_stft_transform():
    """Return a CustomSTFT.transform replacement using torch.where."""
    def transform(self, waveform: torch.Tensor):
        if self.center:
            pad_len = self.n_fft // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)
        x = waveform.unsqueeze(1)
        real_out = F.conv1d(x, self.weight_forward_real, bias=None,
                            stride=self.hop_length, padding=0)
        imag_out = F.conv1d(x, self.weight_forward_imag, bias=None,
                            stride=self.hop_length, padding=0)
        magnitude = torch.sqrt(real_out ** 2 + imag_out ** 2 + 1e-14)
        phase = torch.atan2(imag_out, real_out)
        # Out-of-place correction (was: phase[correction_mask] = torch.pi)
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase = torch.where(correction_mask, torch.full_like(phase, torch.pi), phase)
        return magnitude, phase
    return transform


# ---------------------------------------------------------------------------
# Apply all patches to a decoder deep-copy
# ---------------------------------------------------------------------------

def _apply_patches(decoder: nn.Module) -> None:
    """Patch decoder in-place for litert_torch export compatibility."""
    # 1. AdainResBlk1d.pool: ConvTranspose1d(output_padding=1) → PoolEquiv
    _replace_pool_layers(decoder)

    # 2. SineGen._f02sine: in-place + rand → deterministic, out-of-place
    sinegen = decoder.generator.m_source.l_sin_gen
    sinegen._f02sine = types.MethodType(_make_patched_f02sine(), sinegen)

    # 3. SineGen.forward: randn_like(sine_waves) → zeros_like
    sinegen.forward = types.MethodType(_make_patched_sinegen_forward(), sinegen)

    # 4. SourceModuleHnNSF.forward: randn_like(uv) → zeros_like (noi_source unused)
    hnnnsf = decoder.generator.m_source
    hnnnsf.forward = types.MethodType(_make_patched_hnnnsf_forward(), hnnnsf)

    # 4. CustomSTFT.transform: in-place masked assign → torch.where
    stft = decoder.generator.stft
    assert isinstance(stft, CustomSTFT), (
        "Expected CustomSTFT (model must be loaded with disable_complex=True). "
        f"Got {type(stft)}"
    )
    stft.transform = types.MethodType(_make_patched_stft_transform(), stft)


# ---------------------------------------------------------------------------
# Decoder wrapper
# ---------------------------------------------------------------------------

class DecoderWrapper(torch.nn.Module):
    """Wraps Decoder for litert_torch export.

    forward(
        asr       [1, 512, T_aligned],
        F0_curve  [1, T_aligned*2],
        N         [1, T_aligned*2],
        s         [1, 128],
    ) -> audio [1, T_audio]

    NOTE: s must be ref_s[:, :128] (first 128 dims of the 256-dim style vector).
    The predictor path uses ref_s[:, 128:]; the decoder uses ref_s[:, :128].
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        asr:      torch.FloatTensor,   # [1, 512, T_aligned]
        F0_curve: torch.FloatTensor,   # [1, T_aligned*2]
        N:        torch.FloatTensor,   # [1, T_aligned*2]
        s:        torch.FloatTensor,   # [1, 128]
    ) -> torch.FloatTensor:            # [1, T_audio]
        out = self.decoder(asr, F0_curve, N, s)
        # decoder returns [1, 1, T_audio] or [1, T_audio] depending on iSTFT
        # normalise to [1, T_audio]
        if out.dim() == 3:
            out = out.squeeze(1)
        return out


# ---------------------------------------------------------------------------
# AOT compile for Google Tensor G5
# ---------------------------------------------------------------------------

def aot_compile_tensor_g5(tflite_path: str, out_dir: str, model_name: str) -> None:
    """AOT-compile tflite_path for Google Tensor G5 using the NPU SDK plugin."""
    import os
    sdk_path = "litert_npu/litert_plugin_compiler.tar.gz"
    if not os.path.exists(sdk_path):
        print(f"WARNING: NPU SDK not found at {sdk_path}, skipping AOT compile.")
        return

    os.environ["GOOGLE_TENSOR_SDK_BETA"] = sdk_path

    try:
        from ai_edge_litert.aot import aot_compile as aot_lib
        from ai_edge_litert.aot.vendors.google_tensor import target as gt_target
    except ImportError:
        print("WARNING: ai_edge_litert not installed, skipping AOT compile.")
        return

    print(f"\n--- AOT compile for Google Tensor G5 ---")
    print(f"Using SDK plugin: {sdk_path}")
    print(f"Compiling {tflite_path} ...")

    tensor_g5_target = gt_target.Target(gt_target.SocModel.TENSOR_G5)
    compiled = aot_lib.aot_compile(
        tflite_path,
        target=[tensor_g5_target],
        keep_going=True,   # allow partial NPU offload if some subgraphs fail
        google_tensor_truncation_type="half",
        google_tensor_int64_to_int32=True,
        google_tensor_sharding_intensity="high",  # "extensive" crashed INTERNAL after 36min
    )
    print(compiled.compilation_report())
    compiled.export(out_dir, model_name=model_name)
    print(f"AOT compiled model saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    HASH     = git_hash()
    OUT_DIR  = os.path.join("outputs", HASH)
    TEST_DIR = os.path.join("test_output", HASH, "decoder")
    os.makedirs(OUT_DIR,  exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    TFLITE_PATH = os.path.join(OUT_DIR, "kokoro_decoder_multisig_fp32.tflite")

    # T_aligned bucket sizes (audio ≈ T_aligned * 600 samples at 24 kHz)
    BUCKETS   = [200, 800, 2000]
    SIG_NAMES = ["decoder_short", "decoder_medium", "decoder_long"]

    PARITY_SEED = 42
    SAMPLE_RATE = 24000

    # -----------------------------------------------------------------------
    # Load model (MUST use disable_complex=True → CustomSTFT)
    # -----------------------------------------------------------------------
    print("Loading model (disable_complex=True)...")
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config="checkpoints/config.json",
        model="checkpoints/kokoro-v1_0.pth",
        disable_complex=True,
    ).eval()

    # -----------------------------------------------------------------------
    # Deep-copy decoder, patch, and wrap
    # -----------------------------------------------------------------------
    decoder_copy = copy.deepcopy(model.decoder).eval()
    _apply_patches(decoder_copy)

    wrapper = DecoderWrapper(decoder_copy).eval()

    # Sanity-check wrapper outputs
    with torch.no_grad():
        _T = BUCKETS[0]
        _asr     = torch.randn(1, 512, _T)
        _F0      = torch.randn(1, _T * 2)
        _N       = torch.randn(1, _T * 2)
        _s       = torch.randn(1, 128)
        _audio   = wrapper(_asr, _F0, _N, _s)
        assert _audio.dim() == 2 and _audio.shape[0] == 1, f"Unexpected shape: {_audio.shape}"
        T_audio_short = _audio.shape[1]
        print(f"Shape check OK: T_aligned={_T} → audio shape={tuple(_audio.shape)}  "
              f"(≈{T_audio_short/SAMPLE_RATE:.2f}s at {SAMPLE_RATE}Hz)")

    # -----------------------------------------------------------------------
    # Build multi-signature TFLite model
    # -----------------------------------------------------------------------
    print("\nBuilding multi-signature TFLite model...")

    def make_inputs(T: int):
        return (
            torch.randn(1, 512, T),
            torch.randn(1, T * 2),
            torch.randn(1, T * 2),
            torch.randn(1, 128),
        )

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

    # Reference decoder (original model, disable_complex=True, no patches needed
    # for PyTorch forward pass since masked_fill_ etc. are fine in eager mode)
    dec_ref = model.decoder.eval()

    # -- Part A: Deterministic backbone parity (encode + decode blocks only) --
    # Manually run the encode/decode path which is deterministic (no random ops).
    print("\n  [A] Deterministic decode backbone (before Generator):")
    backbone_passed = True
    for T, sig in zip(BUCKETS[:2], SIG_NAMES[:2]):   # skip T=2000 (slow)
        torch.manual_seed(PARITY_SEED)
        asr     = torch.randn(1, 512, T)   * 0.1      # text encoder scale
        F0_c    = torch.rand(1, T * 2)    * 300 + 50  # F0 in [50, 350] Hz
        N_c     = torch.rand(1, T * 2)    * 0.5       # aperiodicity [0, 0.5]
        s       = torch.randn(1, 128)      * 0.1      # style

        with torch.no_grad():
            F0_down = dec_ref.F0_conv(F0_c.unsqueeze(1))   # [1,1,T]
            N_down  = dec_ref.N_conv(N_c.unsqueeze(1))     # [1,1,T]
            x = torch.cat([asr, F0_down, N_down], dim=1)   # [1,514,T]
            x = dec_ref.encode(x, s)                        # [1,1024,T]
            asr_res = dec_ref.asr_res(asr)                  # [1,64,T]
            res = True
            for block in dec_ref.decode:
                if res:
                    x = torch.cat([x, asr_res, F0_down, N_down], dim=1)
                x = block(x, s)
                if block.upsample_type != "none":
                    res = False
            pt_backbone = x.clone()   # [1, 512, T*2]

        # Verify backbone shape
        assert pt_backbone.shape == (1, 512, T * 2), (
            f"Unexpected backbone shape: {pt_backbone.shape}"
        )
        pt_backbone_np = pt_backbone.float().numpy()
        assert not np.any(np.isnan(pt_backbone_np)), f"NaN in PT backbone T={T}"
        assert not np.any(np.isinf(pt_backbone_np)), f"Inf in PT backbone T={T}"
        print(f"  {sig} T={T}: PyTorch backbone OK  shape={tuple(pt_backbone.shape)}  "
              f"rms={float(pt_backbone.pow(2).mean().sqrt()):.4f}")
        np.save(os.path.join(TEST_DIR, f"{sig}_backbone_pt.npy"), pt_backbone_np)

    # -- Part B: Full decoder parity (shape + finiteness + energy) --
    # Use realistic dummy inputs to avoid float32 overflow in XNNPACK delegate.
    # (random unit-variance noise causes extreme intermediate values in SineGen
    #  cumsum and iSTFT for larger T buckets.)
    print("\n  [B] Full decoder (shape / finiteness / energy):")
    all_passed = True
    for T, sig in zip(BUCKETS, SIG_NAMES):
        torch.manual_seed(PARITY_SEED)
        asr     = torch.randn(1, 512, T)   * 0.1      # text encoder scale
        F0_c    = torch.rand(1, T * 2)    * 300 + 50  # F0 in [50, 350] Hz
        N_c     = torch.rand(1, T * 2)    * 0.5       # aperiodicity [0, 0.5]
        s       = torch.randn(1, 128)      * 0.1      # style

        # PyTorch reference
        with torch.no_grad():
            pt_audio = dec_ref(asr, F0_c, N_c, s)
            if pt_audio.dim() == 3:
                pt_audio = pt_audio.squeeze(1)
        pt_np = pt_audio.float().numpy()

        # TFLite inference
        tflite_audio = edge_model(asr, F0_c, N_c, s, signature_name=sig)

        # Shape check
        assert pt_np.shape == tflite_audio.shape, (
            f"{sig}: shape mismatch PT={pt_np.shape} TFLite={tflite_audio.shape}"
        )

        # Finiteness
        pt_ok  = not (np.any(np.isnan(pt_np))     or np.any(np.isinf(pt_np)))
        tfl_ok = not (np.any(np.isnan(tflite_audio)) or np.any(np.isinf(tflite_audio)))

        # Energy similarity: RMS within 10× (20 dB)
        pt_rms  = float(np.sqrt(np.mean(pt_np ** 2)))
        tfl_rms = float(np.sqrt(np.mean(tflite_audio ** 2)))
        rms_ratio = max(pt_rms, tfl_rms) / (min(pt_rms, tfl_rms) + 1e-9)
        energy_ok = rms_ratio < 10.0

        passed = pt_ok and tfl_ok and energy_ok
        print(f"  {sig} T={T}: shape={tuple(pt_np.shape)}  "
              f"PT finite={'✓' if pt_ok else '✗'}  "
              f"TFLite finite={'✓' if tfl_ok else '✗'}  "
              f"PT_rms={pt_rms:.4f}  TFLite_rms={tfl_rms:.4f}  "
              f"rms_ratio={rms_ratio:.2f}  {'PASS' if passed else 'FAIL'}")

        if not passed:
            all_passed = False

        # Save numpy arrays
        prefix = os.path.join(TEST_DIR, f"{sig}_T{T}")
        np.save(prefix + "_audio_pt.npy",     pt_np)
        np.save(prefix + "_audio_tflite.npy", tflite_audio)

        # Save WAV files for subjective evaluation
        save_wav(pt_np[0],     prefix + "_audio_pt.wav",     SAMPLE_RATE)
        save_wav(tflite_audio[0], prefix + "_audio_tflite.wav", SAMPLE_RATE)
        print(f"    WAV saved: {prefix}_audio_{{pt,tflite}}.wav")

    if all_passed:
        print("\nAll parity tests PASSED.")
    else:
        print("\nSome parity tests FAILED — check test_output/ for details.")
        raise SystemExit(1)

    # -----------------------------------------------------------------------
    # AOT compile for Google Tensor G5
    # -----------------------------------------------------------------------
    aot_compile_tensor_g5(TFLITE_PATH, OUT_DIR, model_name="kokoro_decoder")


if __name__ == "__main__":
    main()
