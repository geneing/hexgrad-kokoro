#!/usr/bin/env python3
"""
streaming_vocos_export_litert.py
================================
Standalone script to export all Kokoro TTS + streaming Vocos modules to
LiteRT (TFLite) format.

Exports:
  bert.tflite               — BERT encoder           STATIC [1, 510] → [1, 512, 510]
  duration_predictor.tflite — Duration predictor      STATIC 510-length I/O
  acoustic_expand.tflite    — BiLSTM acoustic expand  STATIC T_ACOUSTIC_MAX via Keras
  f0n_predictor.tflite      — F0/N predictor          STATIC T_ACOUSTIC_MAX
  vocoder_conditioner.tflite— Vocos conditioner        STATIC T_ACOUSTIC_MAX
  vocoder_stream_chunk.tflite— Vocos backbone+head     STATIC chunk_frames, real ISTFT

Run from the project root:
    python streaming_vocos_export_litert.py [--force]
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import json
import math as _math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Numerics ──────────────────────────────────────────────────────────────────
import numpy as np

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ── Kokoro TTS ────────────────────────────────────────────────────────────────
from kokoro import KModel, KPipeline
from kokoro.model import KModelForONNX

# ── Streaming Vocos ───────────────────────────────────────────────────────────
from streaming_vocos import StreamingVocos

# ── LiteRT / TFLite ──────────────────────────────────────────────────────────
import litert_torch
import tensorflow as tf

# =============================================================================
#  Configuration
# =============================================================================

CONFIG_FILE      = "checkpoints/config.json"
CHECKPOINT_PATH  = "checkpoints/kokoro-v1_0.pth"
VOCOS_CKPT       = "vocos_fp16.pt"
VOICE_PATH       = "checkpoints/voices/af_bella.pt"

EXPORT_DIR       = Path("onnx_streaming_vocos")

SAMPLE_RATE      = 24000
MAX_INPUT_LENGTH = 510
T_ACOUSTIC_MAX   = 8096
VOCOS_CHUNK_FRAMES = 16

TEXT = (
    "I had returned to civil practice and had finally abandoned Holmes in his "
    "Baker Street rooms, although I continually visited him and occasionally "
    "even persuaded him to forgo his Bohemian habits so far as to come and "
    "visit us."
)

# =============================================================================
#  Timing helper
# =============================================================================

def _t() -> float:
    return time.perf_counter()


# =============================================================================
#  Input preparation helpers
# =============================================================================

def load_input_ids(pipeline: KPipeline, text: str):
    if pipeline.lang_code in "ab":
        _, tokens = pipeline.g2p(text)
        for gs, ps, tks in pipeline.en_tokenize(tokens):
            if not ps:
                continue
    else:
        ps, _ = pipeline.g2p(text)

    if len(ps) > 510:
        ps = ps[:510]

    input_ids = [i for i in (pipeline.model.vocab.get(p) for p in ps) if i is not None]
    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(pipeline.model.device)
    return ps, input_ids


def load_voice(pipeline: KPipeline, voice: str, phonemes):
    pack = pipeline.load_voice(voice).to("cpu")
    return pack[len(phonemes) - 1]


def load_sample(model: KModelForONNX, text: str = TEXT, voice: str = VOICE_PATH):
    pipeline = KPipeline(lang_code="a", model=model.kmodel, device="cpu")
    phonemes, input_ids = load_input_ids(pipeline, text)
    style = load_voice(pipeline, voice, phonemes)
    speed = torch.IntTensor([1])
    return input_ids, style, speed


def build_padded_inputs(input_ids: Tensor):
    """Pad input_ids to MAX_INPUT_LENGTH and build matching text_mask."""
    text_mask = torch.zeros(1, MAX_INPUT_LENGTH, dtype=torch.float32)
    text_mask[0, : input_ids.shape[1]] = 1
    input_ids = F.pad(input_ids, (0, MAX_INPUT_LENGTH - input_ids.shape[1]))
    return input_ids, text_mask


# =============================================================================
#  Module definitions
# =============================================================================

class BertEncoderModule(nn.Module):
    """input_ids [1,510] → d_en [1,512,510]  — static shapes, no dynamic axes."""
    def __init__(self, kmodel):
        super().__init__()
        self.bert         = kmodel.bert
        self.bert_encoder = kmodel.bert_encoder

    def forward(self, input_ids: Tensor, text_mask: Tensor) -> Tensor:
        bert_dur = self.bert(input_ids, attention_mask=text_mask)
        return self.bert_encoder(bert_dur).transpose(-1, -2)


class DurationPredictorCore(nn.Module):
    """Predict per-phoneme durations and produce pre-expansion feature tensors.

    All shapes are static (padded to MAX_INPUT_LENGTH=510):
        Inputs:
            d_en       [1, 512, 510]   — BERT encoder output
            style      [1, 256]        — voice style vector
            text_mask  [1, 510]        — valid-token mask (float)
            speed      []  int32       — speed multiplier
            input_ids  [1, 510]        — token IDs for text_encoder

        Outputs:
            pred_dur   [510]           — rounded per-phoneme durations
            d_enc      [1, h, 510]     — predictor text-encoder features (C-first)
            t_en       [1, 512, 510]   — text-encoder features for ASR path
    """
    def __init__(self, kmodel):
        super().__init__()
        self.predictor    = kmodel.predictor
        self.text_encoder = kmodel.text_encoder

    def forward(
        self,
        d_en: Tensor,
        style: Tensor,
        text_mask: Tensor,
        speed: Tensor,
        input_ids: Tensor,
    ):
        d = self.predictor.text_encoder(d_en, style[:, 128:], text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = text_mask * torch.sigmoid(duration).sum(axis=-1) / speed.float()
        pred_dur = torch.round(duration).squeeze()          # [510]

        d_enc = d.transpose(-1, -2)                         # [1, h, 510]
        t_en  = self.text_encoder(input_ids, text_mask)     # [1, 512, 510]

        return pred_dur, d_enc, t_en


def expand_durations(pred_dur: Tensor, T_max: int = T_ACOUSTIC_MAX) -> Tuple[Tensor, int]:
    """Compute expanded indices padded to T_max from rounded per-phoneme durations.

    Args:
        pred_dur: [510] float — rounded durations (0 for padding positions)
        T_max:    fixed output length (default: T_ACOUSTIC_MAX)

    Returns:
        expanded_indices: [T_max] long — index into the 510-length phoneme dim
        T_acoustic:       int — actual number of valid frames (<= T_max)
    """
    boundaries = torch.cumsum(pred_dur, dim=0)
    T_acoustic = min(int(boundaries[-1].item()), T_max)
    values = torch.arange(boundaries[-1])
    expanded_indices = torch.sum(
        boundaries.unsqueeze(1) <= values.unsqueeze(0), dim=0
    ).long()
    return expanded_indices, T_acoustic


class AcousticExpandModule(nn.Module):
    """Apply shared BiLSTM to already-expanded predictor features.

        Input:  d_enc_expanded  [1, h, T_ACOUSTIC_MAX]  — C-first, static T
        Output: en              [1, T_ACOUSTIC_MAX, h']  — after shared BiLSTM
    """
    def __init__(self, kmodel):
        super().__init__()
        self.shared = kmodel.predictor.shared

    def forward(self, d_enc_expanded: Tensor) -> Tensor:
        en, _ = self.shared(d_enc_expanded.transpose(-1, -2))
        return en


class F0NPredictorModule(nn.Module):
    """(en [1,T_ACOUSTIC_MAX,h'], style) → (F0_pred [1,2*T_ACOUSTIC_MAX], N_pred [1,2*T_ACOUSTIC_MAX])"""
    def __init__(self, kmodel):
        super().__init__()
        self.predictor = kmodel.predictor

    def forward(self, en: Tensor, style: Tensor):
        return self.predictor.F0Ntrain(en, style[:, 128:256])


class VocosConditionerModule(nn.Module):
    """features [1,642,T] → conditioned [1,model_input_channels,T]"""
    def __init__(self, vocos: StreamingVocos):
        super().__init__()
        self.conditioner = vocos.model.conditioner

    def forward(self, features: Tensor) -> Tensor:
        return self.conditioner(features)


def build_vocos_features(
    asr: Tensor,       # [1, 512, T_asr]
    F0_pred: Tensor,   # [1, T_f0]
    N_pred: Tensor,    # [1, T_f0]
    style: Tensor,     # [1, 256]
) -> Tensor:
    """Assemble [1, 642, T_f0] feature tensor matching vocos training format."""
    B, _, T_asr = asr.shape
    T_f0 = F0_pred.shape[-1]
    if T_asr != T_f0:
        asr = F.interpolate(asr.float(), size=T_f0, mode="linear", align_corners=False)
    f0    = F0_pred.unsqueeze(1).float()
    n     = N_pred.unsqueeze(1).float()
    s_exp = style[:, :128].unsqueeze(-1).expand(B, -1, T_f0)
    return torch.cat([asr, f0, n, s_exp], dim=1)   # [1, 642, T_f0]


class VocosStreamChunk(nn.Module):
    """Stateless single-chunk backbone+head for ONNX/LiteRT export.

    State layout (embed + 8 blocks + ISTFT buffer):
        embed_prev      : [1, 192, 6]
        block_0_prev    : [1, 384, 6]
        ...
        block_7_prev    : [1, 384, 6]
        istft_prev      : [1, 900]
    """

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
        self.n_fft    = hd.istft.n_fft
        self.hop      = hd.istft.hop
        self.win_len  = hd.istft.win_length
        self.tail     = hd.istft.tail

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

    def state_from_tuple(self, t: Tuple[Tensor, ...]) -> Dict[str, Tensor]:
        keys = ["embed_prev"] + [f"block_{i}_prev" for i in range(self.N_LAYERS)] + ["istft_prev"]
        return dict(zip(keys, t))

    def _causal_conv(
        self,
        norm_conv1d: nn.Module,
        x: Tensor,
        prev: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        xp = torch.cat([prev, x], dim=-1)
        y  = norm_conv1d(xp)
        return y, xp[..., -self.KERNEL_M1:]

    def forward(
        self,
        conditioned_chunk: Tensor,
        embed_prev: Tensor,
        *block_and_istft,
    ) -> Tuple[Tensor, ...]:
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


class VocosStreamChunkReal(VocosStreamChunk):
    """LiteRT / TFLite-compatible variant.

    Replaces the complex-valued irfft with an explicit real-arithmetic IDFT
    using precomputed cosine/sine basis matrices, avoiding any use of
    complex tensors.  All other logic is identical to VocosStreamChunk.
    """

    def __init__(self, vocos: StreamingVocos):
        super().__init__(vocos)
        K = self.n_fft // 2 + 1
        k = torch.arange(K).float()
        t = torch.arange(self.n_fft).float()
        angle = 2 * _math.pi * k.unsqueeze(1) * t.unsqueeze(0) / self.n_fft
        weights = torch.ones(K)
        weights[1:-1] = 2.0
        self.register_buffer("cos_basis", (weights.unsqueeze(1) * torch.cos(angle)) / self.n_fft)
        self.register_buffer("sin_basis", (weights.unsqueeze(1) * torch.sin(angle)) / self.n_fft)

    def forward(
        self,
        conditioned_chunk: Tensor,
        embed_prev: Tensor,
        *block_and_istft,
    ) -> Tuple[Tensor, ...]:
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
        h      = self.head_out(x.transpose(1, 2)).transpose(1, 2)
        mag, phase = h.chunk(2, dim=1)
        mag    = torch.exp(mag).clamp(max=1e2)
        x_real = mag * torch.cos(phase)
        x_imag = mag * torch.sin(phase)

        time_frames = (
            torch.einsum("kt,bkf->bft", self.cos_basis, x_real)
            - torch.einsum("kt,bkf->bft", self.sin_basis, x_imag)
        )[..., : self.win_len]
        time_frames = time_frames * self.window

        ola_in  = time_frames.transpose(1, 2)
        ola_out = self.overlap_add(ola_in)[:, 0, :]

        audio_overlap  = ola_out[:, :self.tail] + istft_prev
        audio_rest     = ola_out[:, self.tail : frames * self.hop]
        audio          = torch.cat([audio_overlap, audio_rest], dim=-1)
        new_istft_prev = ola_out[:, frames * self.hop:]

        return (audio, new_embed_prev) + tuple(new_block_prevs) + (new_istft_prev,)


# =============================================================================
#  LiteRT export helpers
# =============================================================================

class _SafeInstanceNorm1d(nn.Module):
    """InstanceNorm1d via explicit mean/diff formulation (no view, no group_norm).

    Works with static shapes where T is consistently on axis 2.
    Both F.group_norm and aten.var.correction use internal view ops that can
    cause issues with complex tracing backends.
    """
    def __init__(self, src: nn.InstanceNorm1d):
        super().__init__()
        self.eps    = src.eps
        self.affine = src.affine
        if src.affine:
            self.weight = nn.Parameter(src.weight.detach().clone())
            self.bias   = nn.Parameter(src.bias.detach().clone())
        else:
            self.weight = None
            self.bias   = None

    def forward(self, x: Tensor) -> Tensor:
        # x: [N, C, T].  Normalise each (N, C) instance over T (axis 2).
        mean = x.mean(dim=2, keepdim=True)
        diff = x - mean
        var  = (diff * diff).mean(dim=2, keepdim=True)
        x    = diff * (var + self.eps).rsqrt()
        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]
        return x


def _patch_instance_norms(module: nn.Module) -> nn.Module:
    """Recursively replace all InstanceNorm1d with _SafeInstanceNorm1d in-place."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.InstanceNorm1d):
            setattr(module, name, _SafeInstanceNorm1d(child))
        else:
            _patch_instance_norms(child)
    return module


class _SafeConvTranspose1d(nn.Module):
    """ConvTranspose1d with output_padding replaced by explicit post-padding.

    LiteRT does not support output_padding > 0 in transposed convolutions, and
    also has a bug with grouped ConvTranspose (feature_group_count mismatch in
    MLIR verification for the transposed kernel layout).

    This wrapper implements depthwise ConvTranspose1d by:
      1. Zero-interleaving the input (inserting stride-1 zeros between samples).
      2. Padding the zero-interleaved tensor to match ConvTranspose output size.
      3. Applying a forward depthwise conv1d with the time-reversed kernel.

    This is numerically identical to the original operation and avoids both
    the output_padding and the grouped-TransposedConv MLIR issues.
    """
    def __init__(self, src: nn.ConvTranspose1d):
        super().__init__()
        assert src.groups == src.in_channels == src.out_channels, (
            "_SafeConvTranspose1d only handles fully depthwise ConvTranspose1d "
            f"(groups==in==out), got groups={src.groups}, in={src.in_channels}, "
            f"out={src.out_channels}"
        )
        self.stride   = src.stride[0]
        self.padding  = src.padding[0]
        self.out_pad  = src.output_padding[0]
        self.groups   = src.groups
        # Resolve weight_norm hooks (if present) and copy parameters.
        # weight shape: [in_channels, out_channels//groups, kernel] = [C, 1, K]
        self.weight = nn.Parameter(src.weight.detach().clone())
        if src.bias is not None:
            self.bias = nn.Parameter(src.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        N, C, T = x.shape
        S, P, OP, K = self.stride, self.padding, self.out_pad, self.weight.shape[2]

        # Step 1: zero-interleave — insert (S-1) zeros between each time step.
        #   [N, C, T] → stack with zeros → [N, C, T, S] → reshape → [N, C, T*S]
        #   then trim last (S-1) zeros: → [N, C, T*S - (S-1)]
        xz = torch.stack([x] + [torch.zeros_like(x)] * (S - 1), dim=-1)
        xz = xz.reshape(N, C, T * S)
        xz = xz[:, :, : T * S - (S - 1)]  # [N, C, 2T-1] for S=2

        # Step 2: pad to produce the correct ConvTranspose output length.
        #   ConvTranspose1d output length (with OP) = (T-1)*S - 2*P + K + OP
        #   = conv1d(xz_padded, K, stride=1, no-padding) with len(xz_padded)
        #     = (T*S - (S-1)) + 2*(K-1-P) + OP
        side_pad = K - 1 - P
        xz = F.pad(xz, (side_pad, side_pad + OP))

        # Step 3: forward depthwise conv1d with time-reversed kernel (= cross-corr).
        w = self.weight.flip(-1)  # [C, 1, K]
        return F.conv1d(xz, w, self.bias, stride=1, padding=0, groups=self.groups)


def _patch_conv_transpose_output_padding(module: nn.Module) -> nn.Module:
    """Recursively replace ConvTranspose1d layers with output_padding > 0
    with _SafeConvTranspose1d, which is LiteRT-compatible."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ConvTranspose1d) and any(p > 0 for p in child.output_padding):
            setattr(module, name, _SafeConvTranspose1d(child))
        else:
            _patch_conv_transpose_output_padding(child)
    return module


class F0NPredictorModuleStatic(nn.Module):
    """Static-shape variant of F0NPredictorModule for LiteRT export.

    Takes en_t=[1, 512, T_ACOUSTIC_MAX] (C-first) to avoid internal transposes
    that create new symbolic dims during tracing.
    """
    def __init__(self, kmodel):
        super().__init__()
        self.F0      = kmodel.predictor.F0
        self.N       = kmodel.predictor.N
        self.F0_proj = kmodel.predictor.F0_proj
        self.N_proj  = kmodel.predictor.N_proj

    def forward(self, en_t: Tensor, style: Tensor):
        # en_t is [1, 512, T_ACOUSTIC_MAX] — C-first layout
        s = style[:, 128:256]
        F0 = en_t
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = en_t
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)


def _pt_to_keras_bilstm_weights(pt_lstm: nn.LSTM):
    """Convert PyTorch BiLSTM weights to Keras Bidirectional(LSTM) format."""
    def _direction_weights(suffix):
        wih = pt_lstm._parameters[f"weight_ih_l0{suffix}"].detach().numpy()
        whh = pt_lstm._parameters[f"weight_hh_l0{suffix}"].detach().numpy()
        bih = pt_lstm._parameters[f"bias_ih_l0{suffix}"].detach().numpy()
        bhh = pt_lstm._parameters[f"bias_hh_l0{suffix}"].detach().numpy()
        return [wih.T, whh.T, bih + bhh]

    return _direction_weights(""), _direction_weights("_reverse")


def build_acoustic_expand_keras(pt_module: AcousticExpandModule) -> tf.keras.Model:
    """Build a Keras Bidirectional(LSTM) model matching AcousticExpandModule."""
    pt_lstm = pt_module.shared
    H = pt_lstm.hidden_size
    I = pt_lstm.input_size

    inp = tf.keras.Input(shape=(None, I), batch_size=1, name="d_enc_expanded")
    fwd_lstm = tf.keras.layers.LSTM(H, return_sequences=True, name="fwd")
    bwd_lstm = tf.keras.layers.LSTM(H, return_sequences=True, go_backwards=True, name="bwd")
    out = tf.keras.layers.Bidirectional(fwd_lstm, backward_layer=bwd_lstm,
                                        merge_mode="concat", name="bilstm")(inp)
    model = tf.keras.Model(inputs=inp, outputs=out, name="acoustic_expand")
    fwd_w, bwd_w = _pt_to_keras_bilstm_weights(pt_lstm)
    model.get_layer("bilstm").forward_layer.set_weights(fwd_w)
    model.get_layer("bilstm").backward_layer.set_weights(bwd_w)
    return model


def export_acoustic_expand_tflite(
    pt_module: AcousticExpandModule,
    pt_sample: Tensor,
    path: Path,
    force: bool = False,
) -> None:
    """Export AcousticExpandModule to TFLite via Keras BiLSTM path."""
    if not force and path.exists():
        sz = path.stat().st_size / 1e6
        print(f"  ↷ {path.name}  (skipped — already exists, {sz:.2f} MB)")
        return

    t0 = _t()
    try:
        keras_model = build_acoustic_expand_keras(pt_module)

        pt_in  = pt_sample.float()
        with torch.no_grad():
            pt_out = pt_module(pt_in)

        keras_in  = pt_in.permute(0, 2, 1).numpy()
        keras_out = keras_model.predict(keras_in, verbose=0)
        max_diff = float(np.abs(pt_out.numpy() - keras_out).max())
        print(f"  acoustic_expand numerics check: max |PT - Keras| = {max_diff:.2e}")
        assert max_diff < 1e-3, f"Numerical mismatch too large: {max_diff}"

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_bytes = converter.convert()
        path.write_bytes(tflite_bytes)

        elapsed = (_t() - t0) * 1e3
        sz = path.stat().st_size / 1e6
        print(f"  ✓ {path.name}  {elapsed:.0f} ms  {sz:.2f} MB")
    except Exception as e:
        elapsed = (_t() - t0) * 1e3
        print(f"  ✗ {path.name}  {elapsed:.0f} ms  — {e}")
        import traceback; traceback.print_exc()


def export_litert(
    module: nn.Module,
    sample_args: tuple,
    path: Path,
    force: bool = False,
) -> None:
    """Convert nn.Module to TFLite with STATIC shapes and save.

    Skips if file exists and force=False.
    """
    if not force and path.exists():
        sz = path.stat().st_size / 1e6
        print(f"  ↷ {path.name}  (skipped — already exists, {sz:.2f} MB)")
        return

    t0 = _t()
    try:
        edge_model = litert_torch.convert(module.eval(), sample_args, _ai_edge_converter_flags={
            "verbose": True,
            "detailed_conversion": True,
            "dynamic_shapes": False,
        })
        edge_model.export(str(path))
        elapsed = (_t() - t0) * 1e3
        sz = path.stat().st_size / 1e6
        print(f"  ✓ {path.name}  {elapsed:.0f} ms  {sz:.2f} MB")
    except Exception as e:
        elapsed = (_t() - t0) * 1e3
        print(f"  ✗ {path.name}  {elapsed:.0f} ms  — {e}")
        import traceback; traceback.print_exc()


# =============================================================================
#  Main export routine
# =============================================================================

def prepare_sample_inputs(kmodel, model, vocos):
    """Load models, run a forward pass, and collect all intermediate tensors
    needed as sample inputs for the LiteRT exporters."""

    bert_module     = BertEncoderModule(kmodel).eval()
    duration_module = DurationPredictorCore(kmodel).eval()
    acoustic_module = AcousticExpandModule(kmodel).eval()
    f0n_module      = F0NPredictorModule(kmodel).eval()
    cond_module     = VocosConditionerModule(vocos).eval()

    raw_input_ids, style, speed = load_sample(model, text=TEXT, voice=VOICE_PATH)
    input_ids, text_mask = build_padded_inputs(raw_input_ids)

    with torch.no_grad():
        d_en = bert_module(input_ids, text_mask)
        pred_dur, d_enc, t_en_static = duration_module(d_en, style, text_mask, speed, input_ids)
        expanded_indices, T_acoustic = expand_durations(pred_dur)
        d_enc_exp = torch.index_select(d_enc, 2, expanded_indices)
        asr_full  = torch.index_select(t_en_static, 2, expanded_indices)
        en        = acoustic_module(d_enc_exp)
        F0_pred_full, N_pred_full = f0n_module(en, style)
        asr     = asr_full[..., :T_acoustic]
        F0_pred = F0_pred_full[..., :2 * T_acoustic]
        N_pred  = N_pred_full[..., :2 * T_acoustic]
        features    = build_vocos_features(asr, F0_pred, N_pred, style)
        conditioned = cond_module(features.float())

    return dict(
        bert_module=bert_module,
        duration_module=duration_module,
        acoustic_module=acoustic_module,
        f0n_module=f0n_module,
        cond_module=cond_module,
        input_ids=input_ids,
        text_mask=text_mask,
        d_en=d_en,
        style=style,
        speed=speed,
        d_enc_exp=d_enc_exp,
        en=en,
        features=features,
        conditioned=conditioned,
    )


def export_all(force: bool = False):
    EXPORT_DIR.mkdir(exist_ok=True)

    litert_paths = {
        "bert":            EXPORT_DIR / "bert.tflite",
        "duration":        EXPORT_DIR / "duration_predictor.tflite",
        "acoustic_expand": EXPORT_DIR / "acoustic_expand.tflite",
        "f0n":             EXPORT_DIR / "f0n_predictor.tflite",
        "conditioner":     EXPORT_DIR / "vocoder_conditioner.tflite",
        "stream_chunk":    EXPORT_DIR / "vocoder_stream_chunk.tflite",
    }

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading Kokoro model...")
    kmodel = KModel(config=CONFIG_FILE, model=CHECKPOINT_PATH, disable_complex=True).to("cpu")
    model  = KModelForONNX(kmodel).eval()
    print("Loading StreamingVocos checkpoint...")
    vocos  = StreamingVocos.from_checkpoint(VOCOS_CKPT, chunk_frames=VOCOS_CHUNK_FRAMES,
                                            device="cpu", use_fp16=False)

    # ── Prepare sample tensors (single forward pass) ──────────────────────────
    print("Running forward pass to collect sample tensors...")
    s = prepare_sample_inputs(kmodel, model, vocos)

    # ── 1. BERT encoder  (STATIC) ─────────────────────────────────────────────
    print("\n── 1. BERT encoder")
    export_litert(
        s["bert_module"],
        sample_args=(s["input_ids"], s["text_mask"]),
        path=litert_paths["bert"],
        force=force,
    )

    # ── 2. Duration predictor  (STATIC) ───────────────────────────────────────
    print("\n── 2. Duration predictor")
    export_litert(
        s["duration_module"],
        sample_args=(s["d_en"], s["style"], s["text_mask"], s["speed"], s["input_ids"]),
        path=litert_paths["duration"],
        force=force,
    )

    # ── 3. Acoustic expand  (STATIC T_ACOUSTIC_MAX via Keras BiLSTM) ─────────
    print("\n── 3. Acoustic expand (Keras BiLSTM path)")
    export_acoustic_expand_tflite(
        s["acoustic_module"],
        pt_sample=s["d_enc_exp"],
        path=litert_paths["acoustic_expand"],
        force=force,
    )

    # ── 4. F0/N predictor  (STATIC T_ACOUSTIC_MAX) ───────────────────────────
    print("\n── 4. F0/N predictor")
    _f0n_static = F0NPredictorModuleStatic(kmodel)
    _patch_instance_norms(_f0n_static)
    _patch_conv_transpose_output_padding(_f0n_static)

    _en_t = s["en"].transpose(-1, -2).float()   # [1, 512, T_ACOUSTIC_MAX]
    with torch.no_grad():
        _f0n_ref  = s["f0n_module"](s["en"], s["style"])
        _f0n_test = _f0n_static(_en_t, s["style"])
    _f0n_diff = max(
        (_f0n_ref[0] - _f0n_test[0]).abs().max().item(),
        (_f0n_ref[1] - _f0n_test[1]).abs().max().item(),
    )
    print(f"  f0n_static numerics check: max |orig - static| = {_f0n_diff:.2e}")
    assert _f0n_diff < 1e-4, f"F0N numerics mismatch: {_f0n_diff}"

    export_litert(
        _f0n_static,
        sample_args=(_en_t, s["style"]),
        path=litert_paths["f0n"],
        force=force,
    )

    # ── 5. Vocos conditioner  (STATIC T_ACOUSTIC_MAX) ─────────────────────────
    print("\n── 5. Vocos conditioner")
    export_litert(
        s["cond_module"],
        sample_args=(s["features"].float(),),
        path=litert_paths["conditioner"],
        force=force,
    )

    # ── 6. Streaming chunk  (STATIC, real-arithmetic ISTFT) ───────────────────
    print("\n── 6. Vocos streaming chunk (real ISTFT)")
    stream_chunk_real  = VocosStreamChunkReal(vocos).eval()
    _init_state_tup    = stream_chunk_real.state_as_tuple(VocosStreamChunk.initial_state())
    _sample_chunk      = s["conditioned"][..., :VOCOS_CHUNK_FRAMES].float()

    export_litert(
        stream_chunk_real,
        sample_args=(_sample_chunk,) + _init_state_tup,
        path=litert_paths["stream_chunk"],
        force=force,
    )

    print("\n" + "=" * 60)
    print("LiteRT export complete.")
    print("  All modules exported with STATIC shapes.")
    print(f"  Output directory: {EXPORT_DIR.resolve()}")
    for name, path in litert_paths.items():
        if path.exists():
            sz = path.stat().st_size / 1e6
            print(f"    {path.name:<35} {sz:.2f} MB")
        else:
            print(f"    {path.name:<35} MISSING (export failed)")
    print("=" * 60)


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Kokoro TTS streaming Vocos to LiteRT")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-export all modules even if .tflite files already exist",
    )
    args = parser.parse_args()
    export_all(force=args.force)
