# Kokoro → LiteRT Export Agent Guide

## Project Goal

Export the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS model to
`.tflite` format using `litert-torch`, targeting AOT compilation for on-device
inference on Android (Pixel 10) on GPU and NPU. The model will also be
quantized to fp16 and int8.

---

## Repository Layout

```
kokoro/          # Python package – model, pipeline, modules
  model.py       # KModel (top-level nn.Module), KModelForONNX wrapper
  modules.py     # CustomAlbert, ProsodyPredictor, TextEncoder, DurationEncoder
  istftnet.py    # Decoder, Generator, AdaIN blocks, AdainResBlk1d
  custom_stft.py # CustomSTFT – conv1d-based STFT/iSTFT (ONNX-friendly)
  pipeline.py    # KPipeline – G2P, voice management
examples/
  export.py      # Existing ONNX export reference
export/          # LiteRT export scripts for individual Kokoro sub-modules
  export_bert.py
  export_text_encoder.py
  export_predictor_dur.py
  export_predictor_f0n.py
  export_decoder.py
litert-torch/    # Checked-out litert-torch repo (reference & local dev)
  docs/pytorch_converter/README.md   # Conversion API walkthrough
  litert_torch/  # Library source
checkpoints/     # Model weights + config.json
pyproject.toml   # uv-managed dependencies
```

---

## Model Architecture

`KModel` is composed of five independently exportable sub-modules:

| Sub-module | Class | Role | Inputs | Outputs |
|---|---|---|---|---|
| **bert** | `CustomAlbert` | BERT-style phoneme encoder (ALBERT backbone) | `input_ids [B, T]`, `attention_mask [B, T]` | `last_hidden_state [B, T, H]` |
| **bert_encoder** | `nn.Linear` | Projects BERT hidden dim → model hidden dim | `[B, T, H_bert]` | `[B, T, H]` |
| **text_encoder** | `TextEncoder` | CNN + sequence mixer over phoneme embeddings; current export branch uses non-causal TCN instead of BiLSTM | `input_ids, input_lengths, mask` | `[B, H, T]` |
| **predictor** | `ProsodyPredictor` | Duration + F0/energy prediction | `d_en, style_s, input_lengths, mask, alignment` | `duration [B, T, max_dur]`, `F0 [B, T]`, `N [B, T]` |
| **decoder** | `Decoder` → `Generator` → `CustomSTFT` | iSTFTNet waveform synthesis | `asr [B, H, T]`, `F0_curve`, `N`, `style_s` | `audio [T_audio]` |

Full inference data flow in `KModel.forward_with_tokens`:

```
input_ids ──► bert ──► bert_encoder ──► d_en
                                         │
ref_s ──► s (style) ─────────────────────┤
                                         ▼
                                   predictor
                                   ├─ duration ──► pred_dur ──► alignment matrix
                                   └─ F0Ntrain ──► F0_pred, N_pred
                                         │
input_ids ──► text_encoder ──► t_en ─────┤
                                         ▼
                                   decoder(asr, F0_pred, N_pred, style)
                                         │
                                         ▼
                                       audio
```

`ref_s` is a 256-dim style vector; the first 128 dims go to the `decoder`,
the remaining 128 go to `predictor` components.

---

## Current Strategy — Replace LSTM/BiLSTM with Conv1d/TCN

As of 2026-06-03 23:15:57 PDT, the active branch is
`tcn_lstm_replacement`, started from git checkpoint `11e3dd2` (`Add baseline
parity WAV outputs`). This intentionally predates the hybrid conversion work
(`a072f53` and later), which was judged too fragile and complicated for the
mobile path.

The export-facing recurrent layers are being replaced with non-causal
Conv1d/TCN sequence mixers:

| Original recurrent block | Replacement |
|---|---|
| `TextEncoder.lstm` BiLSTM | `TCNSequenceMixer(512 -> 512)` |
| `DurationEncoder.lstms` BiLSTM stack | style-conditioned TCN layers with `640 -> 512`, followed by existing `AdaLayerNorm` and style re-append |
| `ProsodyPredictor.lstm` duration mixer | `TCNSequenceMixer(640 -> 512)` |
| `ProsodyPredictor.shared` F0/N mixer | `TCNSequenceMixer(640 -> 512)` |

Default TCN config:

```json
"sequence_mixer": {
  "type": "tcn",
  "num_blocks": 4,
  "kernel_size": 5,
  "dilations": [1, 2, 4, 8]
}
```

The original LSTM architecture remains supported when `sequence_mixer.type` is
absent or set to `"lstm"`. The Kokoro checkpoint can be loaded into the TCN
model with `strict=False`; non-recurrent weights are reused and the new TCN
weights start randomly initialized until distillation.

### Distillation Plan

Do not try to directly convert LSTM weights into Conv1d weights. Train the TCN
modules as students against frozen LSTM/BiLSTM teacher tensors from the
original checkpoint.

1. Freeze a teacher model using the original LSTM config.
2. Build a student model using `checkpoints/config.json` with
   `sequence_mixer.type = "tcn"`.
3. Reuse loaded checkpoint weights for BERT, embeddings, CNNs, AdaIN blocks,
   decoder, and projection heads.
4. Train only TCN parameters first. Keep projection heads frozen for initial
   intermediate matching unless a head-specific loss is explicitly enabled.
5. Unfreeze local projection heads for a short fine-tune after intermediate
   losses converge.
6. Run TFLite parity on the student model, then generate WAVs for subjective
   comparison against `test_output/baseline/wavs/`.

### Distillation Data Collection

Collect teacher/student pairs from text, not necessarily paired audio. Use the
local text corpora:

| Corpus | Local path |
|---|---|
| LJSpeech | `/export/eingerman/audio/LJSpeech-1.1/` |
| LibriTTS | `/export/eingerman/audio/LibriTTS/LibriTTS/` |
| Distillation data/output | `/export/eingerman/audio/tcl_distil/` |

Use `export/test.txt` only for smoke tests. Full collection should use the
local LJSpeech and LibriTTS roots above, covering short, medium, long,
punctuation-heavy, and prosody-varied lines.

For each line/chunk:

1. Run `KPipeline` G2P to produce `input_ids`.
2. Load one or more voice style vectors; save both `ref_s[:, :128]` and
   `ref_s[:, 128:]`.
3. Run the frozen LSTM teacher and save:
   - `bert_hidden`, `d_en`
   - `text_encoder_out`
   - `duration_encoder_out`
   - `predictor_duration_mixer_out`
   - `duration_logits`
   - `pred_dur`, `pred_aln_trg`
   - `predictor_aligned_en`
   - `f0n_shared_out`
   - `F0`, `N`
   - decoder inputs and optional `audio`
4. Save tensors under `/export/eingerman/audio/tcl_distil/teacher/<git_hash>/`
   as `.npz` files with a manifest containing text, phonemes, token length,
   aligned length, voice id, style file, speed, and source git hash.
5. Bucket by token length and aligned frame length, but keep exact actual
   lengths in metadata for loss masking.

### Retraining Steps

Recommended staged training:

1. `text_encoder` distillation:
   - Input: `input_ids`, mask.
   - Target: teacher `text_encoder_out [B, 512, T]`.
   - Loss: masked MSE + optional cosine loss.
2. `DurationEncoder` distillation:
   - Input: teacher/student `d_en`, style predictor slice.
   - Target: teacher `duration_encoder_out [B, T, 640]`.
   - Loss: masked MSE on the full output and on the non-style 512-channel part.
3. Duration mixer/head distillation:
   - Input: student `duration_encoder_out`.
   - Targets: teacher `predictor_duration_mixer_out`, `duration_logits`.
   - Loss: MSE on mixer hidden plus MSE/KL on duration logits.
4. F0/N shared mixer distillation:
   - Input: teacher `predictor_aligned_en`, style predictor slice.
   - Targets: teacher `f0n_shared_out`, `F0`, `N`.
   - Loss: hidden MSE + F0/N MSE, with optional voiced-region weighting if a
     voiced mask is added later.
5. End-to-end local fine-tune:
   - Freeze BERT and decoder first.
   - Unfreeze TCNs and projection heads.
   - Use duration/F0/N losses and optional decoder mel/audio reconstruction
     against teacher outputs.
6. Export student fp32 TFLite, run PyTorch-vs-TFLite parity, then AOT compile.

Suggested loss mix:

```python
loss = (
    1.0 * masked_mse(student_hidden, teacher_hidden, mask)
    + 0.2 * cosine_loss(student_hidden, teacher_hidden, mask)
    + 0.5 * masked_mse(student_logits, teacher_logits, mask)
)
```

Training outputs should be saved to
`/export/eingerman/audio/tcl_distil/checkpoints/<git_hash>/` with the source git
hash, data manifest hash, and validation summary in the filename or a sidecar
JSON.

---

## Export Strategy — Step by Step

Work one sub-module at a time. Each step should produce a tested `.tflite`
file before proceeding to the next.

### Prerequisites

```python
import torch
import litert_torch
from kokoro import KModel

model = KModel(
    repo_id='hexgrad/Kokoro-82M',
    config='checkpoints/config.json',
    model='checkpoints/kokoro-v1_0.pth',
).eval()
```

---

### Step 1 — Export `bert` (CustomAlbert)

**Why first:** Pure transformer encoder, most likely `torch.export`-compatible
without dynamic control flow issues.

**Wrapper needed:** `CustomAlbert` inherits `AlbertModel` from HuggingFace
transformers. Strip it to tensor I/O only:

```python
class BertWrapper(torch.nn.Module):
    def __init__(self, bert, bert_encoder):
        super().__init__()
        self.bert = bert
        self.bert_encoder = bert_encoder

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        out = self.bert(input_ids, attention_mask=attention_mask)
        return self.bert_encoder(out.last_hidden_state).transpose(-1, -2)  # [B, H, T]

SEQ = 50
wrapper = BertWrapper(model.bert, model.bert_encoder).eval()
sample_inputs = (
    torch.randint(0, 178, (1, SEQ), dtype=torch.long),
    torch.ones(1, SEQ, dtype=torch.long),
)
edge_model = litert_torch.convert(wrapper, sample_inputs)
edge_model.export("kokoro_bert.tflite")
```

**Known challenges:**
- HuggingFace models use keyword-only args; `litert_torch.convert` requires
  positional tensor args only → wrapper is mandatory.
- Dynamic sequence length: use `torch.export` with
  `dynamic_shapes={"input_ids": {1: torch.export.Dim("T", min=1, max=510)}}`
  if the converter supports it; otherwise fix `SEQ` at the max context length
  (512).

---

### Step 2 — Export `text_encoder` (TextEncoder)

Current path: use `sequence_mixer.type = "tcn"` so `TextEncoder` has only
embedding, Conv1d CNN blocks, mask ops, and TCN Conv1d blocks. This avoids
`pack_padded_sequence`, `pad_packed_sequence`, recurrent ops, and the padded
BiLSTM backward-contamination issue.

```python
class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, te):
        super().__init__()
        self.te = te

    def forward(self, input_ids: torch.LongTensor, mask: torch.BoolTensor):
        input_lengths = torch.ones(input_ids.shape[0], dtype=torch.long) * input_ids.shape[1]
        return self.te(input_ids, input_lengths, mask)
```

---

### Step 3 — Export `predictor` (ProsodyPredictor)

Split into two parts to avoid dynamic alignment construction. In the TCN branch,
both parts are Conv1d-based and should be AOT candidates after fp32 parity.

**3a — Duration head** (`predictor.text_encoder` + `predictor.run_duration_mixer` +
`predictor.duration_proj`):

```python
# Input: d_en [B, H, T], style_s [B, 128]
# Output: duration logits [B, T, max_dur]
```

**3b — F0/Energy head** (`predictor.F0Ntrain`):

```python
# Input: aligned_en [B, H, T_aligned], style_s [B, 128]
# Output: F0 [B, T_aligned], N [B, T_aligned]
```

The alignment matrix construction (scatter with `torch.repeat_interleave` and
index assignment) is **not exportable** and must remain on CPU as a
pre-processing step. Pass the pre-built `pred_aln_trg` tensor as an input.

---

### Step 4 — Export `decoder` (Decoder + Generator + CustomSTFT)

`CustomSTFT` was specifically designed to avoid `F.unfold` and complex-number
ops for ONNX compatibility. It should be `torch.export`-compatible.

```python
class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        asr: torch.FloatTensor,       # [B, H, T_aligned]
        F0_curve: torch.FloatTensor,  # [B, T_aligned]
        N: torch.FloatTensor,         # [B, T_aligned]
        style_s: torch.FloatTensor,   # [B, 128]
    ) -> torch.FloatTensor:
        return self.decoder(asr, F0_curve, N, style_s)

SEQ = 200  # aligned frame length
wrapper = DecoderWrapper(model.decoder).eval()
sample_inputs = (
    torch.randn(1, 512, SEQ),
    torch.randn(1, SEQ),
    torch.randn(1, SEQ),
    torch.randn(1, 128),
)
edge_model = litert_torch.convert(wrapper, sample_inputs)
edge_model.export("kokoro_decoder.tflite")
```

**Known challenges:**
- `weight_norm` parametrizations: call `torch.nn.utils.parametrize.remove_parametrizations`
  or `torch.nn.utils.remove_weight_norm` on all conv layers before export.
- `AdaIN1d` uses `nn.InstanceNorm1d(affine=True)` which is fine.
- Check for any in-place ops (`masked_fill_`, `x[...]= `) — replace with
  out-of-place equivalents before calling `torch.export`.

---

### Step 5 — Multi-Signature `.tflite`

Once all parts export cleanly, combine into a single model file using
multi-signature conversion:

```python
edge_model = (
    litert_torch
    .signature("bert", bert_wrapper, bert_inputs)
    .signature("text_encoder", te_wrapper, te_inputs)
    .signature("predictor_dur", dur_wrapper, dur_inputs)
    .signature("predictor_f0n", f0n_wrapper, f0n_inputs)
    .signature("decoder", dec_wrapper, dec_inputs)
    .convert()
)
edge_model.export("kokoro_full.tflite")
```

Invoke each signature independently at runtime:

```python
# Invoke individual signatures
bert_out   = edge_model(*bert_inputs,   signature_name="bert")
te_out     = edge_model(*te_inputs,     signature_name="text_encoder")
dur_out    = edge_model(*dur_inputs,    signature_name="predictor_dur")
f0n_out    = edge_model(*f0n_inputs,    signature_name="predictor_f0n")
audio_out  = edge_model(*dec_inputs,    signature_name="decoder")
```

---

## Multi-Signature Strategy for Variable-Length Inputs

Because `.tflite` models use static shapes, variable-length phoneme sequences
and the audio frames they produce require a deliberate bucketing strategy rather
than a single fixed shape.

### Recommended bucket sizes

| Signature name | Input phoneme tokens `T` | Aligned frames `T_aligned` |
|---|---|---|
| `bert_short` | 32 | — |
| `bert_medium` | 128 | — |
| `bert_long` | 256 | — |
| `bert_max` | 510 | — |
| `text_encoder_short` | 32 | — |
| `text_encoder_medium` | 128 | — |
| `text_encoder_long` | 256 | — |
| `predictor_dur_short` | 32 | — |
| `predictor_dur_medium` | 128 | — |
| `predictor_f0n` | — | 200 |
| `predictor_f0n_long` | — | 800 |
| `decoder_short` | — | 200 |
| `decoder_medium` | — | 800 |
| `decoder_long` | — | 2000 |

At inference time, pick the smallest bucket that fits the actual sequence
length and zero-pad the inputs to that bucket boundary.

### Building the multi-bucket model

```python
def make_bert_sig(seq_len, name):
    inputs = (
        torch.randint(0, 178, (1, seq_len), dtype=torch.long),
        torch.ones(1, seq_len, dtype=torch.long),
    )
    return litert_torch.signature(name, bert_wrapper, inputs)

def make_decoder_sig(t_aligned, name):
    inputs = (
        torch.randn(1, 512, t_aligned),
        torch.randn(1, t_aligned),
        torch.randn(1, t_aligned),
        torch.randn(1, 128),
    )
    return litert_torch.signature(name, decoder_wrapper, inputs)

edge_model = (
    make_bert_sig(32,  "bert_short")
    .signature("bert_medium",         bert_wrapper,    bert_inputs_128)
    .signature("bert_long",           bert_wrapper,    bert_inputs_256)
    .signature("bert_max",            bert_wrapper,    bert_inputs_510)
    .signature("text_encoder_short",  te_wrapper,      te_inputs_32)
    .signature("text_encoder_medium", te_wrapper,      te_inputs_128)
    .signature("predictor_dur_short", dur_wrapper,     dur_inputs_32)
    .signature("predictor_dur_med",   dur_wrapper,     dur_inputs_128)
    .signature("predictor_f0n",       f0n_wrapper,     f0n_inputs_200)
    .signature("predictor_f0n_long",  f0n_wrapper,     f0n_inputs_800)
    .signature("decoder_short",       decoder_wrapper, dec_inputs_200)
    .signature("decoder_medium",      decoder_wrapper, dec_inputs_800)
    .signature("decoder_long",        decoder_wrapper, dec_inputs_2000)
    .convert()
)
edge_model.export("kokoro_multisig.tflite")
```

**Note:** litert_torch multi-signature conversion shares weights across
signatures automatically when the same `nn.Module` instance is reused — this
keeps the `.tflite` file size close to a single-signature model.

---

## Testing: PyTorch vs TFLite Parity

Every sub-module — and key intermediate tensors within each — must be verified
before moving to the next export step. Use the lines of text from export/test.txt to
test and catch shape-dependent bugs.

Do **not** AOT compile after each individual TFLite export step. First complete
and parity-test all fp32 `.tflite` exports, then assemble the final
multi-signature `.tflite`, and only then run Tensor G5 AOT compilation as a
separate packaging/performance phase. This avoids spending compiler time on
intermediate artifacts that may be replaced during later export steps.

With `sequence_mixer.type = "tcn"`, TextEncoder, PredictorDur, and
PredictorF0N should be considered AOT candidates after fp32 parity. Historical
LSTM-configured exports should still skip standalone Tensor G5 AOT experiments
because recurrent subgraphs compile slowly and do not produce useful NPU
kernels.

### Test corpus

Use at least these four lengths for all sequence-dependent tests:

```python
TEST_LENS = [10, 32, 128, 256]   # phoneme token counts
TEST_ALIGNED = [50, 200, 500, 800]  # aligned frame counts (decoder inputs)
```

### Helper

```python
import numpy as np

def assert_close(pt_out, tflite_out, name, atol=1e-4):
    """Compare a PyTorch tensor to a TFLite numpy output."""
    pt = pt_out.detach().float().numpy()
    diff = np.abs(pt - tflite_out).max()
    print(f"{name}: max_abs_diff={diff:.6f}")
    assert diff < atol, f"{name} parity FAILED: max diff {diff} >= {atol}"
```

### Step 1 — BERT parity

```python
for T in TEST_LENS:
    ids  = torch.randint(0, 178, (1, T), dtype=torch.long)
    mask = torch.ones(1, T, dtype=torch.long)

    # PyTorch reference
    with torch.no_grad():
        pt_hidden = model.bert(ids, attention_mask=mask).last_hidden_state  # [1, T, H_bert]
        pt_out    = model.bert_encoder(pt_hidden).transpose(-1, -2)         # [1, H, T]

    # TFLite
    sig = f"bert_{['short','medium','long','max'][[32,128,256,510].index(min([32,128,256,510], key=lambda x: abs(x-T)))]}"
    tflite_out = edge_model(ids, mask, signature_name=sig)

    assert_close(pt_out, tflite_out, f"bert T={T}")
```

### Step 2 — TextEncoder parity (incl. CNN intermediate)

```python
for T in TEST_LENS:
    ids          = torch.randint(0, 178, (1, T), dtype=torch.long)
    input_lengths = torch.tensor([T], dtype=torch.long)
    mask         = torch.zeros(1, T, dtype=torch.bool)

    with torch.no_grad():
        # CNN only (before sequence mixer)
        emb = model.text_encoder.embedding(ids).transpose(1, 2)
        x = emb
        for c in model.text_encoder.cnn:
            x = c(x)
        pt_cnn = x.clone()  # intermediate

        # Full text_encoder
        pt_out = model.text_encoder(ids, input_lengths, mask)  # [1, H, T]

    sig = "text_encoder_short" if T <= 32 else "text_encoder_medium"
    tflite_out = edge_model(ids, signature_name=sig)

    assert_close(pt_out, tflite_out, f"text_encoder T={T}")
```

### Step 3 — Predictor duration head parity

```python
for T in TEST_LENS:
    d_en  = torch.randn(1, 512, T)
    style = torch.randn(1, 128)
    input_lengths = torch.tensor([T], dtype=torch.long)
    mask  = torch.zeros(1, T, dtype=torch.bool)

    with torch.no_grad():
        # DurationEncoder intermediate
        d     = model.predictor.text_encoder(d_en, style, input_lengths, mask)
        # Duration sequence mixer
        x = model.predictor.run_duration_mixer(d)
        pt_dur = model.predictor.duration_proj(x)  # [1, T, max_dur]

    sig = "predictor_dur_short" if T <= 32 else "predictor_dur_med"
    tflite_out = edge_model(d_en, style, signature_name=sig)
    assert_close(pt_dur, tflite_out, f"predictor_dur T={T}")
```

### Step 3b — F0Ntrain parity

```python
for T_aln in TEST_ALIGNED:
    en    = torch.randn(1, 512, T_aln)
    style = torch.randn(1, 128)

    with torch.no_grad():
        pt_f0, pt_n = model.predictor.F0Ntrain(en, style)

    sig = "predictor_f0n" if T_aln <= 200 else "predictor_f0n_long"
    f0_tflite, n_tflite = edge_model(en, style, signature_name=sig)
    assert_close(pt_f0, f0_tflite, f"F0 T_aln={T_aln}")
    assert_close(pt_n,  n_tflite,  f"N  T_aln={T_aln}")
```

### Step 4 — Decoder parity (incl. Generator and CustomSTFT)

```python
for T_aln in TEST_ALIGNED:
    asr   = torch.randn(1, 512, T_aln)
    F0    = torch.randn(1, T_aln)
    N     = torch.randn(1, T_aln)
    style = torch.randn(1, 128)

    with torch.no_grad():
        # Intermediate: after encode block
        F0c = model.decoder.F0_conv(F0.unsqueeze(1))
        Nc  = model.decoder.N_conv(N.unsqueeze(1))
        x   = torch.cat([asr, F0c, Nc], dim=1)
        pt_encode = model.decoder.encode(x, style)

        # Full decoder output
        pt_audio = model.decoder(asr, F0, N, style).squeeze()

    if T_aln <= 200:
        sig = "decoder_short"
    elif T_aln <= 800:
        sig = "decoder_medium"
    else:
        sig = "decoder_long"

    tflite_audio = edge_model(asr, F0, N, style, signature_name=sig)
    assert_close(pt_audio.unsqueeze(0), tflite_audio, f"decoder T_aln={T_aln}", atol=1e-3)
```

### Tolerance guidance

| Precision | Recommended `atol` |
|---|---|
| fp32 export | `1e-4` |
| fp16 / GPU delegate | `5e-3` |
| int8 quantized | `5e-2` |

---

## Quantization

Apply **after** a clean fp32 export is confirmed working.

### fp16

```python
# litert_torch handles fp16 casting during AOT compilation for GPU/NPU.
# No explicit quantization step needed; pass fp16 inputs at runtime or
# enable fp16 mode in the LiteRT delegate.
```

### int8 (PT2E / dynamic)

```python
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config

quantizer = PT2EQuantizer().set_global(
    get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
)
exported = torch.export.export(wrapper, sample_inputs).module()
exported = prepare_pt2e(exported, quantizer)
exported(*sample_inputs)   # calibration pass
exported = convert_pt2e(exported, fold_quantize=False)

edge_model = litert_torch.convert(exported, sample_inputs)
edge_model.export("kokoro_decoder_int8.tflite")
```

Quantize the `decoder` first (most compute-intensive), then `predictor`, then
`bert`.

---

## Known Export Blockers & Mitigations

| Issue | Location | Mitigation |
|---|---|---|
| `pack_padded_sequence` / `pad_packed_sequence` | Historical LSTM config for `TextEncoder`, `DurationEncoder`, `ProsodyPredictor` | Use `sequence_mixer.type = "tcn"` for export-facing models; keep LSTM only as teacher/reference |
| `torch.repeat_interleave` with dynamic count | `KModel.forward_with_tokens` (alignment) | Keep on CPU; pass alignment tensor as input to decoder |
| In-place index assignment `pred_aln_trg[idx, :]= 1` | `KModel.forward_with_tokens` | Keep on CPU pre-processing |
| `weight_norm` parametrizations | `Generator`, `TextEncoder`, conv layers | Remove before export with `remove_weight_norm` |
| HuggingFace keyword args | `CustomAlbert` | Wrap in positional-arg-only `nn.Module` |
| Dynamic audio output length | `Generator` (upsample) | Fix input frame count at export time; use static shapes |

---

## AOT Compilation for Pixel 10 (Google Tensor G5)

The Pixel 10 uses a **Google Tensor G5** SoC. AOT-compiled models run directly
on the EdgeTPU/NPU without JIT overhead.

Run AOT only after Step 5 has produced the final fp32 multi-signature model
(`kokoro_multisig.tflite`) and all TFLite parity tests pass. Treat AOT as a
post-conversion packaging step, not as sub-module export validation.

### NPU plugin

The Google Tensor SDK plugin and a reference notebook live in `litert_npu/`:

```
litert_npu/
  litert_plugin_compiler.tar.gz          # SDK plugin archive
  LiteRT NPU AOT compilation for Google Tensor.ipynb   # walkthrough notebook
```

### Install the SDK

```python
import os
os.environ["GOOGLE_TENSOR_SDK_BETA"] = "/rhome/eingerman/Projects/DeepLearning/TTS/Kokoro/litert_npu/litert_plugin_compiler.tar.gz"

# In the notebook or a setup cell:
# !pip install ai-edge-litert-sdk-google-tensor==2.1.5
# !pip install ai-edge-litert-nightly==2.2.0.dev20260518
```

### Preferred command-line AOT workflow

Prefer the `litert-torch` command-line interface for AOT compilation so long
compiler runs are repeatable, logged, and easy to resume from shell history or
CI. Use it through `uv` so the managed environment and pinned dependencies are
used:

```bash
GOOGLE_TENSOR_SDK_BETA=/rhome/eingerman/Projects/DeepLearning/TTS/Kokoro/litert_npu/litert_plugin_compiler.tar.gz \
  uv run litert-torch export_hf \
  --model <model-or-local-export-config> \
  --output_dir outputs/<git_hash>/aot \
  --aot_backend GOOGLE \
  --aot_soc_model Tensor_G5 \
  --aot_compilation_config_dict='{"google_tensor_truncation_type":"half","google_tensor_int64_to_int32":true,"google_tensor_sharding_intensity":"extensive"}'
```

For Kokoro, keep the custom Python export scripts for the sub-module TFLite
conversion work. Use the CLI AOT path once the final multi-signature model is
available, or mirror the same `litert-torch` AOT flags if a direct compile-only
subcommand is added upstream.

### Python API reference

```python
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.vendors.google_tensor import target as gt_target

tensor_g5_target = gt_target.Target(gt_target.SocModel.TENSOR_G5)

compiled_models = aot_lib.aot_compile(
    "kokoro_multisig.tflite",
    target=[tensor_g5_target],
    keep_going=False,
    # fp16 truncation — best for GPU/NPU throughput:
    google_tensor_truncation_type="half",
    # Allow int64→int32 cast (needed for embedding index ops):
    google_tensor_int64_to_int32=True,
    # Aggressive parallelism for a multi-core EdgeTPU:
    google_tensor_sharding_intensity="extensive",
)
print(compiled_models.compilation_report())
compiled_models.export(".", model_name="kokoro")
```

This produces per-SoC `.tflite` files, e.g.:

| File | Target |
|---|---|
| `kokoro_fallback.tflite` | CPU / GPU (all devices) |
| `kokoro_Google_Tensor_G5.tflite` | Pixel 10 NPU |

### Key compilation flags (Google Tensor)

| Flag | Default | Recommended for Kokoro |
|---|---|---|
| `google_tensor_truncation_type` | `auto` | `"half"` (fp16 weights) |
| `google_tensor_sharding_intensity` | `minimal` | `"extensive"` |
| `google_tensor_int64_to_int32` | `False` | `True` (BERT token indices) |
| `google_tensor_enable_large_model_support` | `False` | `False` (model < 2 GB) |
| `google_tensor_enable_4bit_compilation` | `False` | `False` unless doing 4-bit quant |

Full flag reference: https://ai.google.dev/edge/tensor-sdk/compilation-flags

### Package for Play On-Device AI (PODAI)

```python
from ai_edge_litert.aot.ai_pack import export_lib as ai_pack_export

ai_pack_export.export(
    compiled_models,
    ai_pack_dir="ai_pack",
    ai_pack_name="kokoro_tts",
    litert_model_name="kokoro_model",
)
```

PODAI delivers the correct per-SoC `.tflite` to each device automatically.
See the reference notebook in `litert_npu/` for the full end-to-end workflow.

### GPU delegate (fallback)

Use `GpuDelegateV2` via the LiteRT Android API with `kokoro_fallback.tflite`.
fp16 models run natively; int8 models dequantize to fp16 on GPU.

Reference: `litert-torch/docs/pytorch_converter/README.md`

---

## Environment

Dependencies are managed with **uv**. Key packages:

| Package | Version | Purpose |
|---|---|---|
| `torch` | `==2.12.0` | Model + export |
| `litert-torch-nightly` | `>=0.10.0.dev20260601` | LiteRT conversion backend |
| `torchao` | `>=0.17.0` | PT2E quantization (transitive dep) |
| `ai-edge-litert-nightly` | latest nightly | LiteRT runtime |
| `tensorflow` | `>=2.19` | Required by litert-torch conversion pipeline |

```bash
uv sync          # install all deps from lock file
uv add <pkg>     # add new dependency
uv run python examples/export.py       # ONNX export reference
uv run python export/export_bert.py    # LiteRT export scripts
```

The `litert-torch` repo is checked out at `litert-torch/` for local reference
and is **not** installed as a local package (the PyPI nightly is used instead).

Use PROGRESS.md to keep track of the immediate next steps, steps that have been completed, forks in work. Use LEARNINGS.md to document insights, tips, and general learnings that come up during the project. Use PROBLEMS.md to log any blockers or issues that arise, along with attempted solutions and outcomes, and use DECISIONS.md to record any major decisions made, along with the rationale and alternatives considered. Update these files regularly to maintain a clear and organized record of the project's progress and learnings. Use date and time stamps in md files, add git hash if appropriate. 

Checkpoint commit work regularly with clear messages. Save tflite output to outputs/ with descriptive names for each export step, source git hash and quantization variant. Use test_output/ to save intermediate tensors from PyTorch and TFLite for parity testing, organized by export step and sequence length in directories labeled with git hash. Output wav files where appropriate for subjective evaluation of audio quality at each step, especially after decoder export and quantization.


## Running training:



uv run python export/train_tcn_distill.py   --data-dir /export/eingerman/audio/tcl_distil/teacher/61a6a14   --output-dir /export/eingerman/audio/tcl_distil/checkpoints/$HASH   --device cuda   --batch-size 32   --epochs 20
