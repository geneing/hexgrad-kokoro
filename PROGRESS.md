# Kokoro TTS — Progress Log

## Session Summary (Current)

### Bugs Fixed

| Bug | Symptom | Fix | Result |
|-----|---------|-----|--------|
| `ONNX_DFT` unsupported op in vocoder | Crash on vocoder TFLite | `VocosPreIRFFT`: export backbone up to spectrum, run IRFFT in Python/Kotlin | ✅ diff 5.7e-7 |
| ConvTranspose1d 26× amplitude | Audio way too loud | Remove OLA from model, implement in Python/OLA helper | ✅ diff 5.9e-6 |
| `t_en_static` wrong (diff 1.944) | Fully incorrect audio | Re-export duration predictor with `remove_weight_norm()` before ONNX export | ✅ diff 2.5e-6 |
| en-trim before F0N | 0.052 correlation vs 1.000 | Don't trim `en` to T_acoustic before sending to F0N; send full T_ACOUSTIC output | ✅ corr 1.0000 |

### Validation Results (final)

| Stage | PT vs TFLite max diff | Status |
|-------|----------------------|--------|
| S1 BERT | 3.3e-4 | ✅ |
| S2 Duration (pred_dur) | 0.0 | ✅ |
| S2 Duration (d_enc) | 2.1e-5 | ✅ |
| S2 Duration (t_en_static) | 2.5e-6 | ✅ (was 1.944, bug fixed) |
| S4 Acoustic Expand | 1.5e-5 | ✅ |
| S5 F0N (F0) | 1.5e-3 | ✅ |
| S5 F0N (N) | 2.5e-5 | ✅ |
| S7 Vocoder (standalone) | 5.7e-7 | ✅ |
| **Full pipeline** | **corr=1.0000** | ✅ |

### Code Changes

#### `tflite_inference.py`
- Added `_norm_name()` helper to handle `serving_default_NAME:0` tensor naming from onnx2tf
- Updated `_run_bert()` to use `_norm_name()`
- Updated `_run_duration()`: universal tensor name lookup, auto-detect speed dtype, fallback by shape
- Fixed `generate()`: removed trim of `en` before F0N predictor (was causing 0.052 correlation)
- Updated `KokoroPTTTS.generate()`: pads acoustic_expand input to T_ACOUSTIC (matches TFLite static shapes)

#### `onnx2tf_conversion/duration_predictor.onnx` (re-exported)
- Removed weight_norm from TextEncoder CNN layers before export
- Speed tensor changed: int32 → float32
- input_ids tensor changed: int32 → int64
- CNN weights: `weight_g`/`weight_v` → single `weight` tensor

### Known Limitations
- Static shapes (T_ACOUSTIC=543, T_F0=1086) degrade short-text quality due to BiLSTM backward pass seeing padding zeros
- Duration predictor tensor names use `serving_default_*:0` prefix (handled by `_norm_name()`)
- State tensors from Vocos TFLite: `Identity_3..10` output as NCT [1,384,6] but input as NTC [1,6,384] → must transpose

---

## ONNX Android Export (`onnx_export_android.py`)

### Design Decisions

| Decision | Detail |
|----------|--------|
| `VocosPreIRFFTAndroid` | Same approach as TFLite `VocosPreIRFFT`: backbone up to spectrum, IRFFT+OLA in Kotlin |
| FastGELU = `x·σ(1.702x)` | Replaces `nn.GELU` in conditioner (2×) and ConvNeXt blocks (8×). Exports to {Sigmoid, Mul} only. Max error < 5e-4. |
| NCT conditioner I/O | Conditioner takes NCT directly (no onnx2tf transpose). Unlike TFLite path. |
| NCT state tensors | Vocos state stays NCT [1,C,6] — no transpose needed between chunks (unlike TFLite). |
| Weight-norm folded | `remove_weight_norm_recursive()` before all exports |
| Static shapes | T_ACOUSTIC=543, T_F0=1086, CHUNK_FRAMES=16 (NNAPI-compatible) |

### Output Files (`onnx_android/`)

| Model | FP32 | FP16 |
|-------|------|------|
| `bert.onnx` | ~26 MB | ~13 MB |
| `duration_predictor.onnx` | ~54 MB | ~27 MB |
| `acoustic_expand.onnx` | ~7 MB | ~4 MB |
| `f0n_predictor.onnx` | ~27 MB | ~14 MB |
| `vocoder_conditioner.onnx` | ~1 MB | ~0.6 MB |
| `vocoder_stream_chunk.onnx` | ~33 MB | ~17 MB |
