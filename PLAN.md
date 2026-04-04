# Kokoro TTS — Android TFLite Export Plan

## Objective
Export the Kokoro TTS pipeline (7 stages) to TFLite models compatible with Android LiteRT,
validate numerical correctness against the PyTorch reference, and prepare for deployment.

## Architecture

```
Text → g2p → [BERT] → [Duration] → expand → [AcousticExpand] → [F0N] → [Conditioner] → [Vocos] → Audio
```

All 6 TFLite models are in `onnx2tf_conversion/saved_model/`.

## Decisions

### Vocoder: VocosPreIRFFT + numpy/JTransforms IRFFT
- Root cause: `torch.fft.irfft` → ONNX DFT op → TFLite `ONNX_DFT` (unsupported)
- Fix: Export model up to spectrum outputs (`x_real`, `x_imag`), apply IRFFT at runtime
- Result: max diff PT vs TFLite = 5.7e-7 ✓

### Overlap-Add: Python/Kotlin (not in model)
- Root cause: `onnx2tf` mishandles `ConvTranspose1d(1200,1,1200,stride=300)` → 26× amplitude error
- Fix: Remove OLA from model, implement as pure Python/Kotlin NumPy

### Duration Predictor Re-export
- Root cause: Old ONNX kept `weight_g`/`weight_v` parameters — onnx2tf incorrectly converted weight_norm, producing wrong `t_en_static` (diff 1.944)
- Fix: Call `nn.utils.remove_weight_norm()` before ONNX export → correct `weight` tensor
- Result: `t_en_static` diff 2.5e-6 ✓

### en-trim Bug Fix
- Root cause: TFLite pipeline was trimming acoustic_expand output to `T_acoustic` then zero-padding before sending to F0N predictor. This caused different padding patterns vs PT reference.
- Fix: Pass the full `T_ACOUSTIC` acoustic_expand output to F0N without trimming.
- Result: Full pipeline correlation 1.0000 ✓

## Model Files

| Stage | ONNX | TFLite (float32) | Status |
|-------|------|--------|--------|
| BERT | `onnx2tf_conversion/bert.onnx` | `bert_float32.tflite` | ✅ |
| Duration | `onnx2tf_conversion/duration_predictor.onnx` | `duration_predictor_float32.tflite` | ✅ Re-exported |
| Acoustic Expand | `onnx2tf_conversion/acoustic_expand.onnx` | `acoustic_expand_float32.tflite` | ✅ |
| F0N | `onnx2tf_conversion/f0n_predictor.onnx` | `f0n_predictor_float32.tflite` | ✅ |
| Conditioner | `onnx2tf_conversion/vocoder_conditioner.onnx` | `vocoder_conditioner_float32.tflite` | ✅ |
| Vocos | (see `export_real_vocoder.py`) | `vocoder_stream_chunk_float32.tflite` | ✅ VocosPreIRFFT |

## ONNX for Android (onnxruntime-android + NNAPI)

Separate from the TFLite export, `onnx_export_android.py` produces models in `onnx_android/` optimised for Android ONNX Runtime with NNAPI/GPU delegates.

### Android-specific Changes vs. Base ONNX Export

| Change | Reason |
|--------|---------|
| **VocosPreIRFFTAndroid**: outputs x_real/x_imag spectra (no IRFFT) | ONNX Runtime does not support the ONNX DFT op |
| **FastGELU**: `x · sigmoid(1.702x)` replaces `nn.GELU` | Avoids Erf/Tanh/Pow ops; Sigmoid+Mul are NNAPI-accelerated |
| Weight-norm folded (`remove_weight_norm`) | Clean single weight tensor |
| Static shapes (T_ACOUSTIC=543, T_F0=1086) | NNAPI requires fixed input dimensions |
| IRFFT + OLA in Kotlin (JTransforms) | Matches TFLite inference spec |

### Conditioner tensor layout: NCT (not NTC)

Unlike the onnx2tf TFLite export (which transposes to NTC due to `-osd`), the ONNX Android conditioner takes and returns **NCT** tensors natively.

### State tensor layout: always NCT

All Vocos streaming state tensors are **NCT [1, C, 6]** — no transposition between chunks, unlike TFLite where Identity_3..10 are NCT but inputs are NTC.

## Key Source Files

| File | Purpose |
|------|---------|
| `tflite_inference.py` | Main TFLite inference pipeline |
| `tflite_inference_clean.py` | Clean single-file TFLite inference |
| `export_real_vocoder.py` | Exports `VocosPreIRFFT` to ONNX+TFLite |
| `onnx_export_android.py` | Exports all 6 models to ONNX for Android (FastGELU, VocosPreIRFFTAndroid) |
| `streaming_vocos_export.ipynb` | Exports BERT, duration, acexp, f0n, conditioner |
| `onnx2tf_conversion/main.py` | Runs onnx2tf conversion |
| `kotlin/KokoroStreamingTts.kt` | Android Kotlin prototype |
| `kotlin/Android_Inference_Specification_tflite.md` | Full Android TFLite spec |
| `kotlin/Android_Inference_Specification_onnx.md` | Full Android ONNX Runtime spec |
