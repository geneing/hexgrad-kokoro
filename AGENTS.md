# Kokoro TTS — Agent Documentation

## Overview
This codebase converts the Kokoro TTS PyTorch model to TFLite for Android deployment.

## Key Agents / Commands

### Validate full pipeline
```bash
TF_ENABLE_ONEDNN_OPTS=0 uv run python - << 'EOF'
from tflite_inference import KokoroTFLiteTTS, KokoroPTTTS
import numpy as np
tfl = KokoroTFLiteTTS()
pt  = KokoroPTTTS(
    config_path="checkpoints/config.json",
    kokoro_checkpoint="checkpoints/kokoro-v1_0.pth",
    vocos_checkpoint="vocos_fp16.pt",
    voice_path="checkpoints/voices/af_bella.pt"
)
a = tfl.generate("Hello world, testing Kokoro TTS on TFLite.")
b = pt.generate("Hello world, testing Kokoro TTS on TFLite.")
n = min(len(a), len(b))
print(f"Correlation: {np.corrcoef(a[:n], b[:n])[0,1]:.4f}")
EOF
```

### Re-export duration predictor
```bash
# See: tflite_inference.py or use the DurationPredictorCore code embedded in PROGRESS.md
# Key: remove weight_norm before export
TF_ENABLE_ONEDNN_OPTS=0 uv run python -c "
import torch.nn as nn
# ... (see streaming_vocos_export.ipynb or PROGRESS.md for full export code)
"
```

### Re-convert duration predictor to TFLite
```bash
cd onnx2tf_conversion
TF_ENABLE_ONEDNN_OPTS=0 uv run onnx2tf -i duration_predictor.onnx -osd -o saved_model
```

### Export ONNX for Android (FastGELU + VocosPreIRFFT)
```bash
TF_ENABLE_ONEDNN_OPTS=0 uv run python onnx_export_android.py
```
Outputs to `onnx_android/` — FP32 + opt + FP16 variants + comparison test.

### Re-export vocoder
```bash
TF_ENABLE_ONEDNN_OPTS=0 uv run python export_real_vocoder.py
```

### Re-convert vocoder to TFLite
```bash
cd onnx2tf_conversion
TF_ENABLE_ONEDNN_OPTS=0 uv run onnx2tf -i vocoder_stream_chunk.onnx -osd -o saved_model
```

## Codebase Map

| File | Role |
|------|------|
| `onnx_export_android.py` | Android ONNX export: FastGELU, VocosPreIRFFTAndroid, comparison test |
| `tflite_inference.py` | Main TFLite inference pipeline. `KokoroTFLiteTTS` runs all 7 stages. `KokoroPTTTS` is the PyTorch reference (with matching padding) |
| `export_real_vocoder.py` | Exports `VocosPreIRFFT` (backbone+head minus IRFFT/OLA) to ONNX → TFLite |
| `streaming_vocos_export.ipynb` | Exports BERT, duration, acoustic_expand, F0N, conditioner |
| `onnx2tf_conversion/main.py` | Runs onnx2tf for all models |
| `streaming_vocos.py` | `StreamingVocos` — PyTorch stateful streaming vocoder |
| `kokoro/model.py` | `KModel`, `KModelForONNX` — Kokoro model definition |
| `kokoro/modules.py` | `TextEncoder`, `ProsodyPredictor`, `DurationEncoder` |
| `kokoro/pipeline.py` | `KPipeline` — g2p + voice loading |
| `kotlin/Android_Inference_Specification_tflite.md` | Full Android LiteRT inference spec |
| `kotlin/Android_Inference_Specification_onnx.md` | Full Android ONNX Runtime inference spec |
| `PLAN.md` | Architecture decisions |
| `PROGRESS.md` | Bug fixes and validation results |
| `TODO.md` | Outstanding work |

## Environment

- Python: 3.12 (via `uv`)
- PyTorch: 2.9.1
- onnx2tf: in `onnx2tf_conversion/.venv`
- TFLite runtime: `ai_edge_litert`
- Run all scripts with `TF_ENABLE_ONEDNN_OPTS=0 uv run python ...`

## Critical Implementation Notes

1. **Duration predictor tensor naming**: The re-exported TFLite uses `serving_default_NAME:0` for inputs and `StatefulPartitionedCall:N` for outputs. Use `_norm_name()` helper in `tflite_inference.py` to strip these.

2. **en-trim bug**: Never trim `en` (acoustic_expand output) to `T_acoustic` before sending to F0N predictor. Always send the full `T_ACOUSTIC`-length output.

3. **Vocos state transpose**: `Identity_3..10` outputs are `[1, 384, 6]` NCT but next invocation inputs are `[1, 6, 384]` NTC. Transpose before feeding back.

4. **Duration predictor speed input**: float32 (not int32). input_ids: int64 (not int32).

5. **Weight norm**: Remove with `nn.utils.remove_weight_norm(layer)` before any ONNX export that uses `weight_norm`'d layers. onnx2tf incorrectly handles `weight_g`/`weight_v` parameters.
