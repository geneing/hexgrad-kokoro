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

### Export ONNX models (VocosStreamChunkReal — internal DFT)
```bash
# Run streaming_vocos_export.ipynb (cells 1–12) to export all ONNX + TFLite models.
# Models are written to onnx_streaming_vocos/ and onnx_streaming_vocos/tflite/
```

### Re-export vocoder (VocosStreamChunkReal)
```bash
# Re-run notebook cell 9 (ONNX Export) in streaming_vocos_export.ipynb
# or run streaming_vocos_export_litert.py for the full export pipeline
TF_ENABLE_ONEDNN_OPTS=0 uv run python streaming_vocos_export_litert.py
```

### Re-convert vocoder to TFLite
```bash
cd onnx2tf_conversion
TF_ENABLE_ONEDNN_OPTS=0 uv run onnx2tf -i vocoder_stream_chunk.onnx -osd -o saved_model
```

## Codebase Map

| File | Role |
|------|------|
| `streaming_vocos_export.ipynb` | Main export notebook: BERT, duration, acoustic_expand, F0N, conditioner, VocosStreamChunkReal (internal DFT) → ONNX + TFLite |
| `streaming_vocos_export_litert.py` | Script version of the full ONNX+TFLite export pipeline |
| `tflite_inference.py` | Main TFLite inference pipeline. `KokoroTFLiteTTS` runs all 7 stages. `KokoroPTTTS` is the PyTorch reference (with matching padding) |
| `export_onnx.py` | Packages ONNX models from `onnx_streaming_vocos/` into `export_models/onnx/` with FP16/INT8 variants |
| `export_tflite.py` | Packages TFLite models from `onnx_streaming_vocos/tflite/` into `export_models/tflite/` |
| `onnx2tf_conversion/main.py` | Runs onnx2tf for all models |
| `streaming_vocos.py` | `StreamingVocos` — PyTorch stateful streaming vocoder |
| `kokoro/model.py` | `KModel`, `KModelForONNX` — Kokoro model definition |
| `kokoro/modules.py` | `TextEncoder`, `ProsodyPredictor`, `DurationEncoder` |
| `kokoro/pipeline.py` | `KPipeline` — g2p + voice loading |
| `kotlin/example_onnx_inference.py` | End-to-end ONNX Runtime inference example (audio output directly from model) |
| `kotlin/example_tflite_inference.py` | End-to-end LiteRT inference example (audio output directly from model) |
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

3. **Vocos state transpose**: `Identity_1..9` state outputs may be `[1, 384, 6]` NCT while corresponding inputs expect `[1, 6, 384]` NTC. Transpose before feeding back (handled automatically in inference code).

4. **Duration predictor speed input**: float32 (not int32). input_ids: int64 (not int32).

5. **Weight norm**: Remove with `nn.utils.remove_weight_norm(layer)` before any ONNX export that uses `weight_norm`'d layers. onnx2tf incorrectly handles `weight_g`/`weight_v` parameters.

6. **Vocoder DFT**: All exported ONNX and TFLite vocoder models use `VocosStreamChunkReal` — real-matmul IDFT (cos/sin basis) + pad-sum OLA, fully inside the network. No external `numpy.fft.irfft` is needed at inference time.
