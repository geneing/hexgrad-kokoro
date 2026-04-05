# Kokoro TTS — TODO

## ✅ Completed

- [x] Fix ONNX_DFT error in vocoder TFLite (VocosPreIRFFT + numpy IRFFT)
- [x] Fix ConvTranspose1d amplitude bug (remove OLA from model)
- [x] Re-export duration predictor ONNX with weight_norm removed
- [x] Fix `t_en_static` mismatch (was diff 1.944, now 2.5e-6)
- [x] Fix en-trim bug in `generate()` (was 0.052 corr, now 1.0000)
- [x] Update `tflite_inference.py` for new tensor naming conventions
- [x] Full pipeline validation (corr=1.0000)
- [x] Write Android/Kotlin specification `kotlin/Android_Inference_Specification.md`
- [x] Write PLAN.md, PROGRESS.md
- [x] Replace `torch.fft.irfft` with real-matmul IDFT (cos/sin basis) in `VocosStreamChunkReal` — no DFT op in export
- [x] Replace `ConvTranspose1d` OLA with pad+sum OLA — fixes onnx2tf axis-transposition crash
- [x] FastGELU (`x * sigmoid(1.702x)`) replacing `nn.GELU` throughout Vocos backbone — no unsupported GELU op
- [x] `streaming_vocos_export.ipynb` LiteRT pipeline (cell 16): correlation 1.0 vs ONNX reference
- [x] onnx2tf conversion: use `-osd` (SavedModel path) for `vocoder_stream_chunk` to avoid `flatbuffer_direct` bug
- [x] `package_models.py` — zip ONNX + TFLite weights for download (`kokoro_streaming_vocos_models.zip`)
- [x] Update notebook specs (cell 1) with exact filenames, TFLite implementation notes

## 🔄 In Progress

- [ ] Android LiteRT compatibility test (GPU/NPU delegates)
- [ ] Int8 quantization for conditioner and vocoder stream chunk

## ❌ Not Started

- [ ] Port g2p (grapheme-to-phoneme) to Kotlin (or use precomputed vocabulary)
- [ ] Write complete Kotlin `KokoroTts.kt` implementation matching the spec
- [ ] Profile inference latency on Pixel 10 (NNAPI vs GPU vs CPU)
- [ ] Test float16 model quality vs float32
- [ ] Add voice pack loading in Kotlin (from .pt format or converted to npz)
- [ ] Create end-to-end Android demo app
- [ ] Handle multi-sentence splitting for texts longer than MAX_INPUT_LENGTH=510

## Notes

- Duration predictor TFLite uses `serving_default_*:0` naming and int64 input_ids / float32 speed
- Vocos state outputs `Identity_3..10` are [1,384,6] NCT but inputs expect [1,6,384] NTC — transpose required
- Vocoder TFLite (`vocoder_stream_chunk.tflite`) uses `-osd` onnx2tf conversion; source is `vocoder_stream_chunk_osd/vocoder_stream_chunk_float32.tflite`
- For long texts, split at sentence boundaries and concatenate audio for best quality
