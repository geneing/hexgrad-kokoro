# Learnings

## Environment
- `litert-torch-nightly` pulls in `torchao`, `ai-edge-litert-nightly`,
  `ai-edge-quantizer-nightly`, `tensorflow`, `scipy` etc. as transitive deps —
  no need to list them all explicitly in pyproject.toml.
- `uv add "torch==2.12.0"` correctly pins and resolves the CUDA wheel.
- Baseline Kokoro English generation needs `pip` available inside the uv
  environment because Misaki/spaCy setup invokes `python -m pip` to install
  `en-core-web-sm==3.8.0` when the model package is missing.
- `scipy.io.wavfile.write` is already available through transitive deps and is
  sufficient for writing 24 kHz baseline WAV files; no `soundfile` dependency is
  needed for this path.

## Model
- `CustomAlbert.forward` already returns `last_hidden_state` directly (not the
  full HuggingFace output object). The BertWrapper must call
  `self.bert_encoder(out)` not `self.bert_encoder(out.last_hidden_state)`.
- `ref_s` is 256-dim; first 128 → decoder style, last 128 → predictor style.
- Decoder waveform generation is stochastic in the unmodified PyTorch model:
  `SineGen` uses `torch.rand` for initial phase noise and `torch.randn_like` for
  additive source noise. Use saved `.npz` tensors for exact numeric TFLite
  parity; use accepted WAVs for subjective audio quality baseline.

## litert_torch
- `litert_torch.convert` requires all inputs to be positional tensors — no
  keyword args, no Python scalars in the forward signature.
- Multi-signature `.tflite` shares weights when the same `nn.Module` instance
  is reused across signatures.

## TensorFlow / Keras LSTM bridge
- PyTorch `nn.LSTM` weights can be copied directly into `tf.keras.layers.LSTM`
  for Kokoro's LSTMs:
  - `weight_ih_l0.T` → Keras `kernel`
  - `weight_hh_l0.T` → Keras `recurrent_kernel`
  - `bias_ih_l0 + bias_hh_l0` → Keras `bias`
  - gate order matches (`input, forget, cell, output`)
- For bidirectional LSTM, copy the PyTorch `_reverse` weights into the Keras
  backward layer with `go_backwards=True`; Keras aligns the backward sequence
  output back to original timestep order when using `Bidirectional(...,
  merge_mode="concat")`.
- `tensorflow==2.21.0` / `keras==3.14.1` converts these wrappers to compact
  recurrent `WHILE` subgraphs, not static unrolled trees. This is already a
  useful alternative to litert-torch's unrolled ATen graph for compile size.
- The current TF/Keras converter did not emit `UNIDIRECTIONAL_SEQUENCE_LSTM`
  for the tested Keras 3 wrappers. Treat fused sequence-LSTM lowering as a
  separate optimization/investigation from the compact `WHILE` bridge.
- Force TensorFlow conversion to CPU in local scripts with
  `CUDA_VISIBLE_DEVICES=-1`; the local NVIDIA driver is too old for the
  TensorFlow CUDA runtime.
- The checkpoint's `DurationEncoder` has three LSTM/AdaLayerNorm pairs, not two.
  Split exporters should discover `predictor.text_encoder.lstms` dynamically
  instead of assuming fixed indices.
- When splitting `DurationEncoder`, preserve the original channel-first
  `[B, C, T]` boundary around `AdaLayerNorm`. The exported split path uses:
  LSTM `[B,T,640] -> [B,T,512]`, transpose to `[B,512,T]`, Ada+style concat
  `[B,640,T]`, transpose back to `[B,T,640]` for the next recurrent model.

## Bidirectional LSTM with padding (TextEncoder / DurationEncoder)
- `pack_padded_sequence` / `pad_packed_sequence` are NOT torch.export-friendly;
  replace with direct `self.lstm(x)` call.
- Exact parity (max_abs_diff < 1e-5) only when T_actual == T_bucket (no padding).
- When T_actual < T_bucket: backward LSTM direction is contaminated — it starts
  from the last zero-padded position rather than from T_actual-1. Diffs ~1.0,
  far outside any reasonable atol.
- Baseline-driven parity against `export/test.txt` confirmed the issue is not
  limited to synthetic tests. The real baseline chunks (`T=37`, `T=103`) failed
  TextEncoder and PredictorDur badly when routed through padded `T=128`
  signatures. Exact-length signatures restored parity.
- Deployment constraint: always call text_encoder with T_actual == T_bucket.
  Pick the smallest bucket that fits, then pad input_ids AND fill the padding
  region with a known-neutral token (e.g., repeat last token) for best quality.
- Correct fix (not yet implemented): split bidirectional LSTM into forward +
  backward passes, flip input for backward using a mask-aware `torch.flip`
  with T_actual as a tensor input (requires dynamic shape support).

## Baseline parity harness
- Use `export/parity_baseline_tflite.py` to compare exported TFLite submodules
  against promoted PyTorch `.npz` tensors in `test_output/baseline/tensors`.
- The harness prefers current-hash baseline-compatible exports when present and
  falls back to the historical per-step exports for BERT/decoder.
- Decoder element-wise audio parity is still not the acceptance criterion
  because the exported decoder patches stochastic source generation. The
  baseline harness checks finite output and RMS ratio for decoder audio.

## Weight-norm (new parametrizations API)
- `from torch.nn.utils.parametrizations import weight_norm` (new API, used in
  TextEncoder CNN) is fully traceable by torch.export — no removal needed.
  The old `torch.nn.utils.weight_norm` (deprecated) may require removal.

## AOT compilation (Google Tensor G5)
- `ai-edge-litert-sdk-google-tensor==2.1.5` must be installed with
  `GOOGLE_TENSOR_SDK_BETA=/rhome/eingerman/Projects/DeepLearning/TTS/Kokoro/litert_npu/litert_plugin_compiler.tar.gz`
  set at install time, not just at runtime. The SDK plugin is unpacked into
  site-packages during build.
- All Kokoro sub-modules tested so far achieve 100% NPU offload (no CPU
  fallback). BERT: 677 ops/subgraph × 4; TextEncoder: 1097/4169/8265 ops.
- AOT should run after all fp32 TFLite conversion/parity steps are complete and
  the final multi-signature model exists. Per-step AOT burns a lot of compiler
  time on intermediate files and can distract from export/parity issues.
- Prefer the `uv run litert-torch` CLI for repeatable Tensor G5 AOT compiler
  runs. The Google Tensor backend ID is `GOOGLE`; the Pixel 10 SoC value is
  `Tensor_G5`.
- Current local CLI smoke check fails before help output because
  `litert_torch.generative.export_hf.core.attention` expects
  `transformers.AttentionInterface`, which is missing from the installed
  `transformers` version. Fix the dependency mismatch before relying on the CLI.
- Hybrid package AOT with the Python API succeeds for BERT and non-recurrent
  hybrid components under `outputs/a072f53/hybrid_package/aot/`.
- TensorFlow/Keras recurrent `WHILE` LSTM models are intentionally not compiled
  for Tensor G5; keep them in fallback runtime until a fused sequence-LSTM or
  custom lowering path exists.
- Decoder Tensor G5 AOT still fails even with `google_tensor_sharding_intensity`
  set to `"minimal"`; next useful experiment is a single-signature
  `decoder_short` compile rather than the full 3-signature decoder.
