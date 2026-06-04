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

## TCN replacement path
- 2026-06-03 23:15:57 PDT (git 11e3dd2 starting point): branch
  `tcn_lstm_replacement` starts from the last baseline checkpoint before
  hybrid conversion. Hybrid conversion commits begin at `a072f53`.
- Non-causal Conv1d/TCN sequence mixers avoid the recurrent export problems:
  no `pack_padded_sequence`, no `pad_packed_sequence`, no recurrent LiteRT op,
  no bidirectional backward-state contamination from zero padding.
- A config-driven mixer switch works cleanly:
  `sequence_mixer.type = "lstm"` keeps the historical Kokoro architecture;
  `sequence_mixer.type = "tcn"` builds export-facing TCN modules.
- The existing Kokoro checkpoint can be partially loaded into a TCN model:
  BERT, BERT projection, TextEncoder embedding/CNN, AdaLayerNorm, AdaIN heads,
  F0/N projections, duration projection, and decoder weights are reused. New
  TCN weights are randomly initialized and must be trained/distilled before
  audio quality is meaningful.
- TCN fp32 TFLite conversion is straightforward for TextEncoder,
  PredictorDur, and PredictorF0N. Initial parity against the TCN PyTorch model
  passed with max diffs in the `1e-04` range or lower.
- TextEncoder TCN Tensor G5 AOT compiled quickly and fully offloaded all three
  signatures. PredictorDur and PredictorF0N should be AOT-tested after the
  script messaging is updated and final source-hash exports are regenerated.
- Distillation is the right training method; LSTM weights should not be
  directly transformed into Conv1d weights. Collect teacher tensors from the
  frozen original LSTM model and train the TCN student on intermediate outputs
  before any end-to-end fine-tune.
- LJSpeech and LibriTTS are useful here as text corpora rather than paired
  audio. Use `/export/eingerman/audio/LJSpeech-1.1/` and
  `/export/eingerman/audio/LibriTTS/LibriTTS/`; the collector reads LJSpeech
  `metadata.csv` and LibriTTS transcript text, then uses Kokoro G2P chunking
  and multiple Kokoro voice packs to generate teacher tensor targets.
- Store distillation datasets, checkpoints, and TensorBoard runs under
  `/export/eingerman/audio/tcl_distil/` so large generated artifacts stay out of
  the repo workspace.
- Save `f0n_shared` separately during collection. It lets the student train the
  shared F0/N sequence mixer directly instead of only learning through the
  heavier F0/N heads.
- Training variable-length TTS tensors benefits from two safeguards: bucket-ish
  padding inside each DataLoader batch, and recursive batch splitting on CUDA
  OOM for unusually long examples.
- `torch.utils.tensorboard.SummaryWriter` requires the `tensorboard` package;
  it is now a direct uv dependency.

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
