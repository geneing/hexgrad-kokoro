# Learnings

## Environment
- `litert-torch-nightly` pulls in `torchao`, `ai-edge-litert-nightly`,
  `ai-edge-quantizer-nightly`, `tensorflow`, `scipy` etc. as transitive deps —
  no need to list them all explicitly in pyproject.toml.
- `uv add "torch==2.12.0"` correctly pins and resolves the CUDA wheel.

## Model
- `CustomAlbert.forward` already returns `last_hidden_state` directly (not the
  full HuggingFace output object). The BertWrapper must call
  `self.bert_encoder(out)` not `self.bert_encoder(out.last_hidden_state)`.
- `ref_s` is 256-dim; first 128 → decoder style, last 128 → predictor style.

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
- Deployment constraint: always call text_encoder with T_actual == T_bucket.
  Pick the smallest bucket that fits, then pad input_ids AND fill the padding
  region with a known-neutral token (e.g., repeat last token) for best quality.
- Correct fix (not yet implemented): split bidirectional LSTM into forward +
  backward passes, flip input for backward using a mask-aware `torch.flip`
  with T_actual as a tensor input (requires dynamic shape support).

## Weight-norm (new parametrizations API)
- `from torch.nn.utils.parametrizations import weight_norm` (new API, used in
  TextEncoder CNN) is fully traceable by torch.export — no removal needed.
  The old `torch.nn.utils.weight_norm` (deprecated) may require removal.

## AOT compilation (Google Tensor G5)
- `ai-edge-litert-sdk-google-tensor==2.1.5` must be installed with
  `GOOGLE_TENSOR_SDK_BETA=<path-to-tar.gz>` set at install time, not just at
  runtime. The SDK plugin is unpacked into site-packages during build.
- All Kokoro sub-modules tested so far achieve 100% NPU offload (no CPU
  fallback). BERT: 677 ops/subgraph × 4; TextEncoder: 1097/4169/8265 ops.
