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
