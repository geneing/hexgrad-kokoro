# Export Progress

## Project Setup
- [x] uv environment, torch==2.11.0, litert-torch-nightly==0.10.0.dev20260601, ai-edge-litert-nightly==2.2.0.dev20260518
- [x] AGENTS.md written with full export plan, test strategy, and AOT guide
- [x] Output directories created: `outputs/`, `test_output/`

## In Progress
- [ ] **Step 2 — TextEncoder export** (`examples/export_text_encoder.py`)
  - Requires LSTM pack/unpad fix (replace `pack_padded_sequence`/`pad_packed_sequence` with fixed-length call)

## Next Steps
- [ ] Step 3a — ProsodyPredictor duration head export
- [ ] Step 3b — F0Ntrain export
- [ ] Step 4 — Decoder export (requires weight_norm removal)
- [ ] Step 5 — Multi-signature assembly (kokoro_multisig.tflite)
- [ ] Quantization: fp16 AOT, int8 PT2E
- [ ] AOT compile for Tensor G5 (litert_npu/)

## Completed
- [x] **Step 1 — BERT export** (`examples/export_bert.py`) — `outputs/426e5d5/kokoro_bert_multisig_fp32.tflite`
  - 4 signatures: bert_short (32), bert_medium (128), bert_long (256), bert_max (510)
  - All 8 parity tests PASSED (max_abs_diff < 1e-4)
  - ~63 GMACs per forward pass
- [ ] Step 3a — ProsodyPredictor duration head export
- [ ] Step 3b — F0Ntrain export
- [ ] Step 4 — Decoder export (requires weight_norm removal)
- [ ] Step 5 — Multi-signature assembly (kokoro_multisig.tflite)
- [ ] Quantization: fp16 AOT, int8 PT2E
- [ ] AOT compile for Tensor G5 (litert_npu/)

## Forks / Open Questions
- TextEncoder LSTM: mask-based rewrite vs fixed-length wrapper — TBD after Step 1
- Bucket sizes may need tuning once real phoneme length distributions are measured
- int8 calibration data: use a short corpus of English sentences from examples/
