# Export Progress

## Project Setup
- [x] uv environment, torch==2.11.0, litert-torch-nightly==0.10.0.dev20260601, ai-edge-litert-nightly==2.2.0.dev20260518
- [x] AGENTS.md written with full export plan, test strategy, and AOT guide
- [x] Output directories created: `outputs/`, `test_output/`

## In Progress
_None — paused at Step 4 AOT._

## Next Steps
- [ ] Step 4 — Decoder AOT (INTERNAL compiler crash; try `minimal` sharding or split subgraphs; see PROBLEMS.md)
- [ ] Step 5 — Multi-signature assembly (kokoro_multisig.tflite)
- [ ] Quantization: fp16 AOT, int8 PT2E

## Completed
- [x] **Step 4 — Decoder TFLite export** (`examples/export_decoder.py`) — `outputs/c46f2e1/kokoro_decoder_multisig_fp32.tflite`
  - 3 signatures: decoder_short (T=200→120k samples), decoder_medium (T=800→480k), decoder_long (T=2000→1.2M)
  - Parity tests PASSED: finite outputs, rms_ratio 1.01–4.19 across all buckets
  - WAVs in `test_output/c46f2e1/decoder/`
  - Patches: PoolEquiv (ConvTranspose1d op=1), SineGen noise→zeros_like, CustomSTFT.transform torch.where, disable_complex=True
  - **AOT: BLOCKED** — G5 compiler crashes with `error type: INTERNAL` after 36 min for `extensive` sharding. `medium`/`high` are invalid values. `minimal` started but was paused. See PROBLEMS.md.

- [x] **Step 3b — F0Ntrain export** (`examples/export_predictor_f0n.py`) — `outputs/c46f2e1/kokoro_predictor_f0n_multisig_fp32.tflite`
  - 2 signatures: predictor_f0n (T_aligned=200), predictor_f0n_long (T_aligned=800)
  - All parity tests PASSED (max_abs_diff < 2e-3)
  - ~9.6 GMACs; **AOT skipped** — LSTM (shared) + ConvTranspose1d output_padding unsupported by litert_torch
  - Export fix: `ConvTranspose1d(output_padding=1)` replaced with `PoolEquiv` (zero-interleave + flipped-weight Conv1d)

- [x] **Step 3a — ProsodyPredictor duration head export** (`examples/export_predictor_dur.py`) — `outputs/c46f2e1/kokoro_predictor_dur_multisig_fp32.tflite`
  - 3 signatures: predictor_dur_short (32), predictor_dur_medium (128), predictor_dur_long (256)
  - All parity tests PASSED (max_abs_diff < 2e-3)
  - ~6.4 GMACs; **AOT skipped** — LSTM layers (DurationEncoder + predictor.lstm) cause >20 min compile times on Tensor G5 plugin; runs on CPU/GPU fallback

- [x] **Step 2 — TextEncoder export** (`examples/export_text_encoder.py`) — `outputs/5c9f727/kokoro_text_encoder_multisig_fp32.tflite`
  - 3 signatures: text_encoder_short (32), text_encoder_medium (128), text_encoder_long (256)
  - Exact parity tests PASSED (max_abs_diff < 1e-5 for full-bucket inputs)
  - Approx tests [INFORMATIONAL]: bidirectional LSTM backward contaminated by zero-padding (~1.0 diff) — deployment constraint: use full-bucket inputs only
  - ~2.3 GMACs; 100% NPU offload (1097/4169/8265 ops per subgraph)
  - AOT: `outputs/5c9f727/kokoro_text_encoder_Google_Tensor_G5.tflite` (98.0 MB)

- [x] **Step 1 — BERT export** (`examples/export_bert.py`) — `outputs/426e5d5/kokoro_bert_multisig_fp32.tflite`
  - 4 signatures: bert_short (32), bert_medium (128), bert_long (256), bert_max (510)
  - All 8 parity tests PASSED (max_abs_diff < 1e-4)
  - ~63 GMACs per forward pass

## Forks / Open Questions
- TextEncoder LSTM: mask-based rewrite vs fixed-length wrapper — TBD after Step 1
- Bucket sizes may need tuning once real phoneme length distributions are measured
- int8 calibration data: use a short corpus of English sentences from examples/
