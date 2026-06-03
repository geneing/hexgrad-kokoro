# Export Progress

## Project Setup
- [x] uv environment, torch==2.11.0, litert-torch-nightly==0.10.0.dev20260601, ai-edge-litert-nightly==2.2.0.dev20260518
- [x] AGENTS.md written with full export plan, test strategy, and AOT guide
- [x] Output directories created: `outputs/`, `test_output/`

## In Progress
_None — next work is Step 5 multi-signature assembly._

## Next Steps
- [ ] Step 5 — Multi-signature assembly (kokoro_multisig.tflite)
- [ ] Post-conversion AOT: compile final `kokoro_multisig.tflite` for Tensor G5 using the `uv run litert-torch` CLI where possible
- [ ] Quantization: fp16 AOT, int8 PT2E

## Completed
- [x] **2026-06-02 19:31:06 PDT (git 0f09bf6) — Stable baseline corpus and debug tensors**: promoted the accepted baseline WAVs to `test_output/baseline/wavs/` and saved PyTorch submodule reference tensors under `test_output/baseline/tensors/line_XX/chunk_YY/`.
  - Baseline docs: `test_output/baseline/README.md`
  - Metadata: `test_output/baseline/metadata.json`
  - Tensor format: NumPy `.npz`; files are `inputs.npz`, `bert.npz`, `text_encoder.npz`, `predictor_dur.npz`, `predictor_f0n.npz`, `decoder.npz`
  - Numeric parity should use `.npz` tensors. WAVs are the subjective audio baseline copied from `test_output/c92a93d/baseline_tts/`.

- [x] **2026-06-02 19:20:16 PDT (git c92a93d) — Baseline PyTorch TTS WAVs**: generated three `af_heart` baseline WAV files from `export/test.txt` using the local Kokoro checkpoint.
  - Outputs: `test_output/c92a93d/baseline_tts/baseline_line_01_af_heart.wav`, `baseline_line_02_af_heart.wav`, `baseline_line_03_af_heart.wav`
  - Manifest: `test_output/c92a93d/baseline_tts/manifest.tsv`
  - Added `export/generate_baseline_tts.py` for reproducible baseline generation.
  - Added `pip>=26.1.2` because Misaki's English G2P setup invokes `python -m pip` to install/load `en-core-web-sm==3.8.0`.

- [x] **2026-06-02 19:13:07 PDT (git a25fb6a) — AOT SDK environment docs**: documented the required absolute `GOOGLE_TENSOR_SDK_BETA=/rhome/eingerman/Projects/DeepLearning/TTS/Kokoro/litert_npu/litert_plugin_compiler.tar.gz` setting for Tensor G5 AOT runs.

- [x] **2026-06-02 19:08:31 PDT (git 70e9313) — AOT scheduling doc update**: AOT is now documented as a post-conversion phase after final multi-signature TFLite assembly, with `uv run litert-torch` preferred for Tensor G5 compiler runs.

- [x] **2026-06-02 19:00:35 PDT (git 7cf44fb) — Directory cleanup**: moved LiteRT export scripts from `examples/export_*.py` to `export/export_*.py`; `examples/export.py` remains the ONNX reference.

- [x] **Step 4 — Decoder TFLite export** (`export/export_decoder.py`) — `outputs/c46f2e1/kokoro_decoder_multisig_fp32.tflite`
  - 3 signatures: decoder_short (T=200→120k samples), decoder_medium (T=800→480k), decoder_long (T=2000→1.2M)
  - Parity tests PASSED: finite outputs, rms_ratio 1.01–4.19 across all buckets
  - WAVs in `test_output/c46f2e1/decoder/`
  - Patches: PoolEquiv (ConvTranspose1d op=1), SineGen noise→zeros_like, CustomSTFT.transform torch.where, disable_complex=True
  - AOT deferred until final multi-signature assembly. Historical standalone decoder AOT attempt failed with `error type: INTERNAL`; see PROBLEMS.md.

- [x] **Step 3b — F0Ntrain export** (`export/export_predictor_f0n.py`) — `outputs/c46f2e1/kokoro_predictor_f0n_multisig_fp32.tflite`
  - 2 signatures: predictor_f0n (T_aligned=200), predictor_f0n_long (T_aligned=800)
  - All parity tests PASSED (max_abs_diff < 2e-3)
  - ~9.6 GMACs; **AOT skipped** — LSTM (shared) + ConvTranspose1d output_padding unsupported by litert_torch
  - Export fix: `ConvTranspose1d(output_padding=1)` replaced with `PoolEquiv` (zero-interleave + flipped-weight Conv1d)

- [x] **Step 3a — ProsodyPredictor duration head export** (`export/export_predictor_dur.py`) — `outputs/c46f2e1/kokoro_predictor_dur_multisig_fp32.tflite`
  - 3 signatures: predictor_dur_short (32), predictor_dur_medium (128), predictor_dur_long (256)
  - All parity tests PASSED (max_abs_diff < 2e-3)
  - ~6.4 GMACs; **AOT skipped** — LSTM layers (DurationEncoder + predictor.lstm) cause >20 min compile times on Tensor G5 plugin; runs on CPU/GPU fallback

- [x] **Step 2 — TextEncoder export** (`export/export_text_encoder.py`) — `outputs/5c9f727/kokoro_text_encoder_multisig_fp32.tflite`
  - 3 signatures: text_encoder_short (32), text_encoder_medium (128), text_encoder_long (256)
  - Exact parity tests PASSED (max_abs_diff < 1e-5 for full-bucket inputs)
  - Approx tests [INFORMATIONAL]: bidirectional LSTM backward contaminated by zero-padding (~1.0 diff) — deployment constraint: use full-bucket inputs only
  - ~2.3 GMACs; 100% NPU offload (1097/4169/8265 ops per subgraph)
  - Historical per-step AOT: `outputs/5c9f727/kokoro_text_encoder_Google_Tensor_G5.tflite` (98.0 MB). Future AOT should run after Step 5 only.

- [x] **Step 1 — BERT export** (`export/export_bert.py`) — `outputs/426e5d5/kokoro_bert_multisig_fp32.tflite`
  - 4 signatures: bert_short (32), bert_medium (128), bert_long (256), bert_max (510)
  - All 8 parity tests PASSED (max_abs_diff < 1e-4)
  - ~63 GMACs per forward pass

## Forks / Open Questions
- TextEncoder LSTM: mask-based rewrite vs fixed-length wrapper — TBD after Step 1
- Bucket sizes may need tuning once real phoneme length distributions are measured
- int8 calibration data: use a short corpus of English sentences from examples/
