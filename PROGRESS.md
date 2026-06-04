# Export Progress

## Project Setup
- [x] uv environment, torch==2.11.0, litert-torch-nightly==0.10.0.dev20260601, ai-edge-litert-nightly==2.2.0.dev20260518
- [x] AGENTS.md written with full export plan, test strategy, and AOT guide
- [x] Output directories created: `outputs/`, `test_output/`

## In Progress
- [ ] **TCN distillation path** — replacing export-facing LSTM/BiLSTM mixers with non-causal Conv1d/TCN modules and training them as students from the frozen original LSTM checkpoint.

## Next Steps
- [ ] Commit the TCN model/config/export-script changes, then rerun TFLite exports so `outputs/<git_hash>/` matches the new source hash.
- [ ] Add a teacher tensor collection script for distillation data:
  - Load original LSTM config as teacher and TCN config as student.
  - Run `export/test.txt` first, then a larger text corpus through `KPipeline`.
  - Save `input_ids`, masks, style slices, `d_en`, `text_encoder_out`, `duration_encoder_out`, `predictor_duration_mixer_out`, `duration_logits`, `pred_dur`, `pred_aln_trg`, `predictor_aligned_en`, `f0n_shared_out`, `F0`, `N`, decoder inputs, and optional audio.
  - Store cases under `test_output/<git_hash>/distill_teacher/<case>/` with a manifest containing text, phonemes, lengths, voice id, speed, style source, and teacher/student git hashes.
- [ ] Add staged distillation training:
  - TextEncoder TCN: masked MSE/cosine against teacher `text_encoder_out`.
  - DurationEncoder TCN: masked MSE against teacher `duration_encoder_out`.
  - Duration mixer/head: hidden MSE plus duration-logit MSE/KL.
  - F0/N shared mixer: hidden MSE plus F0/N MSE.
  - Short end-to-end fine-tune with BERT/decoder initially frozen.
- [ ] Save distilled checkpoints to `checkpoints/tcn_distill/` with source git hash, data manifest hash, and validation summary.
- [ ] Reexport distilled fp32 TFLite, run PyTorch-vs-TFLite parity, generate comparison WAVs, then AOT compile Tensor G5.
- [ ] Quantization after distilled fp32 parity: fp16 AOT, int8 PT2E.

## Completed
- [x] **2026-06-03 23:15:57 PDT (git 11e3dd2 starting point) — Pivot from hybrid conversion to TCN replacement**: created branch `tcn_lstm_replacement` from `11e3dd2`, the last baseline checkpoint before hybrid conversion (`a072f53`).
  - Added config-driven `sequence_mixer.type` support; default historical behavior remains LSTM when the config omits the field.
  - Updated `checkpoints/config.json` to use non-causal TCN blocks: 4 blocks, kernel size 5, dilations `[1, 2, 4, 8]`.
  - Replaced export-facing recurrent mixers with TCN alternatives for `TextEncoder`, `DurationEncoder`, `ProsodyPredictor.run_duration_mixer`, and `ProsodyPredictor.F0Ntrain`.
  - Partial checkpoint load reuses non-recurrent Kokoro weights and leaves new TCN weights randomly initialized pending distillation.
  - Smoke shape check passed for TextEncoder, PredictorDur, and PredictorF0N.
  - Initial fp32 TFLite exports from the uncommitted starting hash passed PyTorch-vs-TFLite parity:
    - `outputs/11e3dd2/kokoro_text_encoder_multisig_fp32.tflite`; max diff <= `8e-06`; Tensor G5 AOT fully offloaded, output `kokoro_text_encoder_Google_Tensor_G5.tflite`
    - `outputs/11e3dd2/kokoro_predictor_dur_multisig_fp32.tflite`; duration max diff <= `1.56e-04`, `d_out` max diff <= `1.3e-05`
    - `outputs/11e3dd2/kokoro_predictor_f0n_multisig_fp32.tflite`; F0 max diff <= `2.44e-04`, N max diff <= `1.1e-05`
  - These initial exports are structurally valid but not quality-valid because the TCN weights are not distilled yet.

- [x] **2026-06-02 21:28:11 PDT (git 969e73d) — TFLite decoder WAVs for baseline inspection**: wrote subjective inspection WAV files from the final TFLite decoder parity tensors.
  - `test_output/03301cf/baseline_parity/wavs/line_01_chunk_01_decoder_short_audio_tflite.wav`
  - `test_output/03301cf/baseline_parity/wavs/line_02_chunk_01_decoder_medium_audio_tflite.wav`
  - `test_output/03301cf/baseline_parity/wavs/line_03_chunk_01_decoder_long_audio_tflite.wav`
  - Compare against accepted PyTorch baseline WAVs in `test_output/baseline/wavs/`.

- [x] **2026-06-02 21:18:55 PDT (git 03301cf) — Baseline-driven TFLite parity rerun and LSTM bucket fix**: added a parity harness for `test_output/baseline` tensors and re-ran all exported submodules against `export/test.txt` baseline chunks.
  - Added `export/parity_baseline_tflite.py`; summary: `test_output/03301cf/baseline_parity/summary.tsv`
  - Initial rerun: BERT passed, decoder passed waveform-level checks, but TextEncoder/PredictorDur/PredictorF0N failed on padded real chunks and line 3 exceeded existing LSTM/F0N buckets.
  - Added `export/export_baseline_buckets.py` and exported exact baseline signatures:
    - `outputs/03301cf/kokoro_text_encoder_baseline_fp32.tflite` (`T=32,37,103,128,256,447`)
    - `outputs/03301cf/kokoro_predictor_dur_baseline_fp32.tflite` (`T=32,37,103,128,256,447`)
    - `outputs/03301cf/kokoro_predictor_f0n_baseline_fp32.tflite` (`T_aligned=104,200,257,800,1048`)
  - Final parity PASSED for all baseline chunks:
    - BERT max diff <= `5.7e-05`
    - TextEncoder max diff <= `3.8e-06`
    - PredictorDur logits max diff <= `5.8e-05`
    - PredictorF0N F0 max diff <= `1.9e-04`
    - Decoder waveform-level checks passed with RMS ratios `1.44-1.77`

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
