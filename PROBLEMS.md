# Problems

## Open

### [2026-06-02 21:44:20 PDT] Keras LSTM bridge emits compact `WHILE`, not fused sequence LSTM (git 11e3dd2)

**Symptom:** `export/export_fused_lstm_tf.py` successfully converts Kokoro bare
bidirectional `nn.LSTM` layers through Keras/TFLite with parity, but the
FlatBuffer op list contains two recurrent `WHILE` subgraphs instead of
`UNIDIRECTIONAL_SEQUENCE_LSTM` or `BIDIRECTIONAL_SEQUENCE_LSTM`.

**Details:**
- Tested with `tensorflow==2.21.0` and `keras==3.14.1`.
- All five Kokoro bare LSTM targets passed PyTorch vs Keras and PyTorch vs
  TFLite parity at `T=32` with `max_abs_diff <= 1e-6`.
- Op pattern is compact and not statically unrolled:
  `RESHAPE, REVERSE_V2, WHILE, RESHAPE, REVERSE_V2, WHILE, RESHAPE, CONCATENATION`.
- This likely fixes graph-size and conversion-time blow-up, but may not get
  the same delegate/AOT behavior as a true TFLite fused sequence LSTM op.

**Attempted solutions:**
1. Built `tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(..., unroll=False))`
   wrappers and copied PyTorch weights gate-for-gate.
2. Converted with builtin-only `tf.lite.TFLiteConverter.from_keras_model`.
3. Verified op list with the TFLite interpreter.

**Next to try:**
- Check whether an older `tf_keras` / TensorFlow converter path still emits
  `UNIDIRECTIONAL_SEQUENCE_LSTM` for this structure.
- Try direct SavedModel signatures from `tf.compat.v1.keras.layers.LSTM` or a
  converter-compatible composite function.
- If fused ops are required, implement a custom lowering from PyTorch/Keras LSTM
  weights to TFLite sequence LSTM via MLIR/FlatBuffer generation or upstream
  `litert-torch` lowering.

### [2026-06-02] Decoder AOT compile crashes with `error type: INTERNAL` on Google Tensor G5

**Symptom:** `aot_compile(decoder_multisig_fp32.tflite, sharding_intensity="extensive")` crashes after ~36 min. The `compiler_worker` subprocess exits with a stack trace and `Compilation has failed with error type: INTERNAL`.

**Details:**
- 1811 ops × 3 subgraphs (decoder_short/medium/long), ALL ops selected for NPU (no CPU fallback)
- Heap showed very large allocation before crash — likely OOM in the compiler subprocess
- `"medium"` and `"high"` are invalid sharding values; only `"minimal"` and `"extensive"` are valid
- `keep_going=True` with `"high"` or `"medium"` fails immediately (invalid flag)
- `keep_going=True, sharding="minimal"` was started but paused before result

**Attempted solutions:**
1. `keep_going=False, sharding="extensive"` — INTERNAL crash after 36 min
2. `keep_going=True, sharding="high"` — invalid flag value (fails immediately)
3. `keep_going=True, sharding="medium"` — invalid flag value (fails immediately)
4. `keep_going=True, sharding="minimal"` — started, paused before result

**Next to try:**
- Defer decoder AOT experiments until Step 5 produces the final multi-signature `.tflite`
- Compile the final model through `uv run litert-torch` so flags, logs, and output paths are reproducible
- If final-model AOT also crashes: retry `sharding="minimal"`, then split decoder signatures into single-sig TFLite files and compile `decoder_short` alone
- File a bug at https://github.com/google-ai-edge/LiteRT/issues

### [2026-06-02] `uv run litert-torch --help` fails on missing `transformers.AttentionInterface`

**Symptom:** `uv run litert-torch --help` imports `litert_torch.generative.export_hf`
before printing help, then fails with `AttributeError: module transformers has no
attribute AttentionInterface`.

**Root cause:** The installed `transformers` version is older than the
`litert-torch` CLI expects for its generative `export_hf` attention registration.

**Attempted solutions:**
1. Ran with `UV_CACHE_DIR=/tmp/uv-cache` to avoid the sandbox read-only uv cache;
   the command then reached Python import and exposed the dependency mismatch.

**Next to try:**
- Upgrade or pin `transformers` to a version exposing `AttentionInterface`, then
  rerun `uv run litert-torch --help` and record the exact AOT CLI syntax.

## Resolved

### [2026-06-02 22:06:40 PDT] Hybrid compact-LSTM split pipeline initially predicted wrong alignment length (git 11e3dd2)

**Symptom:** First end-to-end run of `export/export_hybrid_fused_lstm_pipeline.py`
failed before F0/N with `KeyError: 61`; the hybrid duration path predicted
`aligned_len=61` for line 1, while the baseline expected `104`.

**Root cause:** The first split implementation exported only two
`DurationEncoder` LSTM/AdaLayerNorm pairs, but this checkpoint has three pairs.
The duration split also needed to preserve the original channel-first
`[B, C, T]` layout at the AdaLayerNorm boundary.

**Attempted solutions:**
1. Verified standalone Keras/TFLite LSTM parity against direct PyTorch LSTM.
2. Compared split duration intermediates against `PredictorDurWrapper`.
3. Changed Ada split wrappers to use channel-first tensors.
4. Made the exporter discover every `DurationEncoder` LSTM/Ada pair dynamically.

**Resolution:** Re-ran the full hybrid compact-LSTM TFLite pipeline. All three
baseline chunks passed parity, and WAVs were written under
`test_output/11e3dd2/hybrid_fused_lstm/wavs/`.

### [2026-06-02 21:18:55 PDT] Baseline parity failed for padded LSTM signatures (git 03301cf)

**Symptom:** `export/parity_baseline_tflite.py` initially reported large
baseline discrepancies for real `export/test.txt` chunks:
- TextEncoder max diff ~`1.5`
- PredictorDur logits max diff ~`22`
- PredictorF0N F0 max diff ~`160`
- Line 3 skipped TextEncoder/PredictorDur/F0N because existing buckets were too
  small (`T=447`, `T_aligned=1048`)

**Root cause:** Existing LSTM-containing exports use direct LSTM calls instead
of `pack_padded_sequence`. For bidirectional LSTMs, padded signatures change the
backward recurrent state because the backward pass starts at the padded tail.

**Attempted solutions:**
1. Re-ran the historical exported TFLite files against
   `test_output/baseline/tensors`; confirmed BERT passed and decoder waveform
   checks passed, while LSTM-containing modules failed on padded chunks.
2. Added exact baseline signatures for the real token/aligned lengths in
   `export/test.txt`.

**Resolution:** Exported baseline-compatible fp32 TFLite files:
- `outputs/03301cf/kokoro_text_encoder_baseline_fp32.tflite`
- `outputs/03301cf/kokoro_predictor_dur_baseline_fp32.tflite`
- `outputs/03301cf/kokoro_predictor_f0n_baseline_fp32.tflite`

Final baseline parity passed for all chunks; see
`test_output/03301cf/baseline_parity/summary.tsv`.

### [2026-06-02 19:20:16 PDT] Baseline TTS failed because `.venv` had no `pip`

**Symptom:** `uv run python export/generate_baseline_tts.py` exited with
`No module named pip` while initializing English G2P.

**Root cause:** Misaki/spaCy setup invokes `python -m pip` to install/load
`en-core-web-sm==3.8.0` when that package is absent, but the uv environment did
not include `pip`.

**Attempted solutions:**
1. Checked existing WAV-writing deps: `scipy` was installed, `soundfile` was not.
2. Added `pip>=26.1.2` with `uv add pip`.

**Resolution:** Rerunning baseline generation installed `en-core-web-sm==3.8.0`
and wrote all three WAV files to `test_output/c92a93d/baseline_tts/`.

### [2026-06-02] `litert_torch` fails to import — `interpreter` missing from `ai-edge-litert-nightly 2.2.0.dev20260601`

**Symptom:** `ImportError: cannot import name 'interpreter' from 'ai_edge_litert'`

**Root cause:** `litert_torch` 0.10.0.dev uses `from ai_edge_litert import interpreter`.
`ai-edge-litert-nightly 2.2.0.dev20260601` removed the `interpreter` module
(replaced by `CompiledModel` API), breaking the import.

**Attempted solutions:**
1. Installed stable `ai-edge-litert==2.1.5` alongside nightly to add back
   `interpreter.py` — this caused a secondary conflict where
   `_pywrap_tfl_calibration.so` (nightly) had an undefined symbol because
   the stable version's `.so` was overwriting the nightly's.

**Resolution:** Downgraded `ai-edge-litert-nightly` to `2.2.0.dev20260518`
which still includes `interpreter`. Installed with `uv pip install` (not
tracked in pyproject.toml since litert-torch-nightly manages this dep).
**Symptom:** `ImportError: libtorch_cuda.so: undefined symbol: ncclCommResume`
on `import torch`.

**Root cause:** `torch==2.12.0` dynamically links `libtorch_cuda.so` against
the system `libnccl.so.2`. The system has `libnccl2==2.28.9` (CUDA 13.0 apt
package) which does not export `ncclCommResume`. That symbol was introduced in
NCCL 2.29.7. The `nvidia-nccl-cu13==2.29.7` PyPI wheel ships headers only —
no `.so` — so there is no pip-installable fix.

**Attempted solutions:**
1. Searched for a newer `libnccl.so.2` in `/opt`, `/usr/local/cuda*`,
   conda envs — not found.
2. Checked whether `torch/lib/` bundles its own NCCL — it does not for this
   wheel.

**Resolution:** Used `torch==2.11.0` — satisfies litert-torch's `>=2.11.0`
requirement AND resolves to `nvidia-nccl-cu13==2.28.9` (system-compatible).
System NCCL upgrade (via apt `libnccl2>=2.29.7`) would unblock torch 2.12.0.

---
<!-- Template:
### [YYYY-MM-DD hh:mm:ss] Short description [git hash if applicable]
**Symptom:** ...
**Root cause:** ...
**Attempted solutions:**
1. ...
**Resolution:** ...
-->
