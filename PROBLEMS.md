# Problems

## Open

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
