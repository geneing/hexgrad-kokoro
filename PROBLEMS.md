# Problems

## Open
_None yet._

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
### [YYYY-MM-DD] Short description
**Symptom:** ...
**Root cause:** ...
**Attempted solutions:**
1. ...
**Resolution:** ...
-->
