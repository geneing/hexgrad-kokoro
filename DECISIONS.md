# Decisions

## [2026-06-02] Export sub-modules separately, not as a monolithic model

**Decision:** Export each of the 5 sub-modules (`bert`, `text_encoder`,
`predictor_dur`, `predictor_f0n`, `decoder`) as separate TFLite signatures
within a single multi-signature `.tflite` file.

**Rationale:**
- The full `KModel.forward_with_tokens` contains non-exportable dynamic ops
  (alignment scatter, `repeat_interleave` with dynamic counts) that must stay
  on CPU.
- Per-module export makes each step independently testable and debuggable.
- Multi-signature sharing means no weight duplication in the final file.

**Alternatives considered:**
- Single monolithic export: ruled out due to alignment scatter ops.
- ONNX export (already exists in `examples/export.py`): ruled out as the goal
  is LiteRT/TFLite for Android NPU deployment.

---

## [2026-06-02] Bucket-based static shapes for variable-length inputs

**Decision:** Use multiple signatures per sub-module (e.g. bert_short/medium/
long/max) rather than attempting dynamic shapes.

**Rationale:**
- `.tflite` models require static tensor shapes at compile time.
- `torch.export` dynamic shape support in litert-torch is not yet fully
  verified for all ops in this model.
- Bucketing at inference time (pad to nearest bucket) is a standard,
  well-understood approach.

**Alternatives considered:**
- Dynamic shapes via `torch.export.Dim`: will be retried if litert-torch
  gains stable support, as it would reduce model file size.

---

## [2026-06-02] torch==2.6.0 (downgraded from 2.12.0)

**Decision:** Pin to `torch==2.6.0+cu124`.

**Rationale:** torch==2.12.0 links `libtorch_cuda.so` against
`libnccl.so.2` and requires the `ncclCommResume` symbol introduced in
NCCL 2.29.7. The system has NCCL 2.28.9 and upgrading requires root.
torch==2.6.0 imports cleanly with the system NCCL.
See PROBLEMS.md for full details.

**Path to 2.12.0:** Upgrade system NCCL via `apt install libnccl2>=2.29.7`
then `uv add "torch==2.12.0"`.

**Alternatives considered:** CPU-only torch wheel — rejected, CUDA needed
for model conversion performance.

---
<!-- Template:
### [YYYY-MM-DD hh:mm:ss] Short description [git hash if applicable]
**Decision:** ...
**Rationale:** ...
**Alternative solutions:**
**Resolution:** ...
-->
