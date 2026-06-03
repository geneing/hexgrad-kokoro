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

## [2026-06-02] Run AOT after final TFLite assembly

**Decision:** Defer Tensor G5 AOT compilation until all fp32 sub-module exports,
TFLite parity tests, and final multi-signature assembly are complete.

**Rationale:**
- AOT compilation is slow and should validate the artifact that will actually
  ship, not intermediate per-step `.tflite` files.
- Separating export/parity from AOT makes failures easier to classify:
  conversion bugs are handled before compiler/delegate bugs.
- The `uv run litert-torch` CLI gives repeatable arguments, logs, output paths,
  and CI-friendly exit codes when the installed dependency set is compatible.

**Alternatives considered:**
- AOT after every sub-module export: rejected because it wastes compiler time
  and already hit long Tensor G5 compiler failures on decoder-only artifacts.
- Notebook-only or ad hoc Python API AOT: useful for exploration, but rejected
  as the default workflow because long compiler runs need reproducible CLI
  invocations.

---

## [2026-06-02] Add exact baseline signatures for LSTM-containing modules

**Decision:** Add exact `export/test.txt` baseline chunk signatures for
TextEncoder, PredictorDur, and PredictorF0N in separate baseline-compatible
TFLite files under the current git hash.

**Rationale:**
- Direct LSTM exports only match the original packed PyTorch reference when
  `T_actual == T_signature`.
- Baseline parity showed significant discrepancies when `T=37` and `T=103`
  chunks were padded into `T=128` signatures.
- The longest baseline line has `T=447`, which exceeded the previous
  TextEncoder/PredictorDur max bucket (`256`), and `T_aligned=1048`, which
  exceeded the previous F0N max bucket (`800`).

**Alternatives considered:**
- Keep padded direct-LSTM signatures and accept approximate output: rejected for
  parity because differences were large and cascaded into duration/F0/N.
- Rewrite LSTMs manually with mask-aware backward recurrence: still the better
  general runtime fix, but higher risk and not needed to validate the current
  baseline corpus.

---

## [2026-06-02] Use split TFLite artifacts for the compact-LSTM experiment

**Decision:** Export the compact-LSTM experiment as a Python-orchestrated split
TFLite pipeline rather than trying to merge litert-torch and TensorFlow-converted
LSTM flatbuffers into one multi-signature model immediately.

**Rationale:**
- litert-torch still lowers PyTorch LSTM calls into large unrolled graphs.
- TensorFlow/Keras conversion successfully preserves recurrence as compact
  TFLite `WHILE` subgraphs with parity, but those flatbuffers are produced by a
  different converter path.
- Proving end-to-end parity and generating review WAVs is higher value than
  solving cross-converter flatbuffer packaging first.
- The split pipeline exposes clean Android orchestration boundaries:
  non-recurrent litert-torch models, recurrent LSTM models, CPU alignment, then
  decoder.

**Alternatives considered:**
- Single litert-torch export: rejected for this experiment because it reintroduces
  unrolled LSTM graphs.
- Single TensorFlow/Keras reimplementation of all Kokoro modules: higher risk
  and more code than needed to validate the LSTM export path.
- Manual FlatBuffer/MLIR merge: deferred until the split pipeline proves useful
  and the desired recurrent op form is settled.

**Caveat:** Current TensorFlow/Keras recurrent exports emit compact `WHILE`
subgraphs, not `UNIDIRECTIONAL_SEQUENCE_LSTM` fused sequence-LSTM ops.

---

## [2026-06-02] Treat the hybrid package manifest as final assembly

**Decision:** Use `outputs/a072f53/hybrid_package/manifest.json` as the final
multi-component assembly for the accepted hybrid path, instead of forcing all
pieces into one physical multi-signature `.tflite`.

**Rationale:**
- The accepted hybrid path mixes converter outputs: litert-torch flatbuffers for
  non-recurrent modules and TensorFlow/Keras flatbuffers for compact recurrent
  LSTM subgraphs.
- The runtime needs to invoke CPU alignment and choose exact recurrent buckets
  between model calls, so a manifest-driven package maps more directly to the
  real Android orchestration.
- Tensor G5 AOT compilation is per component: BERT and non-recurrent hybrid
  pieces compile for NPU, recurrent `WHILE` models remain fallback, and decoder
  currently fails the Tensor G5 plugin.

**Alternatives considered:**
- Force a single cross-converter flatbuffer: deferred because it is packaging
  work with high risk and no parity benefit.
- Revert to litert-torch unrolled LSTM multisignature assembly: rejected after
  the hybrid audio review because it reintroduces the LSTM graph-size problem.

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
