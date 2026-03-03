# AGENTS.md (kokoro/tf)

## Scope
This directory contains all TensorFlow-specific code for Kokoro streaming-vocos
conversion, training, quantization, and validation.

Current expectation:
- TF generator mirrors the **streaming-causal vocos** architecture used by
  `kokoro/train_vocos.py`.
- Checkpoint mapping supports streaming key layout and weight-norm-derived conv
  weights via `checkpoint_utils.py`.
- Chunked parity semantics match PT:
  - run conditioner once on full features,
  - then chunk backbone + ISTFT head.

## Module Responsibilities
- `model.py`
  - TF streaming-vocos generator layers.
  - Export-safe ISTFT head.
  - TF discriminators/losses used by TF training/export flow.
- `checkpoint_utils.py`
  - PT checkpoint loading and prefix stripping.
  - Streaming-aware config inference from checkpoint keys.
  - PT->TF weight mapping and conversion output helpers.
- `training.py`
  - TF training entrypoint for paired vocoder features -> waveform.
- `convert.py`
  - CLI wrapper to convert PT checkpoint into TF weights/checkpoints.
- `export.py`
  - LiteRT export, optional QAT-like tuning, quantization candidate handling.
- `validate_quant.py`
  - Focused quant validation of TF fp32/fp16/int8 variants on held-out samples.
- `smoke_test_from_pt_ckpt.py`
  - Minimal conversion/inference sanity path.
- `smoke_compare_pt_tf.py`
  - PT vs TF parity checks (full-forward and chunked comparisons using
    conditioner-once chunking).
- `investigate_chunked_streaming_vocos_tf.py`
  - TF full-vs-chunked quality investigation with waveform plots and boundary
    diagnostics.

## Streaming Vocos Notes
- Streaming checkpoints typically contain keys like:
  - `backbone.embed.conv.conv.weight_v`
  - `backbone.convnext.<i>.dwconv.conv.conv.weight_v`
- Legacy checkpoints may still exist; handle intentionally.
- For current migration tasks, default behavior should be streaming-first.

## CLI Examples
### Convert PT checkpoint -> TF artifacts
- `uv run python -m kokoro.tf.convert --pytorch-checkpoint output/checkpoints/last.pt --output-dir output/tf_checkpoints`

### TF quant validation (streaming expected)
- `uv run python -m kokoro.tf.validate_quant --pytorch-checkpoint output/checkpoints/last.pt --val-filelist inputs//filelists/vocos.val.txt --vocos-impl streaming`

### PT/TF parity smoke compare
- `uv run python -m kokoro.tf.smoke_compare_pt_tf --pytorch-checkpoint output/checkpoints/last.pt --pairs-root inputs//pairs --vocos-impl streaming --streaming-vocos-repo third_party/vocos_streaming`

### TF chunked-vs-full investigation (converted TF weights)
- `uv run kokoro-vocos-quant-compare-tf --tf-config output/tf_checkpoints/generator_config.json --tf-weights output/tf_checkpoints/generator.weights.h5 --data-root inputs/ --out-dir output/tf_chunked_vocos_compare --chunked-variant both`

### TF training from paired data
- `uv run python -m kokoro.tf.training --data-root inputs/ --train-filelist inputs//filelists/vocos.train.txt --val-filelist inputs//filelists/vocos.val.txt`

### LiteRT export pipeline
- `uv run python -m kokoro.tf.export --pytorch-checkpoint output/checkpoints/last.pt --data-root inputs/ --train-filelist inputs//filelists/vocos.train.txt --val-filelist inputs//filelists/vocos.val.txt --output-dir output/tf_litert`

## Guardrails
- Keep TensorFlow-only logic inside `kokoro/tf`; keep root-level files as
  compatibility shims where appropriate.
- Preserve checkpoint mapping contract in `checkpoint_utils.py`.
- Keep alignment with project defaults unless intentionally changed:
  - `sample_rate=24000`
  - `hop_length=300`
  - `n_fft=1200`
- Avoid silent architecture drift between PT and TF decoder paths.
- Keep chunked inference semantics consistent across TF tools; avoid legacy
  cache-window chunking helpers.

## Validation Expectations
When editing this directory, run at minimum:
- `uv run python -m py_compile kokoro/tf/*.py`
- `uv run python -m kokoro.tf.convert --help`
- `uv run python -m kokoro.tf.validate_quant --help`
- `uv run python -m kokoro.tf.training --help`
- `uv run python -m kokoro.tf.export --help`
