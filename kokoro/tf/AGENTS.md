# AGENTS.md (kokoro/tf)

## Scope
This directory contains all TensorFlow-specific code for Kokoro Vocos workflows.

## Module Responsibilities
- `model.py`: TensorFlow generator/discriminator/loss definitions and shared model utilities.
- `training.py`: TensorFlow training loop for paired Kokoro vocoder inputs -> waveform targets.
- `convert.py`: CLI for converting PyTorch generator checkpoints to TensorFlow weights/checkpoints.
- `checkpoint_utils.py`: Shared mapping/build/load/save helpers used by training/export/validation scripts.
- `export.py`: Quantization, optional QAT tuning, and LiteRT model export/validation pipeline.
- `validate_quant.py`: Focused quantization validation utility.
- `smoke_test_from_pt_ckpt.py`: Minimal conversion/inference sanity test.

## Guardrails
- Keep TensorFlow-only logic in this directory; avoid spreading TF code back to `kokoro/` root.
- Preserve the checkpoint key-mapping contract in `checkpoint_utils.py` when editing model shapes.
- Maintain `sample_rate=24000`, `hop_length=300`, and `n_fft=1200` alignment unless explicitly changed project-wide.
- Prefer modular changes that keep PyTorch baseline code paths untouched.

## Validation Expectations
When modifying this directory, run at minimum:
- `uv run python -m py_compile kokoro/tf/*.py`
- `uv run python -m kokoro.tf.convert --help`
- `uv run python -m kokoro.tf.training --help`
- `uv run python -m kokoro.tf.export --help`
