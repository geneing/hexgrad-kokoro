# AGENTS.md

## Project Summary
This repository contains Kokoro streaming TTS training and deployment utilities.
Kokoro is a StyleTTS2-derived stack with streaming-oriented decoder paths.

Current decoder migration state:
- The primary vocoder path is **streaming Vocos** (causal ConvNeXt + streaming ISTFT head).
- Legacy non-streaming Vocos and iSTFT baseline code can still exist for comparison.
- PT/TF chunked inference semantics are aligned to:
  - run conditioner once on full feature sequence,
  - then chunk only backbone + ISTFT head.

Primary goals:
- Train decoder-only distillation from paired Kokoro vocoder features to waveform.
- Keep non-decoder Kokoro modules frozen.
- Preserve quality while fitting on a single ~24GB GPU.

## Streaming Vocos Requirements
When working with vocoder training/export/validation:
- Treat streaming-vocos as the default architecture.
- Expect streaming checkpoint keys such as:
  - `backbone.embed.conv.conv.weight_v`
  - `backbone.convnext.<i>.dwconv.conv.conv.weight_v`
- Preserve these config alignments unless intentionally changing project-wide:
  - `sample_rate=24000`
  - `hop_length=300`
  - `n_fft=1200`
  - `n_mels=80`
- Keep compatibility shims intact (`kokoro/*` wrappers that forward to `kokoro.tf.*`).

## Training Objectives and Constraints
- Train only Vocos decoder path for paired-feature distillation.
- Use weighted objective:
  - `L_total = L_GAN + L_FM + L_MR-STFT + L_GroupDelay`
- Default coefficients:
  - `gan=1.0`, `fm=2.0`, `mrstft=45.0`, `group_delay=2.0`, `mrd=1.0`
- Hardware target:
  - Single GPU, ~24GB VRAM
- Stability guidance:
  - Mixed precision (`auto/fp16/bf16`) where safe
  - Dynamic frame budgeting / accumulation as needed
  - Deterministic, reproducible settings preferred over peak throughput

## Core Workflows
### 1) Environment Setup
- `uv sync`
- `uv pip install pip`
- `uv run python -m unidic download`

### 2) Generate Paired Vocoder Data
- Download-only Kokoro assets:
  - `uv run kokoro-vocoder-data --download-only --repo-id hexgrad/Kokoro-82M`
- Smoke data generation:
  - `uv run kokoro-vocoder-data --num-sentences 1 --libritts-root /export/eingerman/audio/LibriTTS/LibriTTS --output-root inputs/ --write-repo-config`
- Full generation:
  - `uv run kokoro-vocoder-data --libritts-root /export/eingerman/audio/LibriTTS/LibriTTS --output-root inputs/ --write-repo-config`

### 3) Train Streaming Vocos Distillation
- Typical run:
  - `uv run python -m kokoro.train_vocos --data-root inputs/ --vocos-impl streaming --streaming-vocos-repo third_party/vocos_streaming --precision auto --tf32 --num-workers 8 --prefetch-factor 4 --log-every 20`

### 4) Prepare Inference and Quantized Weights
- Streaming default:
  - `uv run python prepare_weights.py --input output/checkpoints/last.pt --output-dir output/saved_infer_weights --vocos-impl streaming --streaming-vocos-repo third_party/vocos_streaming`

### 5) Export LiteRT Models
- Streaming default:
  - `uv run python vocos_export.py --weights-dir output/saved_infer_weights --output-dir output/litert --vocos-impl streaming --streaming-vocos-repo third_party/vocos_streaming`

### 6) TF Conversion + Validation
- Convert PT checkpoint to TF:
  - `uv run python -m kokoro.tf.convert --pytorch-checkpoint output/checkpoints/last.pt --output-dir output/tf_checkpoints`
- Validate TF quantized variants:
  - `uv run python -m kokoro.tf.validate_quant --pytorch-checkpoint output/checkpoints/last.pt --val-filelist inputs//filelists/vocos.val.txt --vocos-impl streaming`

### 7) PT/TF Parity Checks
- Full-forward + chunked compare:
  - `uv run python -m kokoro.tf.smoke_compare_pt_tf --pytorch-checkpoint output/checkpoints/last.pt --pairs-root inputs//pairs --vocos-impl streaming --streaming-vocos-repo third_party/vocos_streaming`
- TF full-vs-chunked investigation (converted TF weights):
  - `uv run kokoro-vocos-quant-compare-tf --tf-config output/tf_checkpoints/generator_config.json --tf-weights output/tf_checkpoints/generator.weights.h5 --data-root inputs/ --out-dir output/tf_chunked_vocos_compare --chunked-variant both`

## Important Paths
- Paired features: `inputs//pairs/**/*.pt`
- Target wavs: `inputs//audio/**/*.wav`
- Filelists: `inputs//filelists/vocos.train.txt`, `inputs//filelists/vocos.val.txt`

## Implementation Guardrails
- Keep decoder migration modular so A/B baselines remain reproducible.
- Do not silently switch checkpoint/key conventions; make format assumptions explicit.
- Do not reintroduce cache-window/rolling-cache chunk logic in parity tools.
  Use conditioner-once + chunked backbone/head semantics consistently.
- If touching TF mapping logic, update `kokoro/tf/checkpoint_utils.py` and validate with:
  - `uv run python -m py_compile kokoro/tf/*.py`
  - `uv run python -m kokoro.tf.convert --help`
  - `uv run python -m kokoro.tf.validate_quant --help`
  - `uv run python -m kokoro.tf.export --help`
