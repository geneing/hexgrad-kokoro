# AGENTS.md

## Project Summary
This repository contains Kokoro streaming TTS code. Kokoro is based on the StyleTTS2 architecture and should be treated as a StyleTTS2-derived model with streaming-oriented adaptations.

Primary objective:
- Replace the current iSTFT-based decoder with a Vocos decoder.
- Train only the Vocos decoder while keeping the rest of the Kokoro model fixed (frozen).

Secondary objective:
- Build an efficient training pipeline that preserves perceptual sound quality while fitting within 24GB VRAM.

## Model Context
- **Kokoro lineage:** Kokoro is derived from StyleTTS2.
- **Training code source:** Reuse or adapt training logic from the StyleTTS2 repository where needed, especially for data flow, losses, and training loop structure.
- **Decoder transition:** The decoder backend must move from iSTFT reconstruction to Vocos waveform generation.

## Technical Direction
1. Keep non-decoder modules frozen during Vocos training:
   - Text encoder / linguistic modules
   - Duration/prosody/style predictors
   - Any acoustic backbone components not part of the decoder
2. Train the Vocos decoder to consume the same intermediate acoustic representation currently sent to the iSTFT path (or a well-defined mapped equivalent).
3. Preserve output quality with objective and subjective checks:
   - Spectral losses (e.g., multi-resolution STFT)
   - Adversarial/perceptual components as appropriate for Vocos
   - Listening-based validation on held-out samples
4. Optimize for 24GB VRAM budget:
   - Mixed precision (bf16/fp16 where stable)
   - Gradient accumulation
   - Checkpointing/activation recomputation where useful
   - Efficient batch sizing and sequence bucketing
   - Avoid unnecessary activations from frozen modules

## Implementation Expectations
- Add a clear switch/config to select decoder type (`istft` vs `vocos`) during migration.
- Ensure frozen parameters are explicitly set and verified (`requires_grad=False`) outside Vocos modules.
- Log and monitor:
  - Decoder-only trainable parameter count
  - VRAM usage per step
  - Throughput (steps/sec)
  - Audio quality metrics and checkpoint samples
- Keep changes modular so baseline iSTFT behavior remains reproducible for A/B comparison.

## Training Constraints
- Target hardware: single GPU with ~24GB VRAM.
- Prioritize stability and repeatability over maximum throughput.
- Maintain compatibility with existing datasets and preprocessing conventions used by Kokoro/StyleTTS2 workflows.

## Definition of Done
- Vocos decoder is integrated and selectable.
- Training runs with frozen non-decoder Kokoro components.
- End-to-end training fits within 24GB VRAM.
- Generated audio quality is comparable to or better than the iSTFT baseline on project validation samples.

## Data Generation (Vocoder Distillation)
- Use `kokoro-vocoder-data` to generate paired iSTFTNet vocoder inputs and waveform targets.
- Default sentence count per voice is `5000`.
- Input text source is LibriTTS normalized text files.

### Prerequisites
- Install project deps:
  - `uv sync`
- Ensure `pip` exists inside the environment (needed by some G2P backends):
  - `uv pip install pip`
- Install Japanese tokenizer dictionary (one-time):
  - `uv run python -m unidic download`

### Download Kokoro Weights + Voices
- `uv run kokoro-vocoder-data --download-only --repo-id hexgrad/Kokoro-82M`

### Smoke Test (all voices, 1 sentence each)
- `uv run kokoro-vocoder-data --num-sentences 1 --libritts-root /export/eingerman/audio/LibriTTS/LibriTTS --output-root /export/eingerman/audio/vocoder --write-repo-config`

### Full Dataset Generation (default 5000)
- `uv run kokoro-vocoder-data --libritts-root /export/eingerman/audio/LibriTTS/LibriTTS --output-root /export/eingerman/audio/vocoder --write-repo-config`

### Generated Artifacts
- Wave targets: `/export/eingerman/audio/vocoder/audio/<voice>/*.wav`
- Paired vocoder inputs: `/export/eingerman/audio/vocoder/pairs/<voice>/*.pt`
  - `.pt` payload includes: `asr`, `f0`, `noise`, `style`, plus metadata
- Per-voice manifests: `/export/eingerman/audio/vocoder/manifests/*.jsonl`
- Vocos filelists: `/export/eingerman/audio/vocoder/filelists/vocos.train.txt` and `/export/eingerman/audio/vocoder/filelists/vocos.val.txt`
- Generated Vocos config: `/export/eingerman/audio/vocoder/vocos-kokoro-24khz.yaml`

## Vocos Training
- Install Vocos training dependencies:
  - `uv pip install -r third_party/vocos/requirements-train.txt`
- Train with generated config:
  - `uv run python third_party/vocos/train.py -c /export/eingerman/audio/vocoder/vocos-kokoro-24khz.yaml`
- Repo copy of config is also available at:
  - `configs/vocos-kokoro-24khz.yaml`

## Config Alignment Requirements
- Keep Vocos settings aligned to Kokoro/iSTFTNet output:
  - `sample_rate: 24000`
  - `hop_length: 300` (derived from Kokoro upsampling stack)
  - `n_mels: 80`
  - `n_fft: 1200`
