# Kokoro Vocos/Wavehax Decoder Distillation Plan

## Goal

Train Vocos and Wavehax as drop-in Kokoro decoder replacements for `istftnet.Decoder` using paired distillation data produced by `kokoro/vocoder_data.py` under `data/outputs`.

## Constraints

- Keep Kokoro text/prosody modules frozen; train only the vocoder decoder path.
- Preserve Kokoro feature semantics: `asr`, `f0`, `noise`, and `style` are the decoder inputs.
- Preserve project defaults unless explicitly changed: `sample_rate=24000`, `hop_length=300`, `n_fft=1200`, `n_mels=80`.
- Preserve existing streaming Vocos behavior: run the conditioner on the full feature sequence, then chunk only backbone + ISTFT head.
- Avoid reverting unrelated local work in this dirty worktree.

## Implementation Steps

1. **Create separate third-party training entry points**
   - Add `third_party/vocos/train_kokoro_decoder.py` for Vocos.
   - Add `third_party/wavehax/train_kokoro_decoder.py` for Wavehax.
   - Do not route this work through `kokoro/vocos_decoder.py`.

2. **Share only generic distillation utilities**
   - Add `third_party/kokoro_vocoder_distill.py` for dataset loading, Kokoro feature slicing, losses, checkpointing, and the common training loop.
   - Keep model-specific generator definitions inside each third-party train script.

3. **Handle F0/noise explicitly**
   - Add separate convolutional control branches for F0 and noise before fusion with ASR and style features.
   - Keep the training tensor layout compatible with `vocoder_data.py`: `[asr, f0, noise, style]`.

4. **Checkpoint regularly**
   - Commit `PLAN.md`.
   - Commit separate third-party train scripts and shared utility code.
   - Commit validation fixes.

5. **Validate**
   - Run syntax checks for touched Python modules.
   - Run generator smoke tests with random Kokoro-shaped features for both third-party scripts where dependencies are available.
   - If `data/outputs` contains filelists, run a one-step CPU/GPU smoke command with `--max-steps 1`; otherwise document that dataset validation is blocked by missing local files.

## Initial Training Commands

Vocos:

```bash
uv run python third_party/vocos/train_kokoro_decoder.py \
  --data-root data/outputs \
  --output-dir runs/vocos_kokoro_decoder
```

Wavehax:

```bash
uv run python third_party/wavehax/train_kokoro_decoder.py \
  --data-root data/outputs \
  --output-dir runs/wavehax_kokoro_decoder
```
