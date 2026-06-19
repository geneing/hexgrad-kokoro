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

1. **Unify paired-feature decoder construction**
   - Keep the existing Vocos paired-feature generator.
   - Add a Wavehax paired-feature generator that consumes the same feature tensor layout: `[asr, f0, noise, style]`.
   - Add explicit F0/noise branches before the shared conditioner because Kokoro F0 and noise are higher-rate control streams in the legacy iSTFTNet path.

2. **Extend training entry point**
   - Rename the conceptual backend from Vocos-only to paired vocoder backend while preserving `kokoro.train_vocos` compatibility.
   - Add `--decoder-backend {vocos,wavehax}`.
   - Keep all existing losses and dynamic batching for both backends.
   - Save backend/config metadata in checkpoints so inference can reconstruct the correct decoder.

3. **Wire inference adapters**
   - Add `PTWavehaxDecoder` alongside `PTVocosDecoder`.
   - Add `decoder_type="pt_wavehax"` support in `KModel`.
   - Reuse the same feature assembly logic and checkpoint format.

4. **Checkpoint regularly**
   - Commit `PLAN.md`.
   - Commit the shared backend/adapters.
   - Commit training and inference wiring.
   - Commit validation fixes.

5. **Validate**
   - Run syntax checks for touched Python modules.
   - Run generator smoke tests with random Kokoro-shaped features for both Vocos and Wavehax where dependencies are available.
   - If `data/outputs` contains filelists, run a one-step CPU/GPU smoke command with `--max-steps 1`; otherwise document that dataset validation is blocked by missing local files.

## Initial Training Commands

Vocos:

```bash
uv run python -m kokoro.train_vocos \
  --data-root data/outputs \
  --decoder-backend vocos \
  --vocos-impl streaming \
  --streaming-vocos-repo third_party/vocos_streaming \
  --precision auto \
  --tf32
```

Wavehax:

```bash
uv run python -m kokoro.train_vocos \
  --data-root data/outputs \
  --decoder-backend wavehax \
  --precision auto \
  --tf32
```
