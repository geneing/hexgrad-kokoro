# Third-Party Kokoro Decoder Training

This directory contains third-party vocoder backends wired into the shared Kokoro
decoder distillation loop in `third_party/kokoro_vocoder_distill.py`.

The supported Kokoro wrappers are:

- `third_party/vocos/train_kokoro_decoder.py`
- `third_party/wavehax/train_kokoro_decoder.py`

Both wrappers train from paired Kokoro vocoder features and target waveforms.
They expect the data layout produced by `kokoro-vocoder-data`:

```text
data/outputs/
  audio/**/*.wav
  pairs/**/*.pt
  filelists/vocos.train.txt
  filelists/vocos.val.txt
```

## Setup

From the repository root:

```bash
uv sync
uv pip install pip
uv run python -m unidic download
```

Generate or refresh paired vocoder data before training:

```bash
uv run kokoro-vocoder-data \
  --libritts-root /export/eingerman/audio/LibriTTS/LibriTTS \
  --output-root data/outputs \
  --write-repo-config
```

For a quick data-generation smoke test:

```bash
uv run kokoro-vocoder-data \
  --num-sentences 1 \
  --libritts-root /export/eingerman/audio/LibriTTS/LibriTTS \
  --output-root data/outputs \
  --write-repo-config
```

## Train Vocos

Default Vocos decoder distillation:

```bash
uv run python third_party/vocos/train_kokoro_decoder.py \
  --data-root data/outputs \
  --output-dir data/training/vocos
```

Longer single-GPU run with explicit logging and checkpoint cadence:

```bash
uv run python third_party/vocos/train_kokoro_decoder.py \
  --data-root data/outputs \
  --output-dir runs/vocos_kokoro_decoder \
  --batch-size 8 \
  --min-batch-size 1 \
  --frame-cap 520 \
  --num-workers 4 \
  --max-steps 200000 \
  --pretrain-steps 5000 \
  --log-every 50 \
  --val-every 1000 \
  --val-steps 4 \
  --sample-every 2000 \
  --mel-plot-every 2000 \
  --save-every 1000
```

Useful Vocos architecture overrides:

```bash
uv run python third_party/vocos/train_kokoro_decoder.py \
  --data-root data/outputs \
  --output-dir runs/vocos_kokoro_decoder_large \
  --backbone-dim 384 \
  --backbone-intermediate-dim 1152 \
  --backbone-layers 8
```

## Train Wavehax

Default Wavehax decoder distillation:

```bash
uv run python third_party/wavehax/train_kokoro_decoder.py \
  --data-root data/outputs \
  --output-dir runs/wavehax_kokoro_decoder
```

Longer single-GPU run with explicit logging and checkpoint cadence:

```bash
uv run python third_party/wavehax/train_kokoro_decoder.py \
  --data-root data/outputs \
  --output-dir runs/wavehax_kokoro_decoder \
  --batch-size 8 \
  --min-batch-size 1 \
  --frame-cap 520 \
  --num-workers 4 \
  --max-steps 200000 \
  --pretrain-steps 5000 \
  --log-every 50 \
  --val-every 1000 \
  --val-steps 4 \
  --sample-every 2000 \
  --mel-plot-every 2000 \
  --save-every 1000
```

Useful Wavehax architecture overrides:

```bash
uv run python third_party/wavehax/train_kokoro_decoder.py \
  --data-root data/outputs \
  --output-dir runs/wavehax_kokoro_decoder_large \
  --channels 96 \
  --mult-channels 4 \
  --kernel-size 7 \
  --num-blocks 8 \
  --prior-type pcph
```

Enable Wavehax log-magnitude/phase output mode:

```bash
uv run python third_party/wavehax/train_kokoro_decoder.py \
  --data-root data/outputs \
  --output-dir runs/wavehax_kokoro_decoder_logmag_phase \
  --use-logmag-phase
```

## Resume

Resume either backend from a checkpoint:

```bash
uv run python third_party/vocos/train_kokoro_decoder.py \
  --data-root data/outputs \
  --output-dir runs/vocos_kokoro_decoder \
  --resume runs/vocos_kokoro_decoder/checkpoints/last.pt
```

```bash
uv run python third_party/wavehax/train_kokoro_decoder.py \
  --data-root data/outputs \
  --output-dir runs/wavehax_kokoro_decoder \
  --resume runs/wavehax_kokoro_decoder/checkpoints/last.pt
```

## Monitoring

TensorBoard logs are written under each output directory:

```bash
uv run tensorboard --logdir runs
```

Training logs include generator and discriminator losses, raw and weighted loss
components, waveform metrics, gradient norms, effective batch size, timing, CUDA
memory, validation metrics, baseline target audio, generated audio, and mel plots.

Audio samples default to every 2000 batches and can be changed with:

```bash
--sample-every 2000 --mel-plot-every 2000 --sample-count 2
```

## Outputs

Each run writes:

```text
runs/<run_name>/
  config.json
  tensorboard/
  checkpoints/
    step_XXXXXXXX.pt
    last.pt
    final.pt
```

## OOM Behavior

If CUDA runs out of memory, the shared training loop clears gradients/cache,
halves the effective batch size, retries the current batch slice, and rebuilds
the loader at the smaller batch size. The lower bound is controlled by:

```bash
--min-batch-size 1
```

If memory is still insufficient at the minimum batch size, that batch is skipped
and training continues.

