# Progress

## Wavehax export - 2026-06-24 19:47:14
- Checkpoint: `models/wavehax/last.pt`
- Output: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax`
- Exported: `wavehax_fp32_litert.tflite`, `wavehax_fp16_litert.tflite`
- Samples: af_alloy_00001_00, af_alloy_00002_00, af_alloy_00403_00
- Fixed export shape: `[1, 642, 330]`
- WAVs: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax/sample_audio`
- Diagnostics: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax/diagnostics`
- Elapsed: 24.3s; max RSS: 1631.9 MB

## Wavehax export - 2026-06-24 20:41:04
- Checkpoint: `models/wavehax/last.pt`
- Output: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax`
- Exported: `wavehax_fp32_litert.tflite`, `wavehax_fp16_litert.tflite`
- Optimized prior: True
- Zero-pad reflect convs: True
- Dynamic fp16: not exported (ValueError: length should not be negative

While executing %slice_1 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%features, 1, 0, 512), kwargs = {})
Original traceback:
File "/home/eingerman/Projects/TTS/kokoro/wavehax_export.py", line 154, in forward
    cond = self.conditioner(features)
  File "/home/eingerman/Projects/TTS/kokoro/wavehax_export.py", line 118, in forward
    self.asr_proj(features[:, :asr_end]),
Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs))
- Multisig fp16: `wavehax_fp16_multisig_static_litert.tflite`
- Sample-length fp16 models: 3
- Samples: af_alloy_00001_00, af_alloy_00002_00, af_alloy_00403_00
- Fixed export shape: `[1, 642, 330]`
- WAVs: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax/sample_audio`
- Diagnostics: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax/diagnostics`
- Elapsed: 74.9s; max RSS: 1710.6 MB

## Wavehax export - 2026-06-24 20:42:53
- Checkpoint: `models/wavehax/last.pt`
- Output: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax`
- Exported: `wavehax_fp32_litert.tflite`, `wavehax_fp16_litert.tflite`
- Optimized prior: True
- Zero-pad reflect convs: True
- Dynamic fp16: not exported (TypeError: Shapes must be 1D sequences of concrete values of integer type, got (JitTracer(~int64[]), JitTracer(~int64[]), JitTracer(~int64[]), JitTracer(int32[])).
If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.
The error occurred while tracing the function lower_wrapper at /home/eingerman/Projects/TTS/kokoro/.venv/lib/python3.12/site-packages/litert_torch/backend/jax_bridge/_wrap.py:143 for jit. This concrete value was not available in Python because it depends on the value of the argument args[1][0].
The error occurred while tracing the function lower_wrapper at /home/eingerman/Projects/TTS/kokoro/.venv/lib/python3.12/site-packages/litert_torch/backend/jax_bridge/_wrap.py:143 for jit. This concrete value was not available in Python because it depends on the value of the argument args[1][1].
The error occurred while tracing the function lower_wrapper at /home/eingerman/Projects/TTS/kokoro/.venv/lib/python3.12/site-packages/litert_torch/backend/jax_bridge/_wrap.py:143 for jit. This concrete value was not available in Python because it depends on the value of the argument args[1][2].
The error occurred while tracing the function lower_wrapper at /home/eingerman/Projects/TTS/kokoro/.venv/lib/python3.12/site-packages/litert_torch/backend/jax_bridge/_wrap.py:143 for jit. This concrete value was not available in Python because it depends on the value of the argument args[1][3].

While executing %view : [num_users=2] = call_function[target=torch.ops.aten.view.default](args = (%convolution_11, [1, 5, 61, %sym_size_int_108]), kwargs = {})
Original traceback:
File "/home/eingerman/Projects/TTS/kokoro/wavehax_export.py", line 156, in forward
    audio, _prior = self.generator(cond, f0)
  File "/home/eingerman/Projects/TTS/kokoro/wavehax_export.py", line 329, in forward
    cond = cond.view(b, self.num_splits, self.n_bins, frames)
Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs))
- Multisig fp16: `wavehax_fp16_multisig_static_litert.tflite`
- Sample-length fp16 models: 3
- Samples: af_alloy_00001_00, af_alloy_00002_00, af_alloy_00403_00
- Fixed export shape: `[1, 642, 330]`
- WAVs: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax/sample_audio`
- Diagnostics: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax/diagnostics`
- Elapsed: 65.8s; max RSS: 1546.1 MB

## Wavehax Android benchmark - 2026-06-24 20:50
- Device: Pixel via ADB serial `57220DLCR002R6`
- Command: `uv run python android/run_plan_bench.py --family wavehax --adb /mnt/c/Users/genei/Downloads/platform-tools/adb.exe --log-wait-seconds 80 --case-timeout-seconds 80 --log-wait-seconds-1754 100 --case-timeout-seconds-1754 100`
- Results: `runs/wavehax/android_bench/results.csv`
- Summary: `runs/wavehax/android_bench/summary.md`
- Fixed-shape FP16 GPU timings:
  - `af_alloy_00001_00` / 330 frames: mean `757.524 ms` over 3 runs
  - `af_alloy_00002_00` / 1094 frames: mean `2488.457 ms` over 3 runs
  - `af_alloy_00403_00` / 1754 frames: mean `3972.093 ms` over 3 runs
- Dynamic FP16: not exported; LiteRT dynamic lowering failed on Wavehax's dynamic `[1, 5, 61, T]` view.
- Multisig FP16: exported, but benchmark APK skipped it because `--signature_to_run` is unsupported by this fallback APK.

## Wavehax export - 2026-06-24 20:50:34
- Checkpoint: `models/wavehax/last.pt`
- Output: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax`
- Exported: `wavehax_fp32_litert.tflite`, `wavehax_fp16_litert.tflite`
- Optimized prior: True
- Zero-pad reflect convs: True
- Dynamic fp16: not exported (TypeError: Shapes must be 1D sequences of concrete values of integer type, got (JitTracer(~int64[]), JitTracer(~int64[]), JitTracer(~int64[]), JitTracer(int32[])).
If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.
The error occurred while tracing the function lower_wrapper at /home/eingerman/Projects/TTS/kokoro/.venv/lib/python3.12/site-packages/litert_torch/backend/jax_bridge/_wrap.py:143 for jit. This concrete value was not available in Python because it depends on the value of the argument args[1][0].
The error occurred while tracing the function lower_wrapper at /home/eingerman/Projects/TTS/kokoro/.venv/lib/python3.12/site-packages/litert_torch/backend/jax_bridge/_wrap.py:143 for jit. This concrete value was not available in Python because it depends on the value of the argument args[1][1].
The error occurred while tracing the function lower_wrapper at /home/eingerman/Projects/TTS/kokoro/.venv/lib/python3.12/site-packages/litert_torch/backend/jax_bridge/_wrap.py:143 for jit. This concrete value was not available in Python because it depends on the value of the argument args[1][2].
The error occurred while tracing the function lower_wrapper at /home/eingerman/Projects/TTS/kokoro/.venv/lib/python3.12/site-packages/litert_torch/backend/jax_bridge/_wrap.py:143 for jit. This concrete value was not available in Python because it depends on the value of the argument args[1][3].

While executing %view : [num_users=2] = call_function[target=torch.ops.aten.view.default](args = (%convolution_11, [1, 5, 61, %sym_size_int_107]), kwargs = {})
Original traceback:
File "/home/eingerman/Projects/TTS/kokoro/wavehax_export.py", line 156, in forward
    audio, _prior = self.generator(cond, f0)
  File "/home/eingerman/Projects/TTS/kokoro/wavehax_export.py", line 329, in forward
    cond = cond.view(b, self.num_splits, self.n_bins, frames)
Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs))
- Multisig fp16: `wavehax_fp16_multisig_static_litert.tflite`
- Sample-length fp16 models: 3
- Samples: af_alloy_00001_00, af_alloy_00002_00, af_alloy_00403_00
- Fixed export shape: `[1, 642, 330]`
- WAVs: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax/sample_audio`
- Diagnostics: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax/diagnostics`
- Elapsed: 63.9s; max RSS: 1546.4 MB
## Wavehax export - 2026-06-24 21:40:34
- Checkpoint: `models/wavehax/last.pt`
- Output: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax`
- Exported: `wavehax_fp32_litert.tflite`, `wavehax_fp16_litert.tflite`
- Optimized prior: True
- Zero-pad reflect convs: True
- Dynamic fp16: not exported (TypeError: Shapes must be 1D sequences of concrete values of integer type, got (JitTracer(~int64[]), JitTracer(~int64[]), JitTracer(~int64[]), JitTracer(int32[])).
If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.
The error occurred while tracing the function lower_wrapper at /home/eingerman/Projects/TTS/kokoro/.venv/lib/python3.12/site-packages/litert_torch/backend/jax_bridge/_wrap.py:143 for jit. This concrete value was not available in Python because it depends on the value of the argument args[1][0].
The error occurred while tracing the function lower_wrapper at /home/eingerman/Projects/TTS/kokoro/.venv/lib/python3.12/site-packages/litert_torch/backend/jax_bridge/_wrap.py:143 for jit. This concrete value was not available in Python because it depends on the value of the argument args[1][1].
The error occurred while tracing the function lower_wrapper at /home/eingerman/Projects/TTS/kokoro/.venv/lib/python3.12/site-packages/litert_torch/backend/jax_bridge/_wrap.py:143 for jit. This concrete value was not available in Python because it depends on the value of the argument args[1][2].
The error occurred while tracing the function lower_wrapper at /home/eingerman/Projects/TTS/kokoro/.venv/lib/python3.12/site-packages/litert_torch/backend/jax_bridge/_wrap.py:143 for jit. This concrete value was not available in Python because it depends on the value of the argument args[1][3].

While executing %view : [num_users=2] = call_function[target=torch.ops.aten.view.default](args = (%convolution_11, [1, 5, 61, %sym_size_int_107]), kwargs = {})
Original traceback:
File "/home/eingerman/Projects/TTS/kokoro/wavehax_export.py", line 156, in forward
    audio, _prior = self.generator(cond, f0)
  File "/home/eingerman/Projects/TTS/kokoro/wavehax_export.py", line 329, in forward
    cond = cond.view(b, self.num_splits, self.n_bins, frames)
Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs))
- Multisig fp16: `wavehax_fp16_multisig_static_litert.tflite`
- Sample-length fp16 models: 3
- Samples: af_alloy_00001_00, af_alloy_00002_00, af_alloy_00403_00
- Fixed export shape: `[1, 642, 330]`
- WAVs: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax/sample_audio`
- Diagnostics: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax/diagnostics`
- Elapsed: 79.6s; max RSS: 1714.5 MB

## Wavehax export - 2026-06-26 19:35:00
- Checkpoint: `models/wavehax/last.pt`
- Output: `/tmp/wavehax_noopt`
- Exported: `wavehax_fp32_litert.tflite`, `wavehax_fp16_litert.tflite`
- Optimized prior: False
- Zero-pad reflect convs: True
- Dynamic fp16: not exported (not requested)
- Multisig fp16: not exported (not requested)
- Sample-length fp16 models: 0
- Samples: af_alloy_00001_00
- Fixed export shape: `[1, 642, 330]`
- WAVs: `/tmp/wavehax_noopt/sample_audio`
- Diagnostics: `/tmp/wavehax_noopt/diagnostics`
- Elapsed: 27.3s; max RSS: 1539.6 MB

## Wavehax export - 2026-06-26 19:36:42
- Checkpoint: `models/wavehax/last.pt`
- Output: `/tmp/wavehax_noopt_multisig`
- Exported: `wavehax_fp32_litert.tflite`, `wavehax_fp16_litert.tflite`
- Optimized prior: False
- Zero-pad reflect convs: True
- Dynamic fp16: not exported (not requested)
- Multisig fp16: `wavehax_fp16_multisig_static_litert.tflite`
- Sample-length fp16 models: 0
- Samples: af_alloy_00001_00, af_alloy_00002_00, af_alloy_00403_00
- Fixed export shape: `[1, 642, 330]`
- WAVs: `/tmp/wavehax_noopt_multisig/sample_audio`
- Diagnostics: `/tmp/wavehax_noopt_multisig/diagnostics`
- Elapsed: 53.9s; max RSS: 1546.4 MB

## Wavehax export - 2026-06-26 19:38:58
- Checkpoint: `models/wavehax/last.pt`
- Output: `/tmp/wavehax_noopt_all`
- Exported: `wavehax_fp32_litert.tflite`, `wavehax_fp16_litert.tflite`
- Optimized prior: False
- Zero-pad reflect convs: True
- Dynamic fp16: not exported (not requested)
- Multisig fp16: `wavehax_fp16_multisig_static_litert.tflite`
- Sample-length fp16 models: 3
- Samples: af_alloy_00001_00, af_alloy_00002_00, af_alloy_00403_00
- Fixed export shape: `[1, 642, 330]`
- WAVs: `/tmp/wavehax_noopt_all/sample_audio`
- Diagnostics: `/tmp/wavehax_noopt_all/diagnostics`
- Elapsed: 84.8s; max RSS: 1546.8 MB

## Wavehax export - 2026-06-26 21:42:16
- Checkpoint: `models/wavehax/last.pt`
- Output: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax`
- Exported: `wavehax_fp32_litert.tflite`, `wavehax_fp16_litert.tflite`
- Optimized prior: False
- Zero-pad reflect convs: True
- Dynamic fp16: not exported (TypeError: '<=' not supported between instances of 'int' and 'Node')
- Multisig fp16: `wavehax_fp16_multisig_static_litert.tflite`
- Sample-length fp16 models: 3
- Samples: af_alloy_00001_00, af_alloy_00002_00, af_alloy_00403_00
- Fixed export shape: `[1, 642, 330]`
- WAVs: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax/sample_audio`
- Diagnostics: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax/diagnostics`
- Elapsed: 101.9s; max RSS: 1776.5 MB

## Wavehax export - 2026-06-26 23:20:30
- Checkpoint: `models/wavehax/last.pt`
- Output: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax`
- Exported: `wavehax_fp32_litert.tflite`, `wavehax_fp16_litert.tflite`
- Optimized prior: False
- Zero-pad reflect convs: True
- Dynamic fp16: not exported (TypeError: '<=' not supported between instances of 'int' and 'Node')
- Multisig fp16: `wavehax_fp16_multisig_static_litert.tflite`
- Sample-length fp16 models: 3
- Samples: af_alloy_00001_00, af_alloy_00002_00, af_alloy_00403_00
- Fixed export shape: `[1, 642, 330]`
- WAVs: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax/sample_audio`
- Diagnostics: `/home/eingerman/Projects/TTS/kokoro/runs/wavehax/diagnostics`
- Elapsed: 103.9s; max RSS: 1777.6 MB

## Wavehax export - 2026-06-29 23:14:36
- Checkpoint: `data/training/wavehax_trainable_stft/checkpoints/last.pt`
- Output: `/export/eingerman/audio/training/wavehax_trainable_stft`
- Exported: `wavehax_fp32_litert.tflite`, `wavehax_fp16_litert.tflite`
- Optimized prior: False
- Zero-pad reflect convs: True
- Dynamic fp16: not exported (not requested)
- Multisig fp16: not exported (not requested)
- Sample-length fp16 models: 0
- Samples: af_alloy_00001_00, af_alloy_00304_00, af_alloy_00403_00
- Fixed chunk export shapes: features/current/next/state `[1, 642, 24]`, phase `[1, 1, 1]`
- WAVs: `/export/eingerman/audio/training/wavehax_trainable_stft/sample_audio`
- Diagnostics: `/export/eingerman/audio/training/wavehax_trainable_stft/diagnostics`
- Elapsed: 157.9s; max RSS: 2857.9 MB

