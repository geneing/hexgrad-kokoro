# Progress

## 2026-06-21

- Inspected `models/vocos/last.pt`: checkpoint is from `third_party/vocos/train_kokoro_decoder.py`, step 56000, `backend=vocos`, and includes `backend_config` plus `generator`, `mpd`, and `mrd` state dicts.
- User clarified that deprecated `kokoro/train_vocos.py` checkpoint support is not needed.
- Updated `prepare_weights.py` to require the new checkpoint schema, build `KokoroVocosGenerator`, use `backend_config` as architecture metadata, and load third-party paired data helpers.
- Updated `vocos_export.py` to default to `--checkpoint models/vocos/last.pt`, with `--weights-dir` as an explicit prepared-weights override.
- Validation: `uv run python -m py_compile prepare_weights.py vocos_export.py` passed.
- Runtime smoke note: direct `.venv` import was blocked before the venv refresh by its CUDA-linked `torchaudio` looking for `libcudart.so.13`; use `uv run python ...` for project commands.
- After venv update: compile still passed; checkpoint/model smoke loaded `models/vocos/last.pt`, built `KokoroVocosGenerator`, and produced finite random-feature audio output `(1, 2400)`.
- After venv update: `vocos_export._load_models` successfully built both fp32/fp16 export models directly from `models/vocos/last.pt`.
- Added a direct `kokoro/styletts2_losses.py` loader in `prepare_weights.py` so export tooling does not depend on package-level `kokoro` imports while `kokoro/vocos_decoder.py` is deleted in the worktree.
- Added optional Android ARM GPU delegate testing to `vocos_export.py`: `--android-gpu-test` pushes a selected exported LiteRT model plus an android_arm64 `benchmark_model` binary or APK to a Pixel/Android device over `adb`, runs GPU delegate compile/warmup and benchmark passes, optionally runs a CPU/XNNPACK baseline, and saves logs under `output-dir/android_gpu`.
- Set the default Android GPU test ADB executable to `/mnt/c/Users/genei/Downloads/platform-tools/adb.exe`; it can still be overridden with `--adb`.
- Ran conversion to `runs/litert_vocos_pixel10_realinput` using real `data/af_alloy_*.pt` feature files as model input/calibration samples. Exported fp32/fp16/int8 LiteRT files and validation WAVs for the matched examples.
- Pixel 10 Android benchmark with real input succeeded on CPU/XNNPACK: fp16 model, input `af_alloy_00001_00_42f_float32.bin`, `Inference (avg): 24651.1 us`.
- Pixel 10 Android GPU delegate test with the same real input reached delegate creation but failed before inference: unsupported broadcast/slice ops left 213 ops on CPU, then OpenCL GPU delegate failed with `TfLiteGpuDelegate Init: Unrecognized Write selector` and `Failed to apply GPU delegate`.

## 2026-06-22

- Confirmed this repo should be driven with `uv run python ...`; avoid direct `python` or `.venv/bin/python` commands in saved instructions unless debugging uv itself.
- Added a focused `vocos_export.py --pixel10-fp16-aot` path. It skips the legacy fp32/int8 flow, loads real `.pt` feature inputs, attempts a true fp16 LiteRT export probe, optionally attempts dynamic-frame `litert_torch` export, exports fixed-frame fallback models per input length, runs Google Tensor AOT compilation, and writes diagnostics under `output-dir/diagnostics`.
- Broadened the default feature glob to `data/af_alloy_0*_00.pt` so the currently present examples are all discovered: `af_alloy_00001_00.pt`, `af_alloy_00002_00.pt`, and `af_alloy_00403_00.pt`.
- True fp16 LiteRT export probe failed in LiteRT-Torch converter passes at `LayerNorm`: `expects operand and result to have compatible element type. Got: tensor<330xf32> and tensor<1x330x384xf16>`. Diagnostic: `runs/litert_vocos_pixel10_aot/diagnostics/true_fp16_export_failure.txt`.
- Dynamic-frame export attempt with `litert_torch.convert(dynamic_shapes=({2: Dim("frames", min=16, max=1200)},))` failed before producing TFLite. The failing graph node was `aten.slice.Tensor(features, dim=1, start=0, end=512)` in `KokoroFeatureConditioner.forward`. Diagnostic: `runs/litert_vocos_pixel10_aot/diagnostics/dynamic_export_failure.txt`.
- Fixed-frame fallback LiteRT exports succeeded for all three examples: 330 frames, 1094 frames, and 1754 frames in `runs/litert_vocos_pixel10_aot`. Tensor diagnostics show these are FLOAT32 TFLite graphs (`FLOAT32: 429`, `INT32: 35`, `INT64: 1`) exported from fp16-trained checkpoint weights, not true FLOAT16 TFLite graphs.
- Google Tensor G5 AOT compilation loaded the local SDK library from `tools/google_tensor_ml_sdk`, selected the full TFLite subgraph (`328 / 328 ops`) for each fixed-frame model, then failed internally in the backend compiler for all three frame lengths. The compiler did not emit a per-op unsupported list for that AOT failure; raw logs were copied under `runs/litert_vocos_pixel10_aot/diagnostics/*_tmp*.error`.
- TFLite op inventory from the previous fixed-frame exports: `RESHAPE 76`, `ADD 51`, `TRANSPOSE 40`, `MUL 35`, `MEAN 20`, `FULLY_CONNECTED 17`, `GELU 14`, `SLICE 12`, `SUB 11`, `CONV_2D 10`, `DEPTHWISE_CONV_2D 10`, `SQUARED_DIFFERENCE 10`, `RSQRT 10`, `BATCH_MATMUL 2`, `TRANSPOSE_CONV 2`, and one each of `CONCATENATION`, `EXP`, `MINIMUM`, `COS`, `SIN`, `BROADCAST_TO`, `MAXIMUM`, `DIV`.
- Latest verification command:
  `uv run python vocos_export.py --checkpoint models/vocos/last.pt --output-dir runs/litert_vocos_pixel10_aot --sample-count 3 --pixel10-fp16-aot --dynamic-frames --google-tensor-compiler-lib tools/google_tensor_ml_sdk`
