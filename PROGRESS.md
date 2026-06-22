# Progress

## 2026-06-21

- Inspected `models/vocos/last.pt`: checkpoint is from `third_party/vocos/train_kokoro_decoder.py`, step 56000, `backend=vocos`, and includes `backend_config` plus `generator`, `mpd`, and `mrd` state dicts.
- User clarified that deprecated `kokoro/train_vocos.py` checkpoint support is not needed.
- Updated `prepare_weights.py` to require the new checkpoint schema, build `KokoroVocosGenerator`, use `backend_config` as architecture metadata, and load third-party paired data helpers.
- Updated `vocos_export.py` to default to `--checkpoint models/vocos/last.pt`, with `--weights-dir` as an explicit prepared-weights override.
- Validation: `./.venv/bin/python -m py_compile prepare_weights.py vocos_export.py` passed.
- Runtime smoke note: direct `.venv` import is blocked by its CUDA-linked `torchaudio` looking for `libcudart.so.13`; `uv run` began provisioning the full ML export stack but was stopped after several minutes of large dependency downloads.
- After venv update: compile still passes; checkpoint/model smoke loaded `models/vocos/last.pt`, built `KokoroVocosGenerator`, and produced finite random-feature audio output `(1, 2400)`.
- After venv update: `vocos_export._load_models` successfully built both fp32/fp16 export models directly from `models/vocos/last.pt`.
- Added a direct `kokoro/styletts2_losses.py` loader in `prepare_weights.py` so export tooling does not depend on package-level `kokoro` imports while `kokoro/vocos_decoder.py` is deleted in the worktree.
- Added optional Android ARM GPU delegate testing to `vocos_export.py`: `--android-gpu-test` pushes a selected exported LiteRT model plus an android_arm64 `benchmark_model` binary to a Pixel/Android device over `adb`, runs a one-shot GPU delegate compile/warmup, runs GPU benchmarking, optionally runs a CPU/XNNPACK baseline, and saves logs under `output-dir/android_gpu`.
