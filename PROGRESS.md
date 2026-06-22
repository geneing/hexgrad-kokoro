# Progress

## 2026-06-21

- Inspected `models/vocos/last.pt`: checkpoint is from `third_party/vocos/train_kokoro_decoder.py`, step 56000, `backend=vocos`, and includes `backend_config` plus `generator`, `mpd`, and `mrd` state dicts.
- User clarified that deprecated `kokoro/train_vocos.py` checkpoint support is not needed.
- Updated `prepare_weights.py` to require the new checkpoint schema, build `KokoroVocosGenerator`, use `backend_config` as architecture metadata, and load third-party paired data helpers.
- Updated `vocos_export.py` to default to `--checkpoint models/vocos/last.pt`, with `--weights-dir` as an explicit prepared-weights override.
- Validation: `./.venv/bin/python -m py_compile prepare_weights.py vocos_export.py` passed.
- Runtime smoke note: direct `.venv` import is blocked by its CUDA-linked `torchaudio` looking for `libcudart.so.13`; `uv run` began provisioning the full ML export stack but was stopped after several minutes of large dependency downloads.
