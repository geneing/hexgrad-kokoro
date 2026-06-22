# Progress

## 2026-06-21

- Inspected `models/vocos/last.pt`: checkpoint is from `third_party/vocos/train_kokoro_decoder.py`, step 56000, `backend=vocos`, and includes `backend_config` plus `generator`, `mpd`, and `mrd` state dicts.
- User clarified that deprecated `kokoro/train_vocos.py` checkpoint support is not needed.
- Next: update `prepare_weights.py` and `vocos_export.py` around the new checkpoint/export path only.
