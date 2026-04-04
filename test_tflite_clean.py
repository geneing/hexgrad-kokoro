"""test_tflite_clean.py

Stage-by-stage and end-to-end comparison of KokoroTFLiteTTS
(tflite_inference_clean.py) against ONNX reference models.

Run with:
    TF_ENABLE_ONEDNN_OPTS=0 uv run python test_tflite_clean.py
"""
from __future__ import annotations

import sys
import numpy as np
import torch

sys.path.insert(0, ".")

import onnxruntime as ort

from tflite_inference_clean import (
    KokoroTFLiteTTS,
    _build_conditioner_features,
    _expand_durations,
    _overlap_add,
    CHUNK_FRAMES,
    HOP,
    MAX_INPUT_LEN,
    N_FFT,
    OLA_TAIL,
    T_ACOUSTIC,
    T_F0,
    WIN_LEN,
)

# ── Config ─────────────────────────────────────────────────────────────────────
ONNX_DIR = "onnx2tf_conversion"
TEXT = "Hello world, testing Kokoro TTS on TFLite."
CORR_THRESHOLD = 0.98   # minimum acceptable correlation


# ── ONNX reference pipeline ───────────────────────────────────────────────────

def _load_sessions(onnx_dir: str) -> dict[str, ort.InferenceSession]:
    names = [
        "bert",
        "duration_predictor",
        "acoustic_expand",
        "f0n_predictor",
        "vocoder_conditioner",
        "vocoder_stream_chunk",
    ]
    return {
        n: ort.InferenceSession(
            f"{onnx_dir}/{n}.onnx", providers=["CPUExecutionProvider"]
        )
        for n in names
    }


def run_onnx_pipeline(
    sessions: dict[str, ort.InferenceSession],
    ids_padded: np.ndarray,   # [1, MAX_INPUT_LEN] int32
    style: np.ndarray,        # [1, 256] float32
    mask: np.ndarray,         # [1, MAX_INPUT_LEN] float32
    speed: float = 1.0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run the full ONNX pipeline.

    Returns (intermediates dict, audio array).

    ONNX tensor layouts (native, before onnx2tf transposes for TFLite):
      bert out: d_en       [1, 512, 510] NCT
      dur  out: d_enc      [1, 640, 510] NCT, t_en_static [1, 512, 510] NCT
      acexp in: d_enc_exp  [1, 640, T_ACOUSTIC] NCT
      acexp out: en        [1, T_ACOUSTIC, 512] NTC
      f0n  in : en         [1, T_ACOUSTIC, 512] NTC
      cond in : features   [1, 642, T_F0] NCT
      vocos in: chunk      [1, 192, 16] NCT
      vocos out: x_real/x_imag [1, 16, 601] NTK
    """
    inter: dict[str, np.ndarray] = {}
    speed_arr = np.array([speed], np.float32)

    # Stage 1: BERT
    d_en_nct = sessions["bert"].run(
        None, {"input_ids": ids_padded, "text_mask": mask}
    )[0]   # [1, 512, 510] NCT
    inter["bert_d_en_nct"] = d_en_nct

    # Stage 2: Duration predictor
    pred_dur, d_enc, t_en = sessions["duration_predictor"].run(
        None,
        {
            "d_en": d_en_nct,
            "style": style,
            "text_mask": mask,
            "speed": speed_arr,
            "input_ids": ids_padded.astype(np.int64),
        },
    )  # [510], [1,640,510], [1,512,510]
    inter["pred_dur"] = pred_dur
    inter["d_enc"]    = d_enc
    inter["t_en"]     = t_en

    # Stage 3: Duration expansion (Python)
    idx, T_a = _expand_durations(pred_dur)
    T_a = min(T_a, T_ACOUSTIC)
    idx = idx[:T_a]
    T_f0 = 2 * T_a

    d_enc_exp = np.take(d_enc, idx, axis=2)   # [1, 640, T_a] NCT
    asr       = np.take(t_en,  idx, axis=2)   # [1, 512, T_a] NCT

    # Stage 4: Acoustic expand (ONNX input: NCT)
    d_enc_padded = d_enc_exp
    if T_a < T_ACOUSTIC:
        d_enc_padded = np.pad(d_enc_exp, ((0, 0), (0, 0), (0, T_ACOUSTIC - T_a)))
    en_ntc = sessions["acoustic_expand"].run(
        None, {"d_enc_expanded": d_enc_padded}
    )[0]   # [1, T_ACOUSTIC, 512] NTC
    inter["en_ntc"] = en_ntc

    # Stage 5: F0N predictor (ONNX input: NTC)
    F0_full, N_full = sessions["f0n_predictor"].run(
        None, {"en": en_ntc, "style": style}
    )  # [1, T_F0] each
    f0 = F0_full[:, :T_f0]
    n  = N_full[:,  :T_f0]
    inter["f0"] = f0
    inter["n"]  = n

    # Stage 6: Conditioner (ONNX input: NCT [1, 642, T_F0])
    features_nct = _build_conditioner_features(asr, f0, n, style).astype(np.float32)
    if T_f0 < T_F0:
        features_nct = np.pad(features_nct, ((0, 0), (0, 0), (0, T_F0 - T_f0)))
    cond_full = sessions["vocoder_conditioner"].run(
        None, {"features": features_nct}
    )[0]   # [1, 192, T_F0] NCT
    conditioned = cond_full[:, :, :T_f0]
    inter["conditioned"] = conditioned

    # Stage 7: Streaming vocoder
    vocos = sessions["vocoder_stream_chunk"]
    in_names  = [d.name for d in vocos.get_inputs()]
    out_names = [d.name for d in vocos.get_outputs()]

    state = {n: np.zeros(vocos.get_inputs()[i + 1].shape, np.float32)
             for i, n in enumerate(in_names[1:])}
    ola_tail = np.zeros((1, OLA_TAIL), np.float32)
    hann = torch.hann_window(WIN_LEN).numpy()

    T = conditioned.shape[-1]
    chunks: list[np.ndarray] = []
    for pos in range(0, T, CHUNK_FRAMES):
        valid = min(CHUNK_FRAMES, T - pos)
        chunk = conditioned[..., pos : pos + valid]
        if valid < CHUNK_FRAMES:
            chunk = np.pad(chunk, ((0, 0), (0, 0), (0, CHUNK_FRAMES - valid)))

        feed = {"conditioned_chunk": chunk, **state}
        outs = dict(zip(out_names, vocos.run(None, feed)))

        xr, xi = outs["x_real"], outs["x_imag"]  # [1, F, 601] NTK
        frames = np.fft.irfft(xr + 1j * xi, n=N_FFT, axis=-1)[..., :WIN_LEN].astype(np.float32)
        frames *= hann
        audio_chunk, ola_tail = _overlap_add(frames, ola_tail)
        chunks.append(audio_chunk[0, : valid * HOP])

        state = {n: outs[n + "_new"] for n in in_names[1:]}

    return inter, np.concatenate(chunks)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _diff_stats(a: np.ndarray, b: np.ndarray, label: str) -> bool:
    L = min(len(a.ravel()), len(b.ravel()))
    d = np.abs(a.ravel()[:L] - b.ravel()[:L])
    corr = float(np.corrcoef(a.ravel()[:L], b.ravel()[:L])[0, 1])
    status = "PASS" if d.max() < 1e-2 else "WARN"
    print(
        f"  [{status}] {label:35s}  max_diff={d.max():.2e}  "
        f"mean_diff={d.mean():.2e}  corr={corr:.5f}"
    )
    return status == "PASS"


def _audio_corr(a: np.ndarray, b: np.ndarray) -> float:
    L = min(len(a), len(b))
    return float(np.corrcoef(a[:L], b[:L])[0, 1])


# ── Test runner ───────────────────────────────────────────────────────────────

def run_tests() -> None:
    print(f"\nLoading KokoroTFLiteTTS …")
    tfl = KokoroTFLiteTTS()

    print("Loading ONNX sessions …")
    sessions = _load_sessions(ONNX_DIR)

    print(f'\nText: "{TEXT}"\n')

    # ── Prepare shared inputs ─────────────────────────────────────────────────
    input_ids_raw, style = tfl.text_to_inputs(TEXT)

    T_orig = input_ids_raw.shape[1]
    mask = np.zeros((1, MAX_INPUT_LEN), np.float32)
    mask[0, :T_orig] = 1.0
    pad_len = MAX_INPUT_LEN - T_orig
    ids_padded = (
        np.concatenate([input_ids_raw, np.zeros((1, pad_len), np.int32)], axis=1)
        if pad_len > 0 else input_ids_raw[:, :MAX_INPUT_LEN]
    )

    # ── Run both pipelines ────────────────────────────────────────────────────
    print("Running TFLite pipeline …")
    tfl_audio = np.concatenate(list(tfl.forward(input_ids_raw, style)))

    print("Running ONNX pipeline …")
    onnx_inter, onnx_audio = run_onnx_pipeline(sessions, ids_padded, style, mask)

    # ── Stage-by-stage comparison ─────────────────────────────────────────────
    print("\n── Stage-by-stage comparison ────────────────────────────────────")

    # Access TFLite interpreter outputs for each stage using the same inputs.
    # We call the clean class's internal methods directly.
    tfl_m = tfl._m

    # Stage 1: BERT
    # TFLite outputs d_en NTC [1,510,512]; ONNX outputs NCT [1,512,510] → transpose for comparison
    tfl_d_en_ntc = tfl._run_bert(ids_padded, mask)   # [1,510,512] NTC
    onnx_d_en_nct = onnx_inter["bert_d_en_nct"]      # [1,512,510] NCT
    _diff_stats(tfl_d_en_ntc, onnx_d_en_nct.transpose(0, 2, 1), "BERT d_en")

    # Stage 2: Duration predictor
    # TFLite d_en input is NTC; but we need to pass the same numeric content
    speed_arr = np.array([1.0], np.float32)
    tfl_pred_dur, tfl_d_enc, tfl_t_en = tfl._run_duration(
        tfl_d_en_ntc, style, mask, speed_arr, ids_padded
    )
    _diff_stats(tfl_pred_dur, onnx_inter["pred_dur"], "Duration pred_dur")
    _diff_stats(tfl_d_enc, onnx_inter["d_enc"], "Duration d_enc (NCT)")
    _diff_stats(tfl_t_en, onnx_inter["t_en"], "Duration t_en (NCT)")

    # Stage 3: Expand (Python — deterministic, no TFLite call)
    idx, T_a = _expand_durations(tfl_pred_dur)
    T_a = min(T_a, T_ACOUSTIC)
    idx = idx[:T_a]
    T_f0 = 2 * T_a

    d_enc_exp = np.take(tfl_d_enc, idx, axis=2)    # [1,640,T_a]
    asr       = np.take(tfl_t_en,  idx, axis=2)    # [1,512,T_a]
    print(f"  [INFO] Expansion: T_acoustic={T_a}, T_f0={T_f0}")

    # Stage 4: Acoustic expand
    # TFLite input NTC — transpose from NCT
    d_enc_ntc = d_enc_exp.transpose(0, 2, 1)
    if T_a < T_ACOUSTIC:
        d_enc_ntc = np.pad(d_enc_ntc, ((0, 0), (0, T_ACOUSTIC - T_a), (0, 0)))
    tfl_en_ntc = tfl._run_acoustic_expand(d_enc_ntc)   # [1, T_ACOUSTIC, 512] NTC
    _diff_stats(tfl_en_ntc, onnx_inter["en_ntc"], "AcousticExpand en (NTC)")

    # Stage 5: F0N predictor
    # TFLite input NCT; ONNX uses NTC → compare after transpose
    en_nct = tfl_en_ntc.transpose(0, 2, 1)   # [1,512,T_ACOUSTIC]
    tfl_f0, tfl_n = tfl._run_f0n(en_nct, style)
    _diff_stats(tfl_f0[:, :T_f0], onnx_inter["f0"], "F0N F0_pred")
    _diff_stats(tfl_n[:, :T_f0],  onnx_inter["n"],  "F0N N_pred")

    # Stage 6: Conditioner
    features_nct = _build_conditioner_features(asr, tfl_f0[:, :T_f0], tfl_n[:, :T_f0], style)
    features_ntc = features_nct.transpose(0, 2, 1).astype(np.float32)
    if T_f0 < T_F0:
        features_ntc = np.pad(features_ntc, ((0, 0), (0, T_F0 - T_f0), (0, 0)))
    tfl_cond_full = tfl._run_conditioner(features_ntc)    # [1, 192, T_F0] NCT
    tfl_cond = tfl_cond_full[:, :, :T_f0]
    _diff_stats(tfl_cond, onnx_inter["conditioned"], "Conditioner output (NCT)")

    # ── End-to-end audio comparison ───────────────────────────────────────────
    print("\n── End-to-end audio ─────────────────────────────────────────────")
    L = min(len(tfl_audio), len(onnx_audio))
    diff = np.abs(tfl_audio[:L] - onnx_audio[:L])
    corr = _audio_corr(tfl_audio, onnx_audio)
    print(f"  TFLite samples : {len(tfl_audio)}")
    print(f"  ONNX   samples : {len(onnx_audio)}")
    print(f"  Max |diff|     : {diff.max():.6f}")
    print(f"  Mean |diff|    : {diff.mean():.6f}")
    print(f"  RMS diff       : {float(np.sqrt(np.mean(diff**2))):.6f}")
    print(f"  TFLite RMS     : {float(np.sqrt(np.mean(tfl_audio**2))):.6f}")
    print(f"  ONNX   RMS     : {float(np.sqrt(np.mean(onnx_audio**2))):.6f}")
    print(f"  Correlation    : {corr:.6f}")

    if corr >= CORR_THRESHOLD:
        print(f"\n  PASS  (corr={corr:.4f} >= {CORR_THRESHOLD})")
    else:
        print(f"\n  FAIL  (corr={corr:.4f} < {CORR_THRESHOLD})")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
