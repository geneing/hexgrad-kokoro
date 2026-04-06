#!/usr/bin/env python3
"""int8_quality_eval.py — Layer-by-layer INT8 degradation analysis + UTMOS/PESQ metrics.

Usage
-----
    # Layer-by-layer degradation (ONNX)
    TF_ENABLE_ONEDNN_OPTS=0 uv run python int8_quality_eval.py --mode layer_by_layer

    # Evaluate a specific wav file
    TF_ENABLE_ONEDNN_OPTS=0 uv run python int8_quality_eval.py --mode score --wav path/to/file.wav

    # Full report comparing all precision variants
    TF_ENABLE_ONEDNN_OPTS=0 uv run python int8_quality_eval.py --mode report
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn.functional as F
import onnxruntime as ort

from kokoro import KPipeline, KModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VOICE_PATH = "checkpoints/voices/af_bella.pt"
SAMPLE_RATE = 24_000
MAX_INPUT_LENGTH = 510
T_ACOUSTIC = 543
T_F0 = 1086
CHUNK_FRAMES = 16
HOP = 300
TAIL = 900
KERN = 6
N_BLOCKS = 8

EVAL_SENTENCES = [
    "Hello world. This is Kokoro TTS running on ONNX Runtime with streaming.",
    "The quick brown fox jumps over the lazy dog.",
    "Speech synthesis has improved dramatically with neural network models.",
    "Duration prediction determines how long each phoneme is held in the output.",
    "To be or not to be, that is the question.",
]

MODEL_NAMES = [
    "bert",
    "duration_predictor",
    "acoustic_expand",
    "f0n_predictor",
    "vocoder_conditioner",
    "vocoder_stream_chunk",
]

# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_snr(ref: np.ndarray, deg: np.ndarray) -> float:
    """Signal-to-noise ratio of deg vs ref (dB). Higher is better."""
    n = min(len(ref), len(deg))
    ref, deg = ref[:n].astype(np.float64), deg[:n].astype(np.float64)
    noise = ref - deg
    signal_power = np.mean(ref ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-20:
        return 99.0
    if signal_power < 1e-20:
        return -99.0
    return 10.0 * np.log10(signal_power / noise_power)


def compute_pearson(ref: np.ndarray, deg: np.ndarray) -> float:
    """Pearson correlation between ref and deg waveforms."""
    n = min(len(ref), len(deg))
    ref, deg = ref[:n].astype(np.float64), deg[:n].astype(np.float64)
    if np.std(ref) < 1e-10 or np.std(deg) < 1e-10:
        return 0.0
    return float(np.corrcoef(ref, deg)[0, 1])


def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[float]:
    """PESQ wideband score (1.0–4.5). Requires pesq package. None on failure."""
    try:
        from pesq import pesq as _pesq, NoUtterancesError
        n = min(len(ref), len(deg))
        # PESQ expects 16 kHz; resample if needed
        if sr != 16000:
            ref_t = torch.from_numpy(ref[:n].astype(np.float32)).unsqueeze(0)
            deg_t = torch.from_numpy(deg[:n].astype(np.float32)).unsqueeze(0)
            ref_16 = F.interpolate(ref_t.unsqueeze(0), scale_factor=16000/sr, mode="linear",
                                   align_corners=False).squeeze().numpy()
            deg_16 = F.interpolate(deg_t.unsqueeze(0), scale_factor=16000/sr, mode="linear",
                                   align_corners=False).squeeze().numpy()
        else:
            ref_16, deg_16 = ref[:n].astype(np.float32), deg[:n].astype(np.float32)
        return float(_pesq(16000, ref_16, deg_16, "wb"))
    except Exception:
        return None


def compute_utmos(wav_path: str) -> Optional[float]:
    """UTMOS neural MOS score (1.0–5.0). Higher is better."""
    try:
        import utmos
        predictor = utmos.load_model()
        score = predictor.predict(wav_path)
        if isinstance(score, (list, np.ndarray)):
            return float(np.mean(score))
        return float(score)
    except Exception as e:
        warnings.warn(f"UTMOS failed: {e}")
        return None


def score_wav(wav_path: str, ref_wav: Optional[np.ndarray] = None) -> Dict[str, Optional[float]]:
    """Compute all available quality metrics for a wav file."""
    sr, audio = wavfile.read(wav_path)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

    results: Dict[str, Optional[float]] = {}
    results["utmos"] = compute_utmos(wav_path)
    results["duration_s"] = len(audio) / sr

    if ref_wav is not None:
        results["snr_db"] = compute_snr(ref_wav, audio)
        results["pearson"] = compute_pearson(ref_wav, audio)
        results["pesq"] = compute_pesq(ref_wav, audio, sr=sr)
    return results


# ---------------------------------------------------------------------------
# ONNX runner (wraps example_onnx_inference.generate)
# ---------------------------------------------------------------------------

def _make_session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts,
                                providers=["CPUExecutionProvider"])


_ORT_FLOAT_TYPES = {"tensor(float16)": np.float16, "tensor(float)": np.float32}


def _run(session: ort.InferenceSession, feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
    casted = {}
    for inp in session.get_inputs():
        if inp.name in feed:
            arr = feed[inp.name]
            target = _ORT_FLOAT_TYPES.get(inp.type)
            if target is not None and arr.dtype != target:
                arr = arr.astype(target)
            casted[inp.name] = arr
    return session.run(None, casted)


def _text_to_inputs(kpipeline: KPipeline, text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if kpipeline.lang_code in "ab":
        _, tokens = kpipeline.g2p(text)
        ps: List[str] = []
        for _, p, _ in kpipeline.en_tokenize(tokens):
            ps.extend(p)
    else:
        ps, _ = kpipeline.g2p(text)
    ps = ps[: MAX_INPUT_LENGTH - 2]
    vocab = kpipeline.model.vocab
    ids = [i for i in (vocab.get(p) for p in ps) if i is not None]
    raw_ids = torch.IntTensor([[0, *ids, 0]])
    pack = kpipeline.load_voice(VOICE_PATH).cpu()
    style = pack[len(ps) - 1].detach().numpy().astype(np.float32)  # [1, 256]
    seqlen = raw_ids.shape[1]
    text_mask = np.zeros((1, MAX_INPUT_LENGTH), dtype=np.float32)
    text_mask[0, :seqlen] = 1.0
    pad = MAX_INPUT_LENGTH - seqlen
    if pad > 0:
        raw_ids = F.pad(raw_ids, (0, pad))
    return raw_ids.numpy().astype(np.int32), text_mask, style


def _expand_durations(pred_dur: np.ndarray) -> Tuple[np.ndarray, int]:
    dur = torch.from_numpy(pred_dur.flatten().astype(np.float32))
    boundaries = torch.cumsum(dur, dim=0)
    T_acoustic = min(int(boundaries[-1].item()), T_ACOUSTIC)
    if T_acoustic == 0:
        return np.zeros(0, dtype=np.int64), 0
    values = torch.arange(T_acoustic, dtype=torch.int32)
    idx = torch.sum(boundaries.unsqueeze(1) <= values.unsqueeze(0), dim=0)
    return idx.numpy().astype(np.int64), T_acoustic


def _build_features(asr, f0_pred, n_pred, style):
    T_f0 = f0_pred.shape[-1]
    asr_t = torch.from_numpy(asr.astype(np.float32))
    if asr_t.shape[-1] != T_f0:
        asr_t = F.interpolate(asr_t, size=T_f0, mode="linear", align_corners=False)
    f0 = torch.from_numpy(f0_pred.astype(np.float32)).unsqueeze(1)
    n  = torch.from_numpy(n_pred.astype(np.float32)).unsqueeze(1)
    s  = torch.from_numpy(style.astype(np.float32))[:, :128].unsqueeze(-1).expand(-1, -1, T_f0)
    return torch.cat([asr_t, f0, n, s], dim=1).numpy()  # [1, 642, T_f0]


def generate_onnx(text: str, sessions: Dict[str, ort.InferenceSession],
                  kpipeline: KPipeline) -> np.ndarray:
    """Run full ONNX pipeline, return float32 mono audio."""
    input_ids, text_mask, style = _text_to_inputs(kpipeline, text)

    # S1: BERT
    d_en = _run(sessions["bert"], {"input_ids": input_ids, "text_mask": text_mask})[0]

    # S2: Duration
    dur_outs = _run(sessions["duration_predictor"], {
        "d_en": d_en, "style": style, "text_mask": text_mask,
        "speed": np.array([1], dtype=np.int32), "input_ids": input_ids,
    })
    by_shape = {tuple(a.shape): a for a in dur_outs}
    _pd = by_shape.get((510,))
    if _pd is None:
        _pd = next(a for a in dur_outs if a.size == 510)
    pred_dur = _pd.flatten()
    d_enc = by_shape[(1, 640, 510)]
    t_en_static = by_shape[(1, 512, 510)]

    # S3: Expand
    expanded_idx, T_acoustic = _expand_durations(pred_dur)
    if T_acoustic == 0:
        return np.zeros(0, dtype=np.float32)
    T_f0_actual = 2 * T_acoustic
    d_enc_exp = np.take(d_enc, expanded_idx, axis=2)
    asr = np.take(t_en_static, expanded_idx, axis=2)

    # S4: Acoustic expand (expects NCT [1, 640, T_ACOUSTIC])
    if T_acoustic < T_ACOUSTIC:
        d_enc_exp = np.pad(d_enc_exp, ((0, 0), (0, 0), (0, T_ACOUSTIC - T_acoustic)))
    en_ntc = _run(sessions["acoustic_expand"], {"d_enc_expanded": d_enc_exp.astype(np.float32)})[0]

    # S5: F0N (expects NTC [1, T_ACOUSTIC, 512])
    f0n_outs = _run(sessions["f0n_predictor"], {"en": en_ntc.astype(np.float32), "style": style})
    f0_pred = f0n_outs[0][:, :T_f0_actual]
    n_pred  = f0n_outs[1][:, :T_f0_actual]

    # S6: Conditioner
    features_nct = _build_features(asr, f0_pred, n_pred, style)
    if T_f0_actual < T_F0:
        features_nct = np.pad(features_nct, ((0, 0), (0, 0), (0, T_F0 - T_f0_actual)))
    conditioned_full = _run(sessions["vocoder_conditioner"], {
        "features": features_nct.astype(np.float32)})[0]
    conditioned = conditioned_full[:, :, :T_f0_actual]

    # S7: Vocos stream
    embed_prev  = np.zeros((1, 192, KERN), dtype=np.float32)
    block_prevs = [np.zeros((1, 384, KERN), dtype=np.float32) for _ in range(N_BLOCKS)]
    istft_prev  = np.zeros((1, TAIL), dtype=np.float32)
    audio_chunks: List[np.ndarray] = []
    pos = 0
    while pos < T_f0_actual:
        end   = min(T_f0_actual, pos + CHUNK_FRAMES)
        valid = end - pos
        chunk = conditioned[:, :, pos:end]
        if valid < CHUNK_FRAMES:
            chunk = np.pad(chunk, ((0, 0), (0, 0), (0, CHUNK_FRAMES - valid)))
        feed: Dict[str, np.ndarray] = {
            "conditioned_chunk": chunk.astype(np.float32),
            "embed_prev": embed_prev,
        }
        for b in range(N_BLOCKS):
            feed[f"block_{b}_prev"] = block_prevs[b]
        feed["istft_prev"] = istft_prev
        outs = _run(sessions["vocoder_stream_chunk"], feed)
        embed_prev = outs[1]
        for b in range(N_BLOCKS):
            block_prevs[b] = outs[2 + b]
        istft_prev = outs[2 + N_BLOCKS]
        audio_chunks.append(outs[0][0, : valid * HOP])
        pos = end
    return np.concatenate(audio_chunks).astype(np.float32)


def load_sessions(model_dir: Path, precision_or_map) -> Dict[str, ort.InferenceSession]:
    """Load sessions where precision_or_map is a precision string or per-model suffix dict."""
    if isinstance(precision_or_map, str):
        suffix_map = PRECISION_MAPS[precision_or_map]
    else:
        suffix_map = precision_or_map
    sessions: Dict[str, ort.InferenceSession] = {}
    for name, suffix in suffix_map.items():
        path = model_dir / f"{name}{suffix}.onnx"
        sessions[name] = _make_session(path)
    return sessions


# ---------------------------------------------------------------------------
# Layer-by-layer degradation test
# ---------------------------------------------------------------------------

ONNX_MODEL_DIR = Path("export_models/onnx")

PRECISION_MAPS = {
    "fp32":   {m: "_fp32"         for m in MODEL_NAMES},
    "fp16":   {m: "_fp16"         for m in MODEL_NAMES},
    "int8":   {m: "_int8_static"  for m in MODEL_NAMES},
    # int8_mixed: quality-preserving config, bert fp32 (INT8 cascades badly at 0.57 cosine_sim), conditioner fp32, vocoder fp16
    "int8_mixed": {
        "bert":                 "_fp32",
        "duration_predictor":   "_int8_static",
        "acoustic_expand":      "_int8_static",
        "f0n_predictor":        "_int8_static",
        "vocoder_conditioner":  "_fp32",
        "vocoder_stream_chunk": "_fp16",
    },
}


def layer_by_layer_test(kpipeline: KPipeline, output_dir: Path) -> None:
    """Swap each model to int8 one-at-a-time vs fp32 baseline; report SNR degradation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("LAYER-BY-LAYER INT8 DEGRADATION TEST")
    print("="*70)
    print("Baseline: all fp32. Replacing one model at a time with int8.\n")

    fp32_map = {m: "_fp32" for m in MODEL_NAMES}
    int8_map  = {m: "_int8_static" for m in MODEL_NAMES}

    # Generate fp32 reference
    print("Generating fp32 reference audio...")
    fp32_sessions = load_sessions(ONNX_MODEL_DIR, fp32_map)
    ref_audios = []
    for sent in EVAL_SENTENCES:
        audio = generate_onnx(sent, fp32_sessions, kpipeline)
        ref_audios.append(audio)
    print(f"  Generated {len(ref_audios)} reference clips\n")

    # All-int8 baseline
    print("Generating all-int8 audio...")
    int8_sessions = load_sessions(ONNX_MODEL_DIR, int8_map)
    int8_audios = []
    for i, sent in enumerate(EVAL_SENTENCES):
        audio = generate_onnx(sent, int8_sessions, kpipeline)
        int8_audios.append(audio)
    all_int8_snrs = [compute_snr(ref_audios[i], int8_audios[i])
                     for i in range(len(EVAL_SENTENCES))]
    all_int8_pearson = [compute_pearson(ref_audios[i], int8_audios[i])
                        for i in range(len(EVAL_SENTENCES))]
    print(f"  All-int8 SNR={np.mean(all_int8_snrs):.1f}dB  pearson={np.mean(all_int8_pearson):.3f}\n")

    # Save all-int8 ref for UTMOS
    all_int8_path = output_dir / "layer_test_all_int8.wav"
    wavfile.write(str(all_int8_path), SAMPLE_RATE,
                  np.concatenate(int8_audios).astype(np.float32))

    results: List[Dict] = []

    # Swap one model at a time
    for swap_model in MODEL_NAMES:
        mixed_map = dict(fp32_map)
        mixed_map[swap_model] = "_int8_static"  # swap this one to int8

        mixed_sessions = load_sessions(ONNX_MODEL_DIR, mixed_map)
        mixed_audios = []
        for sent in EVAL_SENTENCES:
            audio = generate_onnx(sent, mixed_sessions, kpipeline)
            mixed_audios.append(audio)

        snrs = [compute_snr(ref_audios[i], mixed_audios[i])
                for i in range(len(EVAL_SENTENCES))]
        pearsons = [compute_pearson(ref_audios[i], mixed_audios[i])
                    for i in range(len(EVAL_SENTENCES))]
        mean_snr = np.mean(snrs)
        mean_pearson = np.mean(pearsons)

        wav_path = output_dir / f"layer_test_{swap_model}_int8.wav"
        wavfile.write(str(wav_path), SAMPLE_RATE,
                      np.concatenate(mixed_audios).astype(np.float32))

        # UTMOS
        utmos = compute_utmos(str(wav_path))

        results.append({
            "model": swap_model,
            "snr_db": mean_snr,
            "pearson": mean_pearson,
            "utmos": utmos,
        })
        utmos_str = f"{utmos:.2f}" if utmos is not None else "n/a"
        print(f"  swap {swap_model:<25s}  SNR={mean_snr:+6.1f}dB  pearson={mean_pearson:.3f}  UTMOS={utmos_str}")

    print("\n── Summary (sorted by SNR impact) ──")
    results.sort(key=lambda x: x["snr_db"])
    print(f"  {'Model':<25s}  {'SNR (dB)':>10s}  {'Pearson':>8s}  {'UTMOS':>7s}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*8}  {'─'*7}")
    for r in results:
        utmos_str = f"{r['utmos']:.2f}" if r['utmos'] is not None else "  n/a"
        print(f"  {r['model']:<25s}  {r['snr_db']:+10.1f}  {r['pearson']:8.3f}  {utmos_str:>7s}")
    print(f"\n  Baseline (all fp32)        SNR=inf  pearson=1.000")
    print(f"  All-int8                   SNR={np.mean(all_int8_snrs):+6.1f}dB  pearson={np.mean(all_int8_pearson):.3f}")

    # Also compute UTMOS for fp32 reference
    fp32_ref_path = output_dir / "layer_test_fp32_ref.wav"
    wavfile.write(str(fp32_ref_path), SAMPLE_RATE,
                  np.concatenate(ref_audios).astype(np.float32))
    fp32_utmos = compute_utmos(str(fp32_ref_path))
    int8_utmos  = compute_utmos(str(all_int8_path))
    if fp32_utmos is not None:
        print(f"\n  UTMOS fp32 ref={fp32_utmos:.2f}  all-int8={int8_utmos:.2f}" if int8_utmos else
              f"\n  UTMOS fp32 ref={fp32_utmos:.2f}")
    return results


# ---------------------------------------------------------------------------
# Full precision comparison report
# ---------------------------------------------------------------------------

def full_report(kpipeline: KPipeline, output_dir: Path) -> None:
    """Compare fp32/fp16/int8/int8_mixed audio quality with UTMOS + SNR/PESQ metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    text = EVAL_SENTENCES[0]

    print("\n" + "="*70)
    print("FULL PRECISION QUALITY REPORT")
    print("="*70)

    ref_audio = None
    rows = []
    for prec in ["fp32", "fp16", "int8_mixed", "int8"]:
        suffix_map = PRECISION_MAPS[prec]
        print(f"\n─── {prec} ───")
        model_sizes = {}
        for name, suffix in suffix_map.items():
            p = ONNX_MODEL_DIR / f"{name}{suffix}.onnx"
            model_sizes[name] = p.stat().st_size / 1e6 if p.exists() else 0.0
        total_mb = sum(model_sizes.values())
        print(f"  Total size: {total_mb:.1f} MB  ({', '.join(f'{n}:{s:.0f}MB' for n,s in model_sizes.items())})")

        try:
            sessions = load_sessions(ONNX_MODEL_DIR, prec)
            audio = generate_onnx(text, sessions, kpipeline)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        wav_path = output_dir / f"report_{prec}.wav"
        wavfile.write(str(wav_path), SAMPLE_RATE, audio)

        if prec == "fp32":
            ref_audio = audio

        utmos = compute_utmos(str(wav_path))
        snr = compute_snr(ref_audio, audio) if ref_audio is not None and prec != "fp32" else None
        pearson = compute_pearson(ref_audio, audio) if ref_audio is not None and prec != "fp32" else None
        pesq_score = compute_pesq(ref_audio, audio) if ref_audio is not None and prec != "fp32" else None

        rows.append({"prec": prec, "utmos": utmos, "snr_db": snr,
                     "pearson": pearson, "pesq": pesq_score, "dur": len(audio)/SAMPLE_RATE,
                     "size_mb": total_mb})
        snr_str = f"{snr:+.1f}" if snr is not None else "  ref"
        pearson_str = f"{pearson:.3f}" if pearson is not None else "  ref"
        pesq_str = f"{pesq_score:.3f}" if pesq_score is not None else " n/a"
        utmos_str = f"{utmos:.2f}" if utmos is not None else "n/a"
        print(f"  SNR={snr_str:>7s}dB  pearson={pearson_str}  PESQ={pesq_str}"
              f"  UTMOS={utmos_str}  dur={len(audio)/SAMPLE_RATE:.1f}s  size={total_mb:.0f}MB")

    print("\n── Summary ──")
    print(f"  {'Precision':<12s}  {'SNR (dB)':>10s}  {'Pearson':>8s}  {'PESQ':>7s}  {'UTMOS':>7s}  {'Size MB':>8s}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}")
    for r in rows:
        snr_str = f"{r['snr_db']:+.1f}" if r['snr_db'] is not None else "    ref"
        p_str = f"{r['pearson']:.3f}" if r['pearson'] is not None else "    ref"
        pesq_str = f"{r['pesq']:.3f}" if r['pesq'] is not None else "    n/a"
        u_str = f"{r['utmos']:.2f}" if r['utmos'] is not None else "    n/a"
        print(f"  {r['prec']:<12s}  {snr_str:>10s}  {p_str:>8s}  {pesq_str:>7s}  "
              f"{u_str:>7s}  {r['size_mb']:>8.0f}")
    return rows


# ---------------------------------------------------------------------------
# Score a single wav
# ---------------------------------------------------------------------------

def score_single(wav_path: str, ref_path: Optional[str]) -> None:
    print(f"\nScoring: {wav_path}")
    ref_audio = None
    if ref_path:
        sr, r = wavfile.read(ref_path)
        if r.dtype != np.float32:
            r = r.astype(np.float32) / np.iinfo(r.dtype).max
        ref_audio = r

    sr, audio = wavfile.read(wav_path)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

    utmos = compute_utmos(wav_path)
    print(f"  Duration:  {len(audio)/sr:.2f}s")
    print(f"  UTMOS:     {utmos:.3f}" if utmos else "  UTMOS:     n/a")
    if ref_audio is not None:
        print(f"  SNR:       {compute_snr(ref_audio, audio):.1f} dB")
        print(f"  Pearson:   {compute_pearson(ref_audio, audio):.4f}")
        pesq = compute_pesq(ref_audio, audio, sr=sr)
        print(f"  PESQ:      {pesq:.3f}" if pesq else "  PESQ:      n/a")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="INT8 quality evaluation for Kokoro TTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["layer_by_layer", "report", "score"],
                        default="layer_by_layer")
    parser.add_argument("--wav",  default=None, help="WAV path for --mode score")
    parser.add_argument("--ref",  default=None, help="Reference WAV for --mode score")
    parser.add_argument("--out",  type=Path, default=Path("int8_eval_output"),
                        help="Output directory for generated wavs")
    args = parser.parse_args()

    print("Loading Kokoro pipeline...")
    kmodel = KModel(config="checkpoints/config.json",
                    model="checkpoints/kokoro-v1_0.pth",
                    disable_complex=True).to("cpu").eval()
    kpipeline = KPipeline(lang_code="a", model=kmodel, device="cpu")

    if args.mode == "layer_by_layer":
        layer_by_layer_test(kpipeline, args.out)
    elif args.mode == "report":
        full_report(kpipeline, args.out)
    elif args.mode == "score":
        if not args.wav:
            parser.error("--wav required for --mode score")
        score_single(args.wav, args.ref)


if __name__ == "__main__":
    main()
