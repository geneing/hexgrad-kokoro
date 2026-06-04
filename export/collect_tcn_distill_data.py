"""
Collect frozen LSTM-teacher tensors for TCN distillation.

Local LJSpeech + LibriTTS usage:
  uv run python export/collect_tcn_distill_data.py \
    --ljspeech-root /export/eingerman/audio/LJSpeech-1.1 \
    --libritts-root /export/eingerman/audio/LibriTTS/LibriTTS \
    --output-dir /export/eingerman/audio/tcl_distil/teacher/$(git rev-parse --short HEAD) \
    --voices af_heart,af_bella,af_sarah,am_michael,am_puck \
    --limit 1000 \
    --device cuda

The output dataset is text/phoneme driven. Audio files from LJSpeech are not
used; LJSpeech/LibriTTS text provides coverage and Kokoro voice packs provide
style vectors.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from kokoro import KModel, KPipeline


DEFAULT_VOICES = (
    "af_heart",
    "af_bella",
    "af_sarah",
    "af_nicole",
    "af_aoede",
    "am_michael",
    "am_puck",
    "am_fenrir",
)

DEFAULT_LJSPEECH_ROOT = Path("/export/eingerman/audio/LJSpeech-1.1")
DEFAULT_LIBRITTS_ROOT = Path("/export/eingerman/audio/LibriTTS/LibriTTS")
DEFAULT_DISTILL_ROOT = Path("/export/eingerman/audio/tcl_distil")


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


def load_config(path: Path, sequence_mixer: str) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if sequence_mixer == "lstm":
        config.pop("sequence_mixer", None)
    else:
        config["sequence_mixer"] = {
            "type": "tcn",
            **config.get("sequence_mixer", {}),
        }
        config["sequence_mixer"]["type"] = sequence_mixer
    return config


def read_ljspeech_metadata(root: Path, limit: int | None) -> list[dict]:
    metadata_path = root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"LJSpeech metadata not found: {metadata_path}")

    rows = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            text = parts[2] if len(parts) >= 3 and parts[2].strip() else parts[1]
            text = text.strip()
            if text:
                rows.append({"id": f"ljspeech_{utt_id}", "source": "ljspeech", "text": text})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def read_libritts_metadata(root: Path, limit: int | None) -> list[dict]:
    if not root.is_dir():
        raise FileNotFoundError(f"LibriTTS root not found: {root}")

    rows = []
    seen_ids = set()

    for trans_path in sorted(root.rglob("*.trans.tsv")):
        with trans_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                utt_id = parts[0].strip()
                text = parts[-1].strip()
                if utt_id and text and utt_id not in seen_ids:
                    rows.append({"id": f"libritts_{utt_id}", "source": "libritts", "text": text})
                    seen_ids.add(utt_id)
                if limit is not None and len(rows) >= limit:
                    return rows

    # Some LibriTTS layouts expose per-utterance normalized text files instead.
    for text_path in sorted(root.rglob("*.normalized.txt")):
        utt_id = text_path.stem.removesuffix(".normalized")
        text = text_path.read_text(encoding="utf-8").strip()
        if utt_id and text and utt_id not in seen_ids:
            rows.append({"id": f"libritts_{utt_id}", "source": "libritts", "text": text})
            seen_ids.add(utt_id)
        if limit is not None and len(rows) >= limit:
            return rows

    return rows


def read_text_file(path: Path, limit: int | None) -> list[dict]:
    rows = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if text:
            rows.append({"id": f"text_{idx:06d}", "source": "text_file", "text": text})
        if limit is not None and len(rows) >= limit:
            break
    return rows


def load_source_rows(
    ljspeech_root: Path | None,
    libritts_root: Path | None,
    text_file: Path | None,
    limit_per_source: int | None,
) -> list[dict]:
    rows = []
    if ljspeech_root is not None:
        rows.extend(read_ljspeech_metadata(ljspeech_root, limit_per_source))
    if libritts_root is not None:
        rows.extend(read_libritts_metadata(libritts_root, limit_per_source))
    if text_file is not None:
        rows.extend(read_text_file(text_file, limit_per_source))
    return rows


def phonemize(pipeline: KPipeline, text: str) -> list[tuple[str, str]]:
    if pipeline.lang_code in "ab":
        _, tokens = pipeline.g2p(text)
        return [
            (graphemes, phonemes)
            for graphemes, phonemes, _ in pipeline.en_tokenize(tokens)
            if phonemes
        ]

    phonemes, _ = pipeline.g2p(text)
    return [(text, phonemes)] if phonemes else []


def input_ids_for_phonemes(model: KModel, phonemes: str) -> torch.LongTensor:
    token_ids = [model.vocab[p] for p in phonemes if model.vocab.get(p) is not None]
    if len(token_ids) + 2 > model.context_length:
        raise ValueError(f"Phoneme sequence too long: {len(token_ids) + 2} > {model.context_length}")
    return torch.LongTensor([[0, *token_ids, 0]]).to(model.device)


def build_alignment(pred_dur: torch.Tensor, token_count: int, device: torch.device) -> torch.Tensor:
    indices = torch.repeat_interleave(torch.arange(token_count, device=device), pred_dur)
    pred_aln_trg = torch.zeros((token_count, indices.shape[0]), device=device)
    pred_aln_trg[indices, torch.arange(indices.shape[0], device=device)] = 1
    return pred_aln_trg.unsqueeze(0)


def run_f0n_heads(model: KModel, shared: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = shared.transpose(-1, -2)

    f0 = x
    for block in model.predictor.F0:
        f0 = block(f0, style)
    f0 = model.predictor.F0_proj(f0).squeeze(1)

    n = x
    for block in model.predictor.N:
        n = block(n, style)
    n = model.predictor.N_proj(n).squeeze(1)
    return f0, n


@torch.inference_mode()
def forward_teacher(
    model: KModel,
    input_ids: torch.Tensor,
    ref_s: torch.Tensor,
    speed: float,
    include_decoder: bool,
) -> dict[str, torch.Tensor]:
    input_lengths = torch.full(
        (input_ids.shape[0],),
        input_ids.shape[-1],
        device=input_ids.device,
        dtype=torch.long,
    )
    text_mask = (
        torch.arange(input_lengths.max(), device=input_ids.device)
        .unsqueeze(0)
        .expand(input_lengths.shape[0], -1)
        .type_as(input_lengths)
    )
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(model.device)
    attention_mask = (~text_mask).int()

    bert_hidden = model.bert(input_ids, attention_mask=attention_mask)
    d_en = model.bert_encoder(bert_hidden).transpose(-1, -2)
    style_predictor = ref_s[:, 128:]
    style_decoder = ref_s[:, :128]

    duration_encoded = model.predictor.text_encoder(d_en, style_predictor, input_lengths, text_mask)
    duration_mixer = model.predictor.run_duration_mixer(duration_encoded)
    duration_logits = model.predictor.duration_proj(duration_mixer)
    duration_scores = torch.sigmoid(duration_logits).sum(axis=-1) / speed
    pred_dur = torch.round(duration_scores).clamp(min=1).long().squeeze(0)

    pred_aln_trg = build_alignment(pred_dur, input_ids.shape[1], model.device)
    predictor_aligned_en = duration_encoded.transpose(-1, -2) @ pred_aln_trg
    f0n_shared = model.predictor.run_shared_mixer(predictor_aligned_en)
    f0, n = run_f0n_heads(model, f0n_shared, style_predictor)

    text_encoder = model.text_encoder(input_ids, input_lengths, text_mask)
    asr = text_encoder @ pred_aln_trg

    tensors = {
        "input_ids": input_ids,
        "input_lengths": input_lengths,
        "text_mask": text_mask,
        "attention_mask": attention_mask,
        "ref_s": ref_s,
        "style_predictor": style_predictor,
        "style_decoder": style_decoder,
        "bert_hidden": bert_hidden,
        "d_en": d_en,
        "text_encoder": text_encoder,
        "duration_encoded": duration_encoded,
        "duration_mixer": duration_mixer,
        "duration_logits": duration_logits,
        "duration_scores": duration_scores,
        "pred_dur": pred_dur,
        "pred_aln_trg": pred_aln_trg,
        "predictor_aligned_en": predictor_aligned_en,
        "f0n_shared": f0n_shared,
        "f0": f0,
        "n": n,
        "asr": asr,
    }
    if include_decoder:
        tensors["audio"] = model.decoder(asr, f0, n, style_decoder).squeeze(0)
    return tensors


def save_npz(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, tensor in tensors.items():
        arrays[name] = tensor.detach().cpu().numpy()
    np.savez_compressed(path, **arrays)


def shape_summary(tensors: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {name: list(tensor.shape) for name, tensor in tensors.items()}


def voice_list(raw: str) -> list[str]:
    return [voice.strip() for voice in raw.split(",") if voice.strip()]


def iter_cases(
    rows: Iterable[dict],
    voices: list[str],
    pipelines: dict[str, KPipeline],
) -> Iterable[tuple[dict, str, int, str, str]]:
    for row in rows:
        for voice in voices:
            lang = voice[0]
            pipeline = pipelines.get(lang) or pipelines["a"]
            for chunk_idx, (graphemes, phonemes) in enumerate(phonemize(pipeline, row["text"]), start=1):
                yield row, voice, chunk_idx, graphemes, phonemes


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Kokoro LSTM-teacher tensors for TCN distillation.")
    parser.add_argument("--ljspeech-root", type=Path, default=DEFAULT_LJSPEECH_ROOT, help="Path to LJSpeech-1.1 containing metadata.csv.")
    parser.add_argument("--libritts-root", type=Path, default=DEFAULT_LIBRITTS_ROOT, help="Path to LibriTTS root containing *.trans.tsv or *.normalized.txt files.")
    parser.add_argument("--text-file", type=Path, default=None, help="Optional plain text file, one utterance per line.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/kokoro-v1_0.pth"))
    parser.add_argument("--voices", default=",".join(DEFAULT_VOICES))
    parser.add_argument("--limit", type=int, default=None, help="Number of rows per source before voice expansion.")
    parser.add_argument("--max-cases", type=int, default=None, help="Hard cap after voice/chunk expansion.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-decoder", action="store_true", help="Also save decoder audio tensors. Expensive.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    run_hash = git_hash()
    output_dir = args.output_dir or DEFAULT_DISTILL_ROOT / "teacher" / run_hash
    tensor_dir = output_dir / "tensors"
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    rows = load_source_rows(args.ljspeech_root, args.libritts_root, args.text_file, args.limit)
    voices = voice_list(args.voices)
    if not rows:
        raise RuntimeError("No text rows found for collection.")
    if not voices:
        raise RuntimeError("No voices selected.")

    teacher_config = load_config(args.config, sequence_mixer="lstm")
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=teacher_config,
        model=str(args.checkpoint),
        disable_complex=True,
    ).to(args.device).eval()

    pipeline_langs = sorted({voice[0] for voice in voices if voice and voice[0] in "ab"} | {"a"})
    pipelines = {
        lang: KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=False)
        for lang in pipeline_langs
    }
    voice_packs = {
        voice: pipelines.get(voice[0], pipelines["a"]).load_voice(
            str(Path("checkpoints/voices") / f"{voice}.pt") if not voice.endswith(".pt") else voice
        ).to(model.device)
        for voice in voices
    }

    manifest = {
        "git_hash": run_hash,
        "sources": {
            "ljspeech_root": str(args.ljspeech_root) if args.ljspeech_root else None,
            "libritts_root": str(args.libritts_root) if args.libritts_root else None,
            "text_file": str(args.text_file) if args.text_file else None,
        },
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "sequence_mixer": "lstm_teacher",
        "voices": voices,
        "speed": args.speed,
        "include_decoder": args.include_decoder,
        "device": args.device,
        "num_source_rows": len(rows),
        "samples_jsonl": "samples.jsonl",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    samples_path = output_dir / "samples.jsonl"
    written = 0
    skipped = 0
    with samples_path.open("w", encoding="utf-8") as samples_file:
        for row, voice, chunk_idx, graphemes, phonemes in iter_cases(rows, voices, pipelines):
            if args.max_cases is not None and written >= args.max_cases:
                break
            safe_id = row["id"].replace("/", "_")
            case_name = f"{safe_id}_chunk{chunk_idx:02d}_{Path(voice).stem}"
            tensor_path = tensor_dir / f"{case_name}.npz"
            if args.skip_existing and tensor_path.is_file():
                skipped += 1
                continue

            try:
                input_ids = input_ids_for_phonemes(model, phonemes)
                ref_s = voice_packs[voice][len(phonemes) - 1]
                if ref_s.dim() == 1:
                    ref_s = ref_s.unsqueeze(0)
                tensors = forward_teacher(model, input_ids, ref_s, args.speed, args.include_decoder)
                save_npz(tensor_path, tensors)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"SKIP runtime_error id={row['id']} voice={voice}: {exc}")
                skipped += 1
                continue
            except Exception as exc:
                print(f"SKIP error id={row['id']} voice={voice}: {exc}")
                skipped += 1
                continue

            sample = {
                "case": case_name,
                "source_id": row["id"],
                "source": row.get("source", "unknown"),
                "voice": voice,
                "chunk_index": chunk_idx,
                "text": row["text"],
                "graphemes": graphemes,
                "phonemes": phonemes,
                "phoneme_length": len(phonemes),
                "token_length": int(tensors["input_ids"].shape[-1]),
                "aligned_length": int(tensors["predictor_aligned_en"].shape[-1]),
                "tensor_path": str(tensor_path.relative_to(output_dir)),
                "shapes": shape_summary(tensors),
            }
            samples_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
            if written % 25 == 0:
                print(f"written={written} skipped={skipped} last={case_name}")

    print(f"Done. output_dir={output_dir} written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
