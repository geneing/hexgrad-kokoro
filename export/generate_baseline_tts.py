import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from kokoro import KModel, KPipeline


SAMPLE_RATE = 24000


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


def read_lines(path: Path, limit: int) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line][:limit]


def write_wav(path: Path, audio: torch.Tensor) -> None:
    samples = audio.detach().cpu().float().numpy()
    if samples.size == 0:
        raise ValueError(f"No audio generated for {path}")
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    samples = np.clip(samples, -1.0, 1.0)
    wavfile.write(path, SAMPLE_RATE, (samples * 32767.0).astype(np.int16))


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def save_npz(path: Path, **tensors: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{name: tensor_to_numpy(tensor) for name, tensor in tensors.items()})


def phonemize(pipeline: KPipeline, text: str) -> list[tuple[str, str]]:
    if pipeline.lang_code in "ab":
        _, tokens = pipeline.g2p(text)
        return [(graphemes, phonemes) for graphemes, phonemes, _ in pipeline.en_tokenize(tokens) if phonemes]

    phonemes, _ = pipeline.g2p(text)
    return [(text, phonemes)] if phonemes else []


def input_ids_for_phonemes(model: KModel, phonemes: str) -> torch.LongTensor:
    token_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
    assert len(token_ids) + 2 <= model.context_length, (
        len(token_ids) + 2,
        model.context_length,
    )
    return torch.LongTensor([[0, *token_ids, 0]]).to(model.device)


def forward_debug(
    model: KModel,
    input_ids: torch.LongTensor,
    ref_s: torch.FloatTensor,
    speed: float,
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
    duration_encoded = model.predictor.text_encoder(
        d_en, style_predictor, input_lengths, text_mask
    )
    predictor_lstm, _ = model.predictor.lstm(duration_encoded)
    duration_logits = model.predictor.duration_proj(predictor_lstm)
    duration_scores = torch.sigmoid(duration_logits).sum(axis=-1) / speed
    pred_dur = torch.round(duration_scores).clamp(min=1).long().squeeze()

    indices = torch.repeat_interleave(
        torch.arange(input_ids.shape[1], device=model.device), pred_dur
    )
    pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=model.device)
    pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
    pred_aln_trg = pred_aln_trg.unsqueeze(0).to(model.device)

    predictor_aligned_en = duration_encoded.transpose(-1, -2) @ pred_aln_trg
    f0_pred, n_pred = model.predictor.F0Ntrain(predictor_aligned_en, style_predictor)

    text_encoded = model.text_encoder(input_ids, input_lengths, text_mask)
    asr = text_encoded @ pred_aln_trg
    audio = model.decoder(asr, f0_pred, n_pred, style_decoder).squeeze()

    return {
        "input_ids": input_ids,
        "input_lengths": input_lengths,
        "text_mask": text_mask,
        "attention_mask": attention_mask,
        "ref_s": ref_s,
        "style_predictor": style_predictor,
        "style_decoder": style_decoder,
        "bert_hidden": bert_hidden,
        "bert": d_en,
        "duration_encoded": duration_encoded,
        "predictor_lstm": predictor_lstm,
        "duration_logits": duration_logits,
        "duration_scores": duration_scores,
        "pred_dur": pred_dur,
        "pred_aln_trg": pred_aln_trg,
        "predictor_aligned_en": predictor_aligned_en,
        "f0": f0_pred,
        "n": n_pred,
        "text_encoder": text_encoded,
        "asr": asr,
        "decoder": audio,
    }


def save_debug_tensors(line_dir: Path, tensors: dict[str, torch.Tensor]) -> None:
    save_npz(
        line_dir / "inputs.npz",
        input_ids=tensors["input_ids"],
        input_lengths=tensors["input_lengths"],
        text_mask=tensors["text_mask"],
        attention_mask=tensors["attention_mask"],
        ref_s=tensors["ref_s"],
        style_predictor=tensors["style_predictor"],
        style_decoder=tensors["style_decoder"],
    )
    save_npz(line_dir / "bert.npz", hidden=tensors["bert_hidden"], output=tensors["bert"])
    save_npz(line_dir / "text_encoder.npz", output=tensors["text_encoder"])
    save_npz(
        line_dir / "predictor_dur.npz",
        duration_encoded=tensors["duration_encoded"],
        lstm=tensors["predictor_lstm"],
        logits=tensors["duration_logits"],
        scores=tensors["duration_scores"],
        pred_dur=tensors["pred_dur"],
        pred_aln_trg=tensors["pred_aln_trg"],
    )
    save_npz(
        line_dir / "predictor_f0n.npz",
        input=tensors["predictor_aligned_en"],
        f0=tensors["f0"],
        n=tensors["n"],
    )
    save_npz(line_dir / "decoder.npz", asr=tensors["asr"], audio=tensors["decoder"])


def shape_summary(tensors: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    keys = [
        "input_ids",
        "bert",
        "text_encoder",
        "duration_logits",
        "pred_dur",
        "pred_aln_trg",
        "predictor_aligned_en",
        "f0",
        "n",
        "asr",
        "decoder",
    ]
    return {key: list(tensors[key].shape) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kokoro baseline WAVs and debug tensors.")
    parser.add_argument("--input", type=Path, default=Path("export/test.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("test_output/baseline"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--voice", default="checkpoints/voices/af_heart.pt")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = args.output_dir
    wav_dir = output_dir / "wavs"
    tensor_dir = output_dir / "tensors"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    lines = read_lines(args.input, args.limit)
    if len(lines) < args.limit:
        raise ValueError(f"Expected at least {args.limit} non-empty lines in {args.input}")

    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config="checkpoints/config.json",
        model="checkpoints/kokoro-v1_0.pth",
        disable_complex=True,
    ).to(args.device).eval()
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
    voice_pack = pipeline.load_voice(args.voice).to(model.device)

    manifest_lines = []
    metadata = {
        "git_hash": git_hash(),
        "input": str(args.input),
        "voice": args.voice,
        "speed": args.speed,
        "sample_rate": SAMPLE_RATE,
        "format": {
            "wavs": "24 kHz mono int16 WAV files in wavs/",
            "tensors": "NumPy .npz files in tensors/line_XX/chunk_YY/",
        },
        "lines": [],
    }
    with torch.inference_mode():
        for index, text in enumerate(lines, start=1):
            chunk_audios = []
            chunk_metadata = []
            for chunk_index, (graphemes, phonemes) in enumerate(phonemize(pipeline, text), start=1):
                ref_s = voice_pack[len(phonemes) - 1]
                input_ids = input_ids_for_phonemes(model, phonemes)
                tensors = forward_debug(model, input_ids, ref_s, args.speed)
                chunk_dir = tensor_dir / f"line_{index:02d}" / f"chunk_{chunk_index:02d}"
                save_debug_tensors(chunk_dir, tensors)
                chunk_audios.append(tensors["decoder"].cpu())

                summary = shape_summary(tensors)
                chunk_metadata.append(
                    {
                        "chunk": chunk_index,
                        "graphemes": graphemes,
                        "phonemes": phonemes,
                        "phoneme_length": len(phonemes),
                        "tensor_dir": str(chunk_dir.relative_to(output_dir)),
                        "shapes": summary,
                    }
                )
                print(
                    f"line={index:02d} chunk={chunk_index:02d} "
                    f"phonemes={len(phonemes)} shapes={summary}"
                )

            if not chunk_audios:
                raise RuntimeError(f"No chunks generated for line {index}: {text}")

            audio = torch.cat(chunk_audios)
            wav_path = wav_dir / f"baseline_line_{index:02d}_af_heart.wav"
            write_wav(wav_path, audio)
            seconds = audio.numel() / SAMPLE_RATE
            manifest_lines.append(
                f"{wav_path.relative_to(output_dir)}\t{seconds:.2f}s\t"
                f"phoneme_lengths={[chunk['phoneme_length'] for chunk in chunk_metadata]}\t{text}"
            )
            metadata["lines"].append(
                {
                    "line": index,
                    "text": text,
                    "wav": str(wav_path.relative_to(output_dir)),
                    "seconds": seconds,
                    "chunks": chunk_metadata,
                }
            )
            print(f"Wrote {wav_path} ({seconds:.2f}s)")

    manifest = output_dir / "manifest.tsv"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
