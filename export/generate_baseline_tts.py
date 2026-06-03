import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kokoro baseline WAVs.")
    parser.add_argument("--input", type=Path, default=Path("export/test.txt"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--voice", default="checkpoints/voices/af_heart.pt")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = args.output_dir or Path("test_output") / git_hash() / "baseline_tts"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    manifest_lines = []
    with torch.inference_mode():
        for index, text in enumerate(lines, start=1):
            chunks = []
            phoneme_lengths = []
            for result in pipeline(
                text,
                voice=args.voice,
                speed=args.speed,
                split_pattern=None,
            ):
                if result.audio is not None:
                    chunks.append(result.audio)
                    phoneme_lengths.append(len(result.phonemes))

            if not chunks:
                raise RuntimeError(f"No chunks generated for line {index}: {text}")

            audio = torch.cat(chunks)
            wav_path = output_dir / f"baseline_line_{index:02d}_af_heart.wav"
            write_wav(wav_path, audio)
            seconds = audio.numel() / SAMPLE_RATE
            manifest_lines.append(
                f"{wav_path.name}\t{seconds:.2f}s\tphoneme_lengths={phoneme_lengths}\t{text}"
            )
            print(f"Wrote {wav_path} ({seconds:.2f}s)")

    manifest = output_dir / "manifest.tsv"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
