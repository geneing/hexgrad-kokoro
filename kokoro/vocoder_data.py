from __future__ import annotations

import argparse
import json
import math
import random
import wave
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger

from kokoro import KModel, KPipeline

DEFAULT_LIBRITTS_ROOT = Path("/export/eingerman/audio/LibriTTS/LibriTTS")
DEFAULT_OUTPUT_ROOT = Path("/export/eingerman/audio/vocoder")
DEFAULT_REPO_ID = "hexgrad/Kokoro-82M"
DEFAULT_SENTENCE_COUNT = 1500
DEFAULT_SAMPLE_RATE = 24000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Prepare paired iSTFTNet input/output data for Vocos training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--libritts-root", type=Path, default=DEFAULT_LIBRITTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--num-sentences", type=int, default=DEFAULT_SENTENCE_COUNT)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--voices",
        type=str,
        default=None,
        help="Comma-separated subset of voices. Default uses all voices from HF repo.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--write-repo-config", action="store_true")
    return parser.parse_args()


def infer_device(requested: str | None) -> str:
    if requested:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _model_file_from_repo(repo_id: str) -> str:
    if repo_id in KModel.MODEL_NAMES:
        return KModel.MODEL_NAMES[repo_id]
    files = list_repo_files(repo_id)
    candidates = sorted(f for f in files if f.endswith(".pth"))
    if not candidates:
        raise RuntimeError(f"No .pth model file found in repo {repo_id}")
    return candidates[0]


def list_and_download_voices(repo_id: str) -> List[str]:
    files = list_repo_files(repo_id)
    voice_files = sorted(f for f in files if f.startswith("voices/") and f.endswith(".pt"))
    if not voice_files:
        raise RuntimeError(f"No voices/*.pt files found in repo {repo_id}")

    logger.info(f"Downloading {len(voice_files)} voices from {repo_id}")
    for vf in voice_files:
        hf_hub_download(repo_id=repo_id, filename=vf)
    voices = [Path(vf).stem for vf in voice_files]
    return voices


def download_kokoro_assets(repo_id: str) -> dict:
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    model_file = _model_file_from_repo(repo_id)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_file)
    voices = list_and_download_voices(repo_id)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    logger.info(f"Model file: {model_path}")
    logger.info(f"Config file: {config_path}")
    return {
        "config_path": config_path,
        "model_path": model_path,
        "config": config,
        "voices": voices,
    }


def iter_libritts_sentences(root: Path) -> Iterable[str]:
    for path in root.rglob("*.normalized.txt"):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            yield text


def reservoir_sample_sentences(root: Path, sample_size: int, seed: int) -> tuple[List[str], int]:
    if sample_size <= 0:
        return [], 0

    rng = random.Random(seed)
    sample: List[str] = []
    seen = 0

    for sentence in iter_libritts_sentences(root):
        seen += 1
        if len(sample) < sample_size:
            sample.append(sentence)
            continue
        j = rng.randrange(seen)
        if j < sample_size:
            sample[j] = sentence

    return sample, seen


def write_wav_16bit(path: Path, audio: torch.FloatTensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = audio.detach().cpu().numpy().astype(np.float32)
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm16 = (waveform * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def read_wav_num_samples(path: Path) -> int:
    with wave.open(str(path), "rb") as wav_file:
        return int(wav_file.getnframes())


def collect_phoneme_chunks(pipeline: KPipeline, text: str) -> List[tuple[str, str]]:
    chunks: List[tuple[str, str]] = []
    for result in pipeline(text, model=False, split_pattern=None):
        ps = (result.phonemes or "").strip()
        if not ps:
            continue
        if len(ps) > 510:
            ps = ps[:510]
        chunk_text = result.graphemes if result.graphemes else text
        chunks.append((chunk_text, ps))
    return chunks


def split_train_val(items: Sequence[str], val_ratio: float, seed: int) -> tuple[List[str], List[str]]:
    items = list(items)
    if not items:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(items)

    val_count = int(len(items) * val_ratio)
    if val_count <= 0 and len(items) > 1:
        val_count = 1
    if val_count >= len(items):
        val_count = max(0, len(items) - 1)

    val = items[:val_count]
    train = items[val_count:]
    return train, val


def render_vocos_config(
    filelist_train: Path,
    filelist_val: Path,
    sample_rate: int,
    n_mels: int,
    hop_length: int,
    n_fft: int,
    train_num_samples: int,
    val_num_samples: int,
) -> str:
    return f"""# Generated for Kokoro iSTFTNet distillation into Vocos
seed_everything: 4444

data:
  class_path: vocos.dataset.VocosDataModule
  init_args:
    train_params:
      filelist_path: {filelist_train}
      sampling_rate: {sample_rate}
      num_samples: {train_num_samples}
      batch_size: 8
      num_workers: 8

    val_params:
      filelist_path: {filelist_val}
      sampling_rate: {sample_rate}
      num_samples: {val_num_samples}
      batch_size: 8
      num_workers: 8

model:
  class_path: vocos.experiment.VocosExp
  init_args:
    sample_rate: {sample_rate}
    initial_learning_rate: 5e-4
    mel_loss_coeff: 45
    mrd_loss_coeff: 0.1
    num_warmup_steps: 0
    pretrain_mel_steps: 0

    evaluate_utmos: false
    evaluate_pesq: false
    evaluate_periodicty: false

    feature_extractor:
      class_path: vocos.feature_extractors.MelSpectrogramFeatures
      init_args:
        sample_rate: {sample_rate}
        n_fft: {n_fft}
        hop_length: {hop_length}
        n_mels: {n_mels}
        padding: center

    backbone:
      class_path: vocos.models.VocosBackbone
      init_args:
        input_channels: {n_mels}
        dim: 512
        intermediate_dim: 1536
        num_layers: 8

    head:
      class_path: vocos.heads.ISTFTHead
      init_args:
        dim: 512
        n_fft: {n_fft}
        hop_length: {hop_length}
        padding: center

trainer:
  logger:
    class_path: pytorch_lightning.loggers.TensorBoardLogger
    init_args:
      save_dir: logs/
      name: vocos-kokoro

  callbacks:
    - class_path: pytorch_lightning.callbacks.LearningRateMonitor
    - class_path: pytorch_lightning.callbacks.ModelSummary
      init_args:
        max_depth: 2
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args:
        monitor: val_loss
        filename: vocos_kokoro_{{epoch}}_{{step}}_{{val_loss:.4f}}
        save_top_k: 3
        save_last: true

  max_steps: 1000000
  limit_val_batches: 32
  accelerator: gpu
  devices: [0]
  log_every_n_steps: 100
"""


def main() -> None:
    logger.enable("kokoro.vocoder_data")
    args = parse_args()
    device = infer_device(args.device)

    if not args.libritts_root.exists():
        raise FileNotFoundError(f"LibriTTS root not found: {args.libritts_root}")

    assets = download_kokoro_assets(args.repo_id)

    voices = assets["voices"]
    if args.voices:
        requested = [v.strip() for v in args.voices.split(",") if v.strip()]
        missing = [v for v in requested if v not in voices]
        if missing:
            raise ValueError(f"Requested voices not found in repo {args.repo_id}: {missing}")
        voices = requested

    logger.info(f"Total voices selected: {len(voices)}")

    if args.download_only:
        logger.info("Download-only mode enabled; exiting after model/voice download")
        return

    sentences, seen = reservoir_sample_sentences(args.libritts_root, args.num_sentences, args.seed)
    if not sentences:
        raise RuntimeError("No LibriTTS .normalized.txt sentences were found")
    if len(sentences) < args.num_sentences:
        logger.warning(
            f"Requested {args.num_sentences} sentences but only found {len(sentences)} usable sentences "
            f"(scanned {seen} total)."
        )

    logger.info(f"Sampled {len(sentences)} sentences from {seen} LibriTTS entries")

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    audio_root = output_root / "audio"
    pair_root = output_root / "pairs"
    manifest_root = output_root / "manifests"
    filelist_root = output_root / "filelists"
    for d in [audio_root, pair_root, manifest_root, filelist_root]:
        d.mkdir(parents=True, exist_ok=True)

    model = KModel(repo_id=args.repo_id).to(device).eval()

    pipelines: dict[str, KPipeline] = {}

    def get_pipeline(lang_code: str) -> KPipeline:
        if lang_code not in pipelines:
            pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id=args.repo_id, model=model, device=device)
        return pipelines[lang_code]

    config = assets["config"]
    upsample_rates = config["istftnet"]["upsample_rates"]
    gen_hop = config["istftnet"]["gen_istft_hop_size"]
    frame_hop = int(math.prod(upsample_rates) * gen_hop)
    n_mels = int(config["n_mels"])

    # Use 4x frame hop for STFT window in Vocos config to match Kokoro frame rate.
    vocos_n_fft = int(frame_hop * 4)

    all_wavs: List[str] = []

    metadata_path = output_root / "dataset_metadata.json"
    metadata = {
        "repo_id": args.repo_id,
        "num_sentences_per_voice": len(sentences),
        "voices": voices,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "frame_hop": frame_hop,
        "n_mels": n_mels,
        "vocos_n_fft": vocos_n_fft,
        "speed": args.speed,
        "libritts_root": str(args.libritts_root),
        "seed": args.seed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    sample_count = 0
    for voice_idx, voice in enumerate(voices):
        logger.info(f"[{voice_idx + 1}/{len(voices)}] Generating data for voice={voice}")
        pipeline = get_pipeline(voice[0])
        pack = pipeline.load_voice(voice).to(model.device)

        voice_audio_root = audio_root / voice
        voice_pair_root = pair_root / voice
        voice_audio_root.mkdir(parents=True, exist_ok=True)
        voice_pair_root.mkdir(parents=True, exist_ok=True)

        manifest_path = manifest_root / f"{voice}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            for sent_idx, sentence in enumerate(sentences):
                chunks = collect_phoneme_chunks(pipeline, sentence)
                if not chunks:
                    continue

                for chunk_idx, (chunk_text, phonemes) in enumerate(chunks):
                    utterance_id = f"{voice}_{sent_idx:05d}_{chunk_idx:02d}"
                    wav_path = voice_audio_root / f"{utterance_id}.wav"
                    pair_path = voice_pair_root / f"{utterance_id}.pt"

                    if args.skip_existing and wav_path.exists() and pair_path.exists():
                        cached = torch.load(pair_path, map_location="cpu", weights_only=False)
                        record = {
                            "utterance_id": utterance_id,
                            "voice": cached.get("voice", voice),
                            "text": cached.get("text", chunk_text),
                            "phonemes": cached.get("phonemes", phonemes),
                            "wav_path": str(wav_path.resolve()),
                            "pair_path": str(pair_path.resolve()),
                            "audio_num_samples": read_wav_num_samples(wav_path),
                            "asr_frames": int(cached["asr"].shape[-1]),
                        }
                        manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        all_wavs.append(str(wav_path.resolve()))
                        sample_count += 1
                        continue

                    ref_index = min(max(len(phonemes) - 1, 0), pack.shape[0] - 1)
                    ref_s = pack[ref_index]

                    audio, vocoder_io = model(
                        phonemes,
                        ref_s,
                        speed=args.speed,
                        return_vocoder_io=True,
                    )

                    write_wav_16bit(wav_path, audio, DEFAULT_SAMPLE_RATE)

                    pair_payload = {
                        "asr": vocoder_io.asr.to(torch.float16),
                        "f0": vocoder_io.f0.to(torch.float16),
                        "noise": vocoder_io.noise.to(torch.float16),
                        "style": vocoder_io.style.to(torch.float16),
                        "sample_rate": DEFAULT_SAMPLE_RATE,
                        "frame_hop": frame_hop,
                        "voice": voice,
                        "text": chunk_text,
                        "phonemes": phonemes,
                        "wav_path": str(wav_path.resolve()),
                    }
                    torch.save(pair_payload, pair_path)

                    record = {
                        "utterance_id": utterance_id,
                        "voice": voice,
                        "text": chunk_text,
                        "phonemes": phonemes,
                        "wav_path": str(wav_path.resolve()),
                        "pair_path": str(pair_path.resolve()),
                        "audio_num_samples": int(audio.numel()),
                        "asr_frames": int(vocoder_io.asr.shape[-1]),
                    }
                    manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                    all_wavs.append(str(wav_path.resolve()))
                    sample_count += 1
                    if sample_count % 100 == 0:
                        logger.info(f"Generated {sample_count} utterances")

    train_wavs, val_wavs = split_train_val(all_wavs, args.val_ratio, args.seed)
    train_filelist = filelist_root / "vocos.train.txt"
    val_filelist = filelist_root / "vocos.val.txt"
    train_filelist.write_text("\n".join(train_wavs) + ("\n" if train_wavs else ""), encoding="utf-8")
    val_filelist.write_text("\n".join(val_wavs) + ("\n" if val_wavs else ""), encoding="utf-8")

    # Make num_samples multiples of frame_hop for cleaner frame alignment.
    train_num_samples = frame_hop * 64
    val_num_samples = frame_hop * 128
    vocos_config_text = render_vocos_config(
        filelist_train=train_filelist.resolve(),
        filelist_val=val_filelist.resolve(),
        sample_rate=DEFAULT_SAMPLE_RATE,
        n_mels=n_mels,
        hop_length=frame_hop,
        n_fft=vocos_n_fft,
        train_num_samples=train_num_samples,
        val_num_samples=val_num_samples,
    )

    output_config = output_root / "vocos-kokoro-24khz.yaml"
    output_config.write_text(vocos_config_text, encoding="utf-8")

    if args.write_repo_config:
        repo_config_dir = Path(__file__).resolve().parent.parent / "configs"
        repo_config_dir.mkdir(parents=True, exist_ok=True)
        repo_config_path = repo_config_dir / "vocos-kokoro-24khz.yaml"
        repo_config_path.write_text(vocos_config_text, encoding="utf-8")
        logger.info(f"Wrote repo config: {repo_config_path}")

    logger.info(f"Generated {sample_count} utterances across {len(voices)} voices")
    logger.info(f"Output root: {output_root}")
    logger.info(f"Train filelist: {train_filelist}")
    logger.info(f"Val filelist: {val_filelist}")
    logger.info(f"Vocos config: {output_config}")


if __name__ == "__main__":
    main()
