from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import random
import signal
import traceback
import wave
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.multiprocessing
from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger

from kokoro import KModel, KPipeline

# Set spawn start method for CUDA multiprocessing compatibility
try:
    torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

DEFAULT_LIBRITTS_ROOT = Path("data/inputs/")
DEFAULT_OUTPUT_ROOT = Path("data/outputs/")
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
        help="Comma-separated subset of voices. Default uses English voices with prefix a/b only.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--write-repo-config", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of parallel worker processes")
    parser.add_argument("--queue-size", type=int, default=32, help="Max jobs in queue before blocking")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Save checkpoint every N samples")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for failed samples")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum total samples to generate (None = unlimited)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache directory for model and voices (default: ~/.cache/huggingface)")
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


def list_and_download_voices(repo_id: str, cache_dir: Optional[Path] = None) -> List[str]:
    files = list_repo_files(repo_id)
    voice_files = sorted(f for f in files if f.startswith("voices/") and f.endswith(".pt"))
    if not voice_files:
        raise RuntimeError(f"No voices/*.pt files found in repo {repo_id}")

    logger.info(f"Caching {len(voice_files)} voices from {repo_id}")
    for vf in voice_files:
        hf_hub_download(repo_id=repo_id, filename=vf, force_download=False)
    voices = [Path(vf).stem for vf in voice_files]
    return voices


def download_kokoro_assets(repo_id: str, cache_dir: Optional[Path] = None) -> dict:
    # Set HF cache directory if provided
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        logger.info(f"Using cache directory: {cache_dir}")
    
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", force_download=False)
    model_file = _model_file_from_repo(repo_id)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_file, force_download=False)
    voices = list_and_download_voices(repo_id, cache_dir=cache_dir)

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


@dataclass
class GenerationJob:
    """Job for parallel worker to process."""
    voice: str
    sent_idx: int
    chunk_idx: int
    chunk_text: str
    phonemes: str
    sentence: str
    ref_index: int
    utterance_id: str
    wav_path: str
    pair_path: str
    retry_count: int = 0


# Global worker state (initialized per-worker by pool initializer)
_worker_model = None
_worker_pipelines = None
_worker_device = None


def _worker_init_fn(repo_id: str, device: str) -> None:
    """Initialize worker process with model and pipelines (called once per worker)."""
    global _worker_model, _worker_pipelines, _worker_device
    
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already set, ignore
        pass
    
    _worker_device = device
    _worker_model = KModel(repo_id=repo_id).to(device).eval()
    _worker_pipelines = {}


def worker_generate_sample(job: GenerationJob, speed: float, frame_hop: int) -> dict:
    """Generate a single sample in worker process."""
    global _worker_model, _worker_pipelines, _worker_device
    
    try:
        # Create output directories
        Path(job.wav_path).parent.mkdir(parents=True, exist_ok=True)
        Path(job.pair_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already exists
        if Path(job.wav_path).exists() and Path(job.pair_path).exists():
            cached = torch.load(job.pair_path, map_location="cpu", weights_only=False)
            return {
                "status": "cached",
                "utterance_id": job.utterance_id,
                "wav_path": job.wav_path,
                "pair_path": job.pair_path,
                "audio_num_samples": _read_wav_num_samples_internal(job.wav_path),
                "asr_frames": int(cached["asr"].shape[-1]),
                "error": None,
            }
        
        # Get or create pipeline
        lang_code = job.voice[0]
        if lang_code not in _worker_pipelines:
            _worker_pipelines[lang_code] = KPipeline(
                lang_code=lang_code,
                repo_id="hexgrad/Kokoro-82M",  # Using default repo_id
                model=_worker_model,
                device=_worker_device,
            )
        pipeline = _worker_pipelines[lang_code]
        pack = pipeline.load_voice(job.voice).to(_worker_device)
        ref_s = pack[job.ref_index]
        
        audio, vocoder_io = _worker_model(
            job.phonemes,
            ref_s,
            speed=speed,
            return_vocoder_io=True,
        )
        
        # Write audio
        _write_wav_16bit_internal(Path(job.wav_path), audio, DEFAULT_SAMPLE_RATE)
        
        # Save pair
        pair_payload = {
            "asr": vocoder_io.asr.to(torch.float16),
            "f0": vocoder_io.f0.to(torch.float16),
            "noise": vocoder_io.noise.to(torch.float16),
            "style": vocoder_io.style.to(torch.float16),
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "frame_hop": frame_hop,
            "voice": job.voice,
            "text": job.chunk_text,
            "phonemes": job.phonemes,
            "wav_path": job.wav_path,
        }
        torch.save(pair_payload, job.pair_path)
        
        return {
            "status": "success",
            "utterance_id": job.utterance_id,
            "voice": job.voice,
            "text": job.chunk_text,
            "phonemes": job.phonemes,
            "wav_path": job.wav_path,
            "pair_path": job.pair_path,
            "audio_num_samples": int(audio.numel()),
            "asr_frames": int(vocoder_io.asr.shape[-1]),
            "error": None,
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            return {
                "status": "oom",
                "utterance_id": job.utterance_id,
                "error": str(e),
            }
        return {
            "status": "failed",
            "utterance_id": job.utterance_id,
            "error": str(e),
        }
    except Exception as e:
        return {
            "status": "failed",
            "utterance_id": job.utterance_id,
            "error": f"{type(e).__name__}: {str(e)}",
        }


def _write_wav_16bit_internal(path: Path, audio: torch.FloatTensor, sample_rate: int) -> None:
    """Write audio to 16-bit WAV file."""
    waveform = audio.detach().cpu().numpy().astype(np.float32)
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm16 = (waveform * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def _read_wav_num_samples_internal(path: str) -> int:
    """Read number of samples from WAV file."""
    with wave.open(path, "rb") as wav_file:
        return int(wav_file.getnframes())


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

    assets = download_kokoro_assets(args.repo_id, cache_dir=args.cache_dir)

    voices = assets["voices"]
    
    # Filter to English voices with prefix "a" or "b" by default
    if not args.voices:
        filtered_voices = [
            v for v in voices 
            if v[0].lower() in ("a", "b")
        ]
        if filtered_voices:
            logger.info(f"Filtering to English voices with prefix a/b: {len(filtered_voices)}/{len(voices)} available")
            voices = filtered_voices
        else:
            logger.warning("No English voices with prefix a/b found; using all available voices")
    
    if args.voices:
        requested = [v.strip() for v in args.voices.split(",") if v.strip()]
        missing = [v for v in requested if v not in assets["voices"]]
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

    # Load model once in main process for reference info
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

    # Setup checkpointing
    checkpoint_path = output_root / "generation_checkpoint.json"
    completed_samples = set()
    job_id_to_info = {}
    
    if checkpoint_path.exists():
        logger.info("Loading previous checkpoint...")
        ckpt = json.loads(checkpoint_path.read_text())
        completed_samples = set(ckpt.get("completed", []))
        logger.info(f"Resuming from {len(completed_samples)} completed samples")

    # Prepare all jobs
    all_jobs: List[GenerationJob] = []
    job_counter = 0
    
    for voice_idx, voice in enumerate(voices):
        logger.info(f"[{voice_idx + 1}/{len(voices)}] Preparing jobs for voice={voice}")
        pipeline = get_pipeline(voice[0])
        pack = pipeline.load_voice(voice).to(model.device)
        
        voice_audio_root = audio_root / voice
        voice_pair_root = pair_root / voice
        voice_audio_root.mkdir(parents=True, exist_ok=True)
        voice_pair_root.mkdir(parents=True, exist_ok=True)
        
        for sent_idx, sentence in enumerate(sentences):
            chunks = collect_phoneme_chunks(pipeline, sentence)
            if not chunks:
                continue
            
            for chunk_idx, (chunk_text, phonemes) in enumerate(chunks):
                # Check if we've reached max samples limit
                if args.max_samples is not None and len(all_jobs) >= args.max_samples:
                    logger.info(f"Reached max-samples limit: {args.max_samples}")
                    break
                
                utterance_id = f"{voice}_{sent_idx:05d}_{chunk_idx:02d}"
                wav_path = voice_audio_root / f"{utterance_id}.wav"
                pair_path = voice_pair_root / f"{utterance_id}.pt"
                
                # Skip if already completed
                if utterance_id in completed_samples:
                    continue
                
                ref_index = min(max(len(phonemes) - 1, 0), pack.shape[0] - 1)
                
                job = GenerationJob(
                    voice=voice,
                    sent_idx=sent_idx,
                    chunk_idx=chunk_idx,
                    chunk_text=chunk_text,
                    phonemes=phonemes,
                    sentence=sentence,
                    ref_index=ref_index,
                    utterance_id=utterance_id,
                    wav_path=str(wav_path),
                    pair_path=str(pair_path),
                )
                all_jobs.append(job)
                job_id_to_info[job_counter] = (utterance_id, job)
                job_counter += 1
        
        # Break outer voice loop if max samples reached
        if args.max_samples is not None and len(all_jobs) >= args.max_samples:
            break
    
    logger.info(f"Total jobs to process: {len(all_jobs)}")
    
    # Process jobs with parallelization
    _process_jobs_parallel(
        all_jobs,
        output_root,
        args.repo_id,
        device,
        args.speed,
        frame_hop,
        args.num_workers,
        args.queue_size,
        args.checkpoint_interval,
        args.max_retries,
        checkpoint_path,
        completed_samples,
    )
    
    # Finalize filelists
    all_wavs = _collect_completed_wavs(output_root)
    
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

    logger.info(f"Generated {len(completed_samples)} utterances across {len(voices)} voices")
    logger.info(f"Output root: {output_root}")
    logger.info(f"Train filelist: {train_filelist}")
    logger.info(f"Val filelist: {val_filelist}")
    logger.info(f"Vocos config: {output_config}")


def _process_jobs_parallel(
    all_jobs: List[GenerationJob],
    output_root: Path,
    repo_id: str,
    device: str,
    speed: float,
    frame_hop: int,
    num_workers: int,
    queue_size: int,
    checkpoint_interval: int,
    max_retries: int,
    checkpoint_path: Path,
    completed_samples: set,
) -> None:
    """Process generation jobs in parallel with OOM handling and checkpointing."""
    
    failed_jobs: List[tuple[GenerationJob, str]] = []
    retry_queue: List[GenerationJob] = []
    pending_jobs = list(all_jobs)
    current_num_workers = num_workers
    processed_count = 0
    
    # Setup signal handler for graceful shutdown
    interrupted = False
    
    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        logger.info("Interrupt signal received; saving checkpoint and exiting...")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        while pending_jobs or retry_queue:
            if interrupted:
                break
            
            # Process retry queue first
            if retry_queue:
                current_batch = retry_queue
                retry_queue = []
                logger.info(f"Processing {len(current_batch)} retry jobs with {current_num_workers} workers")
            else:
                # Take next batch
                current_batch = pending_jobs[:queue_size]
                pending_jobs = pending_jobs[queue_size:]
            
            if not current_batch:
                break
            
            # Process batch with current worker count
            with Pool(
                processes=current_num_workers,
                initializer=_worker_init_fn,
                initargs=(repo_id, device),
            ) as pool:
                results = []
                oom_detected = False
                
                for job in current_batch:
                    if interrupted:
                        break
                    
                    try:
                        result = pool.apply(_worker_wrapper, (job, speed, frame_hop))
                        results.append((job, result))
                        
                        # Handle results
                        if result["status"] == "success" or result["status"] == "cached":
                            completed_samples.add(result["utterance_id"])
                            processed_count += 1
                            
                            if processed_count % checkpoint_interval == 0:
                                _save_checkpoint(checkpoint_path, completed_samples)
                                logger.info(f"Checkpoint: {processed_count} samples completed")
                        
                        elif result["status"] == "oom":
                            oom_detected = True
                            if job.retry_count < max_retries:
                                job.retry_count += 1
                                retry_queue.append(job)
                                logger.warning(
                                    f"OOM on {result['utterance_id']}, retry {job.retry_count}/{max_retries}"
                                )
                            else:
                                failed_jobs.append((job, result["error"]))
                                logger.error(
                                    f"OOM failed after {max_retries} retries: {result['utterance_id']}"
                                )
                        
                        elif result["status"] == "failed":
                            if job.retry_count < max_retries:
                                job.retry_count += 1
                                retry_queue.append(job)
                                logger.warning(
                                    f"Failed {result['utterance_id']}, retry {job.retry_count}/{max_retries}: {result['error']}"
                                )
                            else:
                                failed_jobs.append((job, result["error"]))
                                logger.error(f"Failed after {max_retries} retries: {result['utterance_id']}")
                    
                    except Exception as e:
                        logger.error(f"Error processing job: {e}")
                        if job.retry_count < max_retries:
                            job.retry_count += 1
                            retry_queue.append(job)
                        else:
                            failed_jobs.append((job, str(e)))
                
                # If OOM detected, reduce worker count and retry
                if oom_detected and current_num_workers > 1:
                    current_num_workers = max(1, current_num_workers - 1)
                    logger.info(f"OOM detected, reducing workers to {current_num_workers}")
                    if current_batch:
                        pending_jobs = current_batch + pending_jobs
                
                # Close pool
                pool.close()
                pool.join()

    
    finally:
        # Final checkpoint
        _save_checkpoint(checkpoint_path, completed_samples)
        
        if failed_jobs:
            logger.warning(f"\n{len(failed_jobs)} samples failed:")
            for job, error in failed_jobs[:10]:  # Show first 10
                logger.warning(f"  - {job.utterance_id}: {error}")
            if len(failed_jobs) > 10:
                logger.warning(f"  ... and {len(failed_jobs) - 10} more")
        
        logger.info(f"Processing complete: {len(completed_samples)} samples")


def _worker_wrapper(job: GenerationJob, speed: float, frame_hop: int) -> dict:
    """Wrapper for worker function to handle exceptions."""
    try:
        return worker_generate_sample(job, speed, frame_hop)
    except Exception as e:
        return {
            "status": "failed",
            "utterance_id": job.utterance_id,
            "error": f"Worker wrapper error: {str(e)}",
        }



def _save_checkpoint(checkpoint_path: Path, completed_samples: set) -> None:
    """Save checkpoint of completed samples."""
    ckpt = {"completed": sorted(list(completed_samples))}
    checkpoint_path.write_text(json.dumps(ckpt), encoding="utf-8")


def _collect_completed_wavs(output_root: Path) -> List[str]:
    """Collect all completed wav files."""
    wav_files = []
    audio_root = output_root / "audio"
    if audio_root.exists():
        for wav_file in sorted(audio_root.rglob("*.wav")):
            wav_files.append(str(wav_file.resolve()))
    return wav_files



if __name__ == "__main__":
    main()
