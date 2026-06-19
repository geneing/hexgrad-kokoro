#!/usr/bin/env python
"""
Example script showing programmatic usage of parallelized vocoder_data generation.
This demonstrates how to use the GenerationJob and parallelization features directly.
"""

from pathlib import Path
from kokoro.vocoder_data import (
    parse_args,
    download_kokoro_assets,
    reservoir_sample_sentences,
    collect_phoneme_chunks,
    GenerationJob,
    _process_jobs_parallel,
    _collect_completed_wavs,
    split_train_val,
    render_vocos_config,
    infer_device,
    KModel,
    KPipeline,
)
import json
import math


def example_basic_parallel_generation():
    """
    Example 1: Basic parallel generation with default settings
    """
    print("=" * 60)
    print("Example 1: Basic Parallel Generation")
    print("=" * 60)
    
    args = parse_args([
        "--libritts-root", "/data/libritts",
        "--output-root", "outputs/",
        "--num-workers", "2",
        "--checkpoint-interval", "50",
    ])
    
    print(f"Configuration:")
    print(f"  Workers: {args.num_workers}")
    print(f"  Checkpoint Interval: {args.checkpoint_interval}")
    print(f"  Max Retries: {args.max_retries}")
    print(f"  Queue Size: {args.queue_size}")


def example_resume_from_checkpoint():
    """
    Example 2: Resume from checkpoint
    """
    print("\n" + "=" * 60)
    print("Example 2: Resume from Checkpoint")
    print("=" * 60)
    
    output_root = Path("outputs/")
    checkpoint_path = output_root / "generation_checkpoint.json"
    
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        completed = checkpoint.get("completed", [])
        print(f"Found checkpoint with {len(completed)} completed samples")
        print(f"First few: {completed[:5]}")
    else:
        print("No checkpoint found - will start fresh")


def example_custom_parallelization():
    """
    Example 3: Custom parallelization settings for different scenarios
    """
    print("\n" + "=" * 60)
    print("Example 3: Custom Parallelization Scenarios")
    print("=" * 60)
    
    scenarios = {
        "high_throughput": {
            "num_workers": 4,
            "queue_size": 64,
            "checkpoint_interval": 100,
            "description": "For systems with plenty of GPU memory",
        },
        "memory_constrained": {
            "num_workers": 1,
            "queue_size": 16,
            "checkpoint_interval": 25,
            "description": "For systems with limited GPU memory",
        },
        "balanced": {
            "num_workers": 2,
            "queue_size": 32,
            "checkpoint_interval": 50,
            "description": "Balanced for most ~24GB GPU setups",
        },
    }
    
    for name, settings in scenarios.items():
        print(f"\n{name.upper()}: {settings['description']}")
        for key, value in settings.items():
            if key != "description":
                print(f"  --{key.replace('_', '-')} {value}")


def example_error_handling():
    """
    Example 4: Demonstrates error handling and retry logic
    """
    print("\n" + "=" * 60)
    print("Example 4: Error Handling & Retry Logic")
    print("=" * 60)
    
    print("""
When generation encounters errors:

1. OOM Errors:
   - Detected via RuntimeError with "out of memory"
   - Worker pool size reduced by 1
   - Sample added back to retry queue
   - Automatic retry at reduced parallelism

2. Other Failures:
   - Logged with error message
   - Retried up to --max-retries times
   - If still failing, moved to failed list
   - Job ID recorded for post-processing

3. Graceful Interruption (Ctrl+C):
   - Signal handler saves checkpoint
   - In-flight jobs allowed to complete
   - Progress preserved in generation_checkpoint.json
   - Can resume later with same command
    """)


def example_job_structure():
    """
    Example 5: Understanding the GenerationJob structure
    """
    print("\n" + "=" * 60)
    print("Example 5: GenerationJob Structure")
    print("=" * 60)
    
    print(f"""
GenerationJob contains all parameters for one sample:

    @dataclass
    class GenerationJob:
        voice: str              # Voice name (e.g., "f_english_1")
        sent_idx: int           # Sentence index in source dataset
        chunk_idx: int          # Chunk index within sentence
        chunk_text: str         # Text of this chunk
        phonemes: str           # Phoneme representation
        sentence: str           # Full source sentence
        ref_index: int          # Reference index in voice pack
        utterance_id: str       # Unique identifier
        wav_path: str           # Where to save audio
        pair_path: str          # Where to save vocoder features
        retry_count: int        # Number of retries attempted

This structure allows the worker process to have all needed information
without needing to maintain global state or model references.
    """)


def example_checkpoint_structure():
    """
    Example 6: Understanding checkpoint format
    """
    print("\n" + "=" * 60)
    print("Example 6: Checkpoint Structure")
    print("=" * 60)
    
    example_checkpoint = {
        "completed": [
            "f_english_1_00000_00",
            "f_english_1_00000_01",
            "f_english_1_00001_00",
            "m_english_1_00000_00",
            # ... more IDs ...
        ]
    }
    
    print("Checkpoint file (generation_checkpoint.json):")
    print(json.dumps(example_checkpoint, indent=2)[:200] + "...")
    print(f"""
The checkpoint contains:
- completed: sorted list of utterance IDs that finished successfully
- This allows skipping completed work on resume
- On startup, jobs for IDs in this list are skipped
- Checkpoint updated every N samples (see --checkpoint-interval)
    """)


def example_monitoring():
    """
    Example 7: Monitoring generation progress
    """
    print("\n" + "=" * 60)
    print("Example 7: Monitoring Progress")
    print("=" * 60)
    
    print("""
Monitor progress in real-time:

1. Watch the logs:
   tail -f logs/generation.log | grep "Checkpoint:"

2. Check checkpoint:
   python -c "import json; c=json.load(open('generation_checkpoint.json')); print(f'Done: {len(c[\"completed\"])}')"

3. Count output files:
   find outputs/audio -name "*.wav" | wc -l
   find outputs/pairs -name "*.pt" | wc -l

4. Check for failures:
   grep "ERROR\|FAIL" logs/generation.log | head -20

Expected output format:
[2024-06-17 10:00:00.123] Generated 50 utterances
[2024-06-17 10:00:10.456] Checkpoint: 50 samples completed
[2024-06-17 10:00:20.789] Generated 100 utterances
    """)


def example_post_processing():
    """
    Example 8: Post-processing generated data
    """
    print("\n" + "=" * 60)
    print("Example 8: Post-Processing Generated Data")
    print("=" * 60)
    
    print("""
After generation completes, outputs are organized:

outputs/
├── generation_checkpoint.json    # Resume checkpoint
├── dataset_metadata.json         # Dataset configuration
├── vocos-kokoro-24khz.yaml      # Training config
├── audio/                        # Generated waveforms
│   ├── f_english_1/
│   │   ├── f_english_1_00000_00.wav
│   │   ├── f_english_1_00000_01.wav
│   │   └── ...
│   ├── f_english_2/
│   └── ...
├── pairs/                        # Vocoder feature pairs
│   ├── f_english_1/
│   │   ├── f_english_1_00000_00.pt
│   │   ├── f_english_1_00000_01.pt
│   │   └── ...
│   └── ...
├── manifests/                    # Per-voice manifests
│   ├── f_english_1.jsonl
│   ├── f_english_2.jsonl
│   └── ...
└── filelists/                    # Train/val splits
    ├── vocos.train.txt           # Paths for training
    └── vocos.val.txt             # Paths for validation

Key files for training:
- filelists/vocos.train.txt   ← Use with training pipeline
- vocos-kokoro-24khz.yaml     ← Training configuration
    """)


if __name__ == "__main__":
    example_basic_parallel_generation()
    example_resume_from_checkpoint()
    example_custom_parallelization()
    example_error_handling()
    example_job_structure()
    example_checkpoint_structure()
    example_monitoring()
    example_post_processing()
    
    print("\n" + "=" * 60)
    print("For actual generation, run:")
    print("=" * 60)
    print("""
uv run kokoro-vocoder-data \\
    --libritts-root /path/to/LibriTTS \\
    --output-root inputs/ \\
    --num-workers 2 \\
    --checkpoint-interval 50
    """)
