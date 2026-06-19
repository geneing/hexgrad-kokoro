# Vocoder Data Generation - Parallelization Guide

## Overview

The `kokoro/vocoder_data.py` module has been refactored to support **parallel processing** with advanced features for GPU memory management, error recovery, and resumable workflows.

## Key Features

### 1. Parallel Processing
- **Multi-worker execution**: Use `--num-workers` to spawn N parallel processes
- **GPU-aware**: Each worker maintains its own model instance to maximize throughput
- **Efficient**: Processes kept alive between batches to minimize GPU initialization overhead

### 2. Out-of-Memory (OOM) Handling
- **Automatic Detection**: System detects OOM errors during generation
- **Graceful Degradation**: Worker pool size reduced by 1 on each OOM
- **Automatic Retry**: Failed samples queued for retry at reduced parallelism
- **Prevents Crash**: No complete failure even with memory constraints

### 3. Checkpoint & Resume
- **Automatic Checkpointing**: Progress saved to `generation_checkpoint.json`
- **Resume Support**: Restart interrupted runs from last checkpoint
- **Skip Existing**: Completed samples detected and skipped automatically

### 4. Retry Mechanism
- **Configurable Retries**: Failed samples retried up to `--max-retries` times
- **Error Logging**: Detailed error messages for debugging
- **OOM-aware**: OOM failures treated separately from other errors

### 5. Graceful Interruption
- **Signal Handling**: Catches Ctrl+C and SIGTERM
- **Safe Shutdown**: Saves checkpoint before exiting
- **No Data Loss**: All completed samples preserved

## Command-Line Arguments

### Core Options
- `--libritts-root PATH`: Path to LibriTTS dataset (default: `data/inputs/`)
- `--output-root PATH`: Output directory for generated data (default: `data/outputs/`)
- `--num-sentences N`: Number of sentences per voice (default: 1500)
- `--voices VOICE1,VOICE2`: Comma-separated voice names (optional)

### Parallelization Options
- `--num-workers N` (default: 2)
  - Number of parallel processes
  - Increase for faster generation, decrease to reduce memory
  - Recommended: 2-4 on single GPU with ~24GB VRAM
  
- `--queue-size N` (default: 32)
  - Maximum samples in job queue before blocking
  - Lower values reduce memory footprint
  - Higher values improve throughput
  
- `--checkpoint-interval N` (default: 50)
  - Save progress every N completed samples
  - Lower = more frequent saves, slightly slower
  - Higher = faster generation, larger gaps between saves
  
- `--max-retries N` (default: 3)
  - Maximum retry attempts for failed samples
  - Higher = better fault tolerance (slower if many failures)
  - OOM errors handled separately

### Other Options
- `--skip-existing`: Skip samples already generated
- `--download-only`: Download assets but don't generate
- `--write-repo-config`: Write config to project configs/
- `--seed N`: Random seed (default: 4444)
- `--speed F`: Audio speed multiplier (default: 1.0)

## Usage Examples

### Basic Parallel Generation (2 workers)
```bash
uv run kokoro-vocoder-data \
  --libritts-root /export/eingerman/audio/LibriTTS/LibriTTS \
  --output-root inputs/ \
  --num-workers 2
```

### High-Throughput Mode (4 workers, aggressive checkpointing)
```bash
uv run kokoro-vocoder-data \
  --libritts-root /export/eingerman/audio/LibriTTS/LibriTTS \
  --output-root inputs/ \
  --num-workers 4 \
  --checkpoint-interval 100 \
  --queue-size 64
```

### Memory-Constrained Mode (1 worker, frequent saves)
```bash
uv run kokoro-vocoder-data \
  --libritts-root /export/eingerman/audio/LibriTTS/LibriTTS \
  --output-root inputs/ \
  --num-workers 1 \
  --checkpoint-interval 25 \
  --queue-size 16 \
  --max-retries 5
```

### Smoke Test (Small subset)
```bash
uv run kokoro-vocoder-data \
  --libritts-root /export/eingerman/audio/LibriTTS/LibriTTS \
  --output-root inputs/ \
  --num-sentences 100 \
  --num-workers 2
```

### Resume Interrupted Run
```bash
# Just re-run with same output-root; checkpoint detected automatically
uv run kokoro-vocoder-data \
  --libritts-root /export/eingerman/audio/LibriTTS/LibriTTS \
  --output-root inputs/ \
  --num-workers 2
```

### Specific Voice Generation
```bash
uv run kokoro-vocoder-data \
  --libritts-root /export/eingerman/audio/LibriTTS/LibriTTS \
  --output-root inputs/ \
  --voices f_english_1,f_english_2,m_english_1 \
  --num-workers 3
```

## Monitoring Progress

### Log Output
During generation, you'll see:
```
[...] Sampled 1500 sentences from 40000 LibriTTS entries
[...] Total jobs to process: 45000
[...] Processing 45000 jobs with 2 workers
[...] Checkpoint: 50 samples completed
[...] Failed sample1, retry 1/3
[...] OOM on sample2, retry 1/3
[...] Reducing workers to 1
[...] Processing complete: 45000 samples
```

### Checkpoint Status
Check progress anytime:
```bash
cat inputs/generation_checkpoint.json | grep -o '"completed"' | wc -l
# or
python -c "import json; ckpt = json.load(open('inputs/generation_checkpoint.json')); print(f'Completed: {len(ckpt[\"completed\"])} samples')"
```

## Troubleshooting

### High Memory Usage
- Reduce `--num-workers` (each worker = one model in VRAM)
- Reduce `--queue-size` (fewer jobs waiting in memory)
- Increase `--checkpoint-interval` (don't hold manifests as long)

### OOM Errors
- Starting with high workers? System will auto-reduce
- Reduce `--num-workers` manually to start
- Try `--num-workers 1` to find the bottleneck

### Slow Progress
- Increase `--num-workers` (if memory allows)
- Increase `--queue-size` (more parallelism)
- Decrease `--checkpoint-interval` (less overhead)

### Resuming from Crash
```bash
# Automatic - just run the same command
# System detects generation_checkpoint.json
# All completed samples skipped
# Resumes from where it left off
```

### Cleaning Up for Fresh Start
```bash
rm inputs/generation_checkpoint.json  # Remove checkpoint
# All samples will be re-generated next run (or use --skip-existing)
```

## Implementation Architecture

### Worker Process Model
```
Main Process (setup, job queue, I/O)
    ↓
    ├→ Worker 1 (model in VRAM, generates samples)
    ├→ Worker 2 (model in VRAM, generates samples)
    └→ Worker N (model in VRAM, generates samples)
```

### Error Handling Flow
```
Sample Generation
    ↓
    ├→ Success → Mark completed, continue
    ├→ OOM → Add to retry queue, reduce workers
    ├→ Other Error → Add to retry queue, keep workers
    └→ Max Retries → Log failure, continue
```

### Checkpoint Structure
```json
{
  "completed": [
    "voice1_00000_00",
    "voice1_00000_01",
    "voice1_00001_00",
    ...
  ]
}
```

## Performance Tuning

### Recommended Settings by Hardware

**GPU with ~24GB VRAM:**
```bash
--num-workers 2 --queue-size 32 --checkpoint-interval 50
```

**GPU with ~40GB VRAM:**
```bash
--num-workers 4 --queue-size 64 --checkpoint-interval 100
```

**GPU with <12GB VRAM:**
```bash
--num-workers 1 --queue-size 16 --checkpoint-interval 25
```

## Advanced Usage

### Programmatic Interface
```python
from kokoro.vocoder_data import main, parse_args

# Override defaults
args = parse_args([
    '--num-workers', '3',
    '--checkpoint-interval', '100',
    '--libritts-root', '/data/libritts',
])
main()  # Uses sys.argv
```

## Known Limitations

1. **Single Machine**: Current implementation runs on one machine. For distributed generation, would need job coordinator.
2. **Model Compatibility**: Requires model that supports `return_vocoder_io` parameter
3. **GPU-Only**: CPU mode supported but very slow; parallelization benefits diminish
4. **Memory Accumulation**: Long-lived processes may accumulate memory over time (mitigated by process pool lifecycle)

## FAQ

**Q: How many workers should I use?**
A: Start with 2. If no OOM and fast enough, try 3-4. If OOM occurs, system auto-reduces.

**Q: Will it resume automatically?**
A: Yes! Run same command again. Checkpoint automatically detected.

**Q: Can I interrupt safely?**
A: Yes! Press Ctrl+C anytime. Progress saved to checkpoint.

**Q: What if generation fails?**
A: Check logs for error. Failed samples logged. Re-run with `--max-retries` increased.

**Q: How do I clean up and restart?**
A: Delete `generation_checkpoint.json` in output root to discard progress.
