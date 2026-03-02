"""Compatibility shim for streaming-Vocos TF quantization validation.

Canonical module:
- `kokoro.tf.validate_quant`

This shim exists for backward compatibility with legacy entrypoint usage.
It forwards directly to the canonical module `main()`.

Examples:

1) Validate streaming-vocos checkpoint with defaults
   uv run python -m kokoro.validate_tf_quantizations \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --val-filelist inputs/filelists/vocos.val.txt

2) Validate a small subset
   uv run python -m kokoro.validate_tf_quantizations \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --val-filelist inputs/filelists/vocos.val.txt \
     --num-samples 4

3) Write quant/metrics to custom output dirs
   uv run python -m kokoro.validate_tf_quantizations \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --val-filelist inputs/filelists/vocos.val.txt \
     --quant-output-dir output/tf_quant_streaming \
     --validation-output-dir output/tf_quant_streaming_eval
"""

from .tf.validate_quant import main


if __name__ == "__main__":
    main()
