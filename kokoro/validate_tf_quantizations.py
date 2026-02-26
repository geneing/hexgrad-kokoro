"""Compatibility shim for moved TensorFlow quantization validation entrypoint.

New location: `kokoro.tf.validate_quant`.
"""

from .tf.validate_quant import main


if __name__ == "__main__":
    main()
