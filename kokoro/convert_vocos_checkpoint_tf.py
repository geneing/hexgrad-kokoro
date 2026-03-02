"""Compatibility shim for moved streaming-Vocos PyTorch->TensorFlow conversion entrypoint.

New location: `kokoro.tf.convert`.
"""

from .tf.convert import main


if __name__ == "__main__":
    main()
