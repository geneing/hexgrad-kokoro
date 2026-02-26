"""Compatibility shim for moved TensorFlow LiteRT export pipeline.

New location: `kokoro.tf.export`.
"""

from .tf.export import main


if __name__ == "__main__":
    main()
