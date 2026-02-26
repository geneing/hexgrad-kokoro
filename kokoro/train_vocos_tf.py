"""Compatibility shim for moved TensorFlow training entrypoint.

New location: `kokoro.tf.training`.
"""

from .tf.training import main


if __name__ == "__main__":
    main()
