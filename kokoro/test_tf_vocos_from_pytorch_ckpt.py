"""Compatibility shim for moved TensorFlow smoke test entrypoint.

New location: `kokoro.tf.smoke_test_from_pt_ckpt`.
"""

from .tf.smoke_test_from_pt_ckpt import main


if __name__ == "__main__":
    main()
