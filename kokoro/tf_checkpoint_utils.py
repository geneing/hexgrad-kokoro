"""Compatibility shim for moved TensorFlow checkpoint utilities.

New location: `kokoro.tf.checkpoint_utils`.
"""

from .tf.checkpoint_utils import *  # noqa: F401,F403
