"""
TensorFlow Keras implementation of custom STFT operations.
Converted from PyTorch implementation.
"""

import tensorflow as tf
import numpy as np


class TorchSTFT(tf.keras.layers.Layer):
    """
    TensorFlow equivalent of PyTorch STFT using tf.signal.
    Simpler but may have different behavior than custom implementation.
    """

    def __init__(self, filter_length: int = 800, hop_length: int = 200, 
                 win_length: int = 800, window: str = 'hann', **kwargs):
        super(TorchSTFT, self).__init__(**kwargs)
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        
        assert window == 'hann', f"Only hann window supported, got {window}"
        
        # Create and store window tensor to match PyTorch behavior
        # PyTorch uses periodic=True for Hann window
        self.window = tf.signal.hann_window(win_length, periodic=True, dtype=tf.float32)

    def transform(self, input_data):
        """Forward STFT using tf.signal.stft.
        
        PyTorch torch.stft uses center=True by default, adding reflection padding.
        TensorFlow tf.signal.stft has no center parameter, so we add padding manually.
        """
        # Add reflection padding to match PyTorch center=True behavior
        # PyTorch pads by n_fft // 2 on each side
        pad_length = self.filter_length // 2
        input_padded = tf.pad(
            input_data,
            [[0, 0], [pad_length, pad_length]],
            mode='REFLECT'
        )
        
        stft_result = tf.signal.stft(
            input_padded,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.filter_length,
            window_fn=tf.signal.hann_window,
            pad_end=False
        )
        # Transpose to [batch, freq_bins, time_frames] to match PyTorch layout
        stft_result = tf.transpose(stft_result, [0, 2, 1])
        
        # Extract magnitude and phase
        magnitude = tf.math.abs(stft_result)
        phase = tf.math.angle(stft_result)
        
        return magnitude, phase

    def inverse(self, magnitude, phase):
        """Inverse STFT using tf.signal.inverse_stft.
        
        CRITICAL: PyTorch uses different normalization than TensorFlow's default.
        PyTorch uses the same window for both directions with specific normalization.
        TensorFlow's inverse_stft_window_fn creates a different window for perfect reconstruction.
        
        To match PyTorch: we need to use the inverse_stft_window_fn which handles
        overlap-add normalization correctly.
        """
        # Reconstruct complex spectrogram
        # Convert magnitude to complex first, then multiply
        magnitude_complex = tf.cast(magnitude, tf.complex64)
        complex_spec = magnitude_complex * tf.exp(tf.complex(tf.zeros_like(phase), phase))
        
        # Transpose to [batch, time_frames, freq_bins] for tf.signal.inverse_stft
        complex_spec = tf.transpose(complex_spec, [0, 2, 1])
        
        # Use inverse_stft_window_fn for proper overlap-add reconstruction
        # This should match PyTorch's internal normalization behavior
        reconstructed = tf.signal.inverse_stft(
            complex_spec,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.filter_length,
            window_fn=tf.signal.inverse_stft_window_fn(
                self.hop_length,
                forward_window_fn=tf.signal.hann_window
            )
        )
        
        # Remove padding added during forward transform (center=True behavior)
        # We added filter_length // 2 padding on each side
        pad_length = self.filter_length // 2
        reconstructed = reconstructed[:, pad_length:-pad_length]
        
        # Add dimension to stay consistent with conv_transpose1d implementation
        return tf.expand_dims(reconstructed, axis=-2)

    def call(self, input_data):
        """Forward pass for testing."""
        magnitude, phase = self.transform(input_data)
        reconstruction = self.inverse(magnitude, phase)
        return reconstruction

    def get_config(self):
        config = super().get_config()
        config.update({
            'filter_length': self.filter_length,
            'hop_length': self.hop_length,
            'win_length': self.win_length
        })
        return config
