"""
TensorFlow Keras implementation of custom STFT operations.
Converted from PyTorch implementation.
"""

from typing import Callable, Optional
import tensorflow as tf
import numpy as np

class TFLiteSTFT(tf.keras.layers.Layer):
    """
    TFLite-compatible STFT layer that avoids tf.signal.stft.
    
    Implements STFT using manual DFT computation with vectorized operations.
    Supports arbitrary fft_length values (not restricted to power of 2).
    
    Args:
        frame_length: Window length in samples
        frame_step: Hop length in samples
        fft_length: FFT size (can be any positive integer)
        window_fn: Window function (callable that takes size and returns window)
        pad_end: Whether to pad the end of the signal
    """
    
    def __init__(
        self,
        frame_length: int,
        frame_step: int,
        fft_length: Optional[int] = None,
        window_fn: Callable[[int, tf.dtypes.DType], tf.Tensor] = tf.signal.hann_window,
        pad_end: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length if fft_length is not None else frame_length
        self.window_fn = window_fn
        self.pad_end = pad_end
        
        # Pre-compute window
        self.window = None
        
        # Pre-compute DFT matrix for arbitrary FFT length
        self.dft_matrix = None
        
    def build(self, input_shape):
        # Create window tensor
        window = self.window_fn(self.frame_length, dtype=tf.float32)
        
        # Pad window to fft_length if needed
        if self.fft_length > self.frame_length:
            pad_amount = self.fft_length - self.frame_length
            window = tf.pad(window, [[0, pad_amount]], constant_values=0.0)
        
        self.window = tf.Variable(
            initial_value=window,
            trainable=False,
            dtype=tf.float32,
            name='window'
        )
        
        # Create DFT matrix: W[k,n] = exp(-2j * pi * k * n / N)
        # Shape: [fft_length, fft_length]
        n = tf.range(self.fft_length, dtype=tf.float32)
        k = tf.range(self.fft_length, dtype=tf.float32)
        
        # Broadcasting: [fft_length, 1] * [1, fft_length] -> [fft_length, fft_length]
        angles = -2.0 * np.pi * tf.reshape(k, [-1, 1]) * tf.reshape(n, [1, -1]) / self.fft_length
        
        # Create complex exponential
        dft_real = tf.math.cos(angles)
        dft_imag = tf.math.sin(angles)
        dft_matrix = tf.complex(dft_real, dft_imag)
        
        self.dft_matrix = tf.Variable(
            initial_value=dft_matrix,
            trainable=False,
            dtype=tf.complex64,
            name='dft_matrix'
        )
        
        super().build(input_shape)
    
    def call(self, signals):
        """
        Compute STFT using manual DFT.
        
        Args:
            signals: Input signal [batch, time]
            
        Returns:
            Complex STFT [batch, num_frames, fft_length // 2 + 1]
        """
        # Frame the signal
        frames = tf.signal.frame(
            signals,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            pad_end=self.pad_end
        )
        # frames shape: [batch, num_frames, frame_length]
        
        # Apply window
        windowed_frames = frames * self.window[:self.frame_length]
        
        # Pad to fft_length if needed
        if self.fft_length > self.frame_length:
            pad_amount = self.fft_length - self.frame_length
            windowed_frames = tf.pad(
                windowed_frames,
                [[0, 0], [0, 0], [0, pad_amount]],
                constant_values=0.0
            )
        
        # Convert to complex for DFT
        windowed_frames_complex = tf.cast(windowed_frames, tf.complex64)
        
        # Apply DFT: [batch, num_frames, fft_length] @ [fft_length, fft_length]
        # -> [batch, num_frames, fft_length]
        stft_full = tf.linalg.matmul(
            windowed_frames_complex,
            self.dft_matrix,
            transpose_b=True
        )
        
        # Keep only positive frequencies (first fft_length // 2 + 1 bins)
        num_spectrogram_bins = self.fft_length // 2 + 1
        stft_result = stft_full[:, :, :num_spectrogram_bins]
        
        return stft_result
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'frame_length': self.frame_length,
            'frame_step': self.frame_step,
            'fft_length': self.fft_length,
            'pad_end': self.pad_end,
        })
        return config


class TFLiteISTFT(tf.keras.layers.Layer):
    """
    TFLite-compatible ISTFT layer that avoids tf.signal.inverse_stft.
    
    Implements ISTFT using manual inverse DFT and overlap-add reconstruction.
    Supports arbitrary fft_length values (not restricted to power of 2).
    
    Note: This implementation uses a simplified windowing scheme that may differ
    slightly from tf.signal.inverse_stft's inverse_stft_window_fn. For perfect
    reconstruction, use matching STFT/ISTFT from this module.
    
    Args:
        frame_length: Window length in samples
        frame_step: Hop length in samples
        fft_length: FFT size (can be any positive integer)
        window_fn: Window function for overlap-add synthesis
    """
    
    def __init__(
        self,
        frame_length: int,
        frame_step: int,
        fft_length: Optional[int] = None,
        window_fn: Optional[Callable[[int, tf.dtypes.DType], tf.Tensor]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length if fft_length is not None else frame_length
        self.window_fn = window_fn
        
        # Pre-compute inverse DFT matrix and synthesis window
        self.idft_matrix = None
        self.synthesis_window = None
        
    def build(self, input_shape):
        # Create inverse DFT matrix
        # For inverse DFT: W[n,k] = exp(2j * pi * k * n / N)
        n = tf.range(self.fft_length, dtype=tf.float32)
        k = tf.range(self.fft_length, dtype=tf.float32)
        
        angles = 2.0 * np.pi * tf.reshape(k, [-1, 1]) * tf.reshape(n, [1, -1]) / self.fft_length
        
        idft_real = tf.math.cos(angles)
        idft_imag = tf.math.sin(angles)
        idft_matrix = tf.complex(idft_real, idft_imag)
        
        self.idft_matrix = tf.Variable(
            initial_value=idft_matrix,
            trainable=False,
            dtype=tf.complex64,
            name='idft_matrix'
        )
        
        # Compute synthesis window for overlap-add
        if self.window_fn is not None:
            window = self.window_fn(self.frame_length, dtype=tf.float32)
        else:
            # Use Hann window by default
            window = tf.signal.hann_window(self.frame_length, periodic=True, dtype=tf.float32)
        
        # Compute normalization factor for overlap-add
        # This ensures proper reconstruction when frames overlap
        # Calculate the sum of squared windows at each output position
        hop_ratio = self.frame_step / self.frame_length
        
        # For Hann window with 75% overlap (hop=frame/4), sum of squares = 1.5
        # For 50% overlap (hop=frame/2), sum of squares = 1.0
        # General formula: normalize by sqrt(1/hop_ratio) for proper energy
        normalization = tf.sqrt(1.0 / (1.0 - hop_ratio + 1e-8))
        normalized_window = window / normalization
        
        # Pad to fft_length if needed
        if self.fft_length > self.frame_length:
            pad_amount = self.fft_length - self.frame_length
            normalized_window = tf.pad(normalized_window, [[0, pad_amount]], constant_values=0.0)
        
        self.synthesis_window = tf.Variable(
            initial_value=normalized_window,
            trainable=False,
            dtype=tf.float32,
            name='synthesis_window'
        )
        
        super().build(input_shape)
        
        super().build(input_shape)
    
    def call(self, stfts):
        """
        Compute ISTFT using manual inverse DFT and overlap-add.
        
        Args:
            stfts: Complex STFT [batch, num_frames, num_bins]
            
        Returns:
            Reconstructed signal [batch, time]
        """
        # Reconstruct full spectrum by mirroring
        # Input is [batch, num_frames, fft_length // 2 + 1]
        num_bins = tf.shape(stfts)[-1]
        
        # Conjugate and reverse the bins (excluding DC and Nyquist)
        if self.fft_length % 2 == 0:
            # Even FFT length: mirror bins 1 to fft_length//2 - 1
            middle_bins = stfts[:, :, 1:-1]
        else:
            # Odd FFT length: mirror bins 1 to fft_length//2
            middle_bins = stfts[:, :, 1:]
        
        conjugate_bins = tf.reverse(tf.math.conj(middle_bins), axis=[-1])
        full_spectrum = tf.concat([stfts, conjugate_bins], axis=-1)
        
        # Apply inverse DFT (without additional 1/N in matrix - normalize after)
        # [batch, num_frames, fft_length] @ [fft_length, fft_length]
        frames = tf.linalg.matmul(full_spectrum, self.idft_matrix, transpose_b=True)
        
        # Normalize by fft_length (DFT normalization)
        frames = frames / tf.cast(self.fft_length, tf.complex64)
        
        # Take real part
        frames_real = tf.math.real(frames)
        
        # Apply synthesis window
        windowed_frames = frames_real * self.synthesis_window
        
        # Truncate to frame_length
        windowed_frames = windowed_frames[:, :, :self.frame_length]
        
        # Overlap-add reconstruction
        batch_size = tf.shape(windowed_frames)[0]
        num_frames = tf.shape(windowed_frames)[1]
        
        # Calculate output length
        output_length = (num_frames - 1) * self.frame_step + self.frame_length
        
        # Use tf.signal.overlap_and_add for the reconstruction
        reconstructed = tf.signal.overlap_and_add(
            windowed_frames,
            frame_step=self.frame_step
        )
        
        return reconstructed
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'frame_length': self.frame_length,
            'frame_step': self.frame_step,
            'fft_length': self.fft_length,
        })
        return config

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

        self.stft = TFLiteSTFT(fft_length=filter_length, frame_step=hop_length, frame_length=win_length)
        self.istft = TFLiteISTFT(fft_length=filter_length, frame_step=hop_length, frame_length=win_length)
        
        
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
        
        stft_result = self.stft(input_padded)
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
        reconstructed = self.istft( complex_spec )
        
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