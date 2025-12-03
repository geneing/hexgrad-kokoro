"""
TensorFlow Keras implementation of custom STFT operations.
Converted from PyTorch implementation.
"""

from typing import Callable, Optional
import tensorflow as tf
import numpy as np

class TFLiteSTFT(tf.keras.layers.Layer):
    """Real-valued STFT layer that avoids tf.signal.stft and complex tensors."""

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

        self.window = None
        self.dft_cos = None
        self.dft_sin = None

    def build(self, input_shape):
        window = self.window_fn(self.frame_length, dtype=tf.float32)
        if self.fft_length > self.frame_length:
            pad_amount = self.fft_length - self.frame_length
            window = tf.pad(window, [[0, pad_amount]], constant_values=0.0)

        self.window = tf.Variable(
            initial_value=window,
            trainable=False,
            dtype=tf.float32,
            name='window'
        )

        n = tf.range(self.fft_length, dtype=tf.float32)
        k = tf.range(self.fft_length, dtype=tf.float32)
        fft_len = tf.cast(self.fft_length, tf.float32)
        angles = -2.0 * np.pi * tf.reshape(k, [-1, 1]) * tf.reshape(n, [1, -1]) / fft_len
        self.dft_cos = tf.Variable(
            initial_value=tf.math.cos(angles),
            trainable=False,
            dtype=tf.float32,
            name='dft_cos'
        )
        self.dft_sin = tf.Variable(
            initial_value=tf.math.sin(angles),
            trainable=False,
            dtype=tf.float32,
            name='dft_sin'
        )

        super().build(input_shape)

    def _frame_signal(self, signals: tf.Tensor) -> tf.Tensor:
        signal_length = tf.shape(signals)[1]
        frame_length = tf.constant(self.frame_length, dtype=tf.int32)
        frame_step = tf.constant(self.frame_step, dtype=tf.int32)

        if self.pad_end:
            diff = signal_length - frame_length
            numerator = diff + frame_step - 1
            num_frames = tf.math.floordiv(numerator, frame_step) + 1
            num_frames = tf.maximum(num_frames, 1)
            total_length = (num_frames - 1) * frame_step + frame_length
            pad_amount = tf.maximum(total_length - signal_length, 0)
        else:
            diff = signal_length - frame_length
            clipped = tf.maximum(diff, 0)
            num_frames = tf.where(
                diff >= 0,
                tf.math.floordiv(clipped, frame_step) + 1,
                tf.constant(0, dtype=tf.int32)
            )
            pad_amount = 0

        padded = tf.pad(signals, [[0, 0], [0, pad_amount]])

        frame_starts = tf.range(num_frames, dtype=tf.int32) * frame_step
        frame_offsets = tf.range(frame_length, dtype=tf.int32)
        frame_indices = frame_starts[:, None] + frame_offsets[None, :]
        frames = tf.gather(padded, frame_indices, axis=1)
        return frames

    def call(self, signals):
        frames = self._frame_signal(signals)
        windowed_frames = frames * self.window[:self.frame_length]

        if self.fft_length > self.frame_length:
            pad_amount = self.fft_length - self.frame_length
            windowed_frames = tf.pad(
                windowed_frames,
                [[0, 0], [0, 0], [0, pad_amount]],
                constant_values=0.0
            )

        real_part = tf.linalg.matmul(
            windowed_frames,
            self.dft_cos,
            transpose_b=True
        )
        imag_part = tf.linalg.matmul(
            windowed_frames,
            self.dft_sin,
            transpose_b=True
        )

        num_spectrogram_bins = self.fft_length // 2 + 1
        real_part = real_part[:, :, :num_spectrogram_bins]
        imag_part = imag_part[:, :, :num_spectrogram_bins]
        return tf.stack([real_part, imag_part], axis=-1)

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
    """Inverse STFT implemented with real-valued arithmetic for TFLite."""

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

        self.idft_cos = None
        self.idft_sin = None
        self.synthesis_window = None

    def build(self, input_shape):
        n = tf.range(self.fft_length, dtype=tf.float32)
        k = tf.range(self.fft_length, dtype=tf.float32)
        fft_len = tf.cast(self.fft_length, tf.float32)
        angles = 2.0 * np.pi * tf.reshape(k, [-1, 1]) * tf.reshape(n, [1, -1]) / fft_len

        self.idft_cos = tf.Variable(
            initial_value=tf.math.cos(angles),
            trainable=False,
            dtype=tf.float32,
            name='idft_cos'
        )
        self.idft_sin = tf.Variable(
            initial_value=tf.math.sin(angles),
            trainable=False,
            dtype=tf.float32,
            name='idft_sin'
        )

        if self.window_fn is not None:
            window = self.window_fn(self.frame_length, dtype=tf.float32)
        else:
            window = tf.signal.hann_window(self.frame_length, periodic=True, dtype=tf.float32)

        hop_ratio = self.frame_step / self.frame_length
        normalization = tf.sqrt(1.0 / (1.0 - hop_ratio + 1e-8))
        normalized_window = window / normalization

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

    def call(self, stfts):
        real_part = stfts[..., 0]
        imag_part = stfts[..., 1]

        if self.fft_length % 2 == 0:
            mirror_real = real_part[:, :, 1:-1]
            mirror_imag = imag_part[:, :, 1:-1]
        else:
            mirror_real = real_part[:, :, 1:]
            mirror_imag = imag_part[:, :, 1:]

        reversed_real = tf.reverse(mirror_real, axis=[-1])
        reversed_imag = tf.reverse(mirror_imag, axis=[-1])
        full_real = tf.concat([real_part, reversed_real], axis=-1)
        full_imag = tf.concat([imag_part, -reversed_imag], axis=-1)

        time_real = tf.linalg.matmul(full_real, self.idft_cos, transpose_b=True)
        time_real -= tf.linalg.matmul(full_imag, self.idft_sin, transpose_b=True)
        time_real = time_real / tf.cast(self.fft_length, tf.float32)

        window = tf.reshape(self.synthesis_window, [1, 1, -1])
        windowed_frames = time_real * window
        windowed_frames = windowed_frames[:, :, :self.frame_length]

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
        stft_result = tf.transpose(stft_result, [0, 2, 1, 3])
        real_part = stft_result[..., 0]
        imag_part = stft_result[..., 1]
        magnitude = tf.sqrt(tf.maximum(real_part * real_part + imag_part * imag_part, 1e-12))
        phase = tf.math.atan2(imag_part, real_part)
        
        return magnitude, phase

    def inverse(self, magnitude, phase):
        """Inverse STFT using tf.signal.inverse_stft.
        
        CRITICAL: PyTorch uses different normalization than TensorFlow's default.
        PyTorch uses the same window for both directions with specific normalization.
        TensorFlow's inverse_stft_window_fn creates a different window for perfect reconstruction.
        
        To match PyTorch: we need to use the inverse_stft_window_fn which handles
        overlap-add normalization correctly.
        """
        real_part = magnitude * tf.math.cos(phase)
        imag_part = magnitude * tf.math.sin(phase)
        complex_spec = tf.stack([real_part, imag_part], axis=-1)
        complex_spec = tf.transpose(complex_spec, [0, 2, 1, 3])
        reconstructed = self.istft(complex_spec)
        
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
