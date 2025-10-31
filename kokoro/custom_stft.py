"""
TensorFlow Keras implementation of custom STFT operations.
Converted from PyTorch implementation.
"""

import tensorflow as tf
import numpy as np


class CustomSTFT(tf.keras.layers.Layer):
    """
    Custom STFT/iSTFT implementation using conv1d operations.
    Converted from PyTorch to avoid complex number operations for ONNX compatibility.
    
    Note: This is a complex conversion with potential accuracy differences from PyTorch.
    """

    def __init__(
        self,
        filter_length: int = 800,
        hop_length: int = 200,
        win_length: int = 800,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",  # Note: TensorFlow uses different padding modes
        **kwargs
    ):
        super(CustomSTFT, self).__init__(**kwargs)
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = filter_length
        self.center = center
        self.pad_mode = pad_mode

        # Number of frequency bins for real-valued STFT with onesided=True
        self.freq_bins = self.n_fft // 2 + 1

        # Build window
        assert window == 'hann', f"Only hann window supported, got {window}"
        
        # Create Hann window using numpy (TensorFlow doesn't have hann_window)
        window_np = np.hanning(win_length).astype(np.float32)
        if self.win_length < self.n_fft:
            # Zero-pad up to n_fft
            extra = self.n_fft - self.win_length
            window_np = np.pad(window_np, (0, extra))
        elif self.win_length > self.n_fft:
            window_np = window_np[:self.n_fft]
        
        # Register as non-trainable weight
        self.window = tf.Variable(
            window_np, trainable=False, name='window'
        )

        # Precompute forward DFT (real, imag)
        # Note: This is a complex conversion from PyTorch implementation
        self._build_dft_matrices()

    def _build_dft_matrices(self):
        """Build DFT transformation matrices for conv1d operations."""
        # Create DFT basis functions
        n = np.arange(self.n_fft, dtype=np.float32)
        k = n.reshape(-1, 1)
        
        # Real and imaginary parts of DFT matrix
        # e^(-j 2π kn/N) = cos(2π kn/N) - j sin(2π kn/N)
        angle = 2 * np.pi * k * n / self.n_fft
        
        # Only keep positive frequencies (onesided=True)
        angle = angle[:self.freq_bins, :]
        
        dft_real = np.cos(angle) * self.window.numpy()  # Real part
        dft_imag = -np.sin(angle) * self.window.numpy()  # Imaginary part
        
        # Reshape for conv1d: [out_channels, in_channels, kernel_size]
        # For TensorFlow: [kernel_size, in_channels, out_channels]
        dft_real = dft_real.T.reshape(self.n_fft, 1, self.freq_bins)
        dft_imag = dft_imag.T.reshape(self.n_fft, 1, self.freq_bins)
        
        self.dft_real = tf.Variable(dft_real, trainable=False, name='dft_real')
        self.dft_imag = tf.Variable(dft_imag, trainable=False, name='dft_imag')
        
        # For inverse transform
        # Note: Building inverse DFT matrices - complex conversion
        idft_real = np.cos(angle).T / self.freq_bins
        idft_imag = np.sin(angle).T / self.freq_bins
        
        # Handle DC and Nyquist components for real signals
        idft_real[:, 0] *= 0.5  # DC component
        idft_imag[:, 0] *= 0.5
        if self.n_fft % 2 == 0:  # Nyquist frequency exists
            idft_real[:, -1] *= 0.5
            idft_imag[:, -1] *= 0.5
        
        idft_real = idft_real.reshape(self.n_fft, self.freq_bins, 1)
        idft_imag = idft_imag.reshape(self.n_fft, self.freq_bins, 1)
        # Reorder to match tf.nn.conv1d_transpose filter layout [kernel, out_channels, in_channels]
        idft_real = np.transpose(idft_real, (0, 2, 1))
        idft_imag = np.transpose(idft_imag, (0, 2, 1))

        self.idft_real = tf.Variable(idft_real, trainable=False, name='idft_real')
        self.idft_imag = tf.Variable(idft_imag, trainable=False, name='idft_imag')

    def transform(self, input_data):
        """
        Forward STFT transform using conv1d operations.
        
        Args:
            input_data: [batch, time] audio tensor
            
        Returns:
            magnitude: [batch, freq_bins, time_frames] 
            phase: [batch, freq_bins, time_frames]
        """
        # Padding for center=True
        if self.center:
            pad_length = self.n_fft // 2
            # Note: TensorFlow padding mode differences from PyTorch
            if self.pad_mode == "reflect":
                # TensorFlow reflect padding
                input_data = tf.pad(
                    input_data, [[0, 0], [pad_length, pad_length]], 
                    mode='REFLECT'
                )
            else:  # constant padding
                input_data = tf.pad(
                    input_data, [[0, 0], [pad_length, pad_length]], 
                    mode='CONSTANT'
                )

        # Add channel dimension for conv1d: [batch, time] -> [batch, time, 1]
        input_data = tf.expand_dims(input_data, axis=1)
        
        # Apply STFT using conv1d
        # Note: TensorFlow conv1d parameter order differs from PyTorch
        stft_real = tf.nn.conv1d(
            input_data, 
            self.dft_real, 
            stride=self.hop_length, 
            padding='VALID', 
            data_format='NCW'
        )
        stft_imag = tf.nn.conv1d(
            input_data, 
            self.dft_imag, 
            stride=self.hop_length, 
            padding='VALID', 
            data_format='NCW'
        )
        
        # Transpose to [batch, freq_bins, time_frames]
        # stft_real = tf.transpose(stft_real, [0, 2, 1])
        # stft_imag = tf.transpose(stft_imag, [0, 2, 1])
        
        # Compute magnitude and phase
        magnitude = tf.sqrt(stft_real**2 + stft_imag**2)
        phase = tf.atan2(stft_imag, stft_real)
        correction_mask = tf.logical_and(tf.equal(stft_imag, 0.0), stft_real < 0.0)
        phase = tf.where(correction_mask, tf.cast(np.pi, phase.dtype), phase)
        return magnitude, phase

    def inverse(self, magnitude, phase):
        """
        Inverse STFT transform using conv_transpose1d operations.
        
        Args:
            magnitude: [batch, freq_bins, time_frames]
            phase: [batch, freq_bins, time_frames]
            
        Returns:
            reconstructed: [batch, 1, time] audio tensor
        """
        # Reconstruct real and imaginary parts
        stft_real = magnitude * tf.cos(phase)
        stft_imag = magnitude * tf.sin(phase)
 
        # Transpose for conv operations: [batch, time_frames, freq_bins]
        stft_real = tf.transpose(stft_real, [0, 2, 1])
        stft_imag = tf.transpose(stft_imag, [0, 2, 1])
        print(f"++++++{stft_real.shape=}, {stft_imag.shape=}")
               
        # Inverse transform using conv_transpose1d
        # Note: TensorFlow conv1d_transpose vs PyTorch conv_transpose1d differences
        batch = tf.shape(stft_real)[0]
        frames = tf.shape(stft_real)[1]
        out_width = (frames - 1) * self.hop_length + self.n_fft
        output_shape = tf.stack([batch, out_width, tf.constant(1, dtype=tf.int32)])

        istft_real = tf.nn.conv1d_transpose(
            stft_real,
            self.idft_real,
            output_shape=output_shape,
            strides=self.hop_length,
            padding='VALID', 
            data_format='NWC'
        )
        istft_imag = tf.nn.conv1d_transpose(
            stft_imag,
            self.idft_imag,
            output_shape=output_shape,
            strides=self.hop_length,
            padding='VALID',
            data_format='NWC'	
        )
        
        # Combine real and imaginary parts
        reconstructed = istft_real - istft_imag  # Note: Sign convention
        
        # Remove center padding if applied
        if self.center:
            pad_length = self.n_fft // 2
            reconstructed = reconstructed[:, pad_length:-pad_length, :]
        
        # Transpose to match expected output format
        return tf.transpose(reconstructed, [0, 2, 1])

    def call(self, input_data):
        """Forward pass for testing purposes."""
        magnitude, phase = self.transform(input_data)
        reconstruction = self.inverse(magnitude, phase)
        return reconstruction

    def get_config(self):
        config = super().get_config()
        config.update({
            'filter_length': self.filter_length,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'center': self.center,
            'pad_mode': self.pad_mode
        })
        return config


class TorchSTFTTF(tf.keras.layers.Layer):
    """
    TensorFlow equivalent of PyTorch STFT using tf.signal.
    Simpler but may have different behavior than custom implementation.
    """

    def __init__(self, filter_length: int = 800, hop_length: int = 200, 
                 win_length: int = 800, window: str = 'hann', **kwargs):
        super(TorchSTFTTF, self).__init__(**kwargs)
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        
        assert window == 'hann', f"Only hann window supported, got {window}"

    def transform(self, input_data):
        """Forward STFT using tf.signal.stft."""
        # Note: TensorFlow STFT parameters differ from PyTorch
        stft_result = tf.signal.stft(
            input_data,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.filter_length,
            window_fn=tf.signal.hann_window,
            pad_end=True
        )
        
        magnitude = tf.abs(stft_result)
        phase = tf.angle(stft_result)
        
        return magnitude, phase

    def inverse(self, magnitude, phase):
        """Inverse STFT using tf.signal.inverse_stft."""
        # Reconstruct complex spectrogram
        complex_spec = magnitude * tf.exp(1j * phase)
        
        # Inverse STFT
        reconstructed = tf.signal.inverse_stft(
            complex_spec,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.filter_length,
            window_fn=tf.signal.hann_window
        )
        
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
