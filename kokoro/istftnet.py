"""
TensorFlow Keras implementation of iSTFTNet decoder and related components.
Converted from PyTorch implementation with noted conversion issues.
"""

import tensorflow as tf
import numpy as np
import math
from typing import Optional, List


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for convolution to maintain size."""
    return int((kernel_size * dilation - dilation) / 2)


class AdaIN1d(tf.keras.layers.Layer):
    """Adaptive Instance Normalization for 1D convolutions."""
    
    def __init__(self, style_dim: int, num_features: int, **kwargs):
        super(AdaIN1d, self).__init__(**kwargs)
        self.style_dim = style_dim
        self.num_features = num_features
        
        # Note: TensorFlow doesn't have exact equivalent of PyTorch InstanceNorm1d
        # Using LayerNormalization as approximation - this is a conversion issue
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.fc = tf.keras.layers.Dense(num_features * 2)

    def call(self, x, s):
        # x: [batch, channels, time] or [batch, time, channels]
        # s: [batch, style_dim]
        
        h = self.fc(s)  # [batch, num_features*2]
        h = tf.expand_dims(h, axis=-1)  # [batch, num_features*2, 1]
        gamma, beta = tf.split(h, 2, axis=1)  # Each: [batch, num_features, 1]
        
        # Apply normalization - Note: different from PyTorch InstanceNorm1d
        normalized = self.norm(x)
        return (1 + gamma) * normalized + beta

    def get_config(self):
        config = super().get_config()
        config.update({
            'style_dim': self.style_dim,
            'num_features': self.num_features
        })
        return config


class AdaINResBlock1(tf.keras.layers.Layer):
    """Adaptive Instance Normalization Residual Block for 1D convolutions."""
    
    def __init__(self, channels: int, kernel_size: int = 3, 
                 dilation: tuple = (1, 3, 5), style_dim: int = 64, **kwargs):
        super(AdaINResBlock1, self).__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.style_dim = style_dim
        
        # First set of convolutions
        self.convs1 = []
        for d in dilation:
            # Note: TensorFlow Conv1D handles dilation differently than PyTorch
            # This might be a conversion issue for exact behavior matching
            conv = tf.keras.layers.Conv1D(
                filters=channels,
                kernel_size=kernel_size,
                dilation_rate=d,
                padding='same',
                use_bias=True,
                kernel_initializer='glorot_uniform'  # Approximate xavier_uniform
            )
            self.convs1.append(conv)
        
        # Second set of convolutions  
        self.convs2 = []
        for _ in dilation:
            conv = tf.keras.layers.Conv1D(
                filters=channels,
                kernel_size=kernel_size,
                dilation_rate=1,
                padding='same',
                use_bias=True,
                kernel_initializer='glorot_uniform'
            )
            self.convs2.append(conv)
        
        # Adaptive normalization layers
        self.adain1 = [AdaIN1d(style_dim, channels) for _ in dilation]
        self.adain2 = [AdaIN1d(style_dim, channels) for _ in dilation]
        
        # Alpha parameters for Snake activation - Note: different from PyTorch implementation
        self.alpha1 = []
        self.alpha2 = []
        for _ in dilation:
            alpha1 = self.add_weight(
                name=f'alpha1_{len(self.alpha1)}',
                shape=(1, channels, 1),
                initializer='ones',
                trainable=True
            )
            alpha2 = self.add_weight(
                name=f'alpha2_{len(self.alpha2)}', 
                shape=(1, channels, 1),
                initializer='ones',
                trainable=True
            )
            self.alpha1.append(alpha1)
            self.alpha2.append(alpha2)

    def call(self, x, s, training=None):
        # x: [batch, time, channels] or [batch, channels, time]
        # s: [batch, style_dim]
        
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2
        ):
            # First path
            xt = n1(x, s)
            # Snake activation - Note: Implementation differs from PyTorch
            xt = xt + (1 / a1) * (tf.sin(a1 * xt) ** 2)
            xt = c1(xt, training=training)
            
            # Second path
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (tf.sin(a2 * xt) ** 2)
            xt = c2(xt, training=training)
            
            # Residual connection
            x = xt + x
            
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'dilation': self.dilation,
            'style_dim': self.style_dim
        })
        return config


class CustomSTFTTF(tf.keras.layers.Layer):
    """
    TensorFlow implementation of custom STFT without complex operations.
    Note: This is a simplified version - full conversion would require more work.
    """
    
    def __init__(self, filter_length: int = 800, hop_length: int = 200,
                 win_length: int = 800, window: str = "hann", **kwargs):
        super(CustomSTFTTF, self).__init__(**kwargs)
        self.filter_length = filter_length
        self.hop_length = hop_length 
        self.win_length = win_length
        self.n_fft = filter_length
        
        assert window == 'hann', f"Only hann window supported, got {window}"
        
        # Create window - Note: TensorFlow doesn't have hann_window builtin
        # This is a conversion issue
        window_np = np.hanning(win_length).astype(np.float32)
        if win_length < self.n_fft:
            window_np = np.pad(window_np, (0, self.n_fft - win_length))
        elif win_length > self.n_fft:
            window_np = window_np[:self.n_fft]
            
        self.window = tf.constant(window_np)

    def call(self, x):
        """
        Forward STFT transform.
        Note: This is a placeholder implementation - full STFT requires more complex logic.
        """
        # Note: TensorFlow doesn't have direct equivalent of PyTorch STFT
        # This would require implementing STFT from scratch or using tf.signal.stft
        # This is a major conversion issue
        
        # Using tf.signal.stft as approximation
        stft_result = tf.signal.stft(
            x,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.filter_length,
            window_fn=tf.signal.hann_window
        )
        
        magnitude = tf.abs(stft_result)
        phase = tf.angle(stft_result)
        
        return magnitude, phase

    def inverse(self, magnitude, phase):
        """
        Inverse STFT transform.
        Note: Placeholder implementation with conversion issues.
        """
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
        
        # Add dimension to match PyTorch implementation
        return tf.expand_dims(reconstructed, axis=-2)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filter_length': self.filter_length,
            'hop_length': self.hop_length,
            'win_length': self.win_length
        })
        return config


class SineGen(tf.keras.layers.Layer):
    """Sine wave generator for neural vocoding."""
    
    def __init__(self, samp_rate: int, upsample_scale: int, harmonic_num: int = 0,
                 sine_amp: float = 0.1, noise_std: float = 0.003,
                 voiced_threshold: float = 0, flag_for_pulse: bool = False, **kwargs):
        super(SineGen, self).__init__(**kwargs)
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        """Generate unvoiced/voiced signal from F0."""
        uv = tf.cast(f0 > self.voiced_threshold, tf.float32)
        return uv

    def _f02sine(self, f0_values):
        """Convert F0 to sine waves with harmonics."""
        # f0_values: [batch, length, dim]
        
        # Convert to radians
        rad_values = (f0_values / self.sampling_rate) % 1
        
        # Initial phase noise
        batch_size = tf.shape(f0_values)[0]
        dim_size = tf.shape(f0_values)[2]
        rand_ini = tf.random.uniform([batch_size, dim_size], dtype=f0_values.dtype)
        
        # Set fundamental component phase to 0
        mask = tf.one_hot(0, dim_size, dtype=f0_values.dtype)
        rand_ini = rand_ini * (1 - mask)
        
        # Add initial phase
        rad_values = tf.concat([
            tf.expand_dims(rad_values[:, 0, :] + rand_ini, axis=1),
            rad_values[:, 1:, :]
        ], axis=1)
        
        if not self.flag_for_pulse:
            # Interpolation and phase computation - simplified version
            # Note: TensorFlow interpolation differs from PyTorch - conversion issue
            phase = tf.cumsum(rad_values, axis=1) * 2 * np.pi
            sines = tf.sin(phase)
        else:
            # Pulse train generation - complex logic simplified
            # This is a major conversion issue requiring careful implementation
            uv = self._f02uv(f0_values)
            i_phase = tf.cumsum(rad_values, axis=1)
            sines = tf.cos(i_phase * 2 * np.pi)
            
        return sines

    def call(self, f0):
        """Generate sine waves from F0."""
        # f0: [batch, length, 1]
        batch_size = tf.shape(f0)[0]
        length = tf.shape(f0)[1]
        
        # Create harmonic frequencies
        harmonic_range = tf.range(1, self.harmonic_num + 2, dtype=f0.dtype)
        harmonic_range = tf.reshape(harmonic_range, [1, 1, -1])
        fn = f0 * harmonic_range
        
        # Generate sine waves
        sine_waves = self._f02sine(fn) * self.sine_amp
        
        # Generate UV signal
        uv = self._f02uv(f0)
        
        return sine_waves, uv, None

    def get_config(self):
        config = super().get_config()
        config.update({
            'samp_rate': self.sampling_rate,
            'upsample_scale': self.upsample_scale,
            'harmonic_num': self.harmonic_num,
            'sine_amp': self.sine_amp,
            'noise_std': self.noise_std,
            'voiced_threshold': self.voiced_threshold,
            'flag_for_pulse': self.flag_for_pulse
        })
        return config


class SourceModuleHnNSF(tf.keras.layers.Layer):
    """Harmonic plus noise source module."""
    
    def __init__(self, sampling_rate: int, upsample_scale: int, 
                 harmonic_num: int = 0, voiced_threshod: float = 0, **kwargs):
        super(SourceModuleHnNSF, self).__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.upsample_scale = upsample_scale
        self.sine_amp = 0.1
        
        self.l_sin_gen = SineGen(
            sampling_rate, upsample_scale, harmonic_num,
            self.sine_amp, voiced_threshold=voiced_threshod
        )
        
        self.l_linear = tf.keras.layers.Dense(1)
        self.l_tanh = tf.keras.layers.Activation('tanh')

    def call(self, x):
        """Generate harmonic and noise components."""
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        
        # Generate noise
        noise = tf.random.normal(tf.shape(uv), dtype=uv.dtype) * self.sine_amp / 3
        
        return sine_merge, noise, uv

    def get_config(self):
        config = super().get_config()
        config.update({
            'sampling_rate': self.sampling_rate,
            'upsample_scale': self.upsample_scale
        })
        return config


class Generator(tf.keras.layers.Layer):
    """Main generator network for audio synthesis."""
    
    def __init__(self, style_dim: int, resblock_kernel_sizes: List[int],
                 upsample_rates: List[int], upsample_initial_channel: int,
                 resblock_dilation_sizes: List[List[int]], upsample_kernel_sizes: List[int],
                 gen_istft_n_fft: int, gen_istft_hop_size: int,
                 disable_complex: bool = False, **kwargs):
        super(Generator, self).__init__(**kwargs)
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.style_dim = style_dim
        
        # Source module
        self.m_source = SourceModuleHnNSF(
            sampling_rate=24000,
            upsample_scale=math.prod(upsample_rates) * gen_istft_hop_size,
            harmonic_num=8,
            voiced_threshod=10
        )
        
        # F0 upsampling - Note: TensorFlow UpSampling1D vs PyTorch Upsample difference
        self.f0_upsample_factor = math.prod(upsample_rates) * gen_istft_hop_size
        
        # Upsampling layers
        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # Note: TensorFlow Conv1DTranspose vs PyTorch ConvTranspose1d parameter differences
            up_layer = tf.keras.layers.Conv1DTranspose(
                filters=upsample_initial_channel // (2 ** (i + 1)),
                kernel_size=k,
                strides=u,
                padding='same',
                use_bias=True,
                kernel_initializer='glorot_uniform'
            )
            self.ups.append(up_layer)
        
        # Residual blocks
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                resblock = AdaINResBlock1(ch, k, tuple(d), style_dim)
                self.resblocks.append(resblock)
        
        # Noise convolutions
        self.noise_convs = []
        self.noise_res = []
        for i in range(len(upsample_rates)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                noise_conv = tf.keras.layers.Conv1D(
                    filters=c_cur,
                    kernel_size=stride_f0 * 2,
                    strides=stride_f0,
                    padding='same'
                )
                noise_res = AdaINResBlock1(c_cur, 7, (1, 3, 5), style_dim)
            else:
                noise_conv = tf.keras.layers.Conv1D(
                    filters=c_cur,
                    kernel_size=1,
                    padding='same'
                )
                noise_res = AdaINResBlock1(c_cur, 11, (1, 3, 5), style_dim)
            
            self.noise_convs.append(noise_conv)
            self.noise_res.append(noise_res)
        
        # Final convolution
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = tf.keras.layers.Conv1D(
            filters=self.post_n_fft + 2,
            kernel_size=7,
            padding='same',
            kernel_initializer='glorot_uniform'
        )
        
        # STFT layer
        self.stft = CustomSTFTTF(
            filter_length=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft
        )

    def call(self, x, s, f0, training=None):
        """Forward pass of generator."""
        # F0 processing
        f0_upsampled = tf.image.resize(
            tf.expand_dims(f0, axis=-1),
            [tf.shape(f0)[1] * self.f0_upsample_factor, 1],
            method='linear'
        )
        f0_upsampled = tf.squeeze(f0_upsampled, axis=-1)
        f0_transposed = tf.transpose(f0_upsampled, [0, 2, 1])
        
        # Generate harmonic source
        har_source, noi_source, uv = self.m_source(f0_transposed)
        har_source = tf.transpose(tf.squeeze(har_source, axis=1), [0, 2, 1])
        
        # STFT of harmonic source
        har_spec, har_phase = self.stft(har_source)
        har = tf.concat([har_spec, har_phase], axis=-1)
        
        # Upsampling and processing
        for i in range(self.num_upsamples):
            x = tf.nn.leaky_relu(x, alpha=0.1)
            
            # Process noise/harmonic source
            x_source = self.noise_convs[i](har, training=training)
            x_source = self.noise_res[i](x_source, s, training=training)
            
            # Upsample
            x = self.ups[i](x, training=training)
            
            # Reflection padding for last layer - Note: TensorFlow padding differs
            if i == self.num_upsamples - 1:
                x = tf.pad(x, [[0, 0], [1, 0], [0, 0]], mode='REFLECT')
            
            x = x + x_source
            
            # Apply residual blocks
            xs = None
            for j in range(self.num_kernels):
                block_idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[block_idx](x, s, training=training)
                else:
                    xs += self.resblocks[block_idx](x, s, training=training)
            
            x = xs / self.num_kernels
        
        # Final processing
        x = tf.nn.leaky_relu(x)
        x = self.conv_post(x, training=training)
        
        # Split into magnitude and phase
        spec = tf.exp(x[:, :, :self.post_n_fft // 2 + 1])
        phase = tf.sin(x[:, :, self.post_n_fft // 2 + 1:])
        
        return self.stft.inverse(spec, phase)

    def get_config(self):
        config = super().get_config()
        config.update({
            'style_dim': self.style_dim,
            'num_kernels': self.num_kernels,
            'num_upsamples': self.num_upsamples
        })
        return config


class AdainResBlk1d(tf.keras.layers.Layer):
    """Adaptive Instance Normalization Residual Block with upsampling."""
    
    def __init__(self, dim_in: int, dim_out: int, style_dim: int = 64,
                 actv: str = 'leaky_relu', upsample: str = 'none',
                 dropout_p: float = 0.0, **kwargs):
        super(AdainResBlk1d, self).__init__(**kwargs)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.style_dim = style_dim
        self.upsample_type = upsample
        self.learned_sc = dim_in != dim_out
        
        # Activation function
        if actv == 'leaky_relu':
            self.actv = tf.keras.layers.LeakyReLU(0.2)
        else:
            self.actv = tf.keras.layers.Activation(actv)
        
        # Upsampling
        if upsample == 'none':
            self.upsample = tf.keras.layers.Lambda(lambda x: x)
            self.pool = tf.keras.layers.Lambda(lambda x: x)
        else:
            self.upsample = tf.keras.layers.UpSampling1D(size=2)
            # Note: TensorFlow doesn't have exact equivalent of PyTorch ConvTranspose1d with groups
            # This is a conversion issue
            self.pool = tf.keras.layers.Conv1DTranspose(
                filters=dim_in,
                kernel_size=3,
                strides=2,
                padding='same',
                use_bias=True
            )
        
        # Build main layers
        self.conv1 = tf.keras.layers.Conv1D(dim_out, 3, padding='same')
        self.conv2 = tf.keras.layers.Conv1D(dim_out, 3, padding='same')
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        self.dropout = tf.keras.layers.Dropout(dropout_p)
        
        if self.learned_sc:
            self.conv1x1 = tf.keras.layers.Conv1D(dim_out, 1, use_bias=False)

    def _shortcut(self, x):
        """Shortcut connection with optional learned projection."""
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s, training=None):
        """Residual path through the block."""
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x, training=training))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x, training=training))
        return x

    def call(self, x, s, training=None):
        """Forward pass with shortcut and residual connections."""
        out = self._residual(x, s, training)
        shortcut = self._shortcut(x)
        
        # Note: tf.rsqrt for normalization factor
        return (out + shortcut) * tf.math.rsqrt(2.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim_in': self.dim_in,
            'dim_out': self.dim_out,
            'style_dim': self.style_dim,
            'upsample_type': self.upsample_type
        })
        return config


class Decoder(tf.keras.layers.Layer):
    """Main decoder network combining all components."""
    
    def __init__(self, dim_in: int, style_dim: int, dim_out: int,
                 resblock_kernel_sizes: List[int], upsample_rates: List[int],
                 upsample_initial_channel: int, resblock_dilation_sizes: List[List[int]],
                 upsample_kernel_sizes: List[int], gen_istft_n_fft: int,
                 gen_istft_hop_size: int, disable_complex: bool = False, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        
        self.dim_in = dim_in
        self.style_dim = style_dim
        
        # Encoder block
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        
        # Decoder blocks
        self.decode = []
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample='upsample'))
        
        # F0 and N processing
        self.f0_conv = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=3,
            strides=2,
            padding='same'
        )
        self.n_conv = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=3,
            strides=2,
            padding='same'
        )
        
        # ASR residual processing
        self.asr_res = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 1, padding='same')
        ])
        
        # Generator
        self.generator = Generator(
            style_dim=style_dim,
            resblock_kernel_sizes=resblock_kernel_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gen_istft_n_fft=gen_istft_n_fft,
            gen_istft_hop_size=gen_istft_hop_size,
            disable_complex=disable_complex
        )

    def call(self, asr, f0_curve, n, s, training=None):
        """Forward pass of the decoder."""
        # Process F0 and N
        f0 = self.f0_conv(tf.expand_dims(f0_curve, axis=-1), training=training)
        n_processed = self.n_conv(tf.expand_dims(n, axis=-1), training=training)
        
        # Initial encoding
        x = tf.concat([asr, f0, n_processed], axis=-1)
        x = self.encode(x, s, training=training)
        
        # ASR residual
        asr_res = self.asr_res(asr, training=training)
        
        # Decode through layers
        res = True
        for block in self.decode:
            if res:
                x = tf.concat([x, asr_res, f0, n_processed], axis=-1)
            x = block(x, s, training=training)
            if block.upsample_type != "none":
                res = False
        
        # Generate final audio
        x = self.generator(x, s, f0_curve, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim_in': self.dim_in,
            'style_dim': self.style_dim
        })
        return config
