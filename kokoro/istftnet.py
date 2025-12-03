"""
TensorFlow Keras implementation of iSTFTNet decoder and related components.
Converted from PyTorch implementation with noted conversion issues.
"""
from typing import Optional, List
# from sympy import use
from tarfile import data_filter
from huggingface_hub import dataset_info
import numpy as np
import math
from regex import F
from sympy import primefactors
import tensorflow as tf
from .custom_stft import TorchSTFT


class PointwiseConv1D(tf.keras.layers.Layer):
    """1x1 convolution implemented with matrix multiplications for TFLite."""

    def __init__(self, filters: int, use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.use_bias = use_bias

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        in_channels = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(1, in_channels, self.filters),
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs):
        kernel = tf.squeeze(self.kernel, axis=0)
        output = tf.linalg.matmul(inputs, kernel)
        if self.bias is not None:
            output = tf.nn.bias_add(output, self.bias)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'use_bias': self.use_bias,
        })
        return config


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for convolution to maintain size."""
    return int((kernel_size * dilation - dilation) / 2)


def _resize_1d_linear(x: tf.Tensor, scale: float | tf.Tensor) -> tf.Tensor:
    """Linear interpolation for 1D sequences using TensorFlow to mirror torch.nn.functional.interpolate."""
    x = tf.convert_to_tensor(x)
    scale_tensor = tf.cast(scale, tf.float32)
    length = tf.shape(x)[1]
    length_float = tf.cast(length, tf.float32)
    new_length_float = tf.math.maximum(length_float * scale_tensor, 1.0)
    new_length = tf.cast(tf.math.floor(new_length_float), tf.int32)

    # tf.image.resize expects shape [B, H, W, C]; treat the time axis as height.
    x_expanded = tf.expand_dims(x, axis=2)
    resized = tf.image.resize(
        x_expanded,
        size=tf.stack([new_length, tf.constant(1, dtype=tf.int32)]),
        method="bilinear",
        antialias=False,
        preserve_aspect_ratio=False,
    )
    return tf.squeeze(resized, axis=2)


class InstanceNorm1D(tf.keras.layers.Layer):
    """Minimal InstanceNorm1d analogue that keeps channel semantics explicit."""

    def __init__(self, num_features: int, epsilon: float = 1e-5, channel_axis: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.epsilon = epsilon
        self.channel_axis = channel_axis if channel_axis != -1 else 2

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.num_features,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(self.num_features,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if x.shape.rank != 3:
            raise ValueError("InstanceNorm1D expects rank-3 inputs [B, C, T] or [B, T, C]")

        if self.channel_axis == 1:
            reduce_axes = [2]
            broadcast_shape = (1, self.num_features, 1)
        elif self.channel_axis == 2:
            reduce_axes = [1]
            broadcast_shape = (1, 1, self.num_features)
        else:
            raise ValueError(f"Unsupported channel axis {self.channel_axis} for InstanceNorm1D")

        mean = tf.reduce_mean(x, axis=reduce_axes, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=reduce_axes, keepdims=True)
        inv_std = tf.math.rsqrt(var + self.epsilon)

        gamma = tf.reshape(tf.cast(self.gamma, x.dtype), broadcast_shape)
        beta = tf.reshape(tf.cast(self.beta, x.dtype), broadcast_shape)

        return (x - mean) * inv_std * gamma + beta

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_features": self.num_features,
                "epsilon": self.epsilon,
                "channel_axis": self.channel_axis,
            }
        )
        return config


class AdaIN1d(tf.keras.layers.Layer):
    """Adaptive Instance Normalization for 1D convolutions."""
    
    def __init__(self, style_dim: int, num_features: int, **kwargs):
        super(AdaIN1d, self).__init__(**kwargs)
        self.style_dim = style_dim
        self.num_features = num_features
        self.norm = InstanceNorm1D(num_features, channel_axis=1)
        self.fc = tf.keras.layers.Dense(num_features * 2)

    def call(self, x, s):
        # x: [batch, channels, time] or [batch, time, channels]
        # s: [batch, style_dim]
        h = self.fc(s)  # [batch, num_features*2]        
        h = tf.expand_dims(h, axis=-1)  # [batch, num_features*2, 1]
        gamma, beta = tf.split(h, 2, axis=1)  # Each: [batch, num_features, 1]
        # Ensure channel-first layout for normalization when possible.
        transpose_back = False
        if x.shape.rank == 3 and x.shape[1] != self.num_features and x.shape[-1] == self.num_features:
            x = tf.transpose(x, [0, 2, 1])
            transpose_back = True

        normalized = self.norm(x)
        output = (1 + gamma) * normalized + beta
        if transpose_back:
            output = tf.transpose(output, [0, 2, 1])
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'style_dim': self.style_dim,
            'num_features': self.num_features
        })
        return config


class AdaINResBlock1(tf.keras.layers.Layer):
    """TensorFlow port of the PyTorch AdaINResBlock1.

    Conversion notes / potential mismatches:
    - PyTorch uses `weight_norm` on every Conv1d; Keras Conv1D here omits weight norm. Expect different statistics
      unless weight normalization is re-applied manually via custom wrappers.
        - Instance normalization uses a hand-rolled variant (`InstanceNorm1D`) to keep channel-first semantics while
            supporting dynamic shapes for TFLite tracing.
    - PyTorch evaluates the Snake1D activation with learnable `alpha` parameters; we recreate it with
      `tf.math.divide_no_nan` to avoid division by zero, but numerical behavior can still diverge.
    - All convolutions operate on channel-last internally, requiring explicit transposes around each call.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5), style_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.style_dim = style_dim

        # First convolution stack (three dilated convs)
        self.convs1: list[tf.keras.layers.Layer] = []
        for idx, dil in enumerate(dilation):
            # NOTE: weight norm is not reproduced here; expect discrepancies vs PyTorch.
            padding = get_padding(kernel_size, dil)
            conv = tf.keras.layers.Conv1D(
                filters=channels,
                kernel_size=kernel_size,
                dilation_rate=dil,
                padding='valid',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                name=f"adain_resblock1_conv1_{idx}"
            )
            setattr(conv, "_manual_padding", padding)
            self.convs1.append(conv)

        # Second convolution stack (three standard convs)
        self.convs2: list[tf.keras.layers.Layer] = []
        for idx in range(len(dilation)):
            padding = get_padding(kernel_size, 1)
            conv = tf.keras.layers.Conv1D(
                filters=channels,
                kernel_size=kernel_size,
                dilation_rate=1,
                padding='valid',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                name=f"adain_resblock1_conv2_{idx}"
            )
            setattr(conv, "_manual_padding", padding)
            self.convs2.append(conv)

        # Adaptive instance normalization layers for each branch
        self.adain1 = [AdaIN1d(style_dim, channels) for _ in range(len(dilation))]
        self.adain2 = [AdaIN1d(style_dim, channels) for _ in range(len(dilation))]

        # Learnable Snake activation coefficients (alpha parameters)
        self.alpha1 = [
            self.add_weight(
                name=f"alpha1_{idx}",
                shape=(1, channels, 1),
                initializer='ones',
                trainable=True
            )
            for idx in range(len(dilation))
        ]
        self.alpha2 = [
            self.add_weight(
                name=f"alpha2_{idx}",
                shape=(1, channels, 1),
                initializer='ones',
                trainable=True
            )
            for idx in range(len(dilation))
        ]

    def _apply_conv(self, x, conv_layer: tf.keras.layers.Layer, training: bool = False):
        """Apply Conv1D assuming channel-first input shape [B, C, T]."""
        x_perm = tf.transpose(x, [0, 2, 1])
        pad = getattr(conv_layer, "_manual_padding", 0)
        if pad:
            x_perm = tf.pad(x_perm, [[0, 0], [pad, pad], [0, 0]])
        x_perm = conv_layer(x_perm, training=training)
        return tf.transpose(x_perm, [0, 2, 1])

    @staticmethod
    def _snake_activation(x, alpha):
        """Snake1D activation: x + (1/alpha) * sin(alpha * x)^2.

        `tf.math.divide_no_nan` guards against alpha approaching zero, though divergence may still
        occur if alpha learns extremely small magnitudes.
        """
        sin_term = tf.math.sin(alpha * x)
        sin_sq = tf.math.square(sin_term)
        return x + tf.math.divide_no_nan(sin_sq, alpha)

    def call(self, x, s, training: bool = False):
        out = x

        for i, (conv1, conv2, n1, n2, a1, a2) in enumerate(zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2)):
            xt = n1(out, s)
            xt = self._snake_activation(xt, a1)
            xt = self._apply_conv(xt, conv1, training=training)
            xt = n2(xt, s)
            xt = self._snake_activation(xt, a2)
            xt = self._apply_conv(xt, conv2, training=training)
            out = out + xt  # Residual connection per branch

        return out


class AdainResBlk1d(tf.keras.layers.Layer):
    """TensorFlow/Keras version of AdainResBlk1d.

    Differences / potential mismatches vs PyTorch:
    - Uses UpSampling1D + Conv1D instead of ConvTranspose1d depthwise for pooling/upsample.
    - Normalization uses AdaIN1d above (custom InstanceNorm-backed) instead of PyTorch's InstanceNorm1d with weight norm.
    - Channel ordering conversions performed (PyTorch conv expects [B,C,T], Keras Conv1D expects [B,T,C]).
    - Dropout placement approximate; may not exactly match training-time semantics.
    - Scaling by 1/sqrt(2) implemented via tf.math.rsqrt(tf.constant(2.0)).
    """
    def __init__(self, dim_in, dim_out, style_dim=64, actv=None, upsample='none', dropout_p=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.style_dim = style_dim
        self.upsample_type = upsample
        self.learned_sc = dim_in != dim_out
        self.actv = actv or tf.keras.layers.LeakyReLU(0.2)
        # Upsample: approximate replacement for ConvTranspose1d depthwise + weight_norm
        if self.upsample_type != 'none':
            self.upsample = tf.keras.layers.UpSampling1D(size=2)  # NOTE: Nearest-neighbor; differs from transposed conv
        else:
            self.upsample = None
        # Convolutions (channel-last internally)
        self.conv1 = tf.keras.layers.Conv1D(dim_out, 3, padding='same')  # NOTE: no weight_norm
        self.conv2 = tf.keras.layers.Conv1D(dim_out, 3, padding='same')  # NOTE: no weight_norm
        if self.learned_sc:
            self.conv1x1 = PointwiseConv1D(dim_out, use_bias=False, name=f"{self.name}_pw" if self.name else None)
        else:
            self.conv1x1 = None
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        self.dropout = tf.keras.layers.Dropout(dropout_p)

    def _apply_conv(self, x, conv, training=False):
        # x: [B, C, T] -> transpose -> conv -> transpose back
        x_perm = tf.transpose(x, [0, 2, 1])  # [B, T, C]
        x_perm = conv(x_perm, training=training)
        return tf.transpose(x_perm, [0, 2, 1])

    def _shortcut(self, x, training=False):
        y = x
        if self.upsample is not None:
            # Upsample along time dimension (channel-first -> channel-last -> upsample -> back)
            y_perm = tf.transpose(y, [0, 2, 1])
            y_perm = self.upsample(y_perm)
            y = tf.transpose(y_perm, [0, 2, 1])
        if self.learned_sc:
            y_perm = tf.transpose(y, [0, 2, 1])
            y_perm = self.conv1x1(y_perm, training=training)
            y = tf.transpose(y_perm, [0, 2, 1])
        return y

    def _residual(self, x, s, training=False):
        h = self.norm1(x, s)  # NOTE: normalization semantics differ from PyTorch InstanceNorm1d
        h = self.actv(h)
        if self.upsample is not None:
            # PyTorch used depthwise ConvTranspose1d after activation; approximated with UpSampling1D
            h_perm = tf.transpose(h, [0, 2, 1])
            h_perm = self.upsample(h_perm)
            h = tf.transpose(h_perm, [0, 2, 1])
        # Conv1
        h = self._apply_conv(self.dropout(h, training=training), self.conv1, training=training)
        h = self.norm2(h, s)  # NOTE: second AdaIN; semantics mismatch possible
        h = self.actv(h)
        # Conv2
        h = self._apply_conv(self.dropout(h, training=training), self.conv2, training=training)
        return h

    def call(self, x, s, training=False):
        out = self._residual(x, s, training=training)
        sc = self._shortcut(x, training=training)
        # Scale by 1/sqrt(2) for residual variance preservation
        out = (out + sc) * tf.math.rsqrt(tf.constant(2.0, dtype=out.dtype))
        return out


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
        
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * torch.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1
        
        # Initial phase noise
        batch_size = tf.shape(f0_values)[0]
        dim_size = tf.shape(f0_values)[2]
        rand_ini = tf.random.uniform([batch_size, dim_size], dtype=f0_values.dtype)
        
        # Set fundamental component phase to 0
        rand_ini = tf.concat([tf.zeros_like(rand_ini[:, :1]), rand_ini[:, 1:]], axis=1)
        
        # Add initial phase
        rad_values = tf.concat([
            tf.expand_dims(rad_values[:, 0, :] + rand_ini, axis=1),
            rad_values[:, 1:, :]
        ], axis=1)
        
        if not self.flag_for_pulse:
           
            down_scale = 1.0 / self.upsample_scale
            # Mirror torch.nn.functional.interpolate sequence (downsample -> cumsum -> upsample)
            rad_down = _resize_1d_linear(rad_values, down_scale)

            phase = tf.math.cumsum(rad_down, axis=1) * tf.constant(2.0 * np.pi, dtype=f0_values.dtype)
            phase = _resize_1d_linear(phase * self.upsample_scale, self.upsample_scale)
           
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
                 upsample_rates: list[int], upsample_initial_channel: int,
                 resblock_dilation_sizes: list[List[int]], upsample_kernel_sizes: List[int],
                 gen_istft_n_fft: int, gen_istft_hop_size: int,
                 disable_complex: bool = False, **kwargs):
        super(Generator, self).__init__(**kwargs)
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.style_dim = style_dim
        
        self.f0_upsample_factor = math.prod(upsample_rates) * gen_istft_hop_size
        
        # Source module
        self.m_source = SourceModuleHnNSF(
            sampling_rate=24000,
            upsample_scale=self.f0_upsample_factor,
            harmonic_num=8,
            voiced_threshod=10
        )
        
        # F0 upsampling - Note: TensorFlow UpSampling1D vs PyTorch Upsample difference
        
        self.f0_upsample = tf.keras.layers.UpSampling1D(size=self.f0_upsample_factor)
        
        # Upsampling layers - use channel-last for TFLite
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
        self.noise_conv_pads: list[int] = []
        print(f"{upsample_rates=} {len(upsample_rates)=}")
        for i in range(len(upsample_rates)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                pad = (stride_f0 + 1) // 2
                noise_conv = tf.keras.layers.Conv1D(
                    filters=c_cur,
                    kernel_size=stride_f0 * 2,
                    strides=stride_f0,
                    padding='valid'
                )
                noise_res = AdaINResBlock1(c_cur, 7, (1, 3, 5), style_dim)
                self.noise_conv_pads.append(pad)
            else:
                noise_conv = tf.keras.layers.Conv1D(
                    filters=c_cur,
                    kernel_size=1,
                    padding='same'
                )
                noise_res = AdaINResBlock1(c_cur, 11, (1, 3, 5), style_dim)
                self.noise_conv_pads.append(0)
            
            self.noise_convs.append(noise_conv)
            self.noise_res.append(noise_res)
        
        # Final convolution - use channel-last for TFLite
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = tf.keras.layers.Conv1D(
            filters=self.post_n_fft + 2,
            kernel_size=7,
            padding='same',
            kernel_initializer='glorot_uniform'
        )
        
        # STFT layer - using TorchSTFT for accurate complex STFT (requires Flex delegate)
        # CustomSTFT alternative exists but is not accurate enough for production
        # self.stft = CustomSTFT(
        #     filter_length=gen_istft_n_fft,
        #     hop_length=gen_istft_hop_size,
        #     win_length=gen_istft_n_fft
        # )
        
        self.stft = TorchSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)
        

    def call(self, x, s, f0, training=None):
        """Forward pass of generator. x is [B,C,T] channel-first."""
        # F0 processing
        f0_upsampled = self.f0_upsample(
            tf.expand_dims(f0, axis=-1)) # [B, T, 1]
                
        # Generate harmonic source
        har_source, noi_source, uv = self.m_source(f0_upsampled)
        har_source = tf.squeeze(tf.transpose(har_source, [0, 2, 1]), axis=1) 
        # STFT of harmonic source

        har_spec, har_phase = self.stft.transform(har_source)
        har = tf.concat([har_spec, har_phase], axis=1)  # [B,C,T] channel-first
        
        # Upsampling and processing
        for i in range(self.num_upsamples):
            x = tf.nn.leaky_relu(x, alpha=0.1)

            # Process noise/harmonic source - har is [B,C,T]
            pad = self.noise_conv_pads[i]
            if pad:
                har_input = tf.pad(har, [[0, 0], [0, 0], [pad, pad]])
            else:
                har_input = har
            # noise_conv expects [B,T,C]
            har_input = tf.transpose(har_input, [0, 2, 1])
            x_source = self.noise_convs[i](har_input, training=training)  # [B,T',C]
            x_source = tf.transpose(x_source, [0, 2, 1])  # [B,C,T']
            x_source = self.noise_res[i](x_source, s, training=training)  # [B,C,T']
            
            # Upsample - x is [B,C,T], ups expects [B,T,C]
            x = tf.transpose(x, [0, 2, 1])
            x = self.ups[i](x, training=training)  # [B,T',C]
            x = tf.transpose(x, [0, 2, 1])  # [B,C,T']
            
            # Reflection padding on final stage to mirror PyTorch behavior
            if i == self.num_upsamples - 1:
                x = tf.pad(x, [[0, 0], [0, 0], [1, 0]], mode='REFLECT')
            
            x = x + x_source
            
            # Apply residual blocks - expect [B,C,T]
            xs = None
            for j in range(self.num_kernels):
                block_idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[block_idx](x, s, training=training)
                else:
                    xs = xs + self.resblocks[block_idx](x, s, training=training)

            x = xs / self.num_kernels
        
        # Final processing - x is [B,C,T]
        x = tf.nn.leaky_relu(x, alpha=0.01)
        # conv_post expects [B,T,C]
        x = tf.transpose(x, [0, 2, 1])
        x = self.conv_post(x, training=training)  # [B,T,C]
        x = tf.transpose(x, [0, 2, 1])  # [B,C,T]
        
        # Split into magnitude and phase
        spec = tf.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = tf.sin(x[:, self.post_n_fft // 2 + 1:, :])
        return self.stft.inverse(spec, phase)

    def get_config(self):
        config = super().get_config()
        config.update({
            'style_dim': self.style_dim,
            'num_kernels': self.num_kernels,
            'num_upsamples': self.num_upsamples
        })
        return config

class UpSample1d(tf.keras.layers.Layer):
    """
    TensorFlow/Keras equivalent of PyTorch UpSample1d.

    PyTorch behavior:
        If layer_type != 'none': F.interpolate(x, scale_factor=2, mode='nearest') on [B,C,T].

    Notes / possible mismatches:
    - Uses tf.repeat along the time axis (last dim) to emulate nearest-neighbor.
    - Keeps channel-first layout [B,C,T] (avoids transpose cost).
    - If upstream code switches to channel-last, add transpose externally.
    - No align_corners flag (not applicable for nearest).
    """
    def __init__(self, layer_type: str, **kwargs):
        super().__init__(**kwargs)
        self.layer_type = layer_type

    def call(self, x: tf.Tensor, training=False):
        if self.layer_type == 'none':
            return x
        return tf.repeat(x, repeats=2, axis=-1)  # NOTE: Nearest-neighbor emulation; shape [B,C,2*T]

import tensorflow as tf

# TensorFlow Keras Conv1DTranspose equivalent
class DepthwiseConv1DTranspose(tf.keras.layers.Layer):
    def __init__(self, kernel_size, strides=1, padding="same", use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.use_bias = use_bias
        
    def build(self, input_shape):
        input_shape = [input_shape[0], input_shape[2], input_shape[1]]  # [B, T, C]
        channels = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_size, channels, channels),
            initializer="zeros",
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(channels,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)
        
    def call(self, inputs_in):
        inputs = tf.transpose(inputs_in, [0, 2, 1])  # [B, T, C]
        inputs = tf.concat([inputs, tf.zeros_like(inputs[:, :1, :])], axis=1)      
        batch = tf.shape(inputs)[0]
        width = tf.shape(inputs)[1]
        channels = tf.shape(inputs)[2]
        output_width = width * self.strides
        output_shape = tf.stack([batch, output_width, channels])
        outputs = tf.nn.conv1d_transpose(
            inputs,
            self.kernel,
            output_shape=output_shape,
            strides=[1, self.strides, 1],
            padding=self.padding,
            data_format="NWC",
        )
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format="NWC")
        outputs = tf.transpose(outputs, [0, 2, 1])  # [B, C, T]
        return outputs[:,:,1:-1]
    
    
# Example: PyTorch ConvTranspose1d(in_ch, out_ch, k, s, padding, groups=in_ch)
# becomes:
# layer = ConvTranspose1dGrouped(out_channels=out_ch, kernel_size=k, strides=s, padding="same", groups=in_ch)

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
        self.upsample = UpSample1d(upsample)
        
        if upsample == 'none':
            self.pool = tf.keras.layers.Lambda(lambda x: x)
        else:
            # Note: TensorFlow doesn't have exact equivalent of PyTorch ConvTranspose1d with groups
            # This is a conversion issue
            self.pool = DepthwiseConv1DTranspose(
                kernel_size=3,
                strides=2,
                padding='same',
                use_bias=True
            )
        
        # Build main layers - use channel-last Conv1D for TFLite compatibility
        self.conv1 = tf.keras.layers.Conv1D(dim_out, 3, padding='same')
        self.conv2 = tf.keras.layers.Conv1D(dim_out, 3, padding='same')
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        self.dropout = tf.keras.layers.Dropout(dropout_p)
        
        if self.learned_sc:
            self.conv1x1 = PointwiseConv1D(dim_out, use_bias=False, name=f"{self.name}_pw" if self.name else None)

    def _shortcut(self, x):
        """Shortcut connection with optional learned projection."""
        x = self.upsample(x)
        if self.learned_sc:
            # x: [B,C,T] -> [B,T,C] -> conv1x1 -> [B,T,C] -> [B,C,T]
            x = tf.transpose(x, [0, 2, 1])
            x = self.conv1x1(x)
            x = tf.transpose(x, [0, 2, 1])
        return x

    def _residual(self, x, s, training=None):
        """Residual path through the block."""
        x = self.norm1(x, s)  # expects [B,C,T]
        x = self.actv(x)
        x = self.pool(x)  # expects [B,C,T], outputs [B,C,T]
        # conv1: [B,C,T] -> [B,T,C] -> conv -> [B,T,C] -> [B,C,T]
        x = tf.transpose(x, [0, 2, 1])
        x = self.conv1(x)
        x = tf.transpose(x, [0, 2, 1])
        x = self.norm2(x, s)
        x = self.actv(x)
        # conv2: [B,C,T] -> [B,T,C] -> conv -> [B,T,C] -> [B,C,T]
        x = tf.transpose(x, [0, 2, 1])
        x = self.conv2(x)
        x = tf.transpose(x, [0, 2, 1])
        return x

    def call(self, x, s, training=None):
        """Forward pass with shortcut and residual connections."""
        out = self._residual(x, s, training)
        scut = self._shortcut(x)
        outp = (out + scut) * tf.math.rsqrt(2.0)
        return outp

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
        
        # F0 and N processing - use channel-last for TFLite
        self.f0_conv = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=3,
            strides=2,
            padding='valid'
        )
        self.n_conv = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=3,
            strides=2,
            padding='valid'
        )
        
        # ASR residual processing - 1x1 conv implemented via pointwise dense op
        self.asr_res = tf.keras.Sequential([
            PointwiseConv1D(64, name='asr_res_linear')
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
        pad_spec = [[0, 0], [0, 0], [1, 1]]
        f0_input = tf.pad(tf.expand_dims(f0_curve, axis=1), pad_spec)  # [B,1,T]
        n_input = tf.pad(tf.expand_dims(n, axis=1), pad_spec)  # [B,1,T]
        # Conv1D expects [B,T,C], so transpose
        f0 = tf.transpose(f0_input, [0, 2, 1])  # [B,T,1]
        f0 = self.f0_conv(f0, training=training)  # [B,T',1]
        f0 = tf.transpose(f0, [0, 2, 1])  # [B,1,T']
        n_processed = tf.transpose(n_input, [0, 2, 1])  # [B,T,1]
        n_processed = self.n_conv(n_processed, training=training)  # [B,T',1]
        n_processed = tf.transpose(n_processed, [0, 2, 1])  # [B,1,T']
        # Initial encoding
        x = tf.concat([asr, f0, n_processed], axis=1)  # [B,C,T]
        x = self.encode(x, s, training=training)  # expects [B,C,T]
        
        # ASR residual: asr is [B,C,T], conv expects [B,T,C]
        asr_t = tf.transpose(asr, [0, 2, 1])
        asr_res = self.asr_res(asr_t, training=training)
        asr_res = tf.transpose(asr_res, [0, 2, 1])  # back to [B,C,T]
               
        # Decode through layers
        res = True
        for i, block in enumerate(self.decode):
            if res:
                x = tf.concat([x, asr_res, f0, n_processed], axis=1)
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
