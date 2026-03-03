"""TensorFlow Vocos model components for training and LiteRT export.

This module defines the TensorFlow implementation of the Vocos decoder stack
and related discriminators/losses used by this repository's TF workflows.

Scope:
- Generator layers used for paired-feature -> waveform synthesis.
- Export-safe fixed-frame generator wrappers.
- Discriminators and adversarial/STFT/group-delay loss helpers for TF training.
- Utility functions shared across TF training and export pipelines.

Pipeline entrypoints:
- Training: `kokoro.tf.training`
- Quantization/QAT/export: `kokoro.tf.export`
- Focused quantization validation: `kokoro.tf.validate_quant`

Examples:
1) End-to-end export + QAT tuning + 20-sample validation
   uv run kokoro-vocos-litert-pipeline \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --data-root inputs \
     --train-filelist inputs/filelists/vocos.train.txt \
     --val-filelist inputs/filelists/vocos.val.txt \
     --output-dir output/tf_litert

2) Export fp32/fp16/int8 LiteRT models without QAT
   uv run kokoro-vocos-litert-pipeline \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --data-root inputs \
     --train-filelist inputs/filelists/vocos.train.txt \
     --val-filelist inputs/filelists/vocos.val.txt \
     --qat-steps 0 \
     --output-dir output/tf_litert_no_qat

3) Validation-only waveform generation from existing exports
   uv run kokoro-vocos-litert-pipeline \
     --pytorch-checkpoint output/checkpoints/last.pt \
     --data-root inputs \
     --train-filelist inputs/filelists/vocos.train.txt \
     --val-filelist inputs/filelists/vocos.val.txt \
     --skip-export --skip-qat \
     --output-dir output/tf_litert

Notes:
- LiteRT path is fixed to 520 frames (`DEFAULT_FIXED_FRAMES`).
- Pre/post blocks stay float for quality; ConvNeXt core is tuned for int8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf


_KERNEL_INIT = tf.keras.initializers.TruncatedNormal(stddev=0.02)
DEFAULT_FIXED_FRAMES = 520


def _to_channels_last(x: tf.Tensor) -> tf.Tensor:
    # [B, C, T] -> [B, T, C]
    return tf.transpose(x, [0, 2, 1])


def _to_channels_first(x: tf.Tensor) -> tf.Tensor:
    # [B, T, C] -> [B, C, T]
    return tf.transpose(x, [0, 2, 1])


def _repeat_pad_1d(x: tf.Tensor, target_length: int) -> tf.Tensor:
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    if target_length <= 0:
        return x[:0]
    length = tf.shape(x)[0]

    def _empty():
        return tf.zeros([target_length], dtype=x.dtype)

    def _trim():
        return x[:target_length]

    def _tile():
        repeat = tf.cast(tf.math.floordiv(target_length + length - 1, length), tf.int32)
        y = tf.tile(x, [repeat])
        return y[:target_length]

    return tf.cond(
        tf.equal(length, 0),
        _empty,
        lambda: tf.cond(tf.greater_equal(length, target_length), _trim, _tile),
    )


def _repeat_pad_2d(x: tf.Tensor, target_length: int) -> tf.Tensor:
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    if target_length <= 0:
        return x[:, :0]
    length = tf.shape(x)[1]

    def _empty():
        return tf.zeros([tf.shape(x)[0], target_length], dtype=x.dtype)

    def _trim():
        return x[:, :target_length]

    def _tile():
        repeat = tf.cast(tf.math.floordiv(target_length + length - 1, length), tf.int32)
        y = tf.tile(x, [1, repeat])
        return y[:, :target_length]

    return tf.cond(
        tf.equal(length, 0),
        _empty,
        lambda: tf.cond(tf.greater_equal(length, target_length), _trim, _tile),
    )


class CausalConv1DTF(tf.keras.layers.Layer):
    """Causal Conv1D wrapper matching streaming-vocos non-streaming padding behavior."""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        dilation_rate: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.dilation_rate = int(dilation_rate)
        self.groups = int(groups)
        self.use_bias = bool(use_bias)
        self.conv = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="valid",
            dilation_rate=self.dilation_rate,
            groups=self.groups,
            use_bias=self.use_bias,
            kernel_initializer=_KERNEL_INIT,
            bias_initializer="zeros",
        )

    @property
    def _effective_kernel_size(self) -> int:
        return (self.kernel_size - 1) * self.dilation_rate + 1

    def call(self, x_bct: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = _to_channels_last(tf.convert_to_tensor(x_bct))
        t = tf.shape(x)[1]
        padding_total = self._effective_kernel_size - self.strides
        numer = t - self._effective_kernel_size + padding_total
        rem = tf.math.floormod(numer, self.strides)
        extra_padding = tf.math.floormod(self.strides - rem, self.strides)
        x = tf.pad(x, [[0, 0], [padding_total, extra_padding], [0, 0]])
        y = self.conv(x, training=training)
        return _to_channels_first(y)


class ConvNeXtBlockTF(tf.keras.layers.Layer):
    """Streaming-causal ConvNeXt block used by Vocos backbone."""

    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.dwconv = CausalConv1DTF(filters=dim, kernel_size=kernel_size, groups=dim)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pwconv1 = tf.keras.layers.Dense(intermediate_dim, kernel_initializer=_KERNEL_INIT)
        self.act = tf.keras.layers.Activation("gelu")
        self.pwconv2 = tf.keras.layers.Dense(dim, kernel_initializer=_KERNEL_INIT)
        self.layer_scale_init_value = float(layer_scale_init_value)
        self.dim = int(dim)
        self.gamma: tf.Variable | None = None

    def build(self, input_shape):
        if self.layer_scale_init_value > 0.0:
            self.gamma = self.add_weight(
                name="gamma",
                shape=(self.dim,),
                initializer=tf.keras.initializers.Constant(self.layer_scale_init_value),
                trainable=True,
            )
        super().build(input_shape)

    def call(self, x_bct: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = x_bct
        y = self.dwconv(x_bct, training=training)
        y = _to_channels_last(y)
        y = self.norm(y, training=training)
        y = self.pwconv1(y, training=training)
        y = self.act(y)
        y = self.pwconv2(y, training=training)
        if self.gamma is not None:
            y = y * self.gamma
        y = _to_channels_first(y)
        return residual + y


class VocosBackboneTF(tf.keras.layers.Layer):
    """TensorFlow implementation of streaming-causal Vocos ConvNeXt backbone."""

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        kernel_size: int = 7,
        layer_scale_init_value: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_channels = int(input_channels)
        self.dim = int(dim)
        self.intermediate_dim = int(intermediate_dim)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.embed = CausalConv1DTF(filters=dim, kernel_size=kernel_size)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        scale_init = float(layer_scale_init_value or (1.0 / max(1, num_layers)))
        self.blocks = [
            ConvNeXtBlockTF(
                dim=dim,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=scale_init,
                kernel_size=kernel_size,
            )
            for _ in range(num_layers)
        ]
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x_bct: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Input [B, C, T] -> output [B, H, T]
        x = self.embed(x_bct, training=training)
        x = self.norm(_to_channels_last(x), training=training)
        x = _to_channels_first(x)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.final_layer_norm(_to_channels_last(x), training=training)
        return _to_channels_first(x)


class ISTFTHeadTF(tf.keras.layers.Layer):
    """TensorFlow ISTFT head equivalent to streaming-vocos ISTFT head."""

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same", **kwargs):
        super().__init__(**kwargs)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.padding = str(padding)
        self.out = tf.keras.layers.Dense(self.n_fft + 2, kernel_initializer=_KERNEL_INIT, bias_initializer="zeros")
        # Match torch.hann_window default behavior used by streaming-vocos (periodic=True).
        self.window = tf.signal.hann_window(self.n_fft, periodic=True, dtype=tf.float32)
        ola_kernel = np.zeros((self.n_fft, 1, self.n_fft), dtype=np.float32)
        for i in range(self.n_fft):
            ola_kernel[i, 0, i] = 1.0
        self.ola_kernel = tf.constant(ola_kernel, dtype=tf.float32)

    def call(self, x_bct: tf.Tensor, training: bool = False) -> tf.Tensor:
        # x: [B, H, T]
        x = _to_channels_last(x_bct)
        y = self.out(x, training=training)
        mag, phase = tf.split(y, 2, axis=-1)
        mag = tf.exp(mag)
        mag = tf.clip_by_value(mag, clip_value_min=0.0, clip_value_max=1e2)
        real = tf.cast(mag * tf.cos(phase), tf.float32)
        imag = tf.cast(mag * tf.sin(phase), tf.float32)
        spec = tf.complex(real, imag)  # [B, T, F]
        ifft = tf.signal.irfft(spec, [self.n_fft])  # [B, T, N]
        ifft = ifft * self.window[None, None, :]
        frames = tf.shape(ifft)[1]
        output_size = (frames - 1) * self.hop_length + self.n_fft
        out_shape = tf.stack([tf.shape(ifft)[0], output_size, 1], axis=0)
        ola = tf.nn.conv1d_transpose(
            ifft,
            filters=self.ola_kernel,
            output_shape=out_shape,
            strides=self.hop_length,
            padding="VALID",
        )
        audio = ola[:, :, 0]
        return audio[:, : frames * self.hop_length]


class PairedVocosGeneratorTF(tf.keras.Model):
    """Vocos generator receiving [asr,f0,noise,style] conditioning features."""

    def __init__(
        self,
        in_channels: int,
        model_input_channels: int,
        backbone_dim: int,
        backbone_intermediate_dim: int,
        backbone_layers: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
        backbone_kernel_size: int = 7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = int(in_channels)
        self.model_input_channels = int(model_input_channels)
        self.backbone_dim = int(backbone_dim)
        self.backbone_intermediate_dim = int(backbone_intermediate_dim)
        self.backbone_layers = int(backbone_layers)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.padding = str(padding)
        self.backbone_kernel_size = int(backbone_kernel_size)
        self.conditioner = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=model_input_channels,
                    kernel_size=1,
                    padding="same",
                    kernel_initializer=_KERNEL_INIT,
                    bias_initializer="zeros",
                ),
                tf.keras.layers.Activation("gelu"),
                tf.keras.layers.Conv1D(
                    filters=model_input_channels,
                    kernel_size=3,
                    padding="same",
                    kernel_initializer=_KERNEL_INIT,
                    bias_initializer="zeros",
                ),
                tf.keras.layers.Activation("gelu"),
                tf.keras.layers.Conv1D(
                    filters=model_input_channels,
                    kernel_size=1,
                    padding="same",
                    kernel_initializer=_KERNEL_INIT,
                    bias_initializer="zeros",
                ),
            ]
        )
        self.backbone = VocosBackboneTF(
            input_channels=model_input_channels,
            dim=backbone_dim,
            intermediate_dim=backbone_intermediate_dim,
            num_layers=backbone_layers,
            kernel_size=backbone_kernel_size,
        )
        self.head = ISTFTHeadTF(dim=backbone_dim, n_fft=n_fft, hop_length=hop_length, padding=padding)

    def call(self, features_bct: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = _to_channels_last(features_bct)
        x = self.conditioner(x, training=training)
        x = _to_channels_first(x)
        x = self.backbone(x, training=training)
        return self.head(x, training=training)


class ExportSafeISTFTHeadTF(tf.keras.layers.Layer):
    """ISTFT head variant without complex ops, suitable for TFLite quantization/export."""

    def __init__(self, dim: int, n_fft: int, hop_length: int, fixed_frames: int = DEFAULT_FIXED_FRAMES, padding: str = "same", **kwargs):
        super().__init__(**kwargs)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.fixed_frames = int(fixed_frames)
        self.padding = padding
        self.out = tf.keras.layers.Dense(self.n_fft + 2, kernel_initializer=_KERNEL_INIT, bias_initializer="zeros")

        num_bins = self.n_fft // 2 + 1
        k = np.arange(1, num_bins - 1, dtype=np.float32)  # exclude dc and nyquist
        n = np.arange(self.n_fft, dtype=np.float32)[:, None]
        angle = (2.0 * np.pi * n * k[None, :]) / float(self.n_fft)
        self.cos_basis = tf.constant(np.cos(angle), dtype=tf.float32)  # [N, K]
        self.sin_basis = tf.constant(np.sin(angle), dtype=tf.float32)  # [N, K]
        self.nyquist_sign = tf.constant(np.power(-1.0, np.arange(self.n_fft, dtype=np.float32)), dtype=tf.float32)  # [N]
        # Match torch.hann_window default behavior used by streaming-vocos (periodic=True).
        self.window = tf.signal.hann_window(self.n_fft, periodic=True, dtype=tf.float32)  # [N]

        ola_kernel = np.zeros((self.n_fft, 1, self.n_fft), dtype=np.float32)
        for i in range(self.n_fft):
            ola_kernel[i, 0, i] = 1.0
        self.ola_kernel = tf.constant(ola_kernel, dtype=tf.float32)  # [K, out_ch, in_ch]

        self.output_size = (self.fixed_frames - 1) * self.hop_length + self.n_fft
        self.trimmed_size = self.fixed_frames * self.hop_length

    def _irfft_real(self, real: tf.Tensor, imag: tf.Tensor) -> tf.Tensor:
        # real/imag: [B, T, F], with F = n_fft//2 + 1
        dc = real[:, :, 0:1]  # [B, T, 1]
        nyquist = real[:, :, -1:] * self.nyquist_sign[None, None, :]  # [B, T, N]
        real_mid = real[:, :, 1:-1]  # [B, T, K]
        imag_mid = imag[:, :, 1:-1]  # [B, T, K]
        inner = tf.einsum("btk,nk->btn", real_mid, self.cos_basis)
        inner = inner - tf.einsum("btk,nk->btn", imag_mid, self.sin_basis)
        return (dc + nyquist + (2.0 * inner)) / float(self.n_fft)  # [B, T, N]

    def call(self, x_bct: tf.Tensor, training: bool = False) -> tf.Tensor:
        # x: [B, H, T], with T assumed fixed to self.fixed_frames for export.
        x = _to_channels_last(x_bct)
        y = self.out(x, training=training)
        mag, phase = tf.split(y, 2, axis=-1)  # [B, T, F]
        mag = tf.exp(mag)
        mag = tf.clip_by_value(mag, clip_value_min=0.0, clip_value_max=1e2)
        real = tf.cast(mag * tf.cos(phase), tf.float32)
        imag = tf.cast(mag * tf.sin(phase), tf.float32)

        ifft = self._irfft_real(real, imag)  # [B, T, N]
        ifft = ifft * self.window[None, None, :]  # [B, T, N]

        bsz = tf.shape(ifft)[0]
        out_shape = tf.stack([bsz, self.output_size, 1], axis=0)
        ola = tf.nn.conv1d_transpose(
            ifft,
            filters=self.ola_kernel,
            output_shape=out_shape,
            strides=self.hop_length,
            padding="VALID",
        )  # [B, output_size, 1]
        audio = ola[:, :, 0]
        return audio[:, : self.trimmed_size]


class PairedVocosGeneratorExportTF(tf.keras.Model):
    """Export-focused generator with fixed-frame input and complex-free ISTFT synthesis."""

    def __init__(
        self,
        in_channels: int,
        model_input_channels: int,
        backbone_dim: int,
        backbone_intermediate_dim: int,
        backbone_layers: int,
        n_fft: int,
        hop_length: int,
        fixed_frames: int = DEFAULT_FIXED_FRAMES,
        padding: str = "same",
        backbone_kernel_size: int = 7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = int(in_channels)
        self.model_input_channels = int(model_input_channels)
        self.backbone_dim = int(backbone_dim)
        self.backbone_intermediate_dim = int(backbone_intermediate_dim)
        self.backbone_layers = int(backbone_layers)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.padding = str(padding)
        self.backbone_kernel_size = int(backbone_kernel_size)
        self.fixed_frames = int(fixed_frames)
        self.conditioner = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=model_input_channels,
                    kernel_size=1,
                    padding="same",
                    kernel_initializer=_KERNEL_INIT,
                    bias_initializer="zeros",
                ),
                tf.keras.layers.Activation("gelu"),
                tf.keras.layers.Conv1D(
                    filters=model_input_channels,
                    kernel_size=3,
                    padding="same",
                    kernel_initializer=_KERNEL_INIT,
                    bias_initializer="zeros",
                ),
                tf.keras.layers.Activation("gelu"),
                tf.keras.layers.Conv1D(
                    filters=model_input_channels,
                    kernel_size=1,
                    padding="same",
                    kernel_initializer=_KERNEL_INIT,
                    bias_initializer="zeros",
                ),
            ]
        )
        self.backbone = VocosBackboneTF(
            input_channels=model_input_channels,
            dim=backbone_dim,
            intermediate_dim=backbone_intermediate_dim,
            num_layers=backbone_layers,
            kernel_size=backbone_kernel_size,
        )
        self.head = ExportSafeISTFTHeadTF(
            dim=backbone_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            fixed_frames=fixed_frames,
            padding=padding,
        )

    def call(self, features_bct: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Fixed-frame export path: [B, C, 520] -> [B, 520 * hop]
        x = _to_channels_last(features_bct)
        x = self.conditioner(x, training=training)
        x = _to_channels_first(x)
        x = self.backbone(x, training=training)
        return self.head(x, training=training)


class QuantizableVocosCoreTF(tf.keras.Model):
    """ConvNeXt core block stack only (int8 target)."""

    def __init__(self, export_generator: PairedVocosGeneratorExportTF, **kwargs):
        super().__init__(**kwargs)
        self.blocks = list(export_generator.backbone.blocks)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        for block in self.blocks:
            x = block(x, training=training)
        return x


class FloatPreBlocksTF(tf.keras.Model):
    """Float pre-blocks, kept in fp32/fp16 for quality."""

    def __init__(self, export_generator: PairedVocosGeneratorExportTF, **kwargs):
        super().__init__(**kwargs)
        self.conditioner = export_generator.conditioner
        self.embed = export_generator.backbone.embed
        self.norm = export_generator.backbone.norm

    def call(self, features_bct: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = _to_channels_last(features_bct)
        x = self.conditioner(x, training=training)
        x = _to_channels_first(x)
        x = self.embed(x, training=training)
        x = self.norm(_to_channels_last(x), training=training)
        x = _to_channels_first(x)
        return x


class FloatPostBlocksTF(tf.keras.Model):
    """Float post-blocks, kept in fp32/fp16 for quality."""

    def __init__(self, export_generator: PairedVocosGeneratorExportTF, **kwargs):
        super().__init__(**kwargs)
        self.final_layer_norm = export_generator.backbone.final_layer_norm
        self.head = export_generator.head

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.final_layer_norm(_to_channels_last(x), training=training)
        x = _to_channels_first(x)
        return self.head(x, training=training)


class QuantizedVocosInferenceTF(tf.keras.Model):
    """Inference wrapper matching the PyTorch pre/core/post quantization split."""

    def __init__(self, pre: FloatPreBlocksTF, core: QuantizableVocosCoreTF, post: FloatPostBlocksTF, **kwargs):
        super().__init__(**kwargs)
        self.pre = pre
        self.core = core
        self.post = post

    def call(self, features_bct: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.pre(features_bct, training=training)
        x = self.core(x, training=training)
        return self.post(x, training=training)


def _safe_audio_to_stft_channels(x: tf.Tensor, n_fft: int, hop_factor: float) -> tf.Tensor:
    # [B, T] -> [B, time, freq, 2]
    x = tf.cast(x, tf.float32)
    x = x - tf.reduce_mean(x, axis=-1, keepdims=True)
    x = 0.8 * x / (tf.reduce_max(tf.abs(x), axis=-1, keepdims=True) + 1e-9)
    hop = int(n_fft * hop_factor)
    spec = tf.signal.stft(
        x,
        frame_length=n_fft,
        frame_step=max(1, hop),
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
    )
    return tf.stack([tf.math.real(spec), tf.math.imag(spec)], axis=-1)


class DiscriminatorPTF(tf.keras.layers.Layer):
    def __init__(self, period: int, lrelu_slope: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.period = int(period)
        self.lrelu_slope = float(lrelu_slope)
        self.convs = [
            tf.keras.layers.Conv2D(32, (5, 1), strides=(3, 1), padding="same"),
            tf.keras.layers.Conv2D(128, (5, 1), strides=(3, 1), padding="same"),
            tf.keras.layers.Conv2D(512, (5, 1), strides=(3, 1), padding="same"),
            tf.keras.layers.Conv2D(1024, (5, 1), strides=(3, 1), padding="same"),
            tf.keras.layers.Conv2D(1024, (5, 1), strides=(1, 1), padding="same"),
        ]
        self.conv_post = tf.keras.layers.Conv2D(1, (3, 1), strides=(1, 1), padding="same")

    def call(self, x: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        # x: [B, T]
        x = tf.cast(x, tf.float32)
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        pad = tf.math.floormod(-t, self.period)
        # Constant pad is used for TensorFlow compatibility.
        x = tf.pad(x, [[0, 0], [0, pad]])
        t = tf.shape(x)[1]
        x = tf.reshape(x, [b, t // self.period, self.period, 1])
        fmap: List[tf.Tensor] = []
        for i, conv in enumerate(self.convs):
            x = conv(x, training=training)
            x = tf.nn.leaky_relu(x, alpha=self.lrelu_slope)
            if i > 0:
                fmap.append(x)
        x = self.conv_post(x, training=training)
        fmap.append(x)
        logits = tf.reshape(x, [b, -1])
        return logits, fmap


class MultiPeriodDiscriminatorTF(tf.keras.Model):
    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11), **kwargs):
        super().__init__(**kwargs)
        self.discriminators = [DiscriminatorPTF(period=p) for p in periods]

    def call(
        self, y: tf.Tensor, y_hat: tf.Tensor, training: bool = False
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[List[tf.Tensor]], List[List[tf.Tensor]]]:
        y_d_rs: List[tf.Tensor] = []
        y_d_gs: List[tf.Tensor] = []
        fmap_rs: List[List[tf.Tensor]] = []
        fmap_gs: List[List[tf.Tensor]] = []
        for d in self.discriminators:
            y_r, fmap_r = d(y, training=training)
            y_g, fmap_g = d(y_hat, training=training)
            y_d_rs.append(y_r)
            y_d_gs.append(y_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorRTF(tf.keras.layers.Layer):
    def __init__(
        self,
        window_length: int,
        channels: int = 32,
        hop_factor: float = 0.25,
        bands: Tuple[Tuple[float, float], ...] = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window_length = int(window_length)
        self.hop_factor = float(hop_factor)
        n_fft_bins = self.window_length // 2 + 1
        self.bands = tuple((int(a * n_fft_bins), int(b * n_fft_bins)) for a, b in bands)
        self.band_convs: List[List[tf.keras.layers.Layer]] = []
        for _ in self.bands:
            self.band_convs.append(
                [
                    tf.keras.layers.Conv2D(channels, (3, 9), strides=(1, 1), padding="same"),
                    tf.keras.layers.Conv2D(channels, (3, 9), strides=(1, 2), padding="same"),
                    tf.keras.layers.Conv2D(channels, (3, 9), strides=(1, 2), padding="same"),
                    tf.keras.layers.Conv2D(channels, (3, 9), strides=(1, 2), padding="same"),
                    tf.keras.layers.Conv2D(channels, (3, 3), strides=(1, 1), padding="same"),
                ]
            )
        self.conv_post = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), padding="same")

    def _spectrogram(self, x: tf.Tensor) -> List[tf.Tensor]:
        spec_ri = _safe_audio_to_stft_channels(x, n_fft=self.window_length, hop_factor=self.hop_factor)
        return [spec_ri[:, :, lo:hi, :] for lo, hi in self.bands]

    def call(self, x: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        bands = self._spectrogram(x)
        fmap: List[tf.Tensor] = []
        outs: List[tf.Tensor] = []
        for band, stack in zip(bands, self.band_convs):
            y = band
            for i, layer in enumerate(stack):
                y = layer(y, training=training)
                y = tf.nn.leaky_relu(y, alpha=0.1)
                if i > 0:
                    fmap.append(y)
            outs.append(y)
        x = tf.concat(outs, axis=2)
        x = self.conv_post(x, training=training)
        fmap.append(x)
        return x, fmap


class MultiResolutionDiscriminatorTF(tf.keras.Model):
    def __init__(self, fft_sizes: Sequence[int] = (2048, 1024, 512), **kwargs):
        super().__init__(**kwargs)
        self.discriminators = [DiscriminatorRTF(window_length=w) for w in fft_sizes]

    def call(
        self, y: tf.Tensor, y_hat: tf.Tensor, training: bool = False
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[List[tf.Tensor]], List[List[tf.Tensor]]]:
        y_d_rs: List[tf.Tensor] = []
        y_d_gs: List[tf.Tensor] = []
        fmap_rs: List[List[tf.Tensor]] = []
        fmap_gs: List[List[tf.Tensor]] = []
        for d in self.discriminators:
            y_r, fmap_r = d(y, training=training)
            y_g, fmap_g = d(y_hat, training=training)
            y_d_rs.append(y_r)
            y_d_gs.append(y_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ComplexSTFTDiscriminatorTF(tf.keras.layers.Layer):
    def __init__(self, window_length: int, channels: int = 32, hop_factor: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.window_length = int(window_length)
        self.hop_factor = float(hop_factor)
        self.convs = [
            tf.keras.layers.Conv2D(channels, (3, 9), strides=(1, 1), padding="same"),
            tf.keras.layers.Conv2D(channels, (3, 9), strides=(1, 2), padding="same"),
            tf.keras.layers.Conv2D(channels, (3, 9), strides=(1, 2), padding="same"),
            tf.keras.layers.Conv2D(channels, (3, 9), strides=(1, 2), padding="same"),
            tf.keras.layers.Conv2D(channels, (3, 3), strides=(1, 1), padding="same"),
        ]
        self.conv_post = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), padding="same")

    def call(self, x: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        x = _safe_audio_to_stft_channels(x, n_fft=self.window_length, hop_factor=self.hop_factor)
        fmap: List[tf.Tensor] = []
        for i, layer in enumerate(self.convs):
            x = layer(x, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.1)
            if i > 0:
                fmap.append(x)
        x = self.conv_post(x, training=training)
        fmap.append(x)
        logits = tf.reshape(x, [tf.shape(x)[0], -1])
        return logits, fmap


class MultiResolutionComplexSTFTDiscriminatorTF(tf.keras.Model):
    def __init__(self, fft_sizes: Sequence[int] = (2048, 1024, 512, 256), **kwargs):
        super().__init__(**kwargs)
        self.discriminators = [ComplexSTFTDiscriminatorTF(window_length=w) for w in fft_sizes]

    def call(
        self, y: tf.Tensor, y_hat: tf.Tensor, training: bool = False
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[List[tf.Tensor]], List[List[tf.Tensor]]]:
        y_d_rs: List[tf.Tensor] = []
        y_d_gs: List[tf.Tensor] = []
        fmap_rs: List[List[tf.Tensor]] = []
        fmap_gs: List[List[tf.Tensor]] = []
        for d in self.discriminators:
            y_r, fmap_r = d(y, training=training)
            y_g, fmap_g = d(y_hat, training=training)
            y_d_rs.append(y_r)
            y_d_gs.append(y_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def generator_hinge_loss(disc_outputs: Sequence[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    total = tf.constant(0.0, dtype=tf.float32)
    parts: List[tf.Tensor] = []
    for dg in disc_outputs:
        term = tf.reduce_mean(tf.nn.relu(1.0 - tf.cast(dg, tf.float32)))
        parts.append(term)
        total = total + term
    return total, parts


def discriminator_hinge_loss(
    disc_real_outputs: Sequence[tf.Tensor], disc_generated_outputs: Sequence[tf.Tensor]
) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]:
    total = tf.constant(0.0, dtype=tf.float32)
    real_terms: List[tf.Tensor] = []
    gen_terms: List[tf.Tensor] = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = tf.reduce_mean(tf.nn.relu(1.0 - tf.cast(dr, tf.float32)))
        g_loss = tf.reduce_mean(tf.nn.relu(1.0 + tf.cast(dg, tf.float32)))
        total = total + r_loss + g_loss
        real_terms.append(r_loss)
        gen_terms.append(g_loss)
    return total, real_terms, gen_terms


def feature_matching_loss(fmap_r: Sequence[Sequence[tf.Tensor]], fmap_g: Sequence[Sequence[tf.Tensor]]) -> tf.Tensor:
    total = tf.constant(0.0, dtype=tf.float32)
    for d_r, d_g in zip(fmap_r, fmap_g):
        for r, g in zip(d_r, d_g):
            total = total + tf.reduce_mean(tf.abs(tf.cast(r, tf.float32) - tf.cast(g, tf.float32)))
    return total


def align_audio(pred: tf.Tensor, target: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    n = tf.minimum(tf.shape(pred)[1], tf.shape(target)[1])
    return pred[:, :n], target[:, :n]


@dataclass
class DynamicLossWeights:
    gan: float
    fm: float
    mrstft: float
    group_delay: float


def compute_dynamic_weights(
    step: int,
    max_steps: int,
    pretrain_steps: int,
    adv_ramp_ratio: float,
    gan_base: float,
    fm_base: float,
    mrstft_base: float,
    group_delay_base: float,
    mrstft_final_ratio: float,
) -> DynamicLossWeights:
    post_warmup = max(1, max_steps - pretrain_steps)
    adv_ramp_window = max(1, int(max(0.0, adv_ramp_ratio) * post_warmup))
    if step < pretrain_steps:
        adv_scale = 0.0
    else:
        adv_scale = min(1.0, (step - pretrain_steps) / float(adv_ramp_window))
    progress = min(1.0, step / float(max(1, max_steps)))
    final_ratio = max(0.0, min(1.0, mrstft_final_ratio))
    mrstft_scale = 1.0 - (1.0 - final_ratio) * progress
    return DynamicLossWeights(
        gan=gan_base * adv_scale,
        fm=fm_base * adv_scale,
        mrstft=mrstft_base * mrstft_scale,
        group_delay=group_delay_base * mrstft_scale,
    )


class MultiResolutionSTFTLossTF(tf.keras.losses.Loss):
    """StyleTTS2-aligned multi-resolution mel spectral convergence loss."""

    def __init__(
        self,
        sample_rate: int = 24000,
        n_mels: int = 80,
        fft_sizes: Iterable[int] = (1024, 2048, 512),
        hop_sizes: Iterable[int] = (120, 240, 50),
        win_lengths: Iterable[int] = (600, 1200, 240),
        name: str = "mrstft_loss",
    ):
        super().__init__(name=name)
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = list(hop_sizes)
        self.win_lengths = list(win_lengths)
        if not (len(self.fft_sizes) == len(self.hop_sizes) == len(self.win_lengths)):
            raise ValueError("fft_sizes, hop_sizes and win_lengths must have equal length")

    def _mel(self, x: tf.Tensor, fft_size: int, hop_size: int, win_length: int) -> tf.Tensor:
        del win_length  # tf.signal.stft uses frame_length and internal Hann window.
        stft = tf.signal.stft(
            x,
            frame_length=fft_size,
            frame_step=hop_size,
            fft_length=fft_size,
            window_fn=tf.signal.hann_window,
        )
        mag = tf.abs(stft)
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=fft_size // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=float(self.sample_rate) / 2.0,
            dtype=mag.dtype,
        )
        mel = tf.tensordot(mag, mel_matrix, axes=[[2], [0]])
        mel = tf.math.log(1e-5 + mel)
        mel = (mel - (-4.0)) / 4.0
        return mel

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        total = tf.constant(0.0, dtype=tf.float32)
        for fs, hs, wl in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            pred_mel = self._mel(y_pred, fs, hs, wl)
            true_mel = self._mel(y_true, fs, hs, wl)
            sc = tf.reduce_sum(tf.abs(true_mel - pred_mel)) / (tf.reduce_sum(tf.abs(true_mel)) + 1e-9)
            total = total + sc
        return total / float(len(self.fft_sizes))


class MultiResolutionGroupDelayLossTF(tf.keras.losses.Loss):
    """StyleTTS2-like multi-resolution group delay loss."""

    def __init__(
        self,
        fft_sizes: Iterable[int] = (1024, 2048, 512),
        hop_sizes: Iterable[int] = (120, 240, 50),
        win_lengths: Iterable[int] = (600, 1200, 240),
        phase_eps: float = 1e-6,
        mag_floor: float = 1e-4,
        name: str = "group_delay_loss",
    ):
        super().__init__(name=name)
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = list(hop_sizes)
        self.win_lengths = list(win_lengths)
        if not (len(self.fft_sizes) == len(self.hop_sizes) == len(self.win_lengths)):
            raise ValueError("fft_sizes, hop_sizes and win_lengths must have equal length")
        self.phase_eps = float(max(1e-12, phase_eps))
        self.mag_floor = float(max(0.0, mag_floor))

    def _stft(self, x: tf.Tensor, fft_size: int, hop_size: int, win_length: int) -> tf.Tensor:
        del win_length
        return tf.signal.stft(
            x,
            frame_length=fft_size,
            frame_step=hop_size,
            fft_length=fft_size,
            window_fn=tf.signal.hann_window,
        )

    def _group_delay(self, spec: tf.Tensor) -> tf.Tensor:
        # spec: [B, time, freq]
        cross = spec[:, :, 1:] * tf.math.conj(spec[:, :, :-1])
        real = tf.math.real(cross)
        imag = tf.math.imag(cross)
        power = tf.square(real) + tf.square(imag)
        safe = power > (self.phase_eps * self.phase_eps)
        real = tf.where(safe, real, tf.fill(tf.shape(real), tf.cast(self.phase_eps, real.dtype)))
        imag = tf.where(safe, imag, tf.zeros_like(imag))
        return tf.math.atan2(imag, real)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        total = tf.constant(0.0, dtype=tf.float32)
        for fs, hs, wl in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            pred_spec = self._stft(y_pred, fs, hs, wl)
            true_spec = self._stft(y_true, fs, hs, wl)
            pred_gd = self._group_delay(pred_spec)
            true_gd = self._group_delay(true_spec)
            mag = 0.5 * (tf.abs(true_spec[:, :, 1:]) + tf.abs(true_spec[:, :, :-1]))
            if self.mag_floor > 0.0:
                mag = tf.where(mag >= self.mag_floor, mag, tf.zeros_like(mag))
            delta = pred_gd - true_gd
            delta = tf.math.atan2(tf.sin(delta), tf.cos(delta))
            denom = tf.maximum(tf.reduce_sum(mag), 1e-6)
            total = total + tf.reduce_sum(tf.abs(delta) * mag) / denom
        return total / float(len(self.fft_sizes))
