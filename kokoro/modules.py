"""
TensorFlow Keras implementation of Kokoro TTS modules.
Converted from PyTorch implementation.
"""

from genericpath import exists
from regex import F
import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_models.nlp as nlp
from transformers import TFAlbertModel

import numpy as np
import math
from typing import Dict, Optional, Union, Tuple


def _ensure_batch1_seq(x: tf.Tensor, channels: int) -> tf.Tensor:
    """Reshape input to enforce a batch size of 1 with a known channel dimension."""
    x = tf.reshape(x, [1, -1, channels])
    x.set_shape([1, None, channels])
    return x

# https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
class CustomAlbert(TFAlbertModel):
    def __init__(self, *args, **kwargs):
        super(CustomAlbert, self).__init__(*args, **kwargs)


    def call(self, *args, **kwargs):
        outputs = super().call(*args, **kwargs)
        
        return outputs


class LinearNorm(tf.keras.layers.Layer):
    """TensorFlow equivalent of PyTorch LinearNorm layer."""
    
    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = True, w_init_gain: str = 'linear', **kwargs):
        super(LinearNorm, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        
        # Xavier uniform initialization (Glorot uniform in TF)
        self.dense = tf.keras.layers.Dense(
            out_dim, 
            use_bias=use_bias,
            kernel_initializer='glorot_uniform'
        )

    def call(self, x):
        return self.dense(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_dim': self.in_dim,
            'out_dim': self.out_dim,
            'use_bias': self.use_bias
        })
        return config


class LayerNorm(tf.keras.layers.Layer):
    """LayerNorm that keeps channel-last layout for TFLite compatibility."""

    def __init__(self, channels: int, eps: float = 1e-5, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.channels = channels
        self.eps = eps
        # Channel axis stays last throughout the TF port to avoid transpose ops during export.
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=self.eps)

    def call(self, x, training=None):
        return self.layer_norm(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'eps': self.eps
        })
        return config


class StaticLSTM(tf.keras.layers.Layer):
    """Simple LSTM without TensorList ops (batch size fixed to 1)."""

    def __init__(self, units: int, go_backwards: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.go_backwards = go_backwards
        self.input_dim = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.input_dim, self.units * 4),
            initializer='glorot_uniform',
            trainable=True
        )
        self.recurrent_kernel = self.add_weight(
            name='recurrent_kernel',
            shape=(self.units, self.units * 4),
            initializer='orthogonal',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units * 4,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        del training
        inputs = tf.reshape(inputs, [1, -1, self.input_dim])
        inputs.set_shape([1, None, self.input_dim])
        if self.go_backwards:
            inputs = tf.reverse(inputs, axis=[1])
        time_steps = tf.shape(inputs)[1]
        state_h = tf.zeros([1, self.units], dtype=inputs.dtype)
        state_c = tf.zeros_like(state_h)
        ta = tf.TensorArray(
            dtype=inputs.dtype,
            size=time_steps,
            element_shape=tf.TensorShape([1, self.units])
        )

        def body(t, h, c, ta):
            x_t = tf.gather(inputs, t, axis=1)
            z = tf.matmul(x_t, self.kernel) + tf.matmul(h, self.recurrent_kernel) + self.bias
            z0, z1, z2, z3 = tf.split(z, 4, axis=1)
            i = tf.sigmoid(z0)
            f = tf.sigmoid(z1)
            g = tf.tanh(z2)
            o = tf.sigmoid(z3)
            c = f * c + i * g
            h = o * tf.tanh(c)
            ta = ta.write(t, h)
            return t + 1, h, c, ta

        _, _, _, ta = tf.while_loop(
            lambda t, *_: t < time_steps,
            body,
            loop_vars=(0, state_h, state_c, ta),
        )

        outputs = ta.stack()
        outputs = tf.transpose(outputs, [1, 0, 2])
        if self.go_backwards:
            outputs = tf.reverse(outputs, axis=[1])
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'go_backwards': self.go_backwards})
        return config


class StaticBiLSTM(tf.keras.layers.Layer):
    """Bidirectional LSTM composed of two StaticLSTM instances."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.forward_layer = StaticLSTM(units)
        self.backward_layer = StaticLSTM(units, go_backwards=True)

    def build(self, input_shape):
        self.forward_layer.build(input_shape)
        self.backward_layer.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        fwd = self.forward_layer(inputs, training=training)
        bwd = self.backward_layer(inputs, training=training)
        return tf.concat([fwd, bwd], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config






class TextEncoder(tf.keras.layers.Layer):
    """TensorFlow implementation of TextEncoder."""
    
    def __init__(self, channels: int, kernel_size: int, depth: int, n_symbols: int, **kwargs):
        super(TextEncoder, self).__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.depth = depth
        self.n_symbols = n_symbols
        
        self.embedding = tf.keras.layers.Embedding(n_symbols, channels)

        # CNN layers operate on channel-last tensors to match TF/TFLite expectations.
        self.cnn_layers = []
        for _ in range(depth):
            cnn_block = tf.keras.Sequential([
                tf.keras.layers.Conv1D(
                    filters=channels,
                    kernel_size=kernel_size,
                    padding='same',
                    use_bias=True
                ),
                LayerNorm(channels=channels),
                tf.keras.layers.LeakyReLU(0.2),
                # tf.keras.layers.Dropout(0.2),
            ])
            self.cnn_layers.append(cnn_block)
        
        # LSTM layer - note different parameter handling
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(channels // 2, return_sequences=True),
            merge_mode='concat'
        )

    def call(self, x, training=None):
        # x: [batch, seq_len] -> embedding -> [batch, seq_len, channels]
        x = self.embedding(x)

        # Apply CNN layers (all channel-last)
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x, training=training)

        # LSTM expects [batch, seq_len, channels]
        x = _ensure_batch1_seq(x, self.channels)
        x = self.lstm(x, training=training)

        # Transpose to [batch, channels, seq_len] for parity with Torch implementation.
        x = tf.transpose(x, [0, 2, 1])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'depth': self.depth,
            'n_symbols': self.n_symbols
        })
        return config


class AdaLayerNorm(tf.keras.layers.Layer):
    """Adaptive Layer Normalization."""
    
    def __init__(self, style_dim: int, channels: int, eps: float = 1e-5, **kwargs):
        super(AdaLayerNorm, self).__init__(**kwargs)
        self.style_dim = style_dim
        self.channels = channels
        self.eps = eps
        
        self.fc = tf.keras.layers.Dense(channels * 2)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.eps)

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            x_shape, style_shape = input_shape
        else:
            x_shape = input_shape
            style_shape = tf.TensorShape([None, self.style_dim])

        x_shape = tf.TensorShape(x_shape)
        style_shape = tf.TensorShape(style_shape)

        # Validate the incoming style embedding dimension matches expectations.
        if style_shape.rank is not None and style_shape[-1] is not None and style_shape[-1] != self.style_dim:
            raise ValueError(
                f"AdaLayerNorm expected style dimension {self.style_dim}, received {style_shape[-1]}."
            )

        if not self.fc.built:
            self.fc.build(style_shape)
        if not self.layer_norm.built:
            self.layer_norm.build(x_shape)

        super(AdaLayerNorm, self).build(input_shape)

    def call(self, x, s):
        # x: [batch, channels, seq_len], s: [batch, style_dim]
        h = self.fc(s)  # [batch, channels*2]
        h = tf.expand_dims(h, axis=1)  # [batch, 1, channels*2]
        
        gamma, beta = tf.split(h, 2, axis=-1)  # Each: [batch, 1, channels]
        # Layer normalization
        x = self.layer_norm(x) #[self.channels],
        # x = (1 + gamma) * x + beta
        x = tf.add( tf.multiply((1 + gamma), x), beta )

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'style_dim': self.style_dim,
            'channels': self.channels,
            'eps': self.eps
        })
        return config


class DurationEncoder(tf.keras.layers.Layer):
    """Duration encoder with LSTM and adaptive layer norm."""
    
    def __init__(self, sty_dim: int, d_model: int, nlayers: int, **kwargs):
        super(DurationEncoder, self).__init__(**kwargs)
        self.sty_dim = sty_dim
        self.d_model = d_model
        self.nlayers = nlayers
        self.lstms = tf.keras.Sequential()
        for i in range(nlayers):
            # LSTM layer
            lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(d_model // 2, return_sequences=True, name=f"DurationEncoder_lstm_{i}"),
                merge_mode='concat', name =f"DurationEncoder_bidi_{i}"
            )
            self.lstms.add(lstm)
            # Adaptive Layer Norm
            ada_norm = AdaLayerNorm(sty_dim, d_model, name=f"AdaLayerNorm_fc_{i}")
            self.lstms.add(ada_norm)

    def build(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and len(input_shape) == 2:
            x_shape, style_shape = input_shape
        else:
            x_shape = input_shape
            style_shape = tf.TensorShape([None, self.sty_dim])

        x_shape = tf.TensorShape(x_shape)
        style_shape = tf.TensorShape(style_shape)

        # DurationEncoder expects x_in shaped [B, C, T]; transpose -> [B, T, C].
        batch_dim = x_shape[0]
        time_dim = x_shape[-1]
        feature_dim = x_shape[-2]

        if style_shape.rank is not None and style_shape.rank > 0:
            style_dim = style_shape[-1]
            if style_dim is not None and style_dim != self.sty_dim:
                raise ValueError(
                    f"DurationEncoder expected style dimension {self.sty_dim}, got {style_dim}."
                )

        lstm_input_channels = None if feature_dim is None else feature_dim + self.sty_dim
        current_shape = tf.TensorShape([batch_dim, time_dim, lstm_input_channels])

        for layer_idx in range(self.nlayers):
            lstm_layer = self.lstms.get_layer(index=2 * layer_idx)
            if not lstm_layer.built:
                lstm_layer.build(current_shape)
            current_shape = tf.TensorShape(lstm_layer.compute_output_shape(current_shape))

            ada_layer = self.lstms.get_layer(index=2 * layer_idx + 1)
            if not ada_layer.built:
                ada_layer.build([current_shape, style_shape])

            merged_dim = current_shape[-1]
            if merged_dim is not None:
                merged_dim += self.sty_dim
            current_shape = tf.TensorShape([current_shape[0], current_shape[1], merged_dim])

        super(DurationEncoder, self).build(input_shape)

    def call(self, x_in, style, training=False):
        x = tf.transpose(x_in, [0, 2, 1]) # [B, C, T] -> [B, T, C]
        seq_len = tf.shape(x)[1]
        x = _ensure_batch1_seq(x, self.d_model)
        style = tf.reshape(style, [1, self.sty_dim])
        style.set_shape([1, self.sty_dim])
        
        # Expand style to match sequence
        s = tf.broadcast_to(
            tf.expand_dims(style, axis=1), 
            [1, seq_len, self.sty_dim]
        ) # [B, C] -> [B, T, C]
        
        # Concatenate x and style
        x = tf.concat([x, s], axis=-1) # [B, T, C+S]
        x = _ensure_batch1_seq(x, self.d_model + self.sty_dim)

        for i in range(self.nlayers):
            x = _ensure_batch1_seq(x, self.d_model + self.sty_dim)
            x = self.lstms.get_layer(index=2*i)(x, training=training)
            x = self.lstms.get_layer(index=2*i+1)(x, style)
            x = tf.concat([x, s], axis=-1) # [B, T, C+S]
        x = _ensure_batch1_seq(x, self.d_model + self.sty_dim)
        return x 

    def get_config(self):
        config = super().get_config()
        config.update({
            'sty_dim': self.sty_dim,
            'd_model': self.d_model,
            'nlayers': self.nlayers,
            'dropout_rate': self.dropout_rate
        })
        return config


class ProsodyPredictor(tf.keras.layers.Layer):
    """Prosody predictor with F0 and energy prediction (TensorFlow/Keras version).

    Converted from PyTorch. Major differences / potential mismatches:
    - Replaces nn.LSTM & nn.ModuleList with tf.keras.layers equivalents and Python lists.
    - Conv1d (channel-first) replaced by Conv1D operating on channel-last with explicit transposes.
    - Dropping pack_padded_sequence / pad_packed_sequence logic (approximation for variable lengths).
    - Dropout semantics differ (PyTorch functional vs tf.nn / layer dropout) -> may change statistics.
    - Residual AdainResBlk1d imported from iSTFTNet module; ensure shapes [B,C,T].
    - Duration projection uses LinearNorm (Dense) -> may need weight transpose when loading PT weights.
    """
    def __init__(self, style_dim: int, d_hid: int, nlayers: int, max_dur: int = 50, dropout: float = 0.1, **kwargs):
        super(ProsodyPredictor, self).__init__(**kwargs)
        self.style_dim = style_dim
        self.d_hid = d_hid
        self.max_dur = max_dur
        self.nlayers = nlayers
        self.dropout_rate = dropout

        # Duration text encoder
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim,
            d_model=d_hid,
            nlayers=nlayers,
            name="duration_encoder"
        )

        # LSTM stack (shared for duration path before projection)
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(d_hid // 2, return_sequences=True),
            merge_mode='concat',
            name='duration_bilstm'
        )

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.duration_proj = LinearNorm(d_hid, max_dur, name='duration_proj')  # NOTE: Dense vs Conv1d mismatch

        # Shared bi-LSTM for F0 / Noise branches (input will be [B,T,d_hid+style_dim])
        self.shared_bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(d_hid // 2, return_sequences=True),
            merge_mode='concat',
            name='shared_bilstm'
        )

        # Residual stacks (lists of AdainResBlk1d operating on channel-first [B,C,T])
        from .istftnet import AdainResBlk1d  # local import to avoid circular issues
        self.F0_blocks = [
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout, name='F0_blk_0'),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample='nearest', dropout_p=dropout, name='F0_blk_1'),  # NOTE: upsample flag semantic differs
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout, name='F0_blk_2'),
        ]
        self.N_blocks = [
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout, name='N_blk_0'),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample='nearest', dropout_p=dropout, name='N_blk_1'),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout, name='N_blk_2'),
        ]

        # Projection heads (Conv1D with kernel_size=1). Expect input channel-first -> transpose.
        self.F0_proj = tf.keras.layers.Conv1D(1, 1, padding='same', name='F0_proj')  # NOTE: weight layout differs from PyTorch Conv1d
        self.N_proj = tf.keras.layers.Conv1D(1, 1, padding='same', name='N_proj')    # NOTE: same mismatch

    def _channel_first_conv1x1(self, x, conv_layer):
        """Apply a Conv1D layer to channel-first tensor x: [B,C,T]."""
        x_perm = tf.transpose(x, [0, 2, 1])  # -> [B,T,C]
        x_perm = conv_layer(x_perm)
        return tf.transpose(x_perm, [0, 2, 1])  # -> [B,C,T]

    def call(self, texts, style, text_lengths, alignment, m, training=False):  # replaces forward
        """Compute duration predictions & encoded features.

        Args:
            texts: Tensor (implementation-dependent) passed to text_encoder (expected [B,C,T] or similar).
            style: [B, style_dim]
            text_lengths: Not used (variable length handling omitted)  # NOTE: pack_padded_sequence removed
            alignment: [B, T_text, T_aln] (matrix)  # NOTE: shape assumption; verify
            m: mask tensor used for padding (unused currently)  # NOTE: mask handling differs vs PyTorch
        Returns:
            duration_pred: [B, T, max_dur]
            en: encoded features after alignment multiplication.
        """
        d = self.text_encoder(texts, style, training=training)  # NOTE: variable length handling lost

        # BiLSTM over duration features (expects [B,T,C])
        # if tf.rank(d) == 3 and tf.shape(d)[1] != self.d_hid:  # heuristic check
        #     x = d
        # else:
        #     x = d  # NOTE: shape assumptions; may differ from original
        d = _ensure_batch1_seq(d, self.d_hid + self.style_dim)
        x = self.lstm(d, training=training)
        x = self.dropout(x, training=training)
        duration = self.duration_proj(x)  # [B,T,max_dur]

        # Alignment energy computation: PyTorch used d.transpose(-1,-2) @ alignment
        # We assume d: [B,T,C] -> transpose -> [B,C,T]; alignment: [B,T,T_aln] -> need [B,T,C]? Potential mismatch.
        d_t = tf.transpose(d, [0, 2, 1])  # [B,C,T]
        en = tf.matmul(d_t, alignment)  # NOTE: requires alignment shape [B,T,C] originally; verify dimension order

        return duration, en  # NOTE: squeeze omitted (shape mismatch potential)

    def f0n_train(self, x, s, training=False):  # replaces F0Ntrain
        """Predict F0 & noise components.

        Args:
            x: [B,C,T] (channel-first features)
            s: [B, style_dim]
        Returns:
            F0: [B,T] (after squeeze)
            N:  [B,T]
        """
        # Shared BiLSTM expects [B,T,C]; we transpose from [B,C,T].
        x_perm = tf.transpose(x, [0, 2, 1])  # [B,T,C]
        x_perm = _ensure_batch1_seq(x_perm, self.d_hid + self.style_dim)
        shared_out = self.shared_bilstm(x_perm, training=training)  # [B,T,d_hid]
        
        shared_cf = tf.transpose(shared_out, [0, 2, 1])            # [B,d_hid,T]

        # F0 branch
        F0_feat = shared_cf
        for ii, blk in enumerate(self.F0_blocks):
            F0_feat = blk(F0_feat, s, training=training)
            
        F0_feat = self._channel_first_conv1x1(F0_feat, self.F0_proj)

        # Noise branch
        N_feat = shared_cf
        for blk in self.N_blocks:
            N_feat = blk(N_feat, s, training=training)
        N_feat = self._channel_first_conv1x1(N_feat, self.N_proj)

        # Remove the singleton channel dimension introduced by the 1x1 projections.
        F0_out = tf.squeeze(F0_feat, axis=1)
        N_out = tf.squeeze(N_feat, axis=1)
        return F0_out, N_out

    def get_config(self):
        config = super().get_config()
        config.update({
            'style_dim': self.style_dim,
            'd_hid': self.d_hid,
            'max_dur': self.max_dur,
            'nlayers': self.nlayers,
            'dropout_rate': self.dropout_rate
        })
        return config
