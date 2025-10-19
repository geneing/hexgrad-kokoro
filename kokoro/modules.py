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
    """Custom layer normalization for 1D convolution outputs."""
    
    def __init__(self, channels: int, eps: float = 1e-5, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.channels = channels
        self.eps = eps
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.eps)
        
    # def build(self, input_shape):
    #     self.gamma = self.add_weight(
    #         name='gamma',
    #         shape=(self.channels,),
    #         initializer='ones'
    #     )
    #     self.beta = self.add_weight(
    #         name='beta',
    #         shape=(self.channels,),
    #         initializer='zeros'
    #     )
    #     self.layer_norm.build(input_shape)
    #     super().build(input_shape)

    def call(self, x):
        # x shape: [batch, channels, time] -> transpose to [batch, time, channels]
        x = tf.transpose(x, [0, 2, 1])
        print(f"tf: LayerNorm input {x.shape=}")
        x = self.layer_norm(x) #, [self.channels], self.gamma, self.beta, self.eps
        # Transpose back to [batch, channels, time]
        return tf.transpose(x, [0, 2, 1])

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'eps': self.eps
        })
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
        
        # CNN layers
        self.cnn_layers = []
        for _ in range(depth):
            cnn_block = tf.keras.Sequential([
                # Note: TensorFlow Conv1D has different parameter order than PyTorch
                # PyTorch: (in_channels, out_channels, kernel_size)
                # TensorFlow: filters=out_channels, kernel_size, input_shape
                tf.keras.layers.Conv1D(
                    filters=channels,
                    kernel_size=kernel_size,
                    padding='same',
                    data_format="channels_first",
                    # Note: Weight normalization not directly available in TF - potential conversion issue
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
        
        # Transpose to [batch, channels, seq_len] for Conv1D processing
        x = tf.transpose(x, [0, 2, 1])
        
        # Apply CNN layers
        for i,cnn_layer in enumerate(self.cnn_layers):
            print(f"tf: TextEncoder {i=} {x[0,0:2,0:2]=}")
            for ii, l in enumerate(cnn_layer.layers):
                print(f"tf: {l}")
                print(f"tf:   -layer {ii=} {x.shape=} {x[0,0:2,0:2]=}")
                x = l(x, training=training) #if hasattr(l, 'call') else l(x)
                print(f"tf:   +layer {ii=} {l} {x[0,0:2,0:2]=}")
            #x = cnn_layer(x, training=training)
        
        # Transpose back for LSTM: [batch, seq_len, channels]
        x = tf.transpose(x, [0, 2, 1])
        
        # Note: TensorFlow LSTM doesn't have pack_padded_sequence equivalent
        # This could be a potential conversion issue for variable length sequences
        x = self.lstm(x, training=training)
        
        # Transpose back to [batch, channels, seq_len]
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

    def call(self, x_in, style, training=False):
        x = tf.transpose(x_in, [0, 2, 1]) # [B, C, T] -> [B, T, C]
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # Expand style to match sequence
        s = tf.broadcast_to(
            tf.expand_dims(style, axis=0), 
            [batch_size, seq_len, self.sty_dim]
        ) # [B, C] -> [B, T, C]
        
        # Concatenate x and style
        x = tf.concat([x, s], axis=-1) # [B, T, C+S]

        for i in range(self.nlayers):
            x = self.lstms.get_layer(index=2*i)(x, training=training)
            x = self.lstms.get_layer(index=2*i+1)(x, style)
            x = tf.concat([x, s], axis=-1) # [B, T, C+S]
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

        # Remove channel dim if it equals 1
        F0_out = tf.squeeze(F0_feat, axis=1) if tf.shape(F0_feat)[1] == 1 else F0_feat
        N_out = tf.squeeze(N_feat, axis=1) if tf.shape(N_feat)[1] == 1 else N_feat
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


