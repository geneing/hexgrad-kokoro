"""
TensorFlow Keras implementation of Kokoro TTS modules.
Converted from PyTorch implementation.
"""

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_models.nlp as nlp
from transformers import TFAlbertModel

import numpy as np
import math
from typing import Dict, Optional, Union, Tuple

# https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
class CustomAlbert(nlp.networks.AlbertEncoder):
    def __init__(self, *args, **kwargs):
        super(CustomAlbert, self).__init__(*args, **kwargs)


    def call(self, *args, **kwargs):
        outputs = super().call(*args, **kwargs)
        # Return only the last hidden state, matching PyTorch implementation
        return outputs['encoder_outputs'][-1]


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
        
    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(self.channels,),
            initializer='ones'
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(self.channels,),
            initializer='zeros'
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: [batch, channels, time] -> transpose to [batch, time, channels]
        x = tf.transpose(x, [0, 2, 1])
        x = tf.nn.layer_norm(x, [self.channels], self.gamma, self.beta, self.eps)
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
                    # Note: Weight normalization not directly available in TF - potential conversion issue
                    use_bias=True
                ),
                LayerNorm(channels),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dropout(0.2),
            ])
            self.cnn_layers.append(cnn_block)
        
        # LSTM layer - note different parameter handling
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(channels // 2, return_sequences=True),
            merge_mode='concat'
        )

    def call(self, x, input_lengths, m, training=None):
        # x: [batch, seq_len] -> embedding -> [batch, seq_len, channels]
        x = self.embedding(x)
        
        # Transpose to [batch, channels, seq_len] for Conv1D processing
        x = tf.transpose(x, [0, 2, 1])
        
        # Apply mask - TensorFlow handles broadcasting differently
        m = tf.expand_dims(m, axis=1)  # [batch, 1, seq_len]
        x = tf.where(m, 0.0, x)  # Note: TensorFlow mask logic may need adjustment
        
        # Apply CNN layers
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x, training=training)
            x = tf.where(m, 0.0, x)
        
        # Transpose back for LSTM: [batch, seq_len, channels]
        x = tf.transpose(x, [0, 2, 1])
        
        # Note: TensorFlow LSTM doesn't have pack_padded_sequence equivalent
        # This could be a potential conversion issue for variable length sequences
        x = self.lstm(x, training=training)
        
        # Transpose back to [batch, channels, seq_len]
        x = tf.transpose(x, [0, 2, 1])
        
        # Pad to match mask size if needed
        current_length = tf.shape(x)[2]
        mask_length = tf.shape(m)[2]
        
        def pad_tensor():
            pad_width = mask_length - current_length
            padding = tf.stack([[0, 0], [0, 0], [0, pad_width]])
            return tf.pad(x, padding)
        
        x = tf.cond(current_length < mask_length, pad_tensor, lambda: x)
        x = tf.where(m, 0.0, x)
        
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

    def call(self, x, s):
        # x: [batch, channels, seq_len], s: [batch, style_dim]
        x = tf.transpose(x, [0, 2, 1])  # [batch, seq_len, channels]
        
        h = self.fc(s)  # [batch, channels*2]
        h = tf.expand_dims(h, axis=1)  # [batch, 1, channels*2]
        
        gamma, beta = tf.split(h, 2, axis=-1)  # Each: [batch, 1, channels]
        
        # Layer normalization
        x = tf.nn.layer_norm(x, [self.channels], epsilon=self.eps)
        x = (1 + gamma) * x + beta
        
        return tf.transpose(x, [0, 2, 1])  # Back to [batch, channels, seq_len]

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
    
    def __init__(self, sty_dim: int, d_model: int, nlayers: int, dropout: float = 0.1, **kwargs):
        super(DurationEncoder, self).__init__(**kwargs)
        self.sty_dim = sty_dim
        self.d_model = d_model
        self.nlayers = nlayers
        self.dropout_rate = dropout
        
        self.lstm_layers = []
        for _ in range(nlayers):
            # LSTM layer
            lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(d_model // 2, return_sequences=True),
                merge_mode='concat'
            )
            self.lstm_layers.append(lstm)
            
            # Adaptive Layer Norm
            ada_norm = AdaLayerNorm(sty_dim, d_model)
            self.lstm_layers.append(ada_norm)
        
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, style, text_lengths, m, training=None):
        # x: [batch, d_model, seq_len] -> [seq_len, batch, d_model]
        x = tf.transpose(x, [2, 0, 1])
        batch_size = tf.shape(x)[1]
        seq_len = tf.shape(x)[0]
        
        # Expand style to match sequence
        s = tf.broadcast_to(
            tf.expand_dims(style, axis=0), 
            [seq_len, batch_size, self.sty_dim]
        )
        
        # Concatenate x and style
        x = tf.concat([x, s], axis=-1)
        
        # Apply mask - Note: mask handling may differ from PyTorch
        masks = tf.transpose(m, [1, 0])  # [seq_len, batch]
        mask_expanded = tf.expand_dims(masks, axis=-1)
        x = tf.where(mask_expanded, 0.0, x)
        
        # Back to [batch, seq_len, features]
        x = tf.transpose(x, [1, 0, 2])
        # Then to [batch, features, seq_len] for conv operations
        x = tf.transpose(x, [0, 2, 1])
        
        for i, layer in enumerate(self.lstm_layers):
            if isinstance(layer, AdaLayerNorm):
                x = layer(x, style)
                # Re-add style dimension
                s_expanded = tf.broadcast_to(
                    tf.expand_dims(style, axis=-1),
                    [batch_size, self.sty_dim, tf.shape(x)[-1]]
                )
                x = tf.concat([x, s_expanded], axis=1)
                # Apply mask
                mask_conv = tf.expand_dims(m, axis=1)
                x = tf.where(mask_conv, 0.0, x)
            else:
                # LSTM processing
                x = tf.transpose(x, [0, 2, 1])  # [batch, seq_len, features]
                # Note: No pack_padded_sequence equivalent - potential conversion issue
                x = layer(x, training=training)
                x = self.dropout(x, training=training)
                x = tf.transpose(x, [0, 2, 1])  # [batch, features, seq_len]
                
                # Pad if needed to match mask size
                current_length = tf.shape(x)[2]
                mask_length = tf.shape(m)[1]
                
                def pad_x():
                    pad_width = mask_length - current_length
                    padding = tf.stack([[0, 0], [0, 0], [0, pad_width]])
                    return tf.pad(x, padding)
                
                x = tf.cond(current_length < mask_length, pad_x, lambda: x)
        
        return tf.transpose(x, [0, 2, 1])  # [batch, seq_len, features]

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
    """Prosody predictor with F0 and energy prediction."""
    
    def __init__(self, style_dim: int, d_hid: int, nlayers: int, max_dur: int = 50, dropout: float = 0.1, **kwargs):
        super(ProsodyPredictor, self).__init__(**kwargs)
        self.style_dim = style_dim
        self.d_hid = d_hid
        self.max_dur = max_dur
        
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim, 
            d_model=d_hid, 
            nlayers=nlayers, 
            dropout=dropout
        )
        
        # LSTM layers
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(d_hid // 2, return_sequences=True),
            merge_mode='concat'
        )
        
        self.duration_proj = LinearNorm(d_hid, max_dur)
        
        self.shared = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(d_hid // 2, return_sequences=True),
            merge_mode='concat'
        )
        
        # F0 prediction blocks - Note: AdainResBlk1d not implemented here
        # This would need to be implemented based on the istftnet.py conversion
        # For now, using placeholder Dense layers - THIS IS A CONVERSION ISSUE
        self.f0_blocks = []
        self.f0_blocks.append(tf.keras.layers.Dense(d_hid, activation='relu'))  # Placeholder
        self.f0_blocks.append(tf.keras.layers.Dense(d_hid // 2, activation='relu'))  # Placeholder
        self.f0_blocks.append(tf.keras.layers.Dense(d_hid // 2, activation='relu'))  # Placeholder
        
        # N (energy) prediction blocks - similar placeholder issue
        self.n_blocks = []
        self.n_blocks.append(tf.keras.layers.Dense(d_hid, activation='relu'))  # Placeholder
        self.n_blocks.append(tf.keras.layers.Dense(d_hid // 2, activation='relu'))  # Placeholder  
        self.n_blocks.append(tf.keras.layers.Dense(d_hid // 2, activation='relu'))  # Placeholder
        
        # Final projection layers
        self.f0_proj = tf.keras.layers.Conv1D(1, 1, padding='same')
        self.n_proj = tf.keras.layers.Conv1D(1, 1, padding='same')

    def call(self, texts, style, text_lengths, alignment, m, training=None):
        d = self.text_encoder(texts, style, text_lengths, m, training=training)
        
        # LSTM processing - Note: no pack_padded_sequence equivalent
        x = self.lstm(d, training=training)
        
        # Duration prediction
        duration = self.duration_proj(tf.nn.dropout(x, rate=0.5 if training else 0.0))
        
        # Encode for F0/N prediction
        d_transposed = tf.transpose(d, [0, 2, 1])  # [batch, features, seq_len]
        en = tf.matmul(d_transposed, alignment)
        
        return tf.squeeze(duration, axis=-1), en

    def f0n_train(self, x, s, training=None):
        """F0 and N prediction during training."""
        # Shared LSTM processing
        x_transposed = tf.transpose(x, [0, 2, 1])  # [batch, seq_len, features]
        x_lstm, _ = self.shared(x_transposed, training=training)
        
        # F0 prediction path
        f0 = tf.transpose(x_lstm, [0, 2, 1])  # [batch, features, seq_len]
        for block in self.f0_blocks:
            # Note: These are placeholder Dense layers, not the actual AdainResBlk1d
            # This is a significant conversion issue that needs proper implementation
            f0_reshaped = tf.transpose(f0, [0, 2, 1])  # For Dense layer
            f0_reshaped = block(f0_reshaped)
            f0 = tf.transpose(f0_reshaped, [0, 2, 1])
        
        f0 = self.f0_proj(tf.transpose(f0, [0, 2, 1]))
        f0 = tf.squeeze(f0, axis=-1)
        
        # N (energy) prediction path
        n = tf.transpose(x_lstm, [0, 2, 1])  # [batch, features, seq_len]
        for block in self.n_blocks:
            # Note: Same placeholder issue as F0 blocks
            n_reshaped = tf.transpose(n, [0, 2, 1])  # For Dense layer
            n_reshaped = block(n_reshaped)
            n = tf.transpose(n_reshaped, [0, 2, 1])
        
        n = self.n_proj(tf.transpose(n, [0, 2, 1]))
        n = tf.squeeze(n, axis=-1)
        
        return f0, n

    def get_config(self):
        config = super().get_config()
        config.update({
            'style_dim': self.style_dim,
            'd_hid': self.d_hid,
            'max_dur': self.max_dur
        })
        return config


