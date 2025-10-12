"""
TensorFlow Keras implementation of Kokoro TTS main model.
Converted from PyTorch implementation with TFAlbert replacement.
"""

import tensorflow as tf
from transformers import AlbertConfig

from .modules import CustomAlbert, ProsodyPredictor, TextEncoder

# from .modules import CustomTFAlbert, ProsodyPredictor
from .istftnet import Decoder
from typing import Dict, Optional, Union
import json
# import numpy as np
from dataclasses import dataclass
# from huggingface_hub import hf_hub_download
from loguru import logger
# import os



class KModelTF(tf.keras.Model):
    """
    TensorFlow Keras implementation of KModel.
    
    Main responsibilities:
    1. Init weights, downloading config.json + model weights from HF if needed
    2. forward(phonemes: str, ref_s: tf.Tensor) -> (audio: tf.Tensor)
    
    Note: This is converted from PyTorch and replaces AlbertModel with TFAlbertModel.
    """
    def __init__(
        self,
        config: str,
        **kwargs
    ):
        super(KModelTF, self).__init__(**kwargs)
        
        # Initialize type-annotated attributes
        self.vocab: Dict[str, int] = {}
        self.config_dict: Dict = {}
        self.context_length: int = 512
        
        # Load configuration
        with open(config, 'r', encoding='utf-8') as r:
            config = json.load(r)
            logger.debug(f"Loaded config: {config}")
        
        # Ensure config is a dictionary at this point
        if not isinstance(config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(config)}")
        
        self.vocab = config['vocab']  # type: ignore
        self.config_dict = config
        
        # Initialize model components
        # Note: Using TFAlbertModel instead of AlbertModel as requested
        
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))

        # # BERT encoder - TensorFlow Dense layer instead of PyTorch Linear
        self.bert_encoder = tf.keras.layers.Dense(
            config['hidden_dim'], 
            input_shape=(self.bert.config.hidden_size,)
        )
        
        self.context_length = self.bert.config.max_position_embeddings
        
        # # Prosody predictor
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], 
            d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], 
            max_dur=config['max_dur']
        )
        
        # # Text encoder
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], 
            kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], 
            n_symbols=config['n_token']
        )
        
        # # Decoder - this will need to be implemented in istftnet.py
        # self.decoder = Decoder(
        #     dim_in=config['hidden_dim'], 
        #     style_dim=config['style_dim'],
        #     dim_out=config['n_mels'], 
        #     disable_complex=False, 
        #     **config['istftnet']
        # )
        
        # # Load pre-trained weights if available
        # # Note: Loading PyTorch weights into TensorFlow model is complex
        # # This is a major conversion issue that would require weight mapping
        # if not model:
        #     model = hf_hub_download(repo_id=repo_id, filename=KModelTF.MODEL_NAMES[repo_id])
        
        # # Load and convert PyTorch weights to TensorFlow
        # if model and os.path.exists(model):
        #     logger.info(f"Loading PyTorch weights from {model}")
        #     self._convert_weights(model)
        # else:
        #     logger.warning("Weight loading from PyTorch checkpoint not implemented - conversion issue")


    @property
    def device(self):
        """Get device info - TensorFlow equivalent of PyTorch device property."""
        # Note: TensorFlow handles devices differently than PyTorch
        return '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

    @dataclass
    class Output:
        audio: tf.Tensor
        pred_dur: Optional[tf.Tensor] = None

    def call(self, input_ids: tf.Tensor, ref_s: tf.Tensor, speed: tf.float32 = 1.0, training=False):
        """
        Main forward pass of the model.
        
        Args:
            inputs: Dictionary containing:
                - input_ids: tf.Tensor of shape [batch, seq_len]
                - ref_s: tf.Tensor of shape [batch, ref_dim] 
                - speed: float (default 1.0)
        """
        
        # BERT processing
        inputs = {'input_ids': input_ids, 'token_type_ids': tf.zeros_like(input_ids)}
        bert_dur = self.bert(inputs, training=training)
        bert_dur = bert_dur.last_hidden_state
    
        # Encoder processing - Note: TensorFlow Dense vs PyTorch Linear difference
        d_en = self.bert_encoder(bert_dur)
        d_en = tf.transpose(d_en, perm=(0, 2, 1))  # Transpose for conv processing
    
        # Style processing
        s = ref_s[:, 128:]  # Take style portion
        
        # Duration prediction through predictor
        d = self.predictor.text_encoder(d_en, s, training=training)
        input_tensor = tf.transpose(d, perm=(0, 2, 1))
        
        x = self.predictor.lstm(d, training=training)
        duration = self.predictor.duration_proj(x)
        # Duration processing - Note: TensorFlow operations vs PyTorch
        duration = tf.nn.sigmoid(duration)
        speed = tf.cast(speed, dtype=duration.dtype)
        duration = tf.reduce_sum(duration, axis=-1) / speed
        pred_dur = tf.round(duration)
        pred_dur = tf.maximum(pred_dur, 1.0)  # Clamp minimum to 1
        pred_dur = tf.cast(pred_dur, tf.int32)
        pred_dur = tf.squeeze(pred_dur)
        
        # boundaries = torch.cumsum(pred_dur, dim=0)
        # values = torch.arange(boundaries[-1], device=pred_dur.device)
        # expanded_indices = torch.sum(boundaries.unsqueeze(1) <= values.unsqueeze(0), dim=0)
        # en = torch.index_select(input_tensor, 2, expanded_indices)
        
        boundaries = tf.math.cumsum(pred_dur, axis=0)

        values = tf.range(boundaries[-1], dtype=tf.int32)        
        expanded_indices = tf.reduce_sum(
            tf.cast(tf.expand_dims(boundaries, axis=1) <= tf.expand_dims(values, axis=0), tf.int32),
            axis=0
        )
        en = tf.gather(input_tensor, expanded_indices, axis=2)

        # F0 and N prediction
        F0_pred, N_pred = self.predictor.f0n_train(en, s, training=training)
        
        
        # Text encoder processing
        t_en = self.text_encoder(input_ids, training=training)
        asr = tf.gather(t_en, expanded_indices, axis=2)
        
        # # # Audio generation through decoder
        # audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128], training=training)
        # audio = tf.squeeze(audio)

        return bert_dur, d_en, d, x, expanded_indices, en, F0_pred, N_pred, t_en, asr
        
        # return audio, pred_dur

    # def predict_text(
    #     self,
    #     phonemes: str,
    #     ref_s: tf.Tensor,
    #     speed: float = 1.0,
    #     return_output: bool = False
    # ) -> Union['KModelTF.Output', tf.Tensor]:
    #     """
    #     Predict audio from phoneme string.
        
    #     Args:
    #         phonemes: String of phonemes
    #         ref_s: Reference style tensor
    #         speed: Speaking speed multiplier
    #         return_output: Whether to return Output dataclass
    #     """
    #     # Convert phonemes to input_ids
    #     input_ids = list(filter(
    #         lambda i: i is not None, 
    #         map(lambda p: self.vocab.get(p), phonemes)
    #     ))
    #     logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        
    #     # Check context length
    #     assert len(input_ids) + 2 <= self.context_length, (len(input_ids) + 2, self.context_length)
        
    #     # Add start/end tokens and convert to tensor
    #     input_ids = tf.constant([[0] + input_ids + [0]], dtype=tf.int32)
        
    #     # Forward pass
    #     audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed, training=False)
        
    #     logger.debug(f"pred_dur: {pred_dur}")
        
    #     if return_output:
    #         return self.Output(audio=audio, pred_dur=pred_dur)
    #     else:
    #         return audio


class KModelForONNXTF(tf.keras.Model):
    """TensorFlow version of ONNX-compatible model wrapper."""
    
    def __init__(self, kmodel: KModelTF, **kwargs):
        super(KModelForONNXTF, self).__init__(**kwargs)
        self.kmodel = kmodel

    def call(self, inputs, training=None):
        """
        Forward pass for ONNX export.
        
        Args:
            inputs: Dictionary containing:
                - input_ids: tf.Tensor
                - ref_s: tf.Tensor  
                - speed: float (optional)
        """
        input_ids = inputs['input_ids']
        ref_s = inputs['ref_s']
        speed = inputs.get('speed', 1.0)
        
        waveform, duration = self.kmodel.forward_with_tokens(
            input_ids, ref_s, speed, training=training
        )
        
        return {'waveform': waveform, 'duration': duration}

    def get_config(self):
        return {'kmodel_config': self.kmodel.get_config()}
