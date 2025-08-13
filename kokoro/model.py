"""
TensorFlow Keras implementation of Kokoro TTS main model.
Converted from PyTorch implementation with TFAlbert replacement.
"""

import tensorflow as tf
from transformers import TFAlbertModel, AlbertConfig
from .modules import CustomTFAlbert, ProsodyPredictor, TextEncoder
from .istftnet import Decoder
from typing import Dict, Optional, Union
import json
import numpy as np
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger


class KModelTF(tf.keras.Model):
    """
    TensorFlow Keras implementation of KModel.
    
    Main responsibilities:
    1. Init weights, downloading config.json + model weights from HF if needed
    2. forward(phonemes: str, ref_s: tf.Tensor) -> (audio: tf.Tensor)
    
    Note: This is converted from PyTorch and replaces AlbertModel with TFAlbertModel.
    """

    MODEL_NAMES = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False,
        **kwargs
    ):
        super(KModelTF, self).__init__(**kwargs)
        
        if repo_id is None:
            repo_id = 'hexgrad/Kokoro-82M'
            print(f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning.")
        
        self.repo_id = repo_id
        
        # Load configuration
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=repo_id, filename='config.json')
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        
        self.vocab = config['vocab']
        self.config_dict = config
        
        # Initialize model components
        # Note: Using TFAlbertModel instead of AlbertModel as requested
        albert_config = AlbertConfig(vocab_size=config['n_token'], **config['plbert'])
        self.bert = CustomTFAlbert(albert_config)
        
        # BERT encoder - TensorFlow Dense layer instead of PyTorch Linear
        self.bert_encoder = tf.keras.layers.Dense(
            config['hidden_dim'], 
            input_shape=(self.bert.config.hidden_size,)
        )
        
        self.context_length = self.bert.config.max_position_embeddings
        
        # Prosody predictor
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], 
            d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], 
            max_dur=config['max_dur'], 
            dropout=config['dropout']
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], 
            kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], 
            n_symbols=config['n_token']
        )
        
        # Decoder - this will need to be implemented in istftnet.py
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], 
            style_dim=config['style_dim'],
            dim_out=config['n_mels'], 
            disable_complex=disable_complex, 
            **config['istftnet']
        )
        
        # Load pre-trained weights if available
        # Note: Loading PyTorch weights into TensorFlow model is complex
        # This is a major conversion issue that would require weight mapping
        if not model:
            model = hf_hub_download(repo_id=repo_id, filename=KModelTF.MODEL_NAMES[repo_id])
        
        # TODO: Implement weight loading from PyTorch checkpoint
        # This is a significant conversion challenge
        logger.warning("Weight loading from PyTorch checkpoint not implemented - conversion issue")

    @property 
    def device(self):
        """Get device info - TensorFlow equivalent of PyTorch device property."""
        # Note: TensorFlow handles devices differently than PyTorch
        return '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

    @dataclass
    class Output:
        audio: tf.Tensor
        pred_dur: Optional[tf.Tensor] = None

    def call(self, inputs, training=None):
        """
        Main forward pass of the model.
        
        Args:
            inputs: Dictionary containing:
                - input_ids: tf.Tensor of shape [batch, seq_len]
                - ref_s: tf.Tensor of shape [batch, ref_dim] 
                - speed: float (default 1.0)
        """
        input_ids = inputs['input_ids']
        ref_s = inputs['ref_s'] 
        speed = inputs.get('speed', 1.0)
        
        return self.forward_with_tokens(input_ids, ref_s, speed, training=training)

    def forward_with_tokens(
        self,
        input_ids: tf.Tensor,
        ref_s: tf.Tensor,
        speed: float = 1.0,
        training=None
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Forward pass with token inputs."""
        
        batch_size = tf.shape(input_ids)[0]
        seq_len = tf.shape(input_ids)[1]
        
        # Create input lengths tensor
        input_lengths = tf.fill([batch_size], seq_len)
        input_lengths = tf.cast(input_lengths, tf.int32)
        
        # Create attention mask
        # Note: TensorFlow mask creation differs from PyTorch
        max_len = tf.reduce_max(input_lengths)
        sequence_mask = tf.sequence_mask(input_lengths, max_len, dtype=tf.bool)
        text_mask = tf.logical_not(sequence_mask)  # Invert for masked positions
        
        # BERT processing
        attention_mask = tf.cast(tf.logical_not(text_mask), tf.int32)
        bert_dur = self.bert(input_ids, attention_mask=attention_mask, training=training)
        
        # Encoder processing - Note: TensorFlow Dense vs PyTorch Linear difference
        d_en = self.bert_encoder(bert_dur)
        d_en = tf.transpose(d_en, [0, 2, 1])  # Transpose for conv processing
        
        # Style processing
        s = ref_s[:, 128:]  # Take style portion
        
        # Duration prediction through predictor
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask, training=training)
        x = self.predictor.lstm(d, training=training)
        duration = self.predictor.duration_proj(x)
        
        # Duration processing - Note: TensorFlow operations vs PyTorch
        duration = tf.nn.sigmoid(duration)
        duration = tf.reduce_sum(duration, axis=-1) / speed
        pred_dur = tf.round(duration)
        pred_dur = tf.maximum(pred_dur, 1.0)  # Clamp minimum to 1
        pred_dur = tf.cast(pred_dur, tf.int32)
        pred_dur = tf.squeeze(pred_dur)
        
        # Alignment computation - This is complex in TensorFlow
        # Note: tf.repeat_interleave equivalent and alignment creation
        # This is a potential conversion issue due to dynamic shapes
        seq_indices = tf.range(seq_len, dtype=tf.int32)
        
        # Create alignment matrix - simplified version
        # Note: This is a simplified implementation and may not match PyTorch exactly
        max_duration = tf.reduce_max(pred_dur)
        pred_aln_trg = tf.zeros([batch_size, seq_len, max_duration], dtype=tf.float32)
        
        # F0 and N prediction
        F0_pred, N_pred = self.predictor.f0n_train(d, s, training=training)
        
        # Text encoder processing
        t_en = self.text_encoder(input_ids, input_lengths, text_mask, training=training)
        asr = tf.matmul(t_en, pred_aln_trg)
        
        # Audio generation through decoder
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128], training=training)
        audio = tf.squeeze(audio)
        
        return audio, pred_dur

    def predict_text(
        self,
        phonemes: str,
        ref_s: tf.Tensor,
        speed: float = 1.0,
        return_output: bool = False
    ) -> Union['KModelTF.Output', tf.Tensor]:
        """
        Predict audio from phoneme string.
        
        Args:
            phonemes: String of phonemes
            ref_s: Reference style tensor
            speed: Speaking speed multiplier
            return_output: Whether to return Output dataclass
        """
        # Convert phonemes to input_ids
        input_ids = list(filter(
            lambda i: i is not None, 
            map(lambda p: self.vocab.get(p), phonemes)
        ))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        
        # Check context length
        assert len(input_ids) + 2 <= self.context_length, (len(input_ids) + 2, self.context_length)
        
        # Add start/end tokens and convert to tensor
        input_ids = tf.constant([[0] + input_ids + [0]], dtype=tf.int32)
        
        # Forward pass
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed, training=False)
        
        logger.debug(f"pred_dur: {pred_dur}")
        
        if return_output:
            return self.Output(audio=audio, pred_dur=pred_dur)
        else:
            return audio

    def get_config(self):
        """Get model configuration for serialization."""
        return {
            'repo_id': self.repo_id,
            'config': self.config_dict,
            'context_length': self.context_length
        }


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
