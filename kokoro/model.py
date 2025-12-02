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
import keras

@keras.saving.register_keras_serializable()
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
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], 
            style_dim=config['style_dim'],
            dim_out=config['n_mels'], 
            disable_complex=False, 
            **config['istftnet']
        )
        
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

    def build(self, input_shape):
        """Configure dynamic input shapes for the model and pre-build submodules.

        The caller may pass the shapes either as a dict (SavedModel style) or as
        a tuple/list following the positional argument order of ``call``.  The
        ``seq_len`` dimension in ``input_ids`` must remain dynamic, while
        ``ref_s`` keeps a known channel size.  ``speed`` is treated as a scalar
        float input.
        """

        # Normalize the incoming shape signature into TensorShape objects.
        if isinstance(input_shape, dict):
            shape_spec = {key: tf.TensorShape(value) for key, value in input_shape.items()}
            try:
                input_ids_shape = shape_spec['input_ids']
                ref_s_shape = shape_spec['ref_s']
            except KeyError as exc:
                raise ValueError("KModelTF.build expects 'input_ids' and 'ref_s' shapes") from exc
            speed_shape = shape_spec.get('speed', tf.TensorShape([]))
            shape_structure = 'dict'
        else:
            if not isinstance(input_shape, (tuple, list)):
                raise ValueError("KModelTF.build expects a dict or a tuple/list of shapes")
            shapes = list(input_shape)
            if len(shapes) < 2:
                raise ValueError("KModelTF.build requires at least input_ids and ref_s shapes")
            if len(shapes) == 2:
                shapes.append(())  # default scalar for speed
            input_ids_shape = tf.TensorShape(shapes[0])
            ref_s_shape = tf.TensorShape(shapes[1])
            speed_shape = tf.TensorShape(shapes[2])
            shape_structure = 'list'

        # Normalize shapes to rank-2 (batch, feature) while keeping seq_len dynamic.
        if input_ids_shape.rank is None:
            input_ids_shape = tf.TensorShape([None, None])
        elif input_ids_shape.rank == 1:
            seq_len_dim = input_ids_shape.as_list()[0]
            input_ids_shape = tf.TensorShape([None, seq_len_dim])
        if ref_s_shape.rank is None:
            raise ValueError("ref_s shape must be known up to rank; received unknown rank")
        elif ref_s_shape.rank == 1:
            style_dim = ref_s_shape.as_list()[0]
            ref_s_shape = tf.TensorShape([None, style_dim])
        if input_ids_shape.rank != 2:
            raise ValueError(f"input_ids must be rank-2, got shape {input_ids_shape}")
        if ref_s_shape.rank != 2:
            raise ValueError(f"ref_s must be rank-2, got shape {ref_s_shape}")

        batch_dim = input_ids_shape[0]
        ref_dim = ref_s_shape[-1]
        if ref_dim is None:
            raise ValueError("ref_s last dimension must be known when building KModelTF")

        # seq_len remains dynamic (None) regardless of the build-time shape.
        dynamic_input_ids_shape = tf.TensorShape([batch_dim, None])
        dynamic_ref_s_shape = tf.TensorShape([batch_dim, ref_dim])

        # speed is treated as a scalar float input. Allow (), (1,), or unknown rank 0.
        if speed_shape.rank in (None, 0):
            scalar_shape = tf.TensorShape([])
        elif speed_shape.rank == 1:
            shape_list = speed_shape.as_list()
            first_dim_value = shape_list[0] if shape_list else None
            if first_dim_value not in (None, 1):
                raise ValueError(f"speed must be scalar, received shape {speed_shape}")
            scalar_shape = tf.TensorShape([])
        else:
            raise ValueError(f"speed must be scalar, received shape {speed_shape}")

        # Register InputSpecs so Keras shape checks honour the dynamic sequence length.
        # Pre-build simple submodules whose weight shapes are known statically.
        if not self.bert_encoder.built:
            self.bert_encoder.build(tf.TensorShape([None, self.bert.config.hidden_size]))

        # Propagate the adjusted shapes to the superclass so Keras marks the model as built.
        if shape_structure == 'dict':
            normalized_shapes = {
                'input_ids': dynamic_input_ids_shape,
                'ref_s': dynamic_ref_s_shape,
                'speed': scalar_shape,
            }
        else:
            normalized_shapes = [dynamic_input_ids_shape, dynamic_ref_s_shape, scalar_shape]

        super(KModelTF, self).build(normalized_shapes)

    def call(self, input_ids: tf.Tensor, ref_s: tf.Tensor, n_inputs: tf.int32, speed: tf.float32 = 1.0, training=False):
        """
        Main forward pass of the model.
        
        Args:
            inputs: Dictionary containing:
                - input_ids: tf.Tensor of shape [batch, seq_len]
                - ref_s: tf.Tensor of shape [batch, ref_dim] 
                - speed: float (default 1.0)
        """
        
        # BERT processing
        n_inputs = tf.cast(tf.squeeze(n_inputs), tf.int32)
        input_ids = tf.slice(input_ids, [0, 0], [1, n_inputs])
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
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128], training=training)
        audio = tf.squeeze(audio)

        # return bert_dur, d_en, d, x, expanded_indices, en, F0_pred, N_pred, t_en, asr, audio
        
        return audio

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
