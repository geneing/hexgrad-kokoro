"""
TensorFlow Keras implementation of Kokoro TTS pipeline.
Converted from PyTorch implementation.
"""

import tensorflow as tf
from .model import KModelTF
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from typing import Callable, Generator, List, Optional, Tuple, Union
import re
import os

# Language aliases and codes from original implementation
ALIASES = {
    'en-us': 'a',
    'en-gb': 'b', 
    'es': 'e',
    'fr-fr': 'f',
    'hi': 'h',
    'it': 'i',
    'pt-br': 'p',
    'ja': 'j',
    'zh': 'z',
}

LANG_CODES = dict(
    # pip install misaki[en] 
    a='American English',
    b='British English',

    # espeak-ng
    e='es',
    f='fr-fr', 
    h='hi',
    i='it',
    p='pt-br',

    # pip install misaki[ja]
    j='Japanese',

    # pip install misaki[zh]
    z='Mandarin Chinese',
)


class KPipelineTF:
    """
    TensorFlow Keras implementation of KPipeline.
    
    Language-aware support class with 2 main responsibilities:
    1. Perform language-specific G2P, mapping (and chunking) text -> phonemes
    2. Manage and store voices, lazily downloaded from HF if needed
    
    You are expected to have one KPipelineTF per language. If you have multiple
    KPipelineTF instances, you should reuse one KModelTF instance across all of them.
    
    Note: This is converted from PyTorch and may have different G2P behavior.
    """

    def __init__(
        self,
        model: KModelTF,
        lang: str = 'en-us',
        voices_dir: Optional[str] = None,
        speed: float = 1.0
    ):
        """
        Initialize TensorFlow Kokoro pipeline.
        
        Args:
            model: KModelTF instance (shared across pipelines)
            lang: Language code or alias
            voices_dir: Directory to store/load voices from
            speed: Default speaking speed multiplier
        """
        self.model = model
        self.speed = speed
        self.voices_dir = voices_dir or 'voices'
        
        # Resolve language code
        if lang in ALIASES:
            self.lang_code = ALIASES[lang]
        else:
            self.lang_code = lang
            
        if self.lang_code not in LANG_CODES:
            raise ValueError(f"Unsupported language: {lang}. Supported: {list(LANG_CODES.keys())}")
            
        self.lang_name = LANG_CODES[self.lang_code]
        
        # Initialize G2P function based on language
        # Note: This is a simplified version - full conversion would need proper G2P libraries
        self._init_g2p()
        
        # Voice cache
        self._voices = {}
        
        logger.info(f"Initialized TensorFlow Kokoro pipeline for {self.lang_name}")

    def _init_g2p(self):
        """Initialize grapheme-to-phoneme conversion for the language."""
        # Note: This is a placeholder implementation
        # Full conversion would require integrating misaki, espeak-ng, etc.
        # This is a significant conversion challenge
        
        if self.lang_code in ['a', 'b']:  # English
            try:
                # Placeholder for misaki English G2P
                logger.warning("English G2P not fully implemented - conversion issue")
                self.g2p_fn = self._dummy_g2p
            except ImportError:
                logger.error("misaki[en] not available for English G2P")
                self.g2p_fn = self._dummy_g2p
                
        elif self.lang_code in ['e', 'f', 'h', 'i', 'p']:  # espeak-ng languages
            try:
                # Placeholder for espeak-ng integration
                logger.warning("espeak-ng G2P not fully implemented - conversion issue")
                self.g2p_fn = self._dummy_g2p
            except ImportError:
                logger.error("espeak-ng not available")
                self.g2p_fn = self._dummy_g2p
                
        elif self.lang_code == 'j':  # Japanese
            try:
                # Placeholder for misaki Japanese G2P
                logger.warning("Japanese G2P not fully implemented - conversion issue")
                self.g2p_fn = self._dummy_g2p
            except ImportError:
                logger.error("misaki[ja] not available for Japanese G2P")
                self.g2p_fn = self._dummy_g2p
                
        elif self.lang_code == 'z':  # Chinese
            try:
                # Placeholder for misaki Chinese G2P
                logger.warning("Chinese G2P not fully implemented - conversion issue")
                self.g2p_fn = self._dummy_g2p
            except ImportError:
                logger.error("misaki[zh] not available for Chinese G2P")
                self.g2p_fn = self._dummy_g2p
        else:
            self.g2p_fn = self._dummy_g2p

    def _dummy_g2p(self, text: str) -> str:
        """
        Dummy G2P function - returns basic phoneme representation.
        This is a conversion issue that needs proper implementation.
        """
        logger.warning("Using dummy G2P - conversion not complete")
        
        # Very basic character-to-phoneme mapping for demonstration
        # This is NOT suitable for production use
        char_map = {
            'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u',
            'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'g',
            'h': 'h', 'j': 'dʒ', 'k': 'k', 'l': 'l', 'm': 'm',
            'n': 'n', 'p': 'p', 'q': 'k', 'r': 'r', 's': 's',
            't': 't', 'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'j', 'z': 'z',
            ' ': ' '
        }
        
        phonemes = []
        for char in text.lower():
            if char in char_map:
                phonemes.append(char_map[char])
            else:
                phonemes.append(char)  # Keep unknown characters
                
        return ''.join(phonemes)

    def load_voice(self, voice_name: str, repo_id: Optional[str] = None) -> tf.Tensor:
        """
        Load a voice embedding tensor.
        
        Args:
            voice_name: Name of the voice file
            repo_id: HuggingFace repository ID (optional)
            
        Returns:
            Voice embedding tensor
        """
        if voice_name in self._voices:
            return self._voices[voice_name]
            
        # Try to load from local directory first
        voice_path = os.path.join(self.voices_dir, f"{voice_name}.pt")
        
        if os.path.exists(voice_path):
            # Note: Loading PyTorch tensor in TensorFlow is a conversion issue
            # Would need to convert .pt files to TensorFlow format
            logger.warning(f"Loading PyTorch voice file {voice_path} - conversion needed")
            # Placeholder: load as numpy then convert to tf.Tensor
            try:
                import torch
                voice_tensor = torch.load(voice_path, map_location='cpu')
                voice_tf = tf.constant(voice_tensor.numpy())
            except ImportError:
                logger.error("PyTorch not available to load voice file")
                # Create dummy voice embedding
                voice_tf = tf.random.normal([1, 256])  # Placeholder dimensions
        else:
            # Download from HuggingFace if repo_id provided
            if repo_id:
                try:
                    voice_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=f"{voice_name}.pt"
                    )
                    # Same conversion issue as above
                    logger.warning("Downloaded PyTorch voice - conversion needed")
                    voice_tf = tf.random.normal([1, 256])  # Placeholder
                except Exception as e:
                    logger.error(f"Failed to download voice {voice_name}: {e}")
                    voice_tf = tf.random.normal([1, 256])  # Placeholder
            else:
                logger.error(f"Voice {voice_name} not found locally and no repo_id provided")
                voice_tf = tf.random.normal([1, 256])  # Placeholder
                
        self._voices[voice_name] = voice_tf
        return voice_tf

    def text_to_phonemes(self, text: str, chunk_size: Optional[int] = None) -> Union[str, List[str]]:
        """
        Convert text to phonemes using language-specific G2P.
        
        Args:
            text: Input text
            chunk_size: Optional chunk size for long texts
            
        Returns:
            Phonemes string or list of phoneme chunks
        """
        # Clean and preprocess text
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        if not text:
            return ""
            
        # Apply G2P conversion
        phonemes = self.g2p_fn(text)
        
        # Chunk if requested
        if chunk_size and len(phonemes) > chunk_size:
            chunks = []
            words = phonemes.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= chunk_size:
                    current_chunk.append(word)
                    current_length += len(word) + 1
                else:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                    
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
            return chunks
        else:
            return phonemes

    def synthesize(
        self,
        text: str,
        voice: Union[str, tf.Tensor],
        speed: Optional[float] = None,
        return_output: bool = False
    ) -> Union[tf.Tensor, 'KModelTF.Output']:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text
            voice: Voice name (str) or voice tensor
            speed: Speaking speed multiplier (uses default if None)
            return_output: Whether to return Output dataclass
            
        Returns:
            Audio tensor or Output dataclass
        """
        if speed is None:
            speed = self.speed
            
        # Load voice if string provided
        if isinstance(voice, str):
            voice_tensor = self.load_voice(voice)
        else:
            voice_tensor = voice
            
        # Convert text to phonemes
        phonemes = self.text_to_phonemes(text)
        
        # Synthesize audio
        if return_output:
            return self.model.predict_text(phonemes, voice_tensor, speed, return_output=True)
        else:
            return self.model.predict_text(phonemes, voice_tensor, speed, return_output=False)

    def synthesize_batch(
        self,
        texts: List[str],
        voices: Union[List[str], List[tf.Tensor], str, tf.Tensor],
        speed: Optional[float] = None
    ) -> List[tf.Tensor]:
        """
        Synthesize multiple texts in batch.
        
        Args:
            texts: List of input texts
            voices: Voice(s) - can be single voice for all or list matching texts
            speed: Speaking speed multiplier
            
        Returns:
            List of audio tensors
        """
        if speed is None:
            speed = self.speed
            
        # Handle single voice for all texts
        if isinstance(voices, (str, tf.Tensor)):
            voices = [voices] * len(texts)
            
        assert len(texts) == len(voices), "Number of texts and voices must match"
        
        results = []
        for text, voice in zip(texts, voices):
            audio = self.synthesize(text, voice, speed, return_output=False)
            results.append(audio)
            
        return results

    def get_available_voices(self) -> List[str]:
        """Get list of available voice names."""
        voices = []
        
        # Scan local voices directory
        if os.path.exists(self.voices_dir):
            for file in os.listdir(self.voices_dir):
                if file.endswith('.pt'):
                    voices.append(file[:-3])  # Remove .pt extension
                    
        # Add cached voices
        voices.extend(self._voices.keys())
        
        return sorted(list(set(voices)))

    def preload_voices(self, voice_names: List[str], repo_id: Optional[str] = None):
        """Preload multiple voices into cache."""
        for voice_name in voice_names:
            self.load_voice(voice_name, repo_id)
            
        logger.info(f"Preloaded {len(voice_names)} voices")

    def clear_voice_cache(self):
        """Clear cached voices to free memory."""
        self._voices.clear()
        logger.info("Cleared voice cache")

    def __repr__(self):
        return f"KPipelineTF(lang={self.lang_code}, voices_cached={len(self._voices)})"
