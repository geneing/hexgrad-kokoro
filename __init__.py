"""
Kokoro TTS - TensorFlow Keras Implementation

A neural text-to-speech system converted from PyTorch to TensorFlow Keras.
Uses TFAlbertModel instead of AlbertModel as requested.

Main components:
- KModelTF: Main TTS model with TFAlbert encoder
- KPipelineTF: Language-aware pipeline for text processing
- Custom layers for neural vocoding and synthesis

Note: This is a conversion from PyTorch with several conversion challenges:
1. Weight loading from PyTorch checkpoints requires manual mapping
2. Some layer behaviors differ between frameworks (normalization, padding, etc.)
3. G2P (grapheme-to-phoneme) libraries may need separate integration
4. STFT/iSTFT implementations use different underlying operations
"""

from .kokoro import (
    KModelTF, KModelForONNXTF, KPipelineTF,
    CustomAlbert, ProsodyPredictor, #TextEncoder, 
    LinearNorm, LayerNorm, AdaLayerNorm, DurationEncoder,
    Decoder, Generator, AdaIN1d, 
    AdainResBlk1d, SineGen, SourceModuleHnNSF,
    CustomSTFT, TorchSTFTTF
)

__version__ = "1.0.0-tf"
__author__ = "Converted to TensorFlow Keras"

# Export main classes
__all__ = [
    'KModelTF',
    'KModelForONNXTF', 
    'KPipelineTF',
    'CustomAlbert',
    'ProsodyPredictor',
    #'TextEncoder',
    'Decoder',
    'Generator',
    'CustomSTFT',
    'TorchSTFTTF'
]

# Conversion notes for users
CONVERSION_NOTES = """
IMPORTANT CONVERSION NOTES:

1. Model Weights:
   - PyTorch .pth files need manual conversion to TensorFlow format
   - Weight mapping between frameworks requires careful attention to layer naming
   
2. AlbertModel Replacement:
   - Successfully replaced with TFAlbertModel as requested
   - Behavior should be equivalent but may have minor numerical differences
   
3. Layer Differences:
   - Some normalization layers have different default behaviors
   - Padding modes may differ between frameworks
   - Conv1D parameter ordering differs (filters vs channels)
   
4. STFT Operations:
   - Custom STFT implementation converted but may have accuracy differences
   - TensorFlow tf.signal.stft provides alternative but different windowing
   
5. G2P Integration:
   - Grapheme-to-phoneme conversion needs language-specific library integration
   - Current implementation uses placeholder functions
   
6. Training Compatibility:
   - Model architecture converted but training loop would need TensorFlow implementation
   - Loss functions and optimizers need separate conversion
   
7. Performance:
   - Memory usage patterns may differ between frameworks
   - GPU utilization characteristics may vary
   
For production use, thorough testing and validation against PyTorch outputs is recommended.
"""

def print_conversion_notes():
    """Print important notes about the PyTorch to TensorFlow conversion."""
    print(CONVERSION_NOTES)

def get_conversion_status():
    """Get status of conversion completeness."""
    return {
        'model_architecture': 'Complete',
        'albert_replacement': 'Complete - using TFAlbertModel', 
        'weight_loading': 'Incomplete - needs manual mapping',
        'stft_operations': 'Complete - with potential accuracy differences',
        'g2p_integration': 'Incomplete - needs library integration',
        'training_compatibility': 'Architecture only - training code needed',
        'testing_status': 'Needs validation against PyTorch outputs'
    }
