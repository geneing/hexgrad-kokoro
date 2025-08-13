# Kokoro TTS - TensorFlow Keras Implementation

This directory contains a TensorFlow Keras implementation of the Kokoro TTS system, converted from the original PyTorch implementation. The conversion includes the requested replacement of `AlbertModel` with `TFAlbertModel`.

## 🔄 Conversion Overview

This is a comprehensive conversion from PyTorch to TensorFlow Keras with the following key changes:

### ✅ Completed Components

1. **Model Architecture** (`model_keras.py`)
   - ✅ `KModelTF` - Main TTS model converted to TensorFlow
   - ✅ `TFAlbertModel` replacement for `AlbertModel` as requested
   - ✅ Forward pass logic adapted for TensorFlow operations

2. **Neural Modules** (`modules_keras.py`)
   - ✅ `CustomTFAlbert` - TensorFlow wrapper for TFAlbertModel
   - ✅ `TextEncoder` - Conv1D + LSTM encoder
   - ✅ `ProsodyPredictor` - Duration and prosody prediction
   - ✅ `LinearNorm`, `LayerNorm`, `AdaLayerNorm` - Normalization layers

3. **Vocoder Components** (`istftnet_keras.py`)
   - ✅ `Decoder` - Main decoder network
   - ✅ `Generator` - Neural vocoder generator
   - ✅ `AdaINResBlock1` - Adaptive instance norm residual blocks
   - ✅ `SineGen` - Harmonic sine wave generation
   - ✅ `SourceModuleHnNSF` - Harmonic + noise source

4. **STFT Operations** (`custom_stft_keras.py`)
   - ✅ `CustomSTFT` - Conv1D-based STFT implementation
   - ✅ `TorchSTFTTF` - tf.signal-based alternative

5. **Pipeline** (`pipeline_keras.py`)
   - ✅ `KPipelineTF` - Language-aware text processing pipeline
   - ✅ Voice management and caching

## ⚠️ Conversion Challenges & Notes

### 1. **AlbertModel → TFAlbertModel Replacement**
- **Status**: ✅ **Complete**
- **Implementation**: Successfully replaced `AlbertModel` with `TFAlbertModel` in `CustomTFAlbert` class
- **Location**: `modules_keras.py` line 233-248
- **Comment Added**: "Using TFAlbertModel instead of AlbertModel as requested"

### 2. **Weight Loading** 🚨 **Major Conversion Issue**
- **Status**: ❌ **Incomplete**
- **Challenge**: PyTorch `.pth` files cannot be directly loaded into TensorFlow
- **Solution Needed**: Manual weight mapping and conversion utility
- **Comments Added**: Lines referencing weight loading issues

### 3. **Layer Behavior Differences** ⚠️ **Potential Accuracy Issues**
- **Instance Normalization**: TensorFlow lacks direct equivalent to PyTorch's `InstanceNorm1d`
- **Weight Normalization**: No direct TensorFlow equivalent to PyTorch's `weight_norm`
- **Padding Modes**: Different behavior for reflection padding
- **Comments Added**: Throughout `istftnet_keras.py` noting these differences

### 4. **Sequence Processing** ⚠️ **Conversion Issues**
- **pack_padded_sequence**: No TensorFlow equivalent
- **Variable length sequences**: Handled differently in TensorFlow
- **Comments Added**: In `modules_keras.py` TextEncoder and DurationEncoder

### 5. **STFT Implementation** ⚠️ **Potential Accuracy Differences**
- **Custom STFT**: Converted to use Conv1D operations but may have numerical differences
- **Complex Numbers**: TensorFlow handles complex operations differently
- **Comments Added**: Throughout `custom_stft_keras.py`

### 6. **G2P Integration** ❌ **Incomplete**
- **Status**: Placeholder implementation only
- **Challenge**: Requires integration with misaki, espeak-ng libraries
- **Comments Added**: In `pipeline_keras.py` G2P functions

## 📁 File Structure

```
kokoro-litert/
├── __init__.py              # Main package exports
├── model_keras.py           # Main TTS model (KModelTF)
├── modules_keras.py         # Neural network modules with TFAlbert
├── istftnet_keras.py        # Vocoder and decoder components
├── custom_stft_keras.py     # STFT/iSTFT implementations
├── pipeline_keras.py        # Text processing pipeline
├── requirements.txt         # Dependencies
├── example_usage.py         # Usage example
└── README.md               # This file
```

## 🔧 Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **For G2P support** (language-specific):
```bash
# English
pip install misaki[en]

# Japanese  
pip install misaki[ja]

# Chinese
pip install misaki[zh]

# Other languages
pip install espeak-ng-python
```

## 💻 Usage Example

```python
import tensorflow as tf
from kokoro_litert import KModelTF, KPipelineTF

# Initialize model with TFAlbert
model = KModelTF(repo_id='hexgrad/Kokoro-82M')

# Initialize pipeline
pipeline = KPipelineTF(model=model, lang='en-us')

# Convert text to phonemes
phonemes = pipeline.text_to_phonemes("Hello world")

# Note: Synthesis requires weight conversion implementation
```

## 🚧 Known Issues & TODOs

### Critical Issues:
1. **Weight Loading**: PyTorch → TensorFlow weight conversion not implemented
2. **G2P Libraries**: Only dummy implementation provided
3. **Numerical Validation**: Need to verify outputs match PyTorch implementation

### Layer-Specific Issues:
4. **InstanceNorm1d**: Using LayerNormalization as approximation
5. **Weight Normalization**: Not available in TensorFlow
6. **pack_padded_sequence**: Different sequence handling approach needed

### Comments in Code:
- 🔍 **Search for "conversion issue"** in code for specific problem areas
- 🔍 **Search for "Note:"** for TensorFlow vs PyTorch differences
- 🔍 **Search for "TFAlbertModel"** for the requested model replacement

## 🎯 Next Steps for Complete Conversion

1. **Implement Weight Converter**:
   ```python
   def convert_pytorch_weights(pytorch_path, tensorflow_model):
       # Load PyTorch checkpoint
       # Map layer names PyTorch → TensorFlow  
       # Convert tensor formats
       # Load into TensorFlow model
   ```

2. **Integrate Real G2P**:
   - Add proper misaki/espeak-ng integration
   - Handle language-specific phoneme mapping

3. **Numerical Validation**:
   - Compare outputs with PyTorch implementation
   - Ensure STFT operations match
   - Validate normalization behaviors

4. **Performance Optimization**:
   - TensorFlow-specific optimizations
   - Memory usage analysis
   - GPU utilization improvements

## 📝 Conversion Summary

| Component | Status | TFAlbert Used | Notes |
|-----------|--------|---------------|-------|
| Model Architecture | ✅ Complete | ✅ Yes | TFAlbertModel replacement done |
| Weight Loading | ❌ Missing | N/A | Major conversion challenge |
| Text Processing | ⚠️ Partial | N/A | G2P needs real implementation |
| Neural Vocoder | ✅ Complete | N/A | Some layer approximations |
| STFT Operations | ⚠️ Converted | N/A | Potential accuracy differences |

**Overall Status**: 🟡 **Architecture Complete, Implementation Needs Work**

The requested `AlbertModel` → `TFAlbertModel` replacement has been successfully implemented, but additional work is needed for a fully functional system.
