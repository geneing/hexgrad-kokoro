from multiprocessing import pool
import os
import matplotlib.pyplot as plt

import keras
from regex import E, F
from sympy import N
import torch
from torch.nn.utils import remove_weight_norm
import numpy as np

def _ensure_conv1d_built(layer, in_channels):
    if getattr(layer, 'built', False):
        return
    data_format = getattr(layer, 'data_format', 'channels_last')
    if data_format == 'channels_first':
        layer.build((None, in_channels, None))
    else:
        layer.build((None, None, in_channels))


def _copy_conv1d_weights(torch_conv, tf_conv):
    weight = torch_conv.weight.detach().cpu().numpy()
    bias = torch_conv.bias.detach().cpu().numpy() if torch_conv.bias is not None else None
    kernel = np.transpose(weight, (2, 1, 0)).astype(np.float32)
    _ensure_conv1d_built(tf_conv, weight.shape[1])
    if bias is not None:
        tf_conv.set_weights([kernel, bias.astype(np.float32)])
    else:
        tf_conv.set_weights([kernel])


def _copy_dense_weights(torch_linear, tf_dense):
    weight = torch_linear.weight.detach().cpu().numpy()
    bias = torch_linear.bias.detach().cpu().numpy() if torch_linear.bias is not None else None
    _ensure_dense_built(tf_dense, weight.shape[1])
    weights = [weight.T.astype(np.float32)]
    if bias is not None:
        weights.append(bias.astype(np.float32))
    tf_dense.set_weights(weights)


def _ensure_dense_built(layer, in_features):
    if getattr(layer, 'built', False):
        return
    layer.build((None, in_features))


def _copy_group_norm(torch_norm, tf_norm):
    gamma = torch_norm.weight.detach().cpu().numpy()
    beta = torch_norm.bias.detach().cpu().numpy()
    if not tf_norm.built:
        tf_norm.build((None, gamma.shape[0], None))
    tf_norm.set_weights([gamma.astype(np.float32), beta.astype(np.float32)])


def _convert_adain1d(torch_adain, tf_adain):
    _copy_dense_weights(torch_adain.fc, tf_adain.fc)
    _copy_group_norm(torch_adain.norm, tf_adain.norm)


def _copy_depthwise_transpose(torch_layer, tf_layer):
    weight = torch_layer.weight.detach().cpu().numpy()  # [channels, 1, k]
    bias = torch_layer.bias.detach().cpu().numpy() if torch_layer.bias is not None else None
    channels, _, kernel_size = weight.shape
    kernel = np.zeros((kernel_size, channels, channels), dtype=np.float32)
    for c in range(channels):
        kernel[:, c, c] = weight[c, 0, :]
    if not tf_layer.built:
        tf_layer.build((None, channels, None))
    tf_layer.kernel.assign(kernel)
    if tf_layer.bias is not None:
        if bias is None:
            tf_layer.bias.assign(np.zeros_like(tf_layer.bias))
        else:
            tf_layer.bias.assign(bias.astype(np.float32))


def _convert_adain_resblk1d(torch_block, tf_block):
    _copy_conv1d_weights(torch_block.conv1, tf_block.conv1)
    _copy_conv1d_weights(torch_block.conv2, tf_block.conv2)
    _convert_adain1d(torch_block.norm1, tf_block.norm1)
    _convert_adain1d(torch_block.norm2, tf_block.norm2)
    if torch_block.learned_sc and hasattr(tf_block, 'conv1x1') and tf_block.conv1x1 is not None:
        _copy_conv1d_weights(torch_block.conv1x1, tf_block.conv1x1)
    if hasattr(torch_block, 'pool') and hasattr(tf_block, 'pool'):
        if isinstance(torch_block.pool, torch.nn.ConvTranspose1d) and isinstance(tf_block.pool, DepthwiseConv1DTranspose):
            _copy_depthwise_transpose(torch_block.pool, tf_block.pool)


def _remove_weight_norm_recursive(module):
    for child in module.children():
        _remove_weight_norm_recursive(child)
    try:
        remove_weight_norm(module)
    except (ValueError, AttributeError):
        pass


def convert_decoder_weights(kmodel_torch, model_tf):
    """Copy decoder weights from PyTorch model into TensorFlow model.

    Note: Layers must be built (e.g., by running a dummy forward pass) before calling.
    """
    decoder_torch = kmodel_torch.decoder
    decoder_tf = model_tf.decoder

    # Remove weight normalization wrappers before reading raw weights
    _remove_weight_norm_recursive(decoder_torch)

    # Core encoder/decoder blocks
    _convert_adain_resblk1d(decoder_torch.encode, decoder_tf.encode)
    for idx, (torch_block, tf_block) in enumerate(zip(decoder_torch.decode, decoder_tf.decode)):
        _convert_adain_resblk1d(torch_block, tf_block)

    # F0 / Noise convs
    _copy_conv1d_weights(decoder_torch.F0_conv, decoder_tf.f0_conv)
    _copy_conv1d_weights(decoder_torch.N_conv, decoder_tf.n_conv)

    # ASR residual
    _copy_conv1d_weights(decoder_torch.asr_res[0], decoder_tf.asr_res.layers[0])

    # Generator post conv (other generator weights pending parity work)
    try:
        _copy_conv1d_weights(decoder_torch.generator.conv_post, decoder_tf.generator.conv_post)
    except Exception as exc:
        print(f"[convert_decoder_weights] Skipped generator.conv_post copy: {exc}")


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import pickle
import tensorflow as tf
from kokoro_litert.kokoro import KModelTF
from kokoro_litert.kokoro.istftnet import DepthwiseConv1DTranspose
from kokoro_torch.kokoro import KModel
import csv
import json


def plot_differences(tf, dbg, title):
    import matplotlib.pyplot as plt
    #use png rendering backend
    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 6))
    plt.clf()
    # bert_dur.numpy().T 
    # dbg['bert_dur'].last_hidden_state.numpy()
    tf = (tf - dbg)
    tf = tf.squeeze()
    if len(tf.shape) > 2:
        tf = tf[0, :, :]
    if len(tf.shape) == 2:
        plt.imshow(tf, aspect='auto', cmap='bwr')
        plt.colorbar()
    if len(tf.shape) == 1:
        plt.plot(tf)
    plt.title(f'{title}')
    plt.savefig(f'{title}.png')
    plt.close()

def plot_plot(tf, torch, title):
    import matplotlib.pyplot as plt
    #use png rendering backend
    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 6))
    # bert_dur.numpy().T 
    # dbg['bert_dur'].last_hidden_state.numpy()
    tf = tf.squeeze()
    torch = torch.squeeze()
    if len(tf.shape) > 2:
        tf = tf[0, :, :]
        torch = torch[0, :, :]
    if len(tf.shape) == 2:
        plt.subplot(2,1,1)
        plt.imshow(tf, aspect='auto', cmap='bwr')
        plt.colorbar(label='value')
        plt.subplot(2,1,2)
        plt.imshow(torch, aspect='auto', cmap='bwr')
        plt.colorbar(label='value')
    if len(tf.shape) == 1:
        plt.plot(tf, color='blue')
        plt.plot(torch, color='red')
    plt.title(f'{title}')
    # plt.title('Difference between TensorFlow and PyTorch Outputs')
    # plt.xlabel('Sequence Length')
    # plt.ylabel('Hidden Size')
    plt.savefig(f'{title}.png')
    plt.close()



config_file = 'checkpoints/config.json'
checkpoint_path = 'checkpoints/kokoro-v1_0.pth'

with open('temp/dbg.pkl', 'rb') as f:
    dbg = pickle.load(f)

model = KModelTF(config_file)

inputs = {'input_ids': tf.convert_to_tensor(dbg['input_ids'].numpy()), 
          'input_mask': tf.cast(tf.math.not_equal(dbg['input_ids'], 0), tf.int32), 
          'ref_s': tf.convert_to_tensor(dbg['ref_s'].numpy()), 'input_type_ids': tf.zeros_like(dbg['input_ids'])}

# model.build(input_shape=[(None, dbg['input_ids'].shape[1]), (None, dbg['ref_s'].shape[1]), ()])

#copying weights for bert encoder
kmodel_torch = KModel(config=config_file, model=checkpoint_path, disable_complex=True)

# ############################################
# output = model( input_ids=tf.convert_to_tensor(dbg['input_ids'].numpy()), 
#                 ref_s=tf.convert_to_tensor(dbg['ref_s'].numpy()), 
#                 speed=dbg['speed'])


asr = dbg['asr']
F0_pred = dbg['F0_pred']
N_pred = dbg['N_pred']
ref_s = dbg['ref_s']
audio_ref = kmodel_torch.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
# print(f"{audio.shape=}")

print("\n\n\n########### TF Decoder #############\n\n\n")
convert_decoder_weights(kmodel_torch, model)
audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128], training=False)
print(f"{audio.shape=}")

f0c_torch = kmodel_torch.decoder.F0_conv
f0c_tf = model.decoder.f0_conv
# print(f"f0c_torch weight: {f0c_torch.weight.shape=}")
# print(f"f0c_tf weight: {f0c_tf.kernel.shape=}")

# print(f"f0c_torch weight: {f0c_torch.weight[0,0,0:3]}")
# print(f"f0c_tf weight: {f0c_tf.kernel.numpy()[0:3,0,0]}")

# x_torch = f0c_torch(F0_pred.unsqueeze(1))
# x_tf = f0c_tf(tf.expand_dims(F0_pred, axis=1), training=True)
# print(f"x_torch shape: {x_torch.shape}")
# print(f"x_tf shape: {x_tf.shape}")
# print(f"f0c_torch output: {x_torch[0,0,0:10]}")
# print(f"f0c_tf output: {x_tf[0,0,0:10].numpy()}")
# diff = x_tf.numpy() - x_torch.detach().cpu().numpy()
# print(f"f0c diff: {diff[0,0,:]}, max diff: {np.max(np.abs(diff))}")

# plt.switch_backend('Agg')
# plt.figure(figsize=(10, 6))
# plt.subplot(2,1,1)
# plt.plot(x_tf.numpy()[0,0,:], color='blue')
# plt.plot(x_torch.detach().cpu().numpy()[0,0,:], color='red')

# # plt.plot(audio_ref[:].detach().cpu().numpy(), label='PyTorch Audio')
# plt.title('PyTorch Generated Audio')
# plt.subplot(2,1,2)
# plt.plot(diff[0,0,:], label='Difference', color='green')
# # plt.plot(audio[0,0,:].numpy(), label='TensorFlow Audio', color='orange')
# plt.title('TensorFlow Generated Audio')
# plt.savefig('audio_comparison.png')

import pickle as pkl
with open("debug_decoder_tf.pkl", "rb") as f:
    tf_dbg = pkl.load(f)
    
with open("debug_decoder_torch.pkl", "rb") as f:
    torch_dbg = pkl.load(f)

print(f"tf_dbg keys: {tf_dbg.keys()}")
for key in tf_dbg.keys():
    if key in torch_dbg:
        tf_out = tf_dbg[key].numpy() if isinstance(tf_dbg[key], tf.Tensor) else tf_dbg[key]
        torch_out = torch_dbg[key].detach().cpu().numpy() if isinstance(torch_dbg[key], torch.Tensor) else torch_dbg[key]
        plot_plot(tf_out, torch_out, f'tf_{key}')
        if tf_out.shape == torch_out.shape:
            diff = tf_out - torch_out
            max_diff = np.max(np.abs(diff))
            print(f"{key}: max difference = {max_diff}")
            plot_differences(tf_out, torch_out, f'diff_{key}')
        else:
            print(f"{key}: shape mismatch for {tf_out.shape} vs {torch_out.shape}")
    else:
        print(f"{key}: not found in torch debug data")
