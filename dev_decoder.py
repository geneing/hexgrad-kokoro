# CUDA_VISIBLE_DEVICES="" uv run python dev_decoder.py 2>&1 | tee log.log


from multiprocessing import pool
import os
from turtle import color
import matplotlib.pyplot as plt

import keras
from regex import E, F
from sympy import N
import pickle as pkl
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
    print(f"[DIAG] Torch weight shape: {weight.shape}, min: {weight.min()}, max: {weight.max()}, mean: {weight.mean()}")
    if bias is not None:
        print(f"[DIAG] Torch bias shape: {bias.shape}, min: {bias.min()}, max: {bias.max()}, mean: {bias.mean()}")
    kernel = np.transpose(weight, (2, 1, 0)).astype(np.float32)
    print(f"[DIAG] Kernel (to TF) shape: {kernel.shape}, min: {kernel.min()}, max: {kernel.max()}, mean: {kernel.mean()}")
    _ensure_conv1d_built(tf_conv, weight.shape[1])
    if bias is not None:
        tf_conv.set_weights([kernel, bias.astype(np.float32)])
    else:
        tf_conv.set_weights([kernel])
    # After setting, print TF weights
    tf_weights = tf_conv.get_weights()
    print(f"[DIAG] TF kernel shape: {tf_weights[0].shape}, min: {tf_weights[0].min()}, max: {tf_weights[0].max()}, mean: {tf_weights[0].mean()}")
    if bias is not None and len(tf_weights) > 1:
        print(f"[DIAG] TF bias shape: {tf_weights[1].shape}, min: {tf_weights[1].min()}, max: {tf_weights[1].max()}, mean: {tf_weights[1].mean()}")


def _ensure_conv1d_transpose_built(layer, in_channels):
    if getattr(layer, 'built', False):
        return
    data_format = getattr(layer, 'data_format', 'channels_last')
    if data_format == 'channels_first':
        layer.build((None, in_channels, None))
    else:
        layer.build((None, None, in_channels))


def _copy_conv1d_transpose_weights(torch_conv, tf_conv):
    weight = torch_conv.weight.detach().cpu().numpy()
    bias = torch_conv.bias.detach().cpu().numpy() if torch_conv.bias is not None else None
    kernel = np.transpose(weight, (2, 1, 0)).astype(np.float32)
    _ensure_conv1d_transpose_built(tf_conv, weight.shape[0])
    weights = [kernel]
    if bias is not None:
        weights.append(bias.astype(np.float32))
    tf_conv.set_weights(weights)


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


def _convert_adain_resblock1(torch_block, tf_block):
    for idx, (torch_conv, tf_conv) in enumerate(zip(torch_block.convs1, tf_block.convs1)):
        _copy_conv1d_weights(torch_conv, tf_conv)
        if idx == 0:
            print("[DEBUG] Torch conv1 weight shape:", torch_conv.weight.shape)
            print("[DEBUG] TF conv1 kernel shape:", tf_conv.kernel.shape)
            print("[DEBUG] Torch conv1 weight (slice):", torch_conv.weight.detach().cpu().numpy().flatten()[:10])
            print("[DEBUG] TF conv1 kernel (slice):", tf_conv.kernel.numpy().flatten()[:10])
    for idx, (torch_conv, tf_conv) in enumerate(zip(torch_block.convs2, tf_block.convs2)):
        _copy_conv1d_weights(torch_conv, tf_conv)
        if idx == 0:
            print("[DEBUG] Torch conv2 weight shape:", torch_conv.weight.shape)
            print("[DEBUG] TF conv2 kernel shape:", tf_conv.kernel.shape)
            print("[DEBUG] Torch conv2 weight (slice):", torch_conv.weight.detach().cpu().numpy().flatten()[:10])
            print("[DEBUG] TF conv2 kernel (slice):", tf_conv.kernel.numpy().flatten()[:10])
    for torch_adain, tf_adain in zip(torch_block.adain1, tf_block.adain1):
        _convert_adain1d(torch_adain, tf_adain)
    for torch_adain, tf_adain in zip(torch_block.adain2, tf_block.adain2):
        _convert_adain1d(torch_adain, tf_adain)
    for torch_alpha, tf_alpha in zip(torch_block.alpha1, tf_block.alpha1):
        tf_alpha.assign(torch_alpha.detach().cpu().numpy().astype(np.float32))
    for torch_alpha, tf_alpha in zip(torch_block.alpha2, tf_block.alpha2):
        tf_alpha.assign(torch_alpha.detach().cpu().numpy().astype(np.float32))


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

    # Source module dense layer
    try:
        _copy_dense_weights(
            decoder_torch.generator.m_source.l_linear,
            decoder_tf.generator.m_source.l_linear,
        )
    except Exception as exc:
        print(f"[convert_decoder_weights] Skipped m_source.l_linear copy: {exc}")

    gen_torch = decoder_torch.generator
    gen_tf = decoder_tf.generator

    # Upsample transposed convolutions
    for idx, (torch_layer, tf_layer) in enumerate(zip(gen_torch.ups, gen_tf.ups)):
        try:
            _copy_conv1d_transpose_weights(torch_layer, tf_layer)
        except Exception as exc:
            print(f"[convert_decoder_weights] Skipped generator.ups[{idx}] copy: {exc}")

    # Generator residual blocks
    for idx, (torch_block, tf_block) in enumerate(zip(gen_torch.resblocks, gen_tf.resblocks)):
        try:
            _convert_adain_resblock1(torch_block, tf_block)
        except Exception as exc:
            print(f"[convert_decoder_weights] Skipped generator.resblocks[{idx}] copy: {exc}")

    # Noise convolution branches
    for idx, (torch_conv, tf_conv) in enumerate(zip(gen_torch.noise_convs, gen_tf.noise_convs)):
        try:
            _copy_conv1d_weights(torch_conv, tf_conv)
        except Exception as exc:
            print(f"[convert_decoder_weights] Skipped generator.noise_convs[{idx}] copy: {exc}")

    for idx, (torch_block, tf_block) in enumerate(zip(gen_torch.noise_res, gen_tf.noise_res)):
        try:
            _convert_adain_resblock1(torch_block, tf_block)
        except Exception as exc:
            print(f"[convert_decoder_weights] Skipped generator.noise_res[{idx}] copy: {exc}")

    # Generator post conv (other generator weights pending parity work)
    try:
        _copy_conv1d_weights(gen_torch.conv_post, gen_tf.conv_post)
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
        plt.plot(torch, color='red', alpha=.5)
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

with open("debug_decoder_torch.pkl", "rb") as f:
        dbg = pkl.load(f)

with open("debug_decoder_tf.pkl", "rb") as f:
        dbg_tf = pkl.load(f)
print(f"{dbg.keys()=}")


def compare_stft_module():
    stft_torch = kmodel_torch.decoder.generator.stft
    stft_tf = model.decoder.generator.stft_c

    har_source = dbg['har_source']

    har_source_torch = har_source.transpose(1, 2).squeeze(1)
    har_spec_torch, har_phase_torch = stft_torch.transform(har_source_torch)

    # har_spec_torch, har_phase_torch = har_spec_torch[:,:,1:], har_phase_torch[:,:,1:]
    print(f"har_source_torch shape: {har_source_torch.shape}")
    print(f"har_source shape: {har_source.shape}")

    har_source_tf = tf.squeeze(tf.transpose(har_source, [0, 2, 1]), axis=1) 
    print(f"har_source_tf shape: {har_source_tf.shape}")

    har_spec_tf, har_phase_tf = stft_tf.transform(har_source_tf)

    har_spec_tf = har_spec_tf.numpy()
    har_phase_tf = har_phase_tf.numpy()
    print(f"har_spec_torch shape: {har_spec_torch.shape}")
    print(f"har_spec_tf shape: {har_spec_tf.shape}")

    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 12))
    har_torch = har_spec_torch[:,:,:]
    har_tf = har_spec_tf[:,:,:]

    for i in range(11):
        plt.subplot(11,1,i+1)
        plt.plot(har_torch[0,i,:].detach().cpu().numpy(), label='PyTorch')
        plt.plot(har_tf[0,i,:], label='TensorFlow', color='orange', alpha=0.3)
        dff = har_tf[0,i,:] - har_torch[0,i,:].detach().cpu().numpy()
        # plt.plot(dff, label='difference', color='red')
        plt.title(f'Frame {i}')

    plt.savefig('stft_difference.png')


    sig_torch = stft_torch.inverse(har_spec_torch, har_phase_torch)
    print(f"{sig_torch.shape=}")
    sig_tf = stft_tf.inverse(har_spec_torch, har_phase_torch)
    print(f"{sig_tf.shape=}")

    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 6))
    plt.subplot(3,1,1)
    plt.plot(sig_torch[0,0,0:].detach().cpu().numpy(), label='PyTorch')
    plt.subplot(3,1,2)
    plt.plot(sig_tf[0,0,5:], label='TensorFlow', color='orange', alpha=0.5)
    plt.subplot(3,1,3)
    plt.plot(har_source_tf[0,:], label='Original Signal', color='green', alpha=0.5)
    plt.savefig('sig_difference.png')



##########################################################################################
# additional debugging functions below. Ignore for now.
###########################################################################################
def compare_generator_module():
    gen_torch = kmodel_torch.decoder.generator
    gen_tf = model.decoder.generator

    (x, s, f0) = dbg['gen_input']

    torch_dbg ={}
    audio_torch, torch_dbg = gen_torch( x, s, f0, torch_dbg)

    tf_dbg ={}
    # har_spec = dbg['har_spec'].detach().cpu().numpy()
    # har_phase = dbg['har_phase'].detach().cpu().numpy()   
    # x_relu = dbg['conv_relu'].detach().cpu().numpy()
    audio_tf, tf_dbg = gen_tf( x.detach().numpy(), s.detach().numpy(), f0.detach().numpy(), tf_dbg, training=None)

    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    # plt.plot(x_tf.numpy()[0,0,:], color='blue')
    # plt.plot(x_torch.detach().cpu().numpy()[0,0,:], color='red')

    plt.plot(audio_torch[0,0,:].detach().cpu().numpy(), label='PyTorch Audio')
    plt.title('PyTorch Generated Audio')
    plt.subplot(2,1,2)
    # plt.plot(diff[0,0,:], label='Difference', color='green')
    plt.plot(audio_tf[0,0,:].numpy(), label='TensorFlow Audio', color='orange')
    plt.title('TensorFlow Generated Audio')
    plt.savefig('audio_comparison.png')


    print(f"tf_dbg keys: {tf_dbg.keys()}")
    for key in tf_dbg.keys():
        if (key in torch_dbg) and (not key.endswith('gen_input')):
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


compare_generator_module()

def compare_resblock_module():  
    res_torch = kmodel_torch.decoder.generator.resblocks[0]
    res_tf = model.decoder.generator.resblocks[0]

    x_torch = dbg['resblock_in_x_1_5']
    s_torch = dbg['resblock_in_s_1_5']
    x_tf = dbg_tf['resblock_in_x_1_5']
    s_tf = dbg_tf['resblock_in_s_1_5']

    dff = (x_tf.numpy() - x_torch.detach().cpu().numpy())[0,:,:]
    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 6))
    plt.imshow(dff, aspect='auto', cmap='bwr')
    plt.colorbar()
    plt.title('Difference in resblock_in_x_0_0')
    plt.savefig('x_difference.png')

    # sys.exit(0)

    print(f"x_torch: {x_torch[0:10]=}")
    print(f"x_tf: {x_tf[0:10]=}")
    print(f"s_torch: {s_torch[0:10]=}")
    print(f"s_tf: {s_tf[0:10]=}")

    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.plot(x_torch[0,0,:].detach().cpu().numpy(), label='PyTorch')
    plt.plot(x_tf[0,0,:].numpy(), label='TensorFlow', color='orange', alpha=0.5)
    dff = x_tf[0,0,:].numpy() - x_torch[0,0,:].detach().cpu().numpy()
    plt.plot(dff, label='difference', color='red')
    plt.subplot(2,1,2)
    plt.plot(x_torch[0,100,:].detach().cpu().numpy(), label='PyTorch')
    plt.plot(x_tf[0,100,:].numpy(), label='TensorFlow', color='orange', alpha=0.5)
    dff = x_tf[0,100,:].numpy() - x_torch[0,100,:].detach().cpu().numpy()
    plt.plot(dff, label='difference', color='red')
    # dff = sine_wavs[0,:,0].numpy() - sine_wavs_t[0,:,0].detach().cpu().numpy()
    # plt.plot(dff, label='difference')
    plt.savefig('x_comparison.png')
    print(f"max difference in x: {np.max(np.abs(x_tf.numpy() - x_torch.detach().cpu().numpy()))}  ")



    # --- Normalization diagnostics ---
    print(f"[DEBUG] res_torch type: {type(res_torch)}")
    print(f"[DEBUG] res_torch dir: {dir(res_torch)}")
    # If adain1 and alpha1 are ModuleList or list, try to access first element
    adain1_torch = None
    alpha1_torch = None
    if hasattr(res_torch, 'adain1'):
        print(f"[DEBUG] res_torch.adain1 type: {type(res_torch.adain1)}")
        print(f"[DEBUG] res_torch.adain1 dir: {dir(res_torch.adain1)}")
        try:
            adain1_torch = res_torch.adain1[0]
            print(f"[DEBUG] adain1_torch type: {type(adain1_torch)}")
        except Exception as e:
            print(f"[ERROR] Could not access res_torch.adain1[0]: {e}")
            adain1_torch = None
    else:
        adain1_torch = None
    if hasattr(res_torch, 'alpha1'):
        print(f"[DEBUG] res_torch.alpha1 type: {type(res_torch.alpha1)}")
        print(f"[DEBUG] res_torch.alpha1 dir: {dir(res_torch.alpha1)}")
        try:
            alpha1_torch = res_torch.alpha1[0]
            print(f"[DEBUG] alpha1_torch type: {type(alpha1_torch)}")
        except Exception as e:
            print(f"[ERROR] Could not access res_torch.alpha1[0]: {e}")
            alpha1_torch = None
    else:
        alpha1_torch = None
    # Continue only if both are found and callable/indexable
    if adain1_torch is not None and alpha1_torch is not None:
        adain1_tf = res_tf.adain1[0]
        x_np = x_torch.detach().cpu().numpy()
        s_np = s_torch.detach().cpu().numpy() if hasattr(s_torch, 'detach') else s_torch
        try:
            norm_torch = adain1_torch(x_torch, s_torch).detach().cpu().numpy()
        except Exception as e:
            print(f"[ERROR] Could not call adain1_torch: {e}")
            norm_torch = None
        norm_tf = adain1_tf(x_np, s_np)
        if hasattr(norm_tf, 'numpy'):
            norm_tf = norm_tf.numpy()
        if norm_torch is not None:
            print(f"[DIAG] AdaIN1d norm_torch: min={norm_torch.min()}, max={norm_torch.max()}, mean={norm_torch.mean()}")
            print(f"[DIAG] AdaIN1d norm_tf: min={norm_tf.min()}, max={norm_tf.max()}, mean={norm_tf.mean()}")
            print(f"[DIAG] AdaIN1d norm max diff: {np.max(np.abs(norm_tf - norm_torch))}")

            # --- Snake1D activation diagnostics ---
            alpha_torch = alpha1_torch.detach().cpu().numpy()
            alpha_tf = res_tf.alpha1[0].numpy() if hasattr(res_tf.alpha1[0], 'numpy') else res_tf.alpha1[0]
            snake_torch = norm_torch + (1.0 / alpha_torch) * np.sin(alpha_torch * norm_torch) ** 2
            snake_tf = res_tf._snake_activation(norm_tf, alpha_tf)
            if hasattr(snake_tf, 'numpy'):
                snake_tf = snake_tf.numpy()
            print(f"[DIAG] Snake1D snake_torch: min={snake_torch.min()}, max={snake_torch.max()}, mean={snake_torch.mean()}")
            print(f"[DIAG] Snake1D snake_tf: min={snake_tf.min()}, max={snake_tf.max()}, mean={snake_tf.mean()}")
            print(f"[DIAG] Snake1D max diff: {np.max(np.abs(snake_tf - snake_torch))}")

    xs = res_torch(x_torch, s_torch)
    xs_tf = res_tf(x_torch.detach().numpy(), s_torch, training=False)
    print(f"xs shape: {xs.shape}")
    print(f"xs_tf shape: {xs_tf.shape}")
    print(f"xs_torch: {xs[0:10]=}")
    print(f"xs_tf: {xs_tf[0:10]=}")

    dff = xs_tf - xs.detach().cpu().numpy()
    print(f"max difference: {np.max(np.abs(dff))} {np.max(np.abs(dff))/np.mean(np.abs(xs.detach().cpu().numpy()))}")


def compare_source_module():    
    ms_torch = kmodel_torch.decoder.generator.m_source
    ms_tf = model.decoder.generator.m_source
            
    f0_upsampled = dbg['f0_upsampled']

    har_source_torch, noi_source_torch, uv_torch = ms_torch(f0_upsampled)
    har_source, noi_source, uv = ms_tf(f0_upsampled)
    print(f"har_source_torch shape: {har_source_torch.shape}")
    print(f"har_source shape: {har_source.shape}")
    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.plot(har_source_torch.squeeze().detach().cpu().numpy(), label='PyTorch Har Source')
    plt.subplot(2,1,2)
    plt.plot(har_source[0,:,0].numpy(), label='TensorFlow Har Source', linestyle='dashed')
    plt.savefig('har_source_comparison.png')

    sine_wavs_t, uv_t, _ = ms_torch.l_sin_gen(f0_upsampled)
    sine_wavs, uv, _ = ms_tf.l_sin_gen(f0_upsampled)

    print(f"torch shape: {sine_wavs_t.shape}")
    print(f"tf shape: {sine_wavs.shape}")

    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.plot(sine_wavs_t[0,:,0].detach().cpu().numpy(), label='PyTorch')
    plt.plot(sine_wavs[0,:,0].numpy(), label='TensorFlow', color='orange', alpha=0.5)
    plt.subplot(2,1,2)
    dff = sine_wavs[0,:,0].numpy() - sine_wavs_t[0,:,0].detach().cpu().numpy()
    plt.plot(dff, label='difference')
    plt.savefig('sine_wavs_comparison.png')

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

    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    # plt.plot(x_tf.numpy()[0,0,:], color='blue')
    # plt.plot(x_torch.detach().cpu().numpy()[0,0,:], color='red')

    plt.plot(audio_ref[:].detach().cpu().numpy(), label='PyTorch Audio')
    plt.title('PyTorch Generated Audio')
    plt.subplot(2,1,2)
    # plt.plot(diff[0,0,:], label='Difference', color='green')
    plt.plot(audio[0,0,:].numpy(), label='TensorFlow Audio', color='orange')
    plt.title('TensorFlow Generated Audio')
    plt.savefig('audio_comparison.png')

    def compare1():
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
                
    compare1()
