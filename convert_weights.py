from multiprocessing import pool
import os
import matplotlib.pyplot as plt

import keras
from regex import E
import torch
from torch.nn.utils import remove_weight_norm
from typing import cast

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import pickle
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
from kokoro_litert.kokoro import KModelTF
from kokoro_torch.kokoro import KModel
from kokoro_litert.kokoro.istftnet import DepthwiseConv1DTranspose
import csv
import json

def map_tf_to_torch_names(tf_name: str) -> str:
    tf_name = tf_name.replace('custom_albert/albert/', '')
    tf_name = tf_name.replace('k_model_tf/albert/', '')
    tf_name = tf_name.replace('embeddings:0', 'weight')
    tf_name = tf_name.replace(':0', '')
    tf_name = tf_name.replace('k_model_tf/custom_albert/albert/', '')
    tf_name = tf_name.replace('k_model_tf/', '')
    tf_name = tf_name.replace('_._', '.')
    tf_name = tf_name.replace('/kernel', '/weight')
    tf_name = tf_name.replace('gamma', 'weight')
    tf_name = tf_name.replace('beta', 'bias')
    tf_name = tf_name.replace('/', '.')
    return tf_name

def parse_shape_string(shape_str: str) -> tuple:
    """Converts a string like '[768, 2048]' to a tuple of ints (768, 2048)."""
    # Using json.loads is a safe and easy way to parse this string format.
    # It handles various spacings and is more robust than manual string splitting.
    return tuple(json.loads(shape_str))


def copy_weights_bert(model, kmodel_torch, dbg):
    bert_tf = model.bert
    bert_torch = kmodel_torch.bert

    s = bert_torch.state_dict()
    d = bert_tf.trainable_variables

    map_tf_names={}
    for i, v in enumerate(d):
        # print(f"{v.name};{list(v.shape)}")
        map_tf_names[v.name] = i
        
        
    # for (v, (name, param)) in zip(d, s.items()):
    #     torchname = map_tf_to_torch_names(v.name)
    #     param = s[torchname]
        
    #     print(f"{torchname};{map_tf_to_torch_names(v.name)};{list(param.shape)};{list(v.shape)}")

    # print(f"{dbg['input_ids']}")

    # output = kmodel_torch( dbg['input_ids'], dbg['ref_s'], dbg['speed'])
    # print(f'torch output: {output}')
    print(f"{s.keys()=}")
    for (v, (name, param)) in zip(d, s.items()):
        torchname = map_tf_to_torch_names(v.name)
        print(f"Mapping {v.name} to {torchname}")
        param = s[torchname]
        # tf_shape = list(v.shape)
        # torch_shape = list(param.shape)
        i = map_tf_names[v.name]
        value = s[torchname].numpy()
        #weights for kernels need to be transposed
        if "kernel" in v.name:
            value = value.T
        torch_shape = value.shape   
        tf_shape = d[i].shape 
        print(f"{i}:{torchname};{v.name};{torch_shape=};{tf_shape=}")
        if torch_shape == tf_shape:
            d[i].assign(tf.constant(value))
        elif torch_shape[0] == tf_shape[1] and torch_shape[1] == tf_shape[0]:
            d[i].assign(tf.constant(value.T))

    # inputs = {'input_word_ids': tf.convert_to_tensor(dbg['input_ids'].numpy()), 
    #         'input_mask': tf.cast(tf.math.not_equal(dbg['input_ids'], 0), tf.int32), 
    #         'ref_s': tf.convert_to_tensor(dbg['ref_s'].numpy()), 'input_type_ids': tf.zeros_like(dbg['input_ids'])}
    # output = model( input_ids=tf.convert_to_tensor(dbg['input_ids'].numpy()), 
    #                 ref_s=tf.convert_to_tensor(dbg['ref_s'].numpy()), 
    #                 speed=dbg['speed'] )
    


    # print(f'tf output: {output.last_hidden_state.shape=} {output.last_hidden_state=}')
    # print(f"torch_data: {dbg['bert_dur'].last_hidden_state.shape=} {dbg['bert_dur'].last_hidden_state=}")
    # v1 = output.last_hidden_state.numpy()
    # v2 = dbg['bert_dur'].last_hidden_state.numpy()
    # print(f"max diff: {abs(v1 - v2).max()}")
    # print(f"{v1-v2}")
    
def convert_bilstm_weights(pytorch_bilstm, tensorflow_bilstm, hidden_size):
    # Extract PyTorch weights
    pt_state_dict = pytorch_bilstm.state_dict()

    # Forward direction weights
    pt_weight_ih_fwd = pt_state_dict['weight_ih_l0'].detach().numpy()
    pt_weight_hh_fwd = pt_state_dict['weight_hh_l0'].detach().numpy()
    pt_bias_ih_fwd = pt_state_dict['bias_ih_l0'].detach().numpy()
    pt_bias_hh_fwd = pt_state_dict['bias_hh_l0'].detach().numpy()

    # Backward direction weights
    pt_weight_ih_bwd = pt_state_dict['weight_ih_l0_reverse'].detach().numpy()
    pt_weight_hh_bwd = pt_state_dict['weight_hh_l0_reverse'].detach().numpy()
    pt_bias_ih_bwd = pt_state_dict['bias_ih_l0_reverse'].detach().numpy()
    pt_bias_hh_bwd = pt_state_dict['bias_hh_l0_reverse'].detach().numpy()

    def split_weights_and_reorder(weights, hidden_size):
        # Split the concatenated weights into individual gates
        w_i, w_f, w_c, w_o = np.split(weights, 4, axis=0)
        # Concatenate in the correct gate order for TensorFlow
        return np.concatenate([w_i, w_f, w_c, w_o], axis=0)

    # Process Forward weights
    tf_weight_ih_fwd = split_weights_and_reorder(pt_weight_ih_fwd, hidden_size).T
    tf_weight_hh_fwd = split_weights_and_reorder(pt_weight_hh_fwd, hidden_size).T
    tf_bias_fwd = split_weights_and_reorder(pt_bias_ih_fwd + pt_bias_hh_fwd, hidden_size)

    # Process Backward weights
    tf_weight_ih_bwd = split_weights_and_reorder(pt_weight_ih_bwd, hidden_size).T
    tf_weight_hh_bwd = split_weights_and_reorder(pt_weight_hh_bwd, hidden_size).T
    tf_bias_bwd = split_weights_and_reorder(pt_bias_ih_bwd + pt_bias_hh_bwd, hidden_size)

    # The TensorFlow BiDirectional layer holds two LSTM layers
    fwd_lstm_layer = tensorflow_bilstm.forward_layer
    bwd_lstm_layer = tensorflow_bilstm.backward_layer

    # Assign weights for the forward layer
    fwd_lstm_layer.set_weights([tf_weight_ih_fwd, tf_weight_hh_fwd, tf_bias_fwd])

    # Assign weights for the backward layer
    bwd_lstm_layer.set_weights([tf_weight_ih_bwd, tf_weight_hh_bwd, tf_bias_bwd])


def convert_predictor_text_encoder(model, kmodel_torch):
    torch_layers = kmodel_torch.predictor.text_encoder.lstms #(d_en, s)
    tf_layers = model.predictor.text_encoder.lstms
    nlayers = kmodel_torch.predictor.text_encoder.nlayers
    hidden_size = kmodel_torch.predictor.text_encoder.lstms[0].hidden_size

    for i in range(nlayers):
        convert_bilstm_weights(torch_layers[2*i], tf_layers.get_layer(index=2*i), hidden_size)
        
        #copying weights for adalayernorm
        w = torch_layers[2*i+1].fc.weight.detach().numpy().T
        b = torch_layers[2*i+1].fc.bias.detach().numpy()
        
        model_weights = [w, b]
        tf_layers.get_layer(index=2*i+1).fc.set_weights(model_weights)
        
        
def convert_text_encoder(kmodel_torch, model_tf):
    """Copy TextEncoder weights from PyTorch model to TensorFlow model."""

    torch_encoder = kmodel_torch.text_encoder
    tf_encoder = model_tf.text_encoder

    # Embedding weights: PyTorch [n_symbols, channels] -> Keras identical layout.
    embedding_weights = torch_encoder.embedding.weight.detach().numpy()
    tf_encoder.embedding.set_weights([embedding_weights])

    # CNN blocks: Conv1d (with weight_norm) + LayerNorm.
    for idx, (torch_block, tf_block) in enumerate(zip(torch_encoder.cnn, tf_encoder.cnn_layers)):
        torch_conv = torch_block[0]
        tf_conv = tf_block.layers[0]

        torch_conv_r = remove_weight_norm(torch_conv)
        conv_weight = torch_conv_r.weight.detach().numpy()  # (out_channels, in_channels, kernel)
        conv_weight_f = torch_conv.weight.detach().numpy()  # (out_channels, in_channels, kernel)
        conv_bias = torch_conv_r.bias.detach().numpy() 
        conv_weight_tf = np.transpose(conv_weight, (2, 1, 0))  # -> (kernel, in_channels, out_channels)
        tf_conv.set_weights([conv_weight_tf, conv_bias])

        # x = np.random.randn(1, 512, 78).astype(np.float32)
        # x [0,10,10] = 1.
        
        # y_torch = torch_conv(torch.tensor(x)).detach().numpy()
        # # x_tf = tf.convert_to_tensor(np.transpose(x, (2, 1, 0)))
        # # y_tf = tf.transpose(tf_conv(x_tf, training=False), perm=[2, 1, 0]).numpy()
        
        # x_tf = tf.convert_to_tensor(x)
        # y_tf = tf_conv(x_tf, training=False).numpy()
        
        # plt.switch_backend('Agg')
        # plt.figure(figsize=(10, 6))
        # plt.subplot(3,1,1)
        # plt.imshow(y_torch[0,0:50,0:70], label='PyTorch Output');plt.colorbar()
        # plt.subplot(3,1,2)
        # plt.imshow(y_tf[0,0:50,0:70], label='TensorFlow Output');plt.colorbar()
        # plt.subplot(3,1,3)
        # plt.imshow(y_torch[0,0:50,0:70]-y_tf[0,0:50,0:70], label='Difference', cmap='bwr');plt.colorbar()  
        # plt.legend()
        # plt.savefig(f'conv1d.png')
        # print(f"{y_torch.shape=} {y_tf.shape=}\n\n{y_torch[0,0:2,0:2]=}\n{y_tf[0,0:2,0:2]=}\n\n")
        # print(f"Conv1D Block {idx}: max diff after conv: {np.abs(y_torch - y_tf).max()}")
        # sys.exit(0)
        
        # LayerNorm parameters
        torch_ln = torch_block[1]
        tf_ln = tf_block.layers[1]
        
        tf_ln.layer_norm.gamma.assign(torch_ln.gamma.detach().numpy())
        tf_ln.layer_norm.beta.assign(torch_ln.beta.detach().numpy())
        
        # print(f"******** {torch_ln=} {torch_ln.beta[0:5]=}")
        # print(f"******** {tf_ln=} {tf_ln.layer_norm.beta[0:5]=}")

        # atorch = torch.ones([1,512,78])
        # atf = tf.ones([1,512,78])
        # atorch = torch_ln(atorch)
        # atf = tf_ln(atf, training=False)
        # print(f"After LN: {atorch[0,0:2,0:2]=} {atf[0,0:2,0:2]=}")

    # LSTM weights: reuse shared conversion helper.
    hidden_size = torch_encoder.lstm.hidden_size
    convert_bilstm_weights(torch_encoder.lstm, tf_encoder.lstm, hidden_size)

        
def convert_adain(kmodel_torch_predictor, tf_predictor):
    """Convert / copy AdainResBlk1d weights (F0 / N branches + 1x1 projections).

    Args:
        kmodel_torch_predictor: ProsodyPredictor (PyTorch) with attributes F0, N, F0_proj, (optionally N_proj)
        tf_predictor: ProsodyPredictor (TF/Keras) with attributes F0_blocks, N_blocks, F0_proj, N_proj

    Notes:
        - PyTorch Conv1d weight shape: (out_channels, in_channels, kernel_size)
          Keras Conv1D kernel shape: (kernel_size, in_channels, out_channels)
        - Linear / Dense: PyTorch (out, in) -> Keras (in, out)
        - We attempt best-effort mapping by attribute names; if mismatch, a warning is printed.
        - AdaIN / style modulation: if a PyTorch block exposes fc / style fc weights, we copy to
          similarly named Keras attributes when shapes are compatible.
    """

    def copy_conv1d(pt_module, keras_conv):
        if pt_module is None or keras_conv is None:
            return
        try:
            w_torch = pt_module.weight.detach().numpy()  # (out,in,k)
            b_torch = pt_module.bias.detach().numpy() if pt_module.bias is not None else None
            w_keras = np.transpose(w_torch, (2, 1, 0))   # -> (k,in,out)
            weights = [w_keras]
            if b_torch is not None:
                weights.append(b_torch)
            keras_conv.set_weights(weights)
            print(f"[convert_adain] Copied Conv1d -> Conv1D weights {w_torch.shape} -> {w_keras.shape}")
        except Exception as e:
            print(f"[convert_adain][WARN] conv copy failed: {e}")

    def copy_convtranspose1d(pt_module, keras_conv):
        """Copy ConvTranspose1d (PyTorch) -> Conv1DTranspose (Keras) weights.
        PyTorch weight: (in_c, out_c, k) -> Keras kernel: (k, out_c, in_c).
        """
        if pt_module is None or keras_conv is None:
            return
        try:
            w = pt_module.weight.detach().numpy()  # (in,out,k)
            b = pt_module.bias.detach().numpy() if getattr(pt_module, 'bias', None) is not None else None
            
            dim_in = w.shape[0]           
            depthwise_kernel = np.zeros((w.shape[2], dim_in, dim_in), dtype=np.float32)
            print(f"{w.shape=}")
            for c in range(dim_in):
                depthwise_kernel[:, c, c] = w[c, 0, :].astype(np.float32)
            if b is None:
                bias_tf = np.zeros((depthwise_kernel.shape[2],), dtype=np.float32)
            else:
                bias_tf = b.astype(np.float32)
            keras_conv.set_weights([depthwise_kernel, bias_tf])

            print(f"[convert_adain] Copied ConvTranspose1d -> Conv1DTranspose weights {w.shape} -> {depthwise_kernel.shape}, {b.shape if b is not None else None}")
        except Exception as e:
            print(f"[convert_adain][WARN] convtranspose copy failed: {e}")
            
    def copy_dense(pt_module, keras_dense):
        if pt_module is None or keras_dense is None:
            return
        try:
            w_torch = pt_module.weight.detach().numpy()  # (out,in)
            b_torch = pt_module.bias.detach().numpy() if pt_module.bias is not None else None
            w_keras = w_torch.T  # -> (in,out)
            weights = [w_keras]
            if b_torch is not None:
                weights.append(b_torch)
            keras_dense.set_weights(weights)
            print(f"[convert_adain] Copied Linear -> Dense weights {w_torch.shape} -> {w_keras.shape}")
        except Exception as e:
            print(f"[convert_adain][WARN] dense copy failed: {e}")

    def copy_adain(pt_block, tf_block, tag=""):
        if pt_block is None or tf_block is None:
            return
        # Conv layers (if present)
        for name_pair in [("conv1", "conv1"), ("conv2", "conv2"), ("conv1x1", "conv1x1")]:
            pt_name, tf_name = name_pair
            pt_sub = getattr(pt_block, pt_name, None)
            tf_sub = getattr(tf_block, tf_name, None)
            if pt_sub is not None and tf_sub is not None:
                copy_conv1d(pt_sub, tf_sub)
        # Pool (ConvTranspose1d) if present
        pt_pool = getattr(pt_block, 'pool', None)
        tf_pool = getattr(tf_block, 'pool', None)
        if pt_pool is not None and tf_pool is not None and hasattr(pt_pool, 'weight'):
            copy_convtranspose1d(pt_pool, tf_pool)
            print(f"PoolCopy: {tf_pool=} \n{pt_pool=}")

        # Style / AdaIN fully-connected(s) at block top-level
        cand_pt = [n for n in ["fc", "style_fc", "style", "mod"] if hasattr(pt_block, n)]
        cand_tf = [n for n in ["fc", "style_fc", "style", "mod"] if hasattr(tf_block, n)]
        for pt_name in cand_pt:
            for tf_name in cand_tf:
                pt_obj = getattr(pt_block, pt_name)
                tf_obj = getattr(tf_block, tf_name)
                if hasattr(pt_obj, 'weight') and hasattr(tf_obj, 'kernel'):
                    try:
                        w = pt_obj.weight.detach().numpy()
                        b = pt_obj.bias.detach().numpy() if hasattr(pt_obj, 'bias') and pt_obj.bias is not None else None
                        tf_obj.set_weights([w.T, b])
                        print(f"[convert_adain] Copied style FC {tag}{pt_name}->{tf_name} {w.shape}->{w.T.shape}")
                        break
                    except Exception as e:
                        print(f"[convert_adain][WARN] style fc copy failed {pt_name}->{tf_name}: {e}")
            else:
                continue
            break
        # Traverse AdaIN1d modules inside the block (norm1/norm2 or adain1/adain2 lists) and copy their internal fc + norm gamma/beta
        def copy_inner_adain(pt_adain, tf_adain, subtag):
            if pt_adain is None or tf_adain is None:
                return
            # Copy fc inside AdaIN1d
            if hasattr(pt_adain, 'fc') and hasattr(tf_adain, 'fc') and \
               hasattr(pt_adain.fc, 'weight') and hasattr(tf_adain.fc, 'kernel'):
                try:
                    w = pt_adain.fc.weight.detach().numpy()  # (out,in)
                    b = pt_adain.fc.bias.detach().numpy() if pt_adain.fc.bias is not None else None
                    tf_adain.fc.set_weights([w.T, b])
                    print(f"[convert_adain] Copied AdaIN1d.fc {tag}{subtag} {w.shape}->{w.T.shape}")
                except Exception as e:
                    print(f"[convert_adain][WARN] AdaIN1d.fc copy failed {tag}{subtag}: {e}")
            # Copy InstanceNorm / normalization gamma(beta)/beta if present
            pt_norm = getattr(pt_adain, 'norm', None)
            tf_norm = getattr(tf_adain, 'norm', None)
            if pt_norm is not None and tf_norm is not None and hasattr(pt_norm, 'weight') and hasattr(pt_norm, 'bias'):
                try:
                    gamma = pt_norm.weight.detach().numpy()
                    beta = pt_norm.bias.detach().numpy()
                    assigned = False
                    if hasattr(tf_norm, 'gamma') and hasattr(tf_norm, 'beta'):
                        tf_norm.gamma.assign(gamma)
                        tf_norm.beta.assign(beta)
                        assigned = True
                    else:
                        try:
                            tw = tf_norm.get_weights()
                            if len(tw) == 2 and tw[0].shape == gamma.shape and tw[1].shape == beta.shape:
                                tf_norm.set_weights([gamma, beta])
                                assigned = True
                        except Exception:
                            pass
                    if assigned:
                        print(f"[convert_adain] Copied AdaIN1d.norm (gamma/beta) {tag}{subtag} {gamma.shape}")
                    else:
                        print(f"[convert_adain][WARN] Could not map norm gamma/beta for {tag}{subtag}")
                except Exception as e:
                    print(f"[convert_adain][WARN] norm gamma/beta copy failed {tag}{subtag}: {e}")
        # norm1 / norm2
        for nm in ['norm1','norm2']:
            copy_inner_adain(getattr(pt_block, nm, None), getattr(tf_block, nm, None), nm)
        # adain1 / adain2 could be lists/ModuleList
        for nm in ['adain1','adain2']:
            pt_list = getattr(pt_block, nm, None)
            tf_list = getattr(tf_block, nm, None)
            if pt_list is not None and tf_list is not None:
                for idx, (pt_a, tf_a) in enumerate(zip(pt_list, tf_list)):
                    copy_inner_adain(pt_a, tf_a, f"{nm}[{idx}]")

    # ---- Begin actual execution (previously missing) ----
    print('[convert_adain] Starting AdaIN weight conversion...')

    # F0 blocks
    pt_F0 = getattr(kmodel_torch_predictor, 'F0', None)
    tf_F0 = getattr(tf_predictor, 'F0_blocks', None)
    if pt_F0 is not None and tf_F0 is not None:
        for i, (pt_blk, tf_blk) in enumerate(zip(pt_F0, tf_F0)):
            print(f'[convert_adain] Copying F0 block {i}')
            copy_adain(pt_blk, tf_blk, tag=f'F0[{i}].')
        if len(pt_F0) != len(tf_F0):
            print(f"[convert_adain][WARN] F0 block count mismatch pt={len(pt_F0)} tf={len(tf_F0)}")
    else:
        print('[convert_adain][WARN] Missing F0 blocks on one side')

    # N blocks
    pt_N = getattr(kmodel_torch_predictor, 'N', None)
    tf_N = getattr(tf_predictor, 'N_blocks', None)
    if pt_N is not None and tf_N is not None:
        for i, (pt_blk, tf_blk) in enumerate(zip(pt_N, tf_N)):
            print(f'[convert_adain] Copying N block {i}')
            copy_adain(pt_blk, tf_blk, tag=f'N[{i}].')
        if len(pt_N) != len(tf_N):
            print(f"[convert_adain][WARN] N block count mismatch pt={len(pt_N)} tf={len(tf_N)}")
    else:
        print('[convert_adain][WARN] Missing N blocks on one side')

    # Projection layers
    pt_F0_proj = getattr(kmodel_torch_predictor, 'F0_proj', None)
    tf_F0_proj = getattr(tf_predictor, 'F0_proj', None)
    if pt_F0_proj is not None and tf_F0_proj is not None:
        print('[convert_adain] Copying F0_proj')
        copy_dense(pt_F0_proj, tf_F0_proj)
    else:
        print('[convert_adain][WARN] Missing F0_proj on one side')

    pt_N_proj = getattr(kmodel_torch_predictor, 'N_proj', None)
    tf_N_proj = getattr(tf_predictor, 'N_proj', None)
    if pt_N_proj is not None and tf_N_proj is not None:
        print('[convert_adain] Copying N_proj')
        copy_dense(pt_N_proj, tf_N_proj)
    else:
        print('[convert_adain][WARN] Missing N_proj on one side')

    print('[convert_adain] AdaIN weight conversion complete.')


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


























config_file = 'checkpoints/config.json'
checkpoint_path = 'checkpoints/kokoro-v1_0.pth'

with open('temp/dbg.pkl', 'rb') as f:
    dbg = pickle.load(f)

model = KModelTF(config_file)

inputs_shapes = {'input_ids': (1, 510),           
          'ref_s': (1, 256), 'speed': (1)}

print(f"[INFO] Building model with input shapes: {inputs_shapes}")
model.build(inputs_shapes)

inputs = {'input_ids': tf.convert_to_tensor(dbg['input_ids'].numpy()), 
          'input_mask': tf.cast(tf.math.not_equal(dbg['input_ids'], 0), tf.int32), 
          'ref_s': tf.convert_to_tensor(dbg['ref_s'].numpy()), 'input_type_ids': tf.zeros_like(dbg['input_ids'])}

# model.build(input_shape=[(None, dbg['input_ids'].shape[1]), (None, dbg['ref_s'].shape[1]), ()])

#copying weights for bert encoder
kmodel_torch = KModel(config=config_file, model=checkpoint_path, disable_complex=True)

# Prepare export inputs for both TensorFlow inference and TFLite conversion.
seq_len = 510  # dbg['input_ids'].shape[1]
style_dim = dbg['ref_s'].shape[1]

# Ensure the requested export length does not exceed BERT's positional capacity.
max_context_len = getattr(model, 'context_length', 512)
if seq_len > max_context_len:
    print(f"[INFO] Requested seq_len={seq_len} exceeds model context_length={max_context_len}. "
          f"Clamping to {max_context_len}.")
seq_len = min(seq_len, max_context_len)


def _pad_or_truncate(np_input_ids: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or truncate tokens to match the target sequence length."""
    current_len = np_input_ids.shape[1]
    if current_len == target_len:
        return np_input_ids
    if current_len > target_len:
        print(f"[INFO] Truncating input_ids from {current_len} to {target_len} tokens")
        return np_input_ids[:, :target_len]
    pad_width = target_len - current_len
    print(f"[INFO] Padding input_ids from {current_len} to {target_len} tokens")
    pad_block = np.zeros((np_input_ids.shape[0], pad_width), dtype=np_input_ids.dtype)
    return np.concatenate([np_input_ids, pad_block], axis=1)


def _to_numpy(value: torch.Tensor | tf.Tensor | np.ndarray | float | int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, tf.Tensor):
        return np.asarray(value)
    return np.asarray(value)


dbg_input_ids_np = _to_numpy(dbg['input_ids'])
n_inputs = dbg_input_ids_np.shape[1]
export_input_ids_np = _pad_or_truncate(dbg_input_ids_np, seq_len)


def _torch_from_numpy(np_input: np.ndarray) -> torch.LongTensor:
    return cast(torch.LongTensor, torch.from_numpy(np_input).long())


export_input_ids_torch = _torch_from_numpy(export_input_ids_np)
export_input_ids_tf = tf.convert_to_tensor(export_input_ids_np.astype(np.int32))
export_ref_s_np = _to_numpy(dbg['ref_s']).astype(np.float32)
export_ref_s_tf = tf.convert_to_tensor(export_ref_s_np)
speed_value = float(np.asarray(dbg['speed']).item())
export_speed_tf = tf.constant([[speed_value]], dtype=tf.float32)
export_n_inputs_tf = tf.constant([[n_inputs]], dtype=tf.int32)

# ############################################
output = model(
    input_ids=export_input_ids_tf,
    ref_s=export_ref_s_tf,
    n_inputs=n_inputs,
    speed=speed_value
)


model.bert_encoder.set_weights([kmodel_torch.bert_encoder.weight.detach().numpy().T, kmodel_torch.bert_encoder.bias.detach().numpy()])

hidden_size = kmodel_torch.predictor.lstm.hidden_size

copy_weights_bert(model=model, kmodel_torch=kmodel_torch, dbg=dbg)
convert_predictor_text_encoder(model=model, kmodel_torch=kmodel_torch)
convert_text_encoder(kmodel_torch=kmodel_torch, model_tf=model)
convert_bilstm_weights(kmodel_torch.predictor.lstm, model.predictor.lstm, hidden_size)
model.predictor.duration_proj.set_weights([kmodel_torch.predictor.duration_proj.linear_layer.weight.detach().numpy().T, kmodel_torch.predictor.duration_proj.linear_layer.bias.detach().numpy()])

convert_bilstm_weights(kmodel_torch.predictor.shared, model.predictor.shared_bilstm, hidden_size)

# Perform Adain weight copy and validation
try:
    convert_adain(kmodel_torch.predictor, model.predictor)
except Exception as e:
    print(f"[convert_adain][ERROR] {e}")

convert_decoder_weights(kmodel_torch, model)

audio_ref = kmodel_torch.forward_with_tokens(export_input_ids_torch, dbg['ref_s'], dbg['speed'])


audio = model(
    input_ids=export_input_ids_tf,
    ref_s=export_ref_s_tf,
    n_inputs=n_inputs,
    speed=speed_value
)

plt.switch_backend('Agg')
plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
# plt.plot(x_tf.numpy()[0,0,:], color='blue')
# plt.plot(x_torch.detach().cpu().numpy()[0,0,:], color='red')

audio_ref_waveform = audio_ref[0] if isinstance(audio_ref, (tuple, list)) else audio_ref
audio_ref_np = _to_numpy(audio_ref_waveform)
audio_tf_np = _to_numpy(audio)

plt.plot(audio_ref_np.squeeze(), label='PyTorch Audio')
plt.title('PyTorch Generated Audio')
plt.subplot(2,1,2)
# plt.plot(diff[0,0,:], label='Difference', color='green')
plt.plot(audio_tf_np.squeeze(), label='TensorFlow Audio', color='orange')
plt.title('TensorFlow Generated Audio')
plt.savefig('audio_comparison.png')

print("\n" + "="*80)
print("SUCCESS: TensorFlow model forward pass completed!")
print("="*80)
print(f"Audio comparison plot saved to audio_comparison.png")

print("\nTrying to save Keras model...")
try:
    save_path = 'kokoro'
    model.save(save_path+'.keras', include_optimizer=False)
    # tf.saved_model.save(model, save_path+'/tf_savedmodel')
    print(f"Saved TF model to {save_path}.keras")
except Exception as e:
    print(f"[WARNING] Failed to save TF model: {e}")

# ============================================================================
# TFLite Conversion
# ============================================================================

print("\n" + "="*80)
print("Attempting TFLite conversion...")
print("="*80)

@tf.function(input_signature=[
    tf.TensorSpec(shape=(1, 510), dtype=tf.int32, name='input_ids'),
    tf.TensorSpec(shape=(1, 256), dtype=tf.float32, name='ref_s'),
    tf.TensorSpec(shape=(1,1), dtype=tf.int32, name='n_inputs'),
    tf.TensorSpec(shape=(1,1), dtype=tf.float32, name='speed'),
])
def _serving_step(input_ids, ref_s, n_inputs, speed):
    # Wrap the model call with a concrete input signature for TFLite tracing.
    return model(input_ids=input_ids, ref_s=ref_s, n_inputs=n_inputs, speed=speed, training=False)


concrete_fn = _serving_step.get_concrete_function(
    export_input_ids_tf,
    export_ref_s_tf,
    export_n_inputs_tf,
    export_speed_tf,
)

print("Creating TFLite converter...")
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)
# converter = tf.lite.TFLiteConverter.from_keras_model(model)


# First try with SELECT_TF_OPS (allows more TF ops)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Converting to TFLite (with SELECT_TF_OPS)...")
tflite_model = converter.convert()
tflite_model_bytes = cast(bytes, tflite_model)

tflite_path = 'kokoro.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model_bytes)

print(f"✓ Successfully saved TFLite model to {tflite_path}")
print(f"  Model size: {len(tflite_model_bytes) / 1024 / 1024:.2f} MB")
    

# ============================================================================
# TFLite Testing (if conversion succeeded)
# ============================================================================

if tflite_path and os.path.exists(tflite_path):
    print(f"\n{'='*80}")
    print("Testing TFLite model inference...")
    print('='*80)
    
    try:
        # Load and test TFLite model
        interpreter = Interpreter(model_path=tflite_path)

        # Resize tensors to match the export shapes when necessary before allocating buffers.
        input_details = interpreter.get_input_details()
        target_shapes = [
            [1, export_input_ids_np.shape[1]],
            [1, style_dim],
            [1, 1],
            [1, 1],
        ]
        for detail, target_shape in zip(input_details, target_shapes):
            if list(detail['shape']) != target_shape:
                interpreter.resize_tensor_input(detail['index'], target_shape)

        interpreter.allocate_tensors()

        # Fetch final tensor metadata for logging.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"TFLite model inputs: {len(input_details)}")
        for i, detail in enumerate(input_details):
            print(f"  Input {i}: {detail['name']}, shape={detail['shape']}, dtype={detail['dtype']}")

        print(f"TFLite model outputs: {len(output_details)}")
        for i, detail in enumerate(output_details):
            print(f"  Output {i}: {detail['name']}, shape={detail['shape']}, dtype={detail['dtype']}")

        # Prepare test inputs aligned with the export shapes.
        test_input_ids = export_input_ids_np.astype(np.int32)
        test_ref_s = export_ref_s_np.astype(np.float32)
        test_n_inputs = np.array([[n_inputs]], dtype=np.int32)
        test_speed = np.array([[speed_value]], dtype=np.float32)

        # Set input tensors
        interpreter.set_tensor(input_details[0]['index'], test_input_ids)
        interpreter.set_tensor(input_details[1]['index'], test_ref_s)
        interpreter.set_tensor(input_details[2]['index'], test_n_inputs)
        interpreter.set_tensor(input_details[3]['index'], test_speed)
        
        # Run inference
        print("\nRunning TFLite inference...")
        interpreter.invoke()
        
        # Get output
        tflite_audio = interpreter.get_tensor(output_details[0]['index'])
        
        # Compare with TensorFlow model output
        print(f"\nTensorFlow audio shape: {audio.shape}")
        print(f"TFLite audio shape:     {tflite_audio.shape}")
        
        # Calculate comparison metrics
        audio_np = audio_tf_np
        print(f"{np.mean(audio_np)=}")
        print(f"{np.mean(tflite_audio)=}")
        # mse = np.((audio_np - tflite_audio))
        mae = np.mean(np.abs(audio_np - tflite_audio))
        max_diff = np.max(np.abs(audio_np - tflite_audio))
        corr = np.corrcoef(audio_np.flatten(), tflite_audio.flatten())[0, 1]
        
        print(f"\nComparison metrics:")
        # print(f"  Mean Squared Error:  {mse:.6e}")
        print(f"  Mean Absolute Error: {mae:.6e}")
        print(f"  Max Absolute Diff:   {max_diff:.6e}")
        print(f"  Correlation:         {corr:.6f}")
        
        # Visual comparison
        plt.figure(figsize=(14, 8))
        plt.subplot(3, 1, 1)
        plt.plot(audio_np.flatten(), label='TensorFlow Audio', alpha=0.7)
        plt.plot(tflite_audio.flatten(), label='TFLite Audio', alpha=0.7, linestyle='--')
        plt.title('Audio Waveform Comparison ')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 2)
        diff = audio_np - tflite_audio
        plt.plot(diff.flatten(), label='Difference', color='red', alpha=0.7)
        plt.title('Absolute Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 3)
        plt.plot(audio_ref_np.squeeze(), label='PyTorch Reference', alpha=0.7)
        plt.title('PyTorch Reference Audio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tflite_audio_comparison.png', dpi=150)
        print(f"\nSaved comparison plot to tflite_audio_comparison.png")
        
        # Determine if models match closely enough
        threshold_corr = 0.99
        if corr > threshold_corr and mae < 0.01:
            print(f"\n✓ TFLite model matches TensorFlow model (correlation={corr:.6f}, MAE={mae:.6e})")
        else:
            print(f"\n✗ WARNING: TFLite model differs from TensorFlow model (correlation={corr:.6f}, MAE={mae:.6e})")
    
    
    except Exception as e:
        print(f"\n✗ TFLite inference test failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("Conversion process completed!")
print("="*80)


