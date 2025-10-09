from multiprocessing import pool
import os

import keras
from regex import E
import torch

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import pickle
import numpy as np
import tensorflow as tf
from kokoro_litert.kokoro import KModelTF
from kokoro_torch.kokoro import KModel
import csv
import json

def map_tf_to_torch_names(tf_name: str) -> str:
    tf_name = tf_name.replace('custom_albert/albert/', '')
    tf_name = tf_name.replace('embeddings:0', 'weight')
    tf_name = tf_name.replace(':0', '')
    tf_name = tf_name.replace('k_model_tf/custom_albert/albert/', '')
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


def convert_text_encoder(model, kmodel_torch):
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
            keras_conv.set_weights([depthwise_kernel, b.astype(np.float32)])

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
        for name_pair in [("conv1", "conv1"), ("conv2", "conv2")]:
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


def validate_adain(kmodel_torch_predictor, tf_predictor):
    """Validate/calculate diffs between PyTorch and Keras AdainResBlk1d branch weights.

    Prints per-layer statistics (max abs diff, mean abs diff, relative max, relative mean)
    for conv1, conv2, shortcut (if any) and style FC layers, plus projection layers.
    """

    def stats(name, w_pt, w_tf, already_aligned=False):
        try:
            if not already_aligned:
                if w_pt.ndim == 3 and w_tf.ndim == 3:  # Conv: pt(out,in,k) -> tf(k,in,out)
                    w_pt_cmp = np.transpose(w_pt, (2, 1, 0))
                elif w_pt.ndim == 2 and w_tf.ndim == 2:  # Linear: pt(out,in) -> tf(in,out)
                    w_pt_cmp = w_pt.T
                else:
                    w_pt_cmp = w_pt
            else:
                w_pt_cmp = w_pt
            if w_pt_cmp.shape != w_tf.shape:
                print(f"[validate_adain][SHAPE_MISMATCH] {name}: pt{w_pt_cmp.shape} tf{w_tf.shape}")
                return
            diff = w_pt_cmp - w_tf
            abs_diff = np.abs(diff)
            max_abs = abs_diff.max()
            mean_abs = abs_diff.mean()
            denom = np.maximum(np.abs(w_pt_cmp), 1e-9)
            rel = abs_diff / denom
            max_rel = rel.max()
            mean_rel = rel.mean()
            print(f"[validate_adain] {name}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} max_rel={max_rel:.6g} mean_rel={mean_rel:.6g}")
        except Exception as e:
            print(f"[validate_adain][ERROR] {name}: {e}")

    def compare_block(pt_blk, tf_blk, prefix):
        if pt_blk is None or tf_blk is None:
            print(f"[validate_adain][WARN] Missing block for {prefix}")
            return
        # conv1 / conv2 / shortcut
        for a_pt, a_tf in [("conv1","conv1"),("conv2","conv2"),("shortcut","shortcut_conv")]:
            pt_sub = getattr(pt_blk, a_pt, None)
            tf_sub = getattr(tf_blk, a_tf, None)
            if pt_sub is not None and tf_sub is not None and hasattr(pt_sub, 'weight') and hasattr(tf_sub, 'kernel'):
                stats(f"{prefix}.{a_pt}", pt_sub.weight.detach().numpy(), tf_sub.kernel.numpy())
        # pool (ConvTranspose1d) validation
        pt_pool = getattr(pt_blk, 'pool', None)
        tf_pool = getattr(tf_blk, 'pool', None)
        if pt_pool is not None and tf_pool is not None and hasattr(pt_pool, 'weight') and hasattr(tf_pool, 'kernel'):
            try:
                w_pt = pt_pool.weight.detach().numpy()  # (in,out,k)
                w_pt_cmp = np.transpose(w_pt, (2,1,0))  # -> (k,out,in)
                w_tf = tf_pool.kernel.numpy()
                diff = w_pt_cmp - w_tf[:,0,:]
                abs_diff = np.abs(diff)
                print(f"[validate_adain] {prefix}.pool: max_abs={abs_diff.max():.6g} mean_abs={abs_diff.mean():.6g}")
            except Exception as e:
                print(f"[validate_adain][WARN] pool stats failed {prefix}: {e}")
        # style FC candidates (top-level)
        for cand in ["fc","style_fc","style","mod"]:
            pt_fc = getattr(pt_blk, cand, None)
            tf_fc = getattr(tf_blk, cand, None)
            if pt_fc is not None and tf_fc is not None and hasattr(pt_fc,'weight') and hasattr(tf_fc,'kernel'):
                stats(f"{prefix}.{cand}", pt_fc.weight.detach().numpy(), tf_fc.kernel.numpy())
                break
        # inner AdaIN1d modules (norm1/norm2) and lists (adain1/adain2) including fc and norm gamma/beta
        def norm_stats(tag, pt_norm, tf_norm):
            if pt_norm is None or tf_norm is None or not hasattr(pt_norm,'weight') or not hasattr(pt_norm,'bias'):
                return
            try:
                gamma_pt = pt_norm.weight.detach().numpy()
                beta_pt = pt_norm.bias.detach().numpy()
                if hasattr(tf_norm,'gamma') and hasattr(tf_norm,'beta'):
                    gamma_tf = tf_norm.gamma.numpy(); beta_tf = tf_norm.beta.numpy()
                else:
                    tw = tf_norm.get_weights()
                    if len(tw) >= 2:
                        gamma_tf, beta_tf = tw[0], tw[1]
                    else:
                        return
                stats(f"{tag}.norm.gamma", gamma_pt, gamma_tf, already_aligned=True)
                stats(f"{tag}.norm.beta", beta_pt, beta_tf, already_aligned=True)
            except Exception as e:
                print(f"[validate_adain][WARN] norm stats failed {tag}: {e}")
        for nm in ['norm1','norm2']:
            pt_adain = getattr(pt_blk, nm, None)
            tf_adain = getattr(tf_blk, nm, None)
            if pt_adain is not None and tf_adain is not None:
                if hasattr(pt_adain, 'fc') and hasattr(tf_adain, 'fc') and hasattr(pt_adain.fc,'weight') and hasattr(tf_adain.fc,'kernel'):
                    stats(f"{prefix}.{nm}.fc", pt_adain.fc.weight.detach().numpy(), tf_adain.fc.kernel.numpy())
                norm_stats(f"{prefix}.{nm}", getattr(pt_adain,'norm', None), getattr(tf_adain,'norm', None))
        for nm in ['adain1','adain2']:
            pt_list = getattr(pt_blk, nm, None)
            tf_list = getattr(tf_blk, nm, None)
            if pt_list is not None and tf_list is not None:
                for idx, (pt_a, tf_a) in enumerate(zip(pt_list, tf_list)):
                    if pt_a is None or tf_a is None:
                        continue
                    if hasattr(pt_a, 'fc') and hasattr(tf_a, 'fc') and hasattr(pt_a.fc,'weight') and hasattr(tf_a.fc,'kernel'):
                        stats(f"{prefix}.{nm}[{idx}].fc", pt_a.fc.weight.detach().numpy(), tf_a.fc.kernel.numpy())
                    norm_stats(f"{prefix}.{nm}[{idx}]", getattr(pt_a,'norm', None), getattr(tf_a,'norm', None))

    # Branch blocks
    pt_F0 = getattr(kmodel_torch_predictor, 'F0', [])
    pt_N = getattr(kmodel_torch_predictor, 'N', [])
    tf_F0 = getattr(tf_predictor, 'F0_blocks', [])
    tf_N = getattr(tf_predictor, 'N_blocks', [])

    print("[validate_adain] ==== F0 Blocks ====")
    for i, (pt_blk, tf_blk) in enumerate(zip(pt_F0, tf_F0)):
        compare_block(pt_blk, tf_blk, f"F0[{i}]")
    if len(pt_F0) != len(tf_F0):
        print(f"[validate_adain][WARN] F0 block count mismatch pt={len(pt_F0)} tf={len(tf_F0)}")

    print("[validate_adain] ==== N Blocks ====")
    for i, (pt_blk, tf_blk) in enumerate(zip(pt_N, tf_N)):
        compare_block(pt_blk, tf_blk, f"N[{i}]")
    if len(pt_N) != len(tf_N):
        print(f"[validate_adain][WARN] N block count mismatch pt={len(pt_N)} tf={len(tf_N)}")

    # Projection layers
    pt_F0_proj = getattr(kmodel_torch_predictor, 'F0_proj', None)
    tf_F0_proj = getattr(tf_predictor, 'F0_proj', None)
    if pt_F0_proj is not None and tf_F0_proj is not None and hasattr(pt_F0_proj,'weight') and hasattr(tf_F0_proj,'kernel'):
        stats('F0_proj', pt_F0_proj.weight.detach().numpy(), tf_F0_proj.kernel.numpy())
    else:
        print('[validate_adain][WARN] Missing F0_proj in one model')

    pt_N_proj = getattr(kmodel_torch_predictor, 'N_proj', None)
    tf_N_proj = getattr(tf_predictor, 'N_proj', None)
    if pt_N_proj is not None and tf_N_proj is not None and hasattr(pt_N_proj,'weight') and hasattr(tf_N_proj,'kernel'):
        stats('N_proj', pt_N_proj.weight.detach().numpy(), tf_N_proj.kernel.numpy())
    else:
        print('[validate_adain][WARN] Missing N_proj in one model')

    print('[validate_adain] Validation complete.')


def plot_differences(tf, dbg, title):
    import matplotlib.pyplot as plt
    #use png rendering backend
    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 6))
    # bert_dur.numpy().T 
    # dbg['bert_dur'].last_hidden_state.numpy()
    plt.imshow(((tf - dbg)/(dbg+1))[0, :, :], aspect='auto', cmap='bwr')
    plt.colorbar(label='Difference')
    plt.title('Difference between TensorFlow and PyTorch Outputs')
    # plt.xlabel('Sequence Length')
    # plt.ylabel('Hidden Size')
    plt.savefig(f'{title}.png')
    plt.close()

def plot_plot(tf, title):
    import matplotlib.pyplot as plt
    #use png rendering backend
    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 6))
    # bert_dur.numpy().T 
    # dbg['bert_dur'].last_hidden_state.numpy()
    plt.imshow(tf, aspect='auto', cmap='bwr')
    plt.colorbar(label='value')
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
output = model( input_ids=tf.convert_to_tensor(dbg['input_ids'].numpy()), 
                ref_s=tf.convert_to_tensor(dbg['ref_s'].numpy()), 
                speed=dbg['speed'])


# AdainResBlk1d_upsampled = kmodel_torch.predictor.F0[1]
# pool = AdainResBlk1d_upsampled.pool
# print(f"{pool=}")
# w, b = pool.weight.detach().numpy(), pool.bias.detach().numpy()
# print(f"{w.shape=}, {b.shape=}")
# print(f"{pool.in_channels=}, {pool.out_channels=}, {pool.kernel_size=}, {pool.stride=}, {pool.padding=}, {pool.output_padding=}, {pool.dilation=}, {pool.groups=}, {pool.bias is not None=}")

# AdainResBlk1d_upsampled_tf = model.predictor.F0_blocks[1]
# pool_tf = AdainResBlk1d_upsampled_tf.pool.conv
# print(f"{pool_tf=}")
# w_tf, b_tf = pool_tf.get_weights()
# print(f"{w_tf.shape=}, {b_tf.shape=}")

# w_transp = np.transpose(w, (2,1,0))
# for i in range(w_tf.shape[1]):
#     w_tf[:,i,:] = w_transp[:,i,:]
# pool_tf.set_weights([w_tf, b])

model.bert_encoder.set_weights([kmodel_torch.bert_encoder.weight.detach().numpy().T, kmodel_torch.bert_encoder.bias.detach().numpy()])


# output = model( input_ids=tf.convert_to_tensor(dbg['input_ids'].numpy()), 
#                 ref_s=tf.convert_to_tensor(dbg['ref_s'].numpy()), 
#                 speed=dbg['speed'])
# print(f"{kmodel_torch.bert_encoder.weight=}")

# output = model( input_ids=tf.convert_to_tensor(dbg['input_ids'].numpy()), 
#                 ref_s=tf.convert_to_tensor(dbg['ref_s'].numpy()), 
#                 speed=dbg['speed'], training=False)
# output = model.bert_encoder(tf.convert_to_tensor(dbg['bert_dur'].last_hidden_state.numpy()), training=False)
# output = tf.transpose(output, [0, 2, 1])  # Transpose for conv processing



# print(f"{model.bert_encoder.get_weights()[0]=}")

hidden_size = kmodel_torch.predictor.lstm.hidden_size

copy_weights_bert(model=model, kmodel_torch=kmodel_torch, dbg=dbg)
convert_text_encoder(model=model, kmodel_torch=kmodel_torch)
convert_bilstm_weights(kmodel_torch.predictor.lstm, model.predictor.lstm, hidden_size)
model.predictor.duration_proj.set_weights([kmodel_torch.predictor.duration_proj.linear_layer.weight.detach().numpy().T, kmodel_torch.predictor.duration_proj.linear_layer.bias.detach().numpy()])

convert_bilstm_weights(kmodel_torch.predictor.shared, model.predictor.shared_bilstm, hidden_size)

# Perform Adain weight copy and validation
try:
    convert_adain(kmodel_torch.predictor, model.predictor)
    validate_adain(kmodel_torch.predictor, model.predictor)
except Exception as e:
    print(f"[convert_adain][ERROR] {e}")

kmodel_torch.forward_with_tokens( dbg['input_ids'], dbg['ref_s'], dbg['speed'])

output = model( input_ids=tf.convert_to_tensor(dbg['input_ids'].numpy()), 
                ref_s=tf.convert_to_tensor(dbg['ref_s'].numpy()), 
                speed=float(dbg['speed']))



a = np.zeros([1, 512, 196], dtype=np.float32)
# a[0,0,0]=1.0
a=np.ones(a.shape, dtype=a.dtype)

azero = np.zeros([1, 512, 196], dtype=np.float32)


AdainResBlk1d_upsampled = kmodel_torch.predictor.F0[1]
pool = AdainResBlk1d_upsampled.pool

w, b = pool.weight.detach().numpy(), pool.bias.detach().numpy()

pickle.dump((w, b, a, azero, pool(torch.tensor(a)), pool(torch.tensor(azero))), open('temp/pool_values.pkl', 'wb'))

print(f"{w.shape=}, {b.shape=}")
print(f"{pool.in_channels=}, {pool.out_channels=}, {pool.kernel_size=}, {pool.stride=}, {pool.padding=}, {pool.output_padding=}, {pool.dilation=}, {pool.groups=}, {pool.bias is not None=}")

AdainResBlk1d_upsampled_tf = model.predictor.F0_blocks[1]
pool_tf = AdainResBlk1d_upsampled_tf.pool
print(f"{pool_tf=}")
w_tf, b_tf = pool_tf.get_weights()
print(f"{w_tf.shape=}, {b_tf.shape=}")

print(f"{b_tf[0:10]=}, {b[0:10]=}")
print(f"{w_tf[0, 0, 0:3]=}, {w[0:3, 0, 0]=}")

torch_x = pool(torch.tensor(a))-pool(torch.tensor(azero))
tf_x = pool_tf(a)-pool_tf(azero)
print(f"{torch_x.shape=}, {tf_x.shape=}")
print(f"\n\n{torch_x[0,0:10,0]=}, \n\n{tf_x[0,0:10,0]=}")

plot_plot(torch_x.detach().numpy()[0,:,:], 'torch_pool_output')
plot_plot(tf_x[0,:,:], 'tf_pool_output')
diff = np.abs(torch_x.detach().numpy()[0,:,:] - tf_x.numpy()[0,:,:])
print(f"{diff.max()=}, {diff.min()=}, {diff.mean()=}")
      
# kmodel_torch.forward_with_tokens( dbg['input_ids'], dbg['ref_s'], dbg['speed'])

# output = model( input_ids=tf.convert_to_tensor(dbg['input_ids'].numpy()), 
#                 ref_s=tf.convert_to_tensor(dbg['ref_s'].numpy()), 
#                 speed=float(dbg['speed']))




# print(f"tf shape: {output[0].shape=} {dbg['bert_dur'].last_hidden_state.shape=}")
# print(f"tf_data: {output[0][0,0,0:10].numpy().T=}")
# print(f"torch_data: dbg['bert_dur'].last_hidden_state[0,0,0:10]={dbg['bert_dur'].last_hidden_state[0,0,0:10]}")
# print(f"max diff: {abs(output[0].numpy().T - dbg['bert_dur'].last_hidden_state.numpy()).max()}")
# plot_differences(output[0].numpy().T, dbg['bert_dur'].last_hidden_state.numpy(), 'bert_dur_diff_1')

# print(f"tf_data: {output[1][0,0,0:10]=}")
# print(f"torch_data: {dbg['d_en'][0,0,0:10]=}")
# print(f"max diff: {abs(output[1].numpy() - dbg['d_en'].numpy()).max()}")
# plot_differences(output[1].numpy(), dbg['d_en'].numpy(), 'd_en_diff')

# print(f"tf_data: {output[2][0,0,0:10]=}")
# print(f"torch_data: {dbg['d'][0,0,0:10]=}")
# print(f"max diff: {abs(output[2].numpy() - dbg['d'].numpy()).max()}")
# plot_differences(output[2].numpy(), dbg['d'].numpy(), 'd_diff')

# print(f"tf_data: {output[3][0,0,0:10]=}")
# print(f"torch_data: {dbg['x'][0,0,0:10]=}")
# print(f"max diff: {abs(output[3].numpy() - dbg['x'].numpy()).max()}")
# plot_differences(output[3].numpy(), dbg['x'].numpy(), 'x_diff')

# print(f"tf_data: {output[4][0,0,0:10]=}")
# print(f"torch_data: {dbg['duration'][0,0,0:10]=}")
# print(f"max diff: {abs(output[4].numpy() - dbg['duration'].numpy()).max()}")
# plot_differences(output[4].numpy(), dbg['duration'].numpy(), 'duration_diff')

# print(f"{output[4]=}")
# print(f"{dbg['expanded_indices']=}")
# print(f"{output[5].shape=} {output[5]=}")
# print(f"{dbg['en'].shape=} {dbg['en']=}")

#####
# print(f"{output[6].shape=} {output[6][0,0:10]=}")
# print(f"{dbg['F0_pred'].shape=} {dbg['F0_pred'][0,0:10]=}")

# print(f"{output[7].shape=} {output[7][0,0:10]=}")
# print(f"{dbg['N_pred'].shape=} {dbg['N_pred'][0,0:10]=}")


# # output = model.predictor.text_encoder(tf.convert_to_tensor(dbg['d_en'].numpy()), tf.convert_to_tensor(dbg['s'].numpy()), training=False)
# inputs = {'input_ids': tf.convert_to_tensor(dbg['input_ids'].numpy()), 'token_type_ids': tf.zeros_like(dbg['input_ids'].numpy())}
# bert_dur = model.bert(inputs, training=False)
# bert_dur = bert_dur.last_hidden_state
        
# print(f"\n\n\n\ntf shape: {bert_dur.shape=} {dbg['bert_dur'].last_hidden_state.shape=}")
# print(f"tf_data: {bert_dur[0,0,0:10].numpy().T=}")
# print(f"torch_data: dbg['bert_dur'].last_hidden_state[0,0,0:10]={dbg['bert_dur'].last_hidden_state[0,0,0:10]}")
# print(f"max diff: {abs(bert_dur.numpy().T - dbg['bert_dur'].last_hidden_state.numpy()).max()}")
# plot_differences(bert_dur.numpy().T, dbg['bert_dur'].last_hidden_state.numpy(), 'bert_dur_diff')
# #plot bert_dur.numpy().T - dbg['bert_dur'].last_hidden_state.numpy() and save to a file


# s = kmodel_torch.predictor.text_encoder.state_dict()
# d = tf_text_encoder.trainable_variables
# print(f"{len(s)=};{len(d)=}")
# for (v, (name, param)) in zip(d, s.items()):
#     print(f"{v.name=};{name=};{list(param.shape)=};{list(v.shape)=}")



