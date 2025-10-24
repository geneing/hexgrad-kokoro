from multiprocessing import pool
import os
import matplotlib.pyplot as plt

import keras
from regex import E, F
from sympy import N
import torch
from torch.nn.utils import remove_weight_norm

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
# output = model( input_ids=tf.convert_to_tensor(dbg['input_ids'].numpy()), 
#                 ref_s=tf.convert_to_tensor(dbg['ref_s'].numpy()), 
#                 speed=dbg['speed'])


asr = dbg['asr']
F0_pred = dbg['F0_pred']
N_pred = dbg['N_pred']
ref_s = dbg['ref_s']
audio = kmodel_torch.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
# print(f"{audio.shape=}")

print("\n\n\n########### TF Decoder #############\n\n\n")
audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128], training=False)
print(f"{audio.shape=}")
