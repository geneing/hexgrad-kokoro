# https://github.com/yl4579/StyleTTS2/blob/main/models.py
from .istftnet import AdainResBlk1d
from torch.nn.utils.parametrizations import weight_norm
from transformers import AlbertModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual


class TCNSequenceMixer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_blocks=4,
        kernel_size=5,
        dilations=(1, 2, 4, 8),
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        blocks = []
        for idx in range(num_blocks):
            dilation = dilations[idx % len(dilations)]
            blocks.append(TCNBlock(output_dim, kernel_size, dilation, dropout))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        # x: [B, T, C] -> [B, T, output_dim]
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.blocks(x)
        return x.transpose(1, 2)


class TextEncoder(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        depth,
        n_symbols,
        actv=nn.LeakyReLU(0.2),
        sequence_mixer="lstm",
        tcn_num_blocks=4,
        tcn_kernel_size=5,
        tcn_dilations=(1, 2, 4, 8),
    ):
        super().__init__()
        self.sequence_mixer_type = sequence_mixer
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        if sequence_mixer == "lstm":
            self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)
        elif sequence_mixer == "tcn":
            self.sequence_mixer = TCNSequenceMixer(
                channels,
                channels,
                num_blocks=tcn_num_blocks,
                kernel_size=tcn_kernel_size,
                dilations=tcn_dilations,
                dropout=0.2,
            )
        else:
            raise ValueError(f"Unsupported text encoder sequence_mixer={sequence_mixer!r}")

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)  # [B, T, chn]
        if self.sequence_mixer_type == "lstm":
            lengths = input_lengths if input_lengths.device == torch.device('cpu') else input_lengths.to('cpu')
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            self.lstm.flatten_parameters()
            x, _ = self.lstm(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            x = x.transpose(-1, -2)
            x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
            x_pad[:, :, :x.shape[-1]] = x
            x = x_pad
        else:
            x = self.sequence_mixer(x).transpose(-1, -2)
        x.masked_fill_(m, 0.0)
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    def __init__(
        self,
        style_dim,
        d_hid,
        nlayers,
        max_dur=50,
        dropout=0.1,
        sequence_mixer="lstm",
        tcn_num_blocks=4,
        tcn_kernel_size=5,
        tcn_dilations=(1, 2, 4, 8),
    ):
        super().__init__()
        self.sequence_mixer_type = sequence_mixer
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim,
            d_model=d_hid,
            nlayers=nlayers,
            dropout=dropout,
            sequence_mixer=sequence_mixer,
            tcn_num_blocks=tcn_num_blocks,
            tcn_kernel_size=tcn_kernel_size,
            tcn_dilations=tcn_dilations,
        )
        if sequence_mixer == "lstm":
            self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
            self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        elif sequence_mixer == "tcn":
            self.duration_mixer = TCNSequenceMixer(
                d_hid + style_dim,
                d_hid,
                num_blocks=tcn_num_blocks,
                kernel_size=tcn_kernel_size,
                dilations=tcn_dilations,
                dropout=dropout,
            )
            self.shared_mixer = TCNSequenceMixer(
                d_hid + style_dim,
                d_hid,
                num_blocks=tcn_num_blocks,
                kernel_size=tcn_kernel_size,
                dilations=tcn_dilations,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported predictor sequence_mixer={sequence_mixer!r}")
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def run_duration_mixer(self, d):
        if self.sequence_mixer_type == "lstm":
            x, _ = self.lstm(d)
            return x
        return self.duration_mixer(d)

    def run_shared_mixer(self, x):
        x = x.transpose(-1, -2)
        if self.sequence_mixer_type == "lstm":
            x, _ = self.shared(x)
        else:
            x = self.shared_mixer(x)
        return x

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        m = m.unsqueeze(1)
        if self.sequence_mixer_type == "lstm":
            lengths = text_lengths if text_lengths.device == torch.device('cpu') else text_lengths.to('cpu')
            x = nn.utils.rnn.pack_padded_sequence(d, lengths, batch_first=True, enforce_sorted=False)
            self.lstm.flatten_parameters()
            x, _ = self.lstm(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
            x_pad[:, :x.shape[1], :] = x
            x = x_pad
        else:
            x = self.duration_mixer(d)
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=False))
        en = (d.transpose(-1, -2) @ alignment)
        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        x = self.run_shared_mixer(x)
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)


class DurationEncoder(nn.Module):
    def __init__(
        self,
        sty_dim,
        d_model,
        nlayers,
        dropout=0.1,
        sequence_mixer="lstm",
        tcn_num_blocks=2,
        tcn_kernel_size=5,
        tcn_dilations=(1, 2, 4, 8),
    ):
        super().__init__()
        self.sequence_mixer_type = sequence_mixer
        if sequence_mixer == "lstm":
            self.lstms = nn.ModuleList()
            for _ in range(nlayers):
                self.lstms.append(nn.LSTM(d_model + sty_dim, d_model // 2, num_layers=1, batch_first=True, bidirectional=True))
                self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        elif sequence_mixer == "tcn":
            self.tcn_layers = nn.ModuleList()
            for _ in range(nlayers):
                self.tcn_layers.append(TCNSequenceMixer(
                    d_model + sty_dim,
                    d_model,
                    num_blocks=tcn_num_blocks,
                    kernel_size=tcn_kernel_size,
                    dilations=tcn_dilations,
                    dropout=dropout,
                ))
                self.tcn_layers.append(AdaLayerNorm(sty_dim, d_model))
        else:
            raise ValueError(f"Unsupported duration encoder sequence_mixer={sequence_mixer!r}")
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        blocks = self.lstms if self.sequence_mixer_type == "lstm" else self.tcn_layers
        for block in blocks:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            elif self.sequence_mixer_type == "lstm":
                lengths = text_lengths if text_lengths.device == torch.device('cpu') else text_lengths.to('cpu')
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad
            else:
                x = block(x.transpose(-1, -2))
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)

        return x.transpose(-1, -2)


# https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state
