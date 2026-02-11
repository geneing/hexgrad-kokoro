"""StyleTTS2-aligned loss components.

These losses are adapted from the StyleTTS2 training code path
(via the published styletts2_fork `losses.py`) to be combined with
Vocos training objectives.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


class SpectralConvergenceLoss(nn.Module):
    """L1 spectral convergence used by StyleTTS2."""

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)


class StyleTTS2STFTLoss(nn.Module):
    """Single-resolution mel-spectral convergence loss used in StyleTTS2."""

    def __init__(
        self,
        fft_size: int = 1024,
        shift_size: int = 120,
        win_length: int = 600,
        sample_rate: int = 24000,
    ) -> None:
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=fft_size,
            win_length=win_length,
            hop_length=shift_size,
            window_fn=torch.hann_window,
        )
        self.spectral_convergence_loss = SpectralConvergenceLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_mag = self.to_mel(x)
        x_mag = (torch.log(1e-5 + x_mag) - (-4.0)) / 4.0

        y_mag = self.to_mel(y)
        y_mag = (torch.log(1e-5 + y_mag) - (-4.0)) / 4.0

        return self.spectral_convergence_loss(x_mag, y_mag)


class StyleTTS2MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution StyleTTS2 spectral convergence objective."""

    def __init__(
        self,
        fft_sizes: Iterable[int] = (1024, 2048, 512),
        hop_sizes: Iterable[int] = (120, 240, 50),
        win_lengths: Iterable[int] = (600, 1200, 240),
        sample_rate: int = 24000,
    ) -> None:
        super().__init__()
        fft_sizes = list(fft_sizes)
        hop_sizes = list(hop_sizes)
        win_lengths = list(win_lengths)
        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError("fft_sizes, hop_sizes and win_lengths must have equal length")
        self.losses = nn.ModuleList(
            [
                StyleTTS2STFTLoss(
                    fft_size=fs,
                    shift_size=ss,
                    win_length=wl,
                    sample_rate=sample_rate,
                )
                for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths)
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sc_loss = 0.0
        for loss in self.losses:
            sc_loss = sc_loss + loss(x, y)
        return sc_loss / len(self.losses)


def styletts2_generator_lsgan_loss(disc_outputs: List[torch.Tensor]) -> torch.Tensor:
    """StyleTTS2 generator adversarial loss: mean((1 - D(G(x)))^2)."""
    loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
    for dg in disc_outputs:
        loss = loss + torch.mean((1 - dg) ** 2)
    return loss


def styletts2_discriminator_lsgan_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
) -> torch.Tensor:
    """StyleTTS2 discriminator adversarial loss."""
    loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        loss = loss + torch.mean((1 - dr) ** 2) + torch.mean(dg**2)
    return loss


def _tprls_component(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
    tau: float = 0.04,
) -> torch.Tensor:
    # Reference: StyleTTS2 TPRLS objective
    loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        m_dg = torch.median(dr - dg)
        mask = dr < (dg + m_dg)
        if torch.any(mask):
            l_rel = torch.mean((((dr - dg) - m_dg) ** 2)[mask])
            loss = loss + tau - F.relu(tau - l_rel)
    return loss


def styletts2_discriminator_tprls_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
) -> torch.Tensor:
    return _tprls_component(disc_real_outputs, disc_generated_outputs)


def styletts2_generator_tprls_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
) -> torch.Tensor:
    # StyleTTS2 uses the same relative statistic for generator branch.
    return _tprls_component(disc_real_outputs, disc_generated_outputs)
