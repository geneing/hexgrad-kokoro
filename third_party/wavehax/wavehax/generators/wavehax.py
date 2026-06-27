# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Wavehax generator modules."""

from functools import partial
from typing import List

import torch
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from torch import Tensor, nn

import wavehax.modules
from wavehax.modules import (
    BatchNorm2d,
    RealDFTSTFT,
    STFT,
    ComplexConv1d,
    ComplexConv2d,
    ComplexConvNeXtBlock2d,
    ComplexLayerNorm2d,
    ConvNeXtBlock2d,
    LayerNorm2d,
    to_log_magnitude_and_phase,
    to_real_imaginary,
)


def deterministic_pcph_closed_form(
    f0: Tensor,
    hop_length: int,
    sample_rate: int,
    power_factor: float = 0.1,
    max_frequency: float | None = None,
    epsilon: float = 1e-6,
) -> Tensor:
    """Deterministic pseudo-constant-power harmonic prior for export-safe inference."""

    f0_upsampled = torch.nn.functional.interpolate(f0, scale_factor=hop_length, mode="linear", align_corners=False)
    phase_increment = f0_upsampled / float(sample_rate)
    phase = torch.cumsum(phase_increment, dim=2) * (2.0 * torch.pi)
    phase = torch.fmod(phase, 2.0 * torch.pi)

    limit_freq = float(max_frequency) if max_frequency is not None else float(sample_rate) / 2.0
    safe_f0 = torch.clamp(f0_upsampled, min=1e-5)
    n_harmonics = torch.floor(limit_freq / safe_f0)

    half_phase = phase / 2.0
    numerator = torch.cos(half_phase) - torch.cos((n_harmonics + 0.5) * phase)
    denominator = 2.0 * torch.sin(half_phase)
    harmonics = torch.where(
        torch.abs(denominator) > float(epsilon),
        numerator / denominator,
        torch.zeros_like(phase),
    )

    amp_scale = float(power_factor) * torch.sqrt(2.0 / torch.clamp(n_harmonics, min=1.0))
    vuv_mask = (f0_upsampled > 0.0).to(dtype=f0_upsampled.dtype)
    return harmonics * amp_scale * vuv_mask


class WavehaxGenerator(nn.Module):
    """
    Wavehax generator module.

    This module produces time-domain waveforms through complex spectrogram estimation
    based on the integration of 2D convolution and harmonic prior spectrograms.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        mult_channels: int,
        kernel_size: int,
        num_blocks: int,
        n_fft: int,
        hop_length: int,
        sample_rate: int,
        prior_type: str,
        drop_prob: float = 0.0,
        use_layer_norm: bool = True,
        framewise_norm: bool = False,
        use_logmag_phase: bool = False,
        use_gradient_checkpointing: bool = False,
        padding_mode: str = "zeros",
    ) -> None:
        """
        Initialize the WavehaxGenerator module.

        Args:
            in_channels (int): Number of conditioning feature channels.
            channels (int): Number of hidden feature channels.
            mult_channels (int): Channel expansion multiplier for ConvNeXt blocks.
            kernel_size (int): Kernel size for ConvNeXt blocks.
            num_blocks (int): Number of ConvNeXt residual blocks.
            n_fft (int): Number of Fourier transform points (FFT size).
            hop_length (int): Hop length (frameshift) in samples.
            sample_rate (int): Sampling frequency of input and output waveforms in Hz.
            prior_type (str): Type of prior waveforms used.
            drop_prob (float): Probability of dropping paths for stochastic depth (default: 0.0).
            use_layer_norm (bool): If True, layer normalization is used; otherwise,
                batch normalization is applied (default: True).
            use_logmag_phase (bool): Whether to use log-magnitude and phase for STFT (default: False).
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.use_logmag_phase = use_logmag_phase
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.padding_mode = str(padding_mode)

        # Prior waveform generator
        self.prior_generator = partial(
            getattr(wavehax.modules, f"generate_{prior_type}"),
            hop_length=self.hop_length,
            sample_rate=sample_rate,
        )

        # STFT layer
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length)

        # Input projection layers
        n_bins = n_fft // 2 + 1
        self.prior_proj = nn.Conv1d(
            n_bins, n_bins, 7, padding=3, padding_mode=self.padding_mode
        )
        self.cond_proj = nn.Conv1d(
            in_channels, n_bins, 7, padding=3, padding_mode=self.padding_mode
        )

        # Input normalization and projection layers
        self.input_proj = nn.Conv2d(5, channels, 1, bias=False)
        self.input_norm = LayerNorm2d(channels, framewise=framewise_norm)

        # ConvNeXt-based residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = ConvNeXtBlock2d(
                channels,
                mult_channels,
                kernel_size,
                drop_prob=drop_prob,
                use_layer_norm=use_layer_norm,
                framewise_norm=framewise_norm,
                padding_mode=self.padding_mode,
                layer_scale_init_value=1 / num_blocks,
            )
            self.blocks += [block]

        # Output normalization and projection layers
        self.output_norm = LayerNorm2d(channels, framewise=framewise_norm)
        self.output_proj = nn.Conv2d(channels, 2, 1)

        self.apply(self.init_weights)

    def init_weights(self, m) -> None:
        """
        Initialize weights of the module.

        Args:
            m (Any): Module to initialize.
        """
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, cond: Tensor, f0: Tensor) -> Tensor:
        """
        Calculate forward propagation.

        Args:
            cond (Tensor): Conditioning features with shape (batch, in_channels, frames).
            f0 (Tensor): F0 sequences with shape (batch, 1, frames).

        Returns:
            Tensor: Generated waveforms with shape (batch, 1, frames * hop_length).
            Tensor: Generated prior waveforms with shape (batch, 1, frames * hop_length).
        """
        # Generate prior waveform and compute spectrogram
        with torch.no_grad():
            prior = self.prior_generator(f0)
            real, imag = self.stft(prior)
            if self.use_logmag_phase:
                prior1, prior2 = to_log_magnitude_and_phase(real, imag)
            else:
                prior1, prior2 = real, imag

        # Apply input projection
        prior1_proj = self.prior_proj(prior1)
        prior2_proj = self.prior_proj(prior2)
        cond = self.cond_proj(cond)

        # Convert to 2d representation
        x = torch.stack([prior1, prior2, prior1_proj, prior2_proj, cond], dim=1)
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Apply residual blocks
        for f in self.blocks:
            if self.use_gradient_checkpointing and self.training and x.requires_grad:
                x = checkpoint(f, x, use_reentrant=False)
            else:
                x = f(x)

        # Apply output projection
        x = self.output_norm(x)
        x = self.output_proj(x)

        # Apply iSTFT followed by overlap and add
        if self.use_logmag_phase:
            real, imag = to_real_imaginary(x[:, 0], x[:, 1])
        else:
            real, imag = x[:, 0], x[:, 1]
        x = self.stft.inverse(real, imag)

        return x, prior

    @torch.inference_mode()
    def inference(self, cond: Tensor, f0: Tensor) -> Tensor:
        return self(cond, f0)[0]


class ComplexWavehaxGenerator(nn.Module):
    """
    Complex-valued Wavehax generator module.

    This class examines whether incorporating the algebraic structure of complex numbers enhances performance.
    Although this complex-valued version slightly increases computational cost compared to the standard Wavehax with
    an almost equivalent number of channels, consistent performance improvement was not observed.
    This code is shared in the hope that it may be useful for further research or for developing more advanced methods.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        mult_channels: int,
        kernel_size: int,
        num_blocks: int,
        n_fft: int,
        hop_length: int,
        sample_rate: int,
        prior_type: str,
        drop_prob: float = 0.0,
        use_layer_norm: bool = True,
        framewise_norm: bool = False,
        init_weights: bool = False,
        padding_mode: str = "zeros",
    ) -> None:
        """
        Initialize the ComplexWavehaxGenerator module.

        Args:
            in_channels (int): Number of conditioning feature channels.
            channels (int): Number of hidden feature channels.
                Note that both real and imaginary parts will retain this number of channels.
            mult_channels (int): Channel expansion multiplier for ConvNeXt blocks.
            kernel_size (int): Kernel size for ConvNeXt blocks.
            num_blocks (int): Number of ConvNeXt residual blocks.
            n_fft (int): Number of Fourier transform points (FFT size).
            hop_length (int): Hop length (frameshift) in samples.
            sample_rate (int): Sampling frequency of input and output waveforms in Hz.
            prior_type (str): Type of prior waveforms used.
            drop_prob (float): Probability of dropping paths for stochastic depth (default: 0.0).
            use_layer_norm (bool): If True, layer normalization is used; otherwise,
                batch normalization is applied (default: True).
            init_weights (bool): If True, apply the weight initialization of the standard Wavehax,
                instead of the weight initialization designed for complex-valued weights (default: False).
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.padding_mode = str(padding_mode)

        # Prior waveform generator
        self.prior_generator = partial(
            getattr(wavehax.modules, f"generate_{prior_type}"),
            hop_length=self.hop_length,
            sample_rate=sample_rate,
        )

        # STFT layer
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length)

        # Input projection layers
        n_bins = n_fft // 2 + 1
        self.prior_proj = ComplexConv1d(
            n_bins, n_bins, 7, padding=3, padding_mode=self.padding_mode
        )
        self.cond_proj = ComplexConv1d(
            in_channels, n_bins, 7, padding=3, padding_mode=self.padding_mode
        )

        # Input normalization and projection layers
        self.input_proj = ComplexConv2d(3, channels, 1, bias=False)
        self.input_norm = ComplexLayerNorm2d(channels, framewise=framewise_norm)

        # ConvNeXt-based residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = ComplexConvNeXtBlock2d(
                channels,
                mult_channels,
                kernel_size,
                drop_prob=drop_prob,
                use_layer_norm=use_layer_norm,
                framewise_norm=framewise_norm,
                padding_mode=self.padding_mode,
                layer_scale_init_value=1 / num_blocks,
            )
            self.blocks += [block]

        # Output normalization and projection layers
        self.output_norm = ComplexLayerNorm2d(channels, framewise=framewise_norm)
        self.output_proj = ComplexConv2d(channels, 1, 1)

        # Apply the standard Wavehax weight initialization, which tends to produce better results.
        if init_weights:
            self.apply(self.init_weights)

    def init_weights(self, m) -> None:
        """
        Initialize weights of the module.

        Args:
            m (Any): Module to initialize.
        """
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, cond: Tensor, f0: Tensor) -> Tensor:
        """
        Calculate forward propagation.

        Args:
            cond (Tensor): Conditioning features with shape (batch, in_channels, frames).
            f0 (Tensor): F0 sequences with shape (batch, 1, frames).

        Returns:
            Tensor: Generated waveforms with shape (batch, 1, frames * hop_length).
            Tensor: Generated prior waveforms with shape (batch, 1, frames * hop_length).
        """
        # Generate prior waveform and compute spectrogram
        with torch.no_grad():
            prior = self.prior_generator(f0)
            real, imag = self.stft(prior)

        # Apply input projection
        real_proj, imag_proj = self.prior_proj(real, imag)
        cond_real, cond_imag = self.cond_proj(cond, cond)

        # Convert to 2d representation
        real = torch.stack([real, real_proj, cond_real], dim=1)
        imag = torch.stack([imag, imag_proj, cond_imag], dim=1)
        real, imag = self.input_proj(real, imag)
        real, imag = self.input_norm(real, imag)

        # Apply residual blocks
        for f in self.blocks:
            real, imag = f(real, imag)

        # Apply output projection
        real, imag = self.output_norm(real, imag)
        real, imag = self.output_proj(real, imag)

        # Apply iSTFT followed by overlap and add
        real, imag = real.squeeze(1), imag.squeeze(1)
        x = self.stft.inverse(real, imag)

        return x, prior

    @torch.inference_mode()
    def inference(self, cond: Tensor, f0: Tensor) -> Tensor:
        return self(cond, f0)[0]


class MultiScaleWavehaxGenerator(nn.Module):
    """Multi-scale Wavehax generator module.

    The prior waveform is first decomposed into multiple scales, transformed into time-frequency representations, processed by
    ConvNeXt-based blocks, and finally synthesized back into a full-scale waveform using the corresponding inverse decomposition.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        mult_channels: int,
        kernel_size: int,
        num_blocks: int,
        decomposer: str,
        num_splits: int,
        n_fft: int,
        hop_length: int,
        sample_rate: int,
        prior_type: str,
        drop_prob: float = 0.0,
        framewise_norm: bool = True,
        use_gradient_checkpointing: bool = False,
        norm_type: str = "layer",
        padding_mode: str = "zeros",
        export_safe_ops: bool = False,
    ) -> None:
        """
        Initialize the MultiScaleWavehaxGenerator module.

        Args:
            in_channels (int): Number of conditioning feature channels.
            channels (int): Number of hidden feature channels.
            mult_channels (int): Channel expansion multiplier for ConvNeXt blocks.
            kernel_size (int): Kernel size for ConvNeXt blocks.
            num_blocks (int): Number of ConvNeXt residual blocks.
            decomposer (str): Name of the decomposer class used to obtain multi-scale waveform representations.
            num_splits (int): Number of subscales produced by the decomposition module.
            n_fft (int): Number of Fourier transform points (FFT size).
            hop_length (int): Hop length (frameshift) in samples.
            sample_rate (int): Sampling frequency of input and output waveforms in Hz.
            prior_type (str): Type of prior waveforms used (default: "pcph").
            drop_prob (float, optional): Probability of dropping paths for stochastic depth. Default: 0.0.
            framewise_norm (bool, optional): If True, normalization is performed independently for each time frame.
        """
        super().__init__()
        assert hop_length % num_splits == 0
        assert n_fft % (hop_length // num_splits) == 0

        self.in_channels = in_channels
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.norm_type = str(norm_type)
        self.padding_mode = str(padding_mode)
        self.export_safe_ops = bool(export_safe_ops)
        if self.norm_type not in {"layer", "batch"}:
            raise ValueError(f"Unsupported norm_type={self.norm_type}")

        # Define signal decomposition module
        self.num_splits = num_splits
        decomposer_cls = getattr(wavehax.modules, decomposer)
        if decomposer == "MultiStream1d":
            self.decomposer = decomposer_cls(
                num_splits,
                padding_mode=self.padding_mode,
                export_safe_synthesis=self.export_safe_ops,
            )
        else:
            self.decomposer = decomposer_cls(num_splits)

        # Prior waveform generator
        if self.export_safe_ops and prior_type == "pcph_closed_form":
            self.prior_generator = partial(
                deterministic_pcph_closed_form,
                hop_length=self.hop_length,
                sample_rate=sample_rate,
            )
        else:
            self.prior_generator = partial(
                getattr(wavehax.modules, f"generate_{prior_type}"),
                hop_length=self.hop_length,
                sample_rate=sample_rate,
            )

        # STFT layer
        stft_cls = RealDFTSTFT if self.export_safe_ops else STFT
        self.stft = stft_cls(n_fft=n_fft, hop_length=hop_length // num_splits)

        # Input projection layers
        self.cond_proj = nn.Conv1d(
            in_channels, num_splits * self.n_bins, 7, padding=3, padding_mode=self.padding_mode
        )

        # Input normalization and projection layers
        self.input_proj = nn.Conv2d(3 * num_splits, channels, 1, bias=False)
        if self.norm_type == "batch":
            self.input_norm = BatchNorm2d(channels)
        else:
            self.input_norm = LayerNorm2d(channels, framewise=framewise_norm)

        # ConvNeXt-based residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = ConvNeXtBlock2d(
                channels,
                mult_channels,
                kernel_size,
                drop_prob=drop_prob,
                use_layer_norm=self.norm_type == "layer",
                framewise_norm=framewise_norm,
                padding_mode=self.padding_mode,
                layer_scale_init_value=1 / num_blocks,
            )
            self.blocks += [block]

        # Output projection and normalization layers
        if self.norm_type == "batch":
            self.output_norm = BatchNorm2d(channels)
        else:
            self.output_norm = LayerNorm2d(channels, framewise=framewise_norm)
        self.output_proj = nn.Conv2d(channels, 2 * num_splits, 1)

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m) -> None:
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, cond: Tensor, f0: Tensor) -> Tensor:
        """
        Calculate forward propagation.

        Args:
            cond (Tensor): Conditioning features with shape (batch, in_channels, frames).
            f0 (Tensor): F0 sequences with shape (batch, 1, frames).

        Returns:
            Tensor: Generated waveforms with shape (batch, 1, frames * hop_length).
            Tensor: Generated prior waveforms with shape (batch, 1, frames * hop_length).
        """
        # Generate prior waveform
        with torch.no_grad():
            prior = self.prior_generator(f0)

        # Decompose prior signal
        priors = self.decomposer.analysis(prior)

        # Compute spectrograms
        specs = []
        for p in priors:
            real, imag = self.stft(p)
            specs += [real, imag]
        specs = torch.stack(specs, dim=1)

        # Apply input projection
        cond = self.cond_proj(cond)
        cond = rearrange(cond, "b (s f) t -> b s f t", s=self.num_splits)

        # Convert to 2d representation
        x = torch.cat([cond, specs], dim=1)
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Apply residual blocks
        for f in self.blocks:
            if self.use_gradient_checkpointing and self.training and x.requires_grad:
                x = checkpoint(f, x, use_reentrant=False)
            else:
                x = f(x)

        # Apply output projection
        x = self.output_norm(x)
        x = self.output_proj(x)

        # Split into real/imag parts for each sub-signal
        xs = list(x.chunk(2 * self.num_splits, dim=1))
        ys: List[Tensor] = []

        # Apply iSTFT with OLA to each sub-signal’s complex spectrogram
        for real, imag in zip(xs[:-1:2], xs[1::2]):
            real, imag = real.squeeze(1), imag.squeeze(1)
            y = self.stft.inverse(real, imag)
            ys += [y]

        # Construct the output signal
        y = self.decomposer.synthesis(ys)

        return y, prior

    @torch.inference_mode()
    def inference(self, cond: Tensor, f0: Tensor) -> Tensor:
        return self(cond, f0)[0]
