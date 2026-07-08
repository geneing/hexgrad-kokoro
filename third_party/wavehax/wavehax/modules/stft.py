# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Modules related to short-time Fourier transform (STFT)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_filters_mel
from torch import Tensor


def to_log_magnitude_and_phase(
    real: Tensor, imag: Tensor, clip_value: Optional[float] = 1e-10
) -> Tuple[Tensor, Tensor]:
    """
    Convert real and imaginary components of a complex signal to log-magnitude and phase.

    Args:
        real (Tensor): Real part of the complex signal.
        imag (Tensor): Imaginary part of the complex signal.
        clip_value (float, optional): Minimum value for magnitude to avoid log of zero (default: 1e-10).

    Returns:
        Tuple[Tensor, Tensor]: Log-magnitude and phase of the input complex signal.
    """
    magnitude = torch.sqrt(torch.clamp(real**2 + imag**2, min=clip_value))
    log_magnitude = torch.log(magnitude)
    phase = torch.atan2(imag, real)
    return log_magnitude, phase


def to_real_imaginary(
    log_magnitude: Tensor, phase: Tensor, clip_value: Optional[float] = 1e2
) -> Tuple[Tensor, Tensor]:
    """
    Convert log-magnitude and implicit phase wrapping back to real and imaginary components of a complex signal.

    Args:
        log_magnitude (Tensor): Log-magnitude of the complex signal.
        phase (Tensor): Implicit phase wrapping spectra as in Vocos.
        clip_value (float, optional): Maximum allowed value for magnitude after exponentiation (default: 1e2).

    Returns:
        Tuple[Tensor, Tensor]: Real and imaginary components of the complex signal.

    References:
        - https://arxiv.org/abs/2306.00814
        - https://github.com/gemelo-ai/vocos
    """
    magnitude = torch.clip(torch.exp(log_magnitude), max=clip_value)
    real, imag = magnitude * torch.cos(phase), magnitude * torch.sin(phase)
    return real, imag


class STFT(nn.Module):
    """
    Short-Time Fourier Transform (STFT) module.

    References:
        - https://github.com/gemelo-ai/vocos
        - https://github.com/echocatzh/torch-mfcc
    """

    def __init__(
        self, n_fft: int, hop_length: int, window: Optional[str] = "hann_window"
    ) -> None:
        """
        Initialize the STFT module.

        Args:
            n_fft (int): Number of Fourier transform points (FFT size).
            hop_length (int): Hop length (frameshift) in samples.
            window (str, optional): Name of the window function (default: "hann_window").
        """
        super().__init__()
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length

        # Create the window function and its squared values for normalization
        window = getattr(torch, window)(self.n_fft).reshape(1, n_fft, 1)
        self.register_buffer("window", window.reshape(1, n_fft, 1))
        window_envelope = window.square()
        self.register_buffer("window_envelope", window_envelope.reshape(1, n_fft, 1))

        # Create the kernel for enframe operation (sliding windows)
        enframe_kernel = torch.eye(self.n_fft).unsqueeze(1)
        self.register_buffer("enframe_kernel", enframe_kernel)

    def forward(self, x: Tensor, norm: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """
        Perform the forward Short-Time Fourier Transform (STFT) on the input waveform.

        Args:
            x (Tensor): Input waveform with shape (batch, samples) or (batch, 1, samples).
            norm (str, optional): Normalization mode for the FFT (default: None).

        Returns:
            Tuple[Tensor, Tensor]: Real and imaginary parts of the STFT result.
        """
        # Apply zero-padding to the input signal
        pad = self.n_fft - self.hop_length
        pad_left = pad // 2
        x = F.pad(x, (pad_left, pad - pad_left))

        # Enframe the padded waveform (sliding windows)
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = F.conv1d(x, self.enframe_kernel, stride=self.hop_length)

        # cuFFT only supports power-of-two half/bfloat16 FFT sizes. Keep the
        # FFT itself in fp32 so non-power-of-two sizes such as n_fft=480 work
        # under autocast, then cast the spectrogram back to the incoming dtype.
        out_dtype = x.dtype
        fft_dtype = torch.float32 if x.is_cuda and x.dtype in {torch.float16, torch.bfloat16} else x.dtype
        x = x.to(dtype=fft_dtype) * self.window.to(dtype=fft_dtype)
        x_stft = torch.fft.rfft(x, n=self.n_fft, dim=1, norm=norm)
        real, imag = x_stft.real, x_stft.imag

        return real.to(dtype=out_dtype), imag.to(dtype=out_dtype)

    def inverse(self, real: Tensor, imag: Tensor, norm: Optional[str] = None) -> Tensor:
        """
        Perform the inverse Short-Time Fourier Transform (iSTFT) to reconstruct the waveform from the complex spectrogram.

        Args:
            real (Tensor): Real part of the complex spectrogram with shape (batch, n_bins, frames).
            imag (Tensor): Imaginary part of the complex spectrogram with shape (batch, n_bins, frames).
            norm (str, optional): Normalization mode for the inverse FFT (default: None).

        Returns:
            Tensor: Reconstructed waveform with shape (batch, 1, samples).
        """
        # Validate shape and dimensionality
        assert real.shape == imag.shape and real.ndim == 3

        # Ensure the input represents a one-sided spectrogram
        assert real.size(1) == self.n_bins

        frames = real.shape[2]
        samples = frames * self.hop_length

        # Inverse RDFT and apply windowing, followed by overlap-add. See the
        # forward path for why the FFT is forced to fp32 under CUDA autocast.
        out_dtype = real.dtype
        fft_dtype = torch.float32 if real.is_cuda and real.dtype in {torch.float16, torch.bfloat16} else real.dtype
        real = real.to(dtype=fft_dtype)
        imag = imag.to(dtype=fft_dtype)
        x = torch.fft.irfft(torch.complex(real, imag), n=self.n_fft, dim=1, norm=norm)
        x = x * self.window.to(dtype=fft_dtype)
        x = F.conv_transpose1d(x, self.enframe_kernel.to(dtype=fft_dtype), stride=self.hop_length)

        # Compute window envelope for normalization
        window_envelope = F.conv_transpose1d(
            self.window_envelope.to(dtype=fft_dtype).repeat(1, 1, frames),
            self.enframe_kernel.to(dtype=fft_dtype),
            stride=self.hop_length,
        )

        # Remove padding
        pad = (self.n_fft - self.hop_length) // 2
        x = x[..., pad : samples + pad]
        window_envelope = window_envelope[..., pad : samples + pad]

        # Normalize the output by the window envelope
        assert (window_envelope > 1e-11).all()
        x = x / window_envelope

        return x.to(dtype=out_dtype)


class RealDFTSTFT(nn.Module):
    """STFT/iSTFT using real DFT projections and regular convolutions only."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        window: Optional[str] = "hann_window",
        trainable_inverse: bool = False,
        trainable_analysis: bool = False,
        trainable_window: bool = False,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.n_bins = self.n_fft // 2 + 1
        self.hop_length = int(hop_length)
        self.trainable_inverse = bool(trainable_inverse)
        self.trainable_analysis = bool(trainable_analysis)
        self.trainable_window = bool(trainable_window)

        window_tensor = getattr(torch, window)(self.n_fft).reshape(1, self.n_fft, 1)
        self._register_tensor("window", window_tensor.reshape(1, self.n_fft, 1), self.trainable_window)
        self.register_buffer("window_initial", window_tensor.reshape(1, self.n_fft, 1).clone(), persistent=False)
        self.register_buffer("window_envelope", window_tensor.square().reshape(1, self.n_fft, 1))
        self.register_buffer("enframe_kernel", torch.eye(self.n_fft).unsqueeze(1))

        n = torch.arange(self.n_fft, dtype=torch.float32).unsqueeze(0)
        k = torch.arange(self.n_bins, dtype=torch.float32).unsqueeze(1)
        angle = 2.0 * torch.pi * k * n / float(self.n_fft)
        dft_cos = torch.cos(angle).unsqueeze(-1)
        dft_sin = torch.sin(angle).unsqueeze(-1)
        self._register_tensor("dft_cos", dft_cos, self.trainable_analysis)
        self._register_tensor("dft_sin", dft_sin, self.trainable_analysis)
        self.register_buffer("dft_cos_initial", dft_cos.clone(), persistent=False)
        self.register_buffer("dft_sin_initial", dft_sin.clone(), persistent=False)

        mid_k = torch.arange(1, self.n_bins - 1, dtype=torch.float32)
        n_col = torch.arange(self.n_fft, dtype=torch.float32).unsqueeze(1)
        mid_angle = 2.0 * torch.pi * n_col * mid_k.unsqueeze(0) / float(self.n_fft)
        idft_cos_mid = torch.cos(mid_angle).unsqueeze(-1)
        idft_sin_mid = torch.sin(mid_angle).unsqueeze(-1)
        self._register_tensor("idft_cos_mid", idft_cos_mid, self.trainable_inverse)
        self._register_tensor("idft_sin_mid", idft_sin_mid, self.trainable_inverse)
        self.register_buffer("idft_cos_mid_initial", idft_cos_mid.clone(), persistent=False)
        self.register_buffer("idft_sin_mid_initial", idft_sin_mid.clone(), persistent=False)
        self.register_buffer(
            "nyquist_sign",
            torch.pow(torch.tensor(-1.0, dtype=torch.float32), torch.arange(self.n_fft, dtype=torch.float32)),
        )

        shift_kernel = torch.zeros(self.n_fft, 1, self.n_fft, dtype=torch.float32)
        for c in range(self.n_fft):
            shift_kernel[c, 0, self.n_fft - 1 - c] = 1.0
        self.register_buffer("ola_shift_kernel", shift_kernel)
        hop_mask = torch.zeros(1, 1, self.hop_length, dtype=torch.float32)
        hop_mask[..., 0] = 1.0
        self.register_buffer("hop_mask", hop_mask)

    def _register_tensor(self, name: str, tensor: Tensor, trainable: bool) -> None:
        if trainable:
            self.register_parameter(name, nn.Parameter(tensor.clone()))
        else:
            self.register_buffer(name, tensor.clone())

    def set_trainable(self, inverse: bool, analysis: bool = False, window: bool | None = None) -> None:
        window = bool(inverse or analysis) if window is None else bool(window)
        for name, param in self.named_parameters(recurse=False):
            if name in {"idft_cos_mid", "idft_sin_mid"}:
                param.requires_grad_(bool(inverse))
            elif name in {"dft_cos", "dft_sin"}:
                param.requires_grad_(bool(analysis))
            elif name == "window":
                param.requires_grad_(window)

    def regularization_loss(self) -> Tensor:
        terms = []
        for name in ("dft_cos", "dft_sin", "idft_cos_mid", "idft_sin_mid", "window"):
            value = getattr(self, name, None)
            initial = getattr(self, f"{name}_initial", None)
            if isinstance(value, nn.Parameter) and initial is not None:
                terms.append(F.mse_loss(value, initial.to(dtype=value.dtype, device=value.device)))
        if not terms:
            return self.window.new_zeros(())
        return torch.stack(terms).sum()

    def reconstruction_loss(self, audio: Tensor) -> Tensor:
        real, imag = self(audio)
        reconstructed = self.inverse(real, imag)
        audio = audio.unsqueeze(1) if audio.dim() == 2 else audio
        n = min(audio.shape[-1], reconstructed.shape[-1])
        return F.l1_loss(reconstructed[..., :n], audio[..., :n])

    def forward(self, x: Tensor, norm: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        if norm is not None:
            raise ValueError("RealDFTSTFT only supports norm=None")
        pad = self.n_fft - self.hop_length
        pad_left = pad // 2
        x = F.pad(x, (pad_left, pad - pad_left))
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = F.conv1d(x, self.enframe_kernel.to(dtype=x.dtype), stride=self.hop_length)
        x = x * self.window.to(dtype=x.dtype)
        real = F.conv1d(x, self.dft_cos.to(dtype=x.dtype))
        imag = -F.conv1d(x, self.dft_sin.to(dtype=x.dtype))
        return real, imag

    def inverse(self, real: Tensor, imag: Tensor, norm: Optional[str] = None) -> Tensor:
        if norm is not None:
            raise ValueError("RealDFTSTFT only supports norm=None")
        assert real.shape == imag.shape and real.ndim == 3
        assert real.size(1) == self.n_bins

        frames = real.shape[2]
        samples = frames * self.hop_length

        dc = real[:, 0:1, :]
        nyquist = real[:, -1:, :] * self.nyquist_sign.to(dtype=real.dtype).view(1, self.n_fft, 1)
        real_mid = real[:, 1:-1, :]
        imag_mid = imag[:, 1:-1, :]
        inner = F.conv1d(real_mid, self.idft_cos_mid.to(dtype=real.dtype))
        inner = inner - F.conv1d(imag_mid, self.idft_sin_mid.to(dtype=imag.dtype))
        x = (dc + nyquist + 2.0 * inner) / float(self.n_fft)

        x = x * self.window.to(dtype=x.dtype)
        x = self._overlap_add(x)
        window_envelope = self._overlap_add(self.window.to(dtype=x.dtype).square().repeat(1, 1, frames))

        pad = (self.n_fft - self.hop_length) // 2
        x = x[..., pad : samples + pad]
        window_envelope = window_envelope[..., pad : samples + pad]
        return x / window_envelope.clamp_min(1e-11)

    def _overlap_add(self, x: Tensor) -> Tensor:
        hop = int(self.hop_length)
        if hop > 1:
            frames = x.shape[-1]
            x_up = x.repeat_interleave(hop, dim=2)
            mask = self.hop_mask.to(dtype=x.dtype, device=x.device).repeat(1, 1, frames)
            x_up = x_up * mask
        else:
            x_up = x
        shifted = F.conv1d(
            x_up,
            self.ola_shift_kernel.to(dtype=x.dtype, device=x.device),
            stride=1,
            padding=self.n_fft - 1,
            groups=self.n_fft,
        )
        out_size = (x.shape[-1] - 1) * hop + self.n_fft
        return shifted.sum(dim=1, keepdim=True)[..., :out_size]


class MelSpectrogram(nn.Module):
    """A module to compute a mel-spectrogram from waveforms."""

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        n_fft: int,
        n_mels: int,
        window: Optional[str] = "hann_window",
        fmin: Optional[float] = 0,
        fmax: Optional[float] = None,
    ) -> None:
        """
        Initialize the MelSpectrogram module.

        Args:
            sample_rate (int): Sampling frequency of input waveforms.
            hop_length (int): Hop length (frameshift) in samples.
            n_fft (int): Number of Fourier transform points (FFT size).
            n_mels (int): Number of mel basis.
            window (str, optional): Name of the window function (default: "hann_window).
            fmin (float, optional): Minimum frequency for mel-filter bank (default: 0).
            fmax (float, optional): Maximum frequency for mel-filter bank (default: None).
        """
        super().__init__()
        self.n_mels = n_mels
        self.stft = STFT(n_fft, hop_length, window)
        mel_basis = librosa_filters_mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )  # (n_mels, n_bins)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis))

    def forward(
        self,
        audio: Tensor,
        log_scale: Optional[bool] = True,
        eps: Optional[float] = 1e-5,
    ) -> Tensor:
        """
        Compute mel-spectrogram from the input waveforms.

        Args:
            audio (Tensor): Input waveforms with shape (batch, samples) or (batch, 1, samples).
            log_scale (bool, optional): Whether to return the log-magnitude of the mel-spectrogram (default: True).
            eps (float, optional): Small value to avoid numerical instability in log calculation (default: 1e-5).

        Returns:
            Tensor: Mel-spectrogram with shape (batch, n_mels, frames).
        """
        real, imag = self.stft(audio)
        mel = torch.matmul(self.mel_basis, torch.complex(real, imag).abs())
        mel = torch.log(torch.clamp(mel, min=eps)) if log_scale else mel
        return mel


def griffin_lim(spectrogram: Tensor, stft: STFT, n_iter: int) -> Tensor:
    """
    Perform the Griffin-Lim algorithm for phase recovery from a magnitude spectrogram.

    Args:
        spectrogram (Tensor): Input complex spectrogram tensor with shape (batch, bins, frames).
        stft (STFT): STFT object with the configuration used for the spectrogram.
        n_iter (int): Number of iterations for phase recovery.

    Returns:
        Tensor: Recovered waveform tensor with shape (batch, bins, frames).
    """
    magnitude = spectrogram.abs()
    phase = spectrogram.angle()
    for _ in range(n_iter):
        inverse = stft.inverse(magnitude * torch.exp(1.0j * phase))
        phase = stft(inverse).angle()
    return phase
