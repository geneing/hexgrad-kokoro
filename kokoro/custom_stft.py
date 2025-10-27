"""TensorFlow port of the Kokoro custom STFT helper."""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf


class CustomSTFT(tf.Module):
	"""Short-time Fourier transform helper matching the PyTorch reference.

	The PyTorch implementation synthesises convolution kernels for the forward and
	inverse transforms.  In TensorFlow we lean on :func:`tf.signal.stft` and
	:func:`tf.signal.inverse_stft` while reproducing the same window handling,
	padding semantics, and output layout (``[B, freq_bins, frames]``).
	"""

	def __init__(
		self,
		filter_length: int = 800,
		hop_length: int = 200,
		win_length: int = 800,
		window: str = "hann",
		center: bool = True,
		pad_mode: str = "replicate",  # or "constant"
		name: Optional[str] = None,
	) -> None:
		super().__init__(name=name)
		self.filter_length = int(filter_length)
		self.hop_length = int(hop_length)
		self.win_length = int(win_length)
		self.n_fft = self.filter_length
		self.center = bool(center)
		self.pad_mode = pad_mode

		if window != "hann":
			raise ValueError(f"Unsupported window '{window}'. Only 'hann' is implemented.")

		if pad_mode not in {"replicate", "constant"}:
			raise ValueError("pad_mode must be 'replicate' or 'constant'.")

		# Number of frequency bins for real-valued STFT with onesided=True
		self.freq_bins = self.n_fft // 2 + 1
		self.pad_len = self.n_fft // 2

		# Build the analysis window and pad/truncate to match n_fft exactly, mirroring PyTorch.
		window_tensor = tf.signal.hann_window(self.win_length, periodic=True, dtype=tf.float32)
		if self.win_length < self.n_fft:
			extra = self.n_fft - self.win_length
			window_tensor = tf.pad(window_tensor, [[0, extra]])
		elif self.win_length > self.n_fft:
			window_tensor = window_tensor[: self.n_fft]

		self.window = tf.constant(window_tensor, dtype=tf.float32, name="window")

		# Store callable references for tf.signal utilities.
		self._forward_window_fn = self._make_forward_window_fn()
		self._inverse_window_fn = tf.signal.inverse_stft_window_fn(
			frame_step=self.hop_length,
			forward_window_fn=self._forward_window_fn,
		)

	# ---------------------------------------------------------------------
	# Window helpers
	def _make_forward_window_fn(self):
		"""Return a tf.signal-compatible window_fn using the cached Hann window."""

		def forward_window_fn(frame_length: int, dtype: tf.dtypes.DType) -> tf.Tensor:
			if frame_length != self.n_fft:
				raise ValueError(
					f"Expected frame_length={self.n_fft}, received {frame_length}."
				)
			return tf.cast(self.window, dtype)

		return forward_window_fn

	# ------------------------------------------------------------------
	# Padding logic
	def _pad_waveform(self, waveform: tf.Tensor) -> tf.Tensor:
		"""Apply PyTorch-style centre padding using replicate or constant mode."""

		if not self.center or self.pad_len == 0:
			return waveform

		pad = self.pad_len
		if self.pad_mode == "replicate":
			left = tf.repeat(waveform[:, :1], repeats=pad, axis=1)
			right = tf.repeat(waveform[:, -1:], repeats=pad, axis=1)
			return tf.concat([left, waveform, right], axis=1)

		# pad_mode == "constant"
		return tf.pad(waveform, [[0, 0], [pad, pad]])

	# ------------------------------------------------------------------
	# Public API
	def transform(self, waveform: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
		"""Compute magnitude and phase using STFT.

		Args:
			waveform: Tensor of shape ``[batch, time]``.

		Returns:
			A tuple ``(magnitude, phase)`` with shape ``[batch, freq_bins, frames]``.
		"""

		waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
		waveform = self._pad_waveform(waveform)

		stft = tf.signal.stft(
			waveform,
			frame_length=self.n_fft,
			frame_step=self.hop_length,
			fft_length=self.n_fft,
			window_fn=self._forward_window_fn,
			pad_end=False,
		)  # [B, frames, freq_bins]

		real = tf.math.real(stft)
		imag = tf.math.imag(stft)

		magnitude = tf.sqrt(tf.square(real) + tf.square(imag) + 1e-14)
		phase = tf.math.atan2(imag, real)

		correction_mask = tf.logical_and(tf.equal(imag, 0.0), tf.less(real, 0.0))
		phase = tf.where(correction_mask, tf.constant(np.pi, dtype=phase.dtype), phase)

		# Match PyTorch layout: [B, freq_bins, frames]
		magnitude = tf.transpose(magnitude, perm=[0, 2, 1])
		phase = tf.transpose(phase, perm=[0, 2, 1])
		return magnitude, phase

	def inverse(
		self,
		magnitude: tf.Tensor,
		phase: tf.Tensor,
		length: Optional[int] = None,
	) -> tf.Tensor:
		"""Reconstruct waveform from magnitude/phase tensors."""

		magnitude = tf.convert_to_tensor(magnitude, dtype=tf.float32)
		phase = tf.convert_to_tensor(phase, dtype=tf.float32)

		real = magnitude * tf.math.cos(phase)
		imag = magnitude * tf.math.sin(phase)

		complex_spec = tf.complex(real, imag)
		complex_spec = tf.transpose(complex_spec, perm=[0, 2, 1])  # [B, frames, freq_bins]

		waveform = tf.signal.inverse_stft(
			complex_spec,
			frame_length=self.n_fft,
			frame_step=self.hop_length,
			fft_length=self.n_fft,
			window_fn=self._inverse_window_fn,
		)

		if self.center and self.pad_len > 0:
			waveform = waveform[:, self.pad_len : -self.pad_len]

		if length is not None:
			waveform = waveform[:, :length]

		return waveform

	def __call__(self, x: tf.Tensor) -> tf.Tensor:
		magnitude, phase = self.transform(x)
		return self.inverse(magnitude, phase, length=tf.shape(x)[-1])


