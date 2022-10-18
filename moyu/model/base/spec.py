from typing import Tuple

import torch
from torchlibrosa.stft import STFT
import numpy as np


class SPEC:
    def __init__(
        self,
        window_size: int,
        hop_size: int,
        window: str,
        center: bool,
        pad_mode: str,
        freeze_parameters: bool
        ):
        r"""Base function for extracting spectrogram, cos, and sin, etc."""
        
        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=freeze_parameters,
        )

    def spectrogram(self, input: torch.Tensor) -> torch.Tensor:
        r"""Calculate spectrogram.

        Args:
            input: (batch_size, segments_num)
            eps: float

        Returns:
            spectrogram: (batch_size, time_steps, freq_bins)
        """
        real, imag = self.stft(input)
        return (real ** 2 + imag ** 2) ** 0.5

    def spectrogram_phase(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Calculate spectrogram.

        Args:
            input: (batch_size, segments_num)
            eps: float

        Returns:
            spectrogram: (batch_size, time_steps, freq_bins)
        """
        real, imag = self.stft(input)
        mag = (real ** 2 + imag ** 2) ** 0.5
        return mag, real / mag, imag / mag

    def wav_to_spectrogram_phase(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Convert waveforms to magnitude, cos, and sin of STFT.

        Args:
            input: (batch_size, channels_num, segment_samples)
            eps: float

        Outputs:
            mag: (batch_size, channels_num, time_steps, freq_bins)
            cos: (batch_size, channels_num, time_steps, freq_bins)
            sin: (batch_size, channels_num, time_steps, freq_bins)
        """
        batch_size, channels_num, segment_samples = input.shape

        # Reshape input with shapes of (n, segments_num) to meet the
        # requirements of the stft function.
        x = input.reshape(batch_size * channels_num, segment_samples)

        mag, cos, sin = self.spectrogram_phase(x)
        # mag, cos, sin: (batch_size * channels_num, 1, time_steps, freq_bins)

        _, _, time_steps, freq_bins = mag.shape
        mag = mag.reshape(batch_size, channels_num, time_steps, freq_bins)
        cos = cos.reshape(batch_size, channels_num, time_steps, freq_bins)
        sin = sin.reshape(batch_size, channels_num, time_steps, freq_bins)

        return mag, cos, sin

    def wav_to_spectrogram(
        self, input: torch.Tensor, eps: float = 1e-10
    ) -> torch.Tensor:

        mag, *_ = self.wav_to_spectrogram_phase(input)
        return mag