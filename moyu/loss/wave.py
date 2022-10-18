import torch
import torch.nn as nn
from torchlibrosa.stft import STFT, magphase

from moyu.utils.audio import magphase, wave_to_spectrogram


class L1_sp(nn.Module):
    def __init__(self):
        r"""L1 loss on the spectrogram."""
        super().__init__()

        self.window_size = 2048
        hop_size = 480
        center = True
        pad_mode = "reflect"
        window = "hann"

        self.stft = STFT(
            n_fft=self.window_size,
            hop_length=hop_size,
            win_length=self.window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def __call__(
        self, input: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        r"""L1 loss in the time-domain and on the spectrogram.

        Args:
            output: torch.Tensor
            target: torch.Tensor

        Returns:
            loss: torch.float
        """

        # L1 loss in the time-domain.
        input_real, input_imag = wave_to_spectrogram(input, self.stft)
        input_mag, *_ = magphase(input_real, input_imag)

        target_real, target_imag = wave_to_spectrogram(target, self.stft)
        target_mag, *_ = magphase(target_real, target_imag)
        
        # L1 loss on the spectrogram.
        # sp_loss = torch.mean(torch.abs(input_real - target_real)) + \
        #     torch.mean(torch.abs(input_imag - target_imag))
        sp_loss = torch.mean(torch.abs(input_mag - target_mag))

        # Total loss.
        return sp_loss


class L1_Wav_L1_Sp(nn.Module):
    def __init__(self):
        r"""L1 loss in the time-domain and L1 loss on the spectrogram."""
        super().__init__()

        self.window_size = 2048
        hop_size = 480
        center = True
        pad_mode = "reflect"
        window = "hann"

        self.stft = STFT(
            n_fft=self.window_size,
            hop_length=hop_size,
            win_length=self.window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def __call__(
        self, input: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        r"""L1 loss in the time-domain and on the spectrogram.

        Args:
            output: torch.Tensor
            target: torch.Tensor

        Returns:
            loss: torch.float
        """

        # L1 loss in the time-domain.
        wav_loss = torch.mean(torch.abs(input - target))

        # L1 loss in the time-domain.
        input_real, input_imag = wave_to_spectrogram(input, self.stft)
        input_mag, *_ = magphase(input_real, input_imag)

        target_real, target_imag = wave_to_spectrogram(target, self.stft)
        target_mag, *_ = magphase(target_real, target_imag)
        
        # L1 loss on the spectrogram.
        # sp_loss = torch.mean(torch.abs(input_real - target_real)) + \
        #     torch.mean(torch.abs(input_imag - target_imag))
        sp_loss = torch.mean(torch.abs(input_mag - target_mag))
        # Total loss.
        return wav_loss + sp_loss


class L2_Wav_L2_Sp(nn.Module):
    def __init__(self):
        r"""L2 loss in the time-domain and L2 loss on the spectrogram."""
        super().__init__()

        self.window_size = 2048
        hop_size = 480
        center = True
        pad_mode = "reflect"
        window = "hann"

        self.stft = STFT(
            n_fft=self.window_size,
            hop_length=hop_size,
            win_length=self.window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def __call__(
        self, input: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        r"""L2 loss in the time-domain and on the spectrogram.

        Args:
            output: torch.Tensor
            target: torch.Tensor

        Returns:
            loss: torch.float
        """

        # L2 loss in the time-domain.
        wav_loss = torch.mean((input - target) ** 2)

        real, imag = wave_to_spectrogram(input, self.stft)
        input_mag, *_ = magphase(real, imag)

        real, imag = wave_to_spectrogram(target, self.stft)
        target_mag, *_ = magphase(real, imag)
        
        # L2 loss on the spectrogram.
        sp_loss = torch.mean((input_mag - target_mag) ** 2)

        # Total loss.
        return wav_loss + sp_loss
