import torch
import torch.nn as nn

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np
import torch
import torch.nn as nn
from torchlibrosa.stft import ISTFT, STFT

from moyu.model.ambisonic_to_binaural.utils import B_to_A_format
from moyu.utils.audio import magphase, wave_to_spectrogram


class Base(nn.Module):
    """
    Assume original ambisonic is B-format.
    """
    def __init__(
        self,
        input_format: Literal["mag", "complex", "mag+relative_phase", "complex+relatve_phase", "complex+GCC"],
        output_format: Literal["complex", "mag+phase", "mask+phase"],
        ambisonic_format: Literal["A", "B"],
        # regular_phase: Literal["none", "omni", "spec"],
        # regular_output: bool,
    ):
        super().__init__()

        window_size = 2048

        hop_size = 480
        center = True
        pad_mode = "reflect"
        window = "hann"

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        if input_format == "mag":
            input_channels = 4
        elif input_format == "complex":
            input_channels = 8
        elif input_format == "absolute_phase":
            input_channels = 8
        elif input_format == "relative_phase":
            input_channels = 12
        elif input_format == "complex+absolute_phase":
            input_channels = 16
        # elif input_format == "complex_repeat":
        #     input_channels = 16
        elif input_format == "mag+relative_phase":
            input_channels = 16
        elif input_format == "complex+relative_phase":
            input_channels = 20
        elif input_format == "complex+GCC":
            input_channels = 20
        else:
            raise Exception("Unrecognized input_format")

        self.input_channels = input_channels

        if output_format == "complex":
            out_channels = 4
        elif output_format == "mag+relative_phase":
            out_channels = 6
        elif output_format == "mask+absolute_phase":
            out_channels = 6
        elif output_format == "mask+relative_phase":
            out_channels = 6
        else:
            raise Exception("Unrecognized output_format")

        self.output_channels = out_channels

        self.input_format = input_format
        self.output_format = output_format
        self.ambisonic_format = ambisonic_format

        self.regular_phase = "omni"
        self.regular_output = False

    def calculate_sine_and_cosine_of_delta(self, cos_1, sin_1, cos_2, sin_2):
        # cos(∠1-∠2), sin(∠1-∠2)
        cos_delta = cos_1 * cos_2 + sin_1 * sin_2
        sin_delta = sin_1 * cos_2 - cos_1 * sin_2
        return cos_delta, sin_delta

    def calcualate_relative_phase(self, cos, sin):
        """Calculate relative phase that is represented by sine and cosine.
        shape: (N, 6, T, F)
        """
        cos_12, sin_12 = self.calculate_sine_and_cosine_of_delta(
            cos[:, 0:1, :, :],
            sin[:, 0:1, :, :],
            cos[:, 1:2, :, :],
            sin[:, 1:2, :, :]
        )

        cos_13, sin_13 = self.calculate_sine_and_cosine_of_delta(
            cos[:, 0:1, :, :],
            sin[:, 0:1, :, :],
            cos[:, 2:3, :, :],
            sin[:, 2:3, :, :]
        )

        cos_14, sin_14 = self.calculate_sine_and_cosine_of_delta(
            cos[:, 0:1, :, :],
            sin[:, 0:1, :, :],
            cos[:, 3:, :, :],
            sin[:, 3:, :, :]
        )

        cos_23, sin_23 = self.calculate_sine_and_cosine_of_delta(
            cos[:, 1:2, :, :],
            sin[:, 1:2, :, :],
            cos[:, 2:3, :, :],
            sin[:, 2:3, :, :]
        )

        cos_24, sin_24 = self.calculate_sine_and_cosine_of_delta(
            cos[:, 1:2, :, :],
            sin[:, 1:2, :, :],
            cos[:, 3:, :, :],
            sin[:, 3:, :, :]
        )

        cos_34, sin_34 = self.calculate_sine_and_cosine_of_delta(
            cos[:, 2:3, :, :],
            sin[:, 2:3, :, :],
            cos[:, 3:, :, :],
            sin[:, 3:, :, :]
        )
        return torch.cat([
            cos_12, sin_12, 
            cos_13, sin_13,
            cos_14, sin_14,
            cos_23, sin_23,
            cos_24, sin_24,    
            cos_34, sin_34], dim=1)
                
    def calculate_GCC_feature(self, real, imag):
        def cal(real1, imag1, real2, imag2):
            real = real1 * real2 - imag1 * imag2
            imag = real1 * imag2 + real2 * imag1
            return torch.cat([real, imag], dim=1)
        
        gcc12 = cal(real[:, 0:1, ...], imag[:, 0:1, ...], real[:, 1:2, ...], imag[:, 1:2, ...])
        gcc13 = cal(real[:, 0:1, ...], imag[:, 0:1, ...], real[:, 2:3, ...], imag[:, 2:3, ...])
        gcc14 = cal(real[:, 0:1, ...], imag[:, 0:1, ...], real[:, 3:, ...], imag[:, 3:, ...])
        gcc23 = cal(real[:, 1:2, ...], imag[:, 1:2, ...], real[:, 2:3, ...], imag[:, 2:3, ...])
        gcc24 = cal(real[:, 1:2, ...], imag[:, 1:2, ...], real[:, 3:, ...], imag[:, 3:, ...])
        gcc34 = cal(real[:, 2:3, ...], imag[:, 2:3, ...], real[:, 3:, ...], imag[:, 3:, ...])
        return torch.cat([
            gcc12, gcc13, gcc14, gcc23, gcc24, gcc34], dim=1)

    def pack_input(self, waveform):
        # ambisonic: (N, 4, L)
        x = waveform
        if self.ambisonic_format == "A":
            x = B_to_A_format(waveform)            

        real, imag = wave_to_spectrogram(x, self.stft)

        mag, cos, sin = magphase(real, imag)

        complex_ = torch.cat([real, imag], dim=1)
        
        regular_phase = 1
        if self.regular_phase == "omni":
            regular_phase = mag[:, 0:1, ...]
        elif self.regular_phase == "spec":
            regular_phase = mag            

        # self._ctxs = (waveform.shape[2], b_mag, b_cos_in, b_sin_in)
        if self.input_format == "mag":
            return mag

        elif self.input_format == "complex":
            return complex_

        elif self.input_format == "absolute_phase":
            # absolute_phase supports regular_phase
            return torch.cat([regular_phase * cos, regular_phase * sin], dim=1)
        
        elif self.input_format == "relative_pahse":
            # relative_phase does not support regular_phase
            relative_phase = self.calcualate_relative_phase(cos, sin)
            return mag[:, 0:1, ...] * relative_phase
        
        elif self.input_format == "complex+absolute_phase":
            return torch.cat([complex_, regular_phase * cos, regular_phase * sin], dim=1)

        elif self.input_format == "complex_repeat":
            return torch.cat([complex_, complex_], dim=1)

        elif self.input_format == "mag+relative_phase":
            relative_phase = self.calcualate_relative_phase(cos, sin)
            return torch.cat([mag, mag[:, 0:1, ...] * relative_phase], dim=1)

        elif self.input_format == "complex+relative_phase":
            relative_phase = self.calcualate_relative_phase(cos, sin)
            return torch.cat([complex_, mag[:, 0:1, ...] * relative_phase], dim=1)

        elif self.input_format == "complex+GCC":
            GCC_feature = self.calculate_GCC_feature(real, imag)
            return torch.cat([complex_, GCC_feature], dim=1)

        else:
            raise NotImplementedError
        
    def render_output(self, feature, raw_waveform):
        audio_length = raw_waveform.shape[-1]

        if self.output_format == "complex":
            # x: (N, 4, T, F)
            # real, imag: (N, 2, T, F)
            real = feature[:, :2, :, :]
            imag = feature[:, 2:, :, :]

        elif self.output_format == "mag+relative_phase":
            # Are those numerical filtering operations important?
            mag = feature[:, :2, :, :]

            real, imag = wave_to_spectrogram(raw_waveform, self.stft)
            _, b_cos, b_sin = magphase(real, imag)

            d_cos = feature[:, 2:4, :, :]
            d_sin = feature[:, 4:, :, :]

            if self.regular_output:
                mag = torch.relu(mag)
                d_cos = torch.tanh(d_cos)
                d_sin = torch.tanh(d_sin)
                _, d_cos, d_sin = magphase(d_cos, d_sin)

            cos = d_cos * b_cos[:, 0:1, :, :] - b_sin[:, 0:1, :, :] * d_sin
            sin = b_sin[:, 0:1, :, :] * d_cos + b_cos[:, 0:1, :, :] * d_sin

            # _, cos, sin = magphase(sin, cos)
            real = mag * cos
            imag = mag * sin

        elif self.output_format == "mask+absolute_phase":
            # x: (N, 4, T, F)
            mask = feature[:, :2, :, :]

            real, imag = wave_to_spectrogram(raw_waveform, self.stft)
            mag, *_ = magphase(real, imag)

            cos = feature[:, 2:4, :, :]
            sin = feature[:, 4:, :, :]

            if self.regular_output:
                mask = torch.sigmoid(mask)
                cos = torch.tanh(cos)
                sin = torch.tanh(sin)
                _, cos, sin = magphase(cos, sin)

            mag = mag[:, 0:1, ...] * mask
            real = mag * cos
            imag = mag * sin

        elif self.output_format == "mask+relative_phase":
            mask = feature[:, :2, :, :]

            real, imag = wave_to_spectrogram(raw_waveform, self.stft)
            mag, b_cos, b_sin = magphase(real, imag)

            d_cos = feature[:, 2:4, :, :]
            d_sin = feature[:, 4:, :, :]

            if self.regular_output:
                mask = torch.sigmoid(mask)
                d_cos = torch.tanh(d_cos)
                d_sin = torch.tanh(d_sin)
                _, d_cos, d_sin = magphase(d_cos, d_sin)

            cos = d_cos * b_cos[:, 0:1, :, :] - b_sin[:, 0:1, :, :] * d_sin
            sin = b_sin[:, 0:1, :, :] * d_cos + b_cos[:, 0:1, :, :] * d_sin
            
            mag = mag[:, 0:1, ...] * mask
            real = mag * cos
            imag = mag * sin

        else:
            raise NotImplementedError
    
        left = self.istft(real[:, 0:, :, :], imag[:, 0:1, :, :], audio_length)
        right = self.istft(real[:, 1:2, :, :], imag[:, 1:2, :, :], audio_length)
        separated_audio = torch.stack([left, right], dim=1)
        return separated_audio
    
    def reset_parameters(self):
        # from moyu.model.ambisonic_to_binaural.reset import init_module
        # self.apply(init_module)
        pass