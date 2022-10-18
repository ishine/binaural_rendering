from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from moyu.model.ambisonic_to_binaural.base import Base


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
    ):
        r"""Convolutional block."""
        super(ConvBlock, self).__init__()

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activ1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activ2 = nn.LeakyReLU()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, in_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, out_feature_maps, time_steps, freq_bins)
        """
        x = self.activ1(self.bn1(self.conv1(input_tensor)))
        x = self.activ2(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        downsample: Tuple[int, int],
    ):
        r"""Encoder block."""
        super(EncoderBlock, self).__init__()

        self.conv_block = ConvBlock(
            in_channels, out_channels, kernel_size
        )
        self.downsample = nn.AvgPool2d(kernel_size=downsample)

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, in_feature_maps, time_steps, freq_bins)

        Returns:
            encoder_pool: (batch_size, out_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            encoder: (batch_size, out_feature_maps, time_steps, freq_bins)
        """
        encoder_tensor = self.conv_block(input_tensor)
        # encoder: (batch_size, out_feature_maps, time_steps, freq_bins)

        encoder_pool = self.downsample(encoder_tensor)
        # encoder_pool: (batch_size, out_feature_maps, downsampled_time_steps, downsampled_freq_bins)

        return encoder_pool, encoder_tensor


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        upsample: Tuple[int, int],
    ):
        r"""Decoder block."""
        super(DecoderBlock, self).__init__()

        self.kernel_size = kernel_size
        self.stride = upsample

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
            dilation=1,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.act1 = nn.LeakyReLU()

        self.conv_block2 = ConvBlock(
            out_channels * 2, out_channels, kernel_size
        )

    def forward(
        self, input_tensor: torch.Tensor, concat_tensor: torch.Tensor
    ) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            torch_tensor: (batch_size, in_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            concat_tensor: (batch_size, in_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, out_feature_maps, time_steps, freq_bins)
        """
        x = self.act1(self.bn1(self.conv1(input_tensor)))
        # (batch_size, in_feature_maps, time_steps, freq_bins)

        x = torch.cat((x, concat_tensor), dim=1)
        # (batch_size, in_feature_maps * 2, time_steps, freq_bins)

        output_tensor = self.conv_block2(x)
        # output_tensor: (batch_size, out_feature_maps, time_steps, freq_bins)

        return output_tensor


class UNet(Base):
    def __init__(
        self, **kwds):
        super().__init__(**kwds)

        # self.bn0 = nn.BatchNorm2d(1025)

        self.encoder_block1 = EncoderBlock(
            in_channels=self.input_channels,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
        )
        self.encoder_block2 = EncoderBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
        )
        self.encoder_block3 = EncoderBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
        )
        self.encoder_block4 = EncoderBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
        )
        self.encoder_block5 = EncoderBlock(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
        )
        self.encoder_block6 = EncoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 2),
        )
        self.conv_block7 = ConvBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
        )
        self.decoder_block1 = DecoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(1, 2),
        )
        self.decoder_block2 = DecoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
        )
        self.decoder_block3 = DecoderBlock(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
        )
        self.decoder_block4 = DecoderBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
        )
        self.decoder_block5 = DecoderBlock(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
        )

        self.decoder_block6 = DecoderBlock(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
        )

        self.after_conv_block1 = ConvBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
        )
       
        self.after_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=self.output_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.reset_parameters()

    def forward(self, input_dict):
        r"""Forward data into the module.

        Args:
            input_dict: dict, e.g., {
                waveform: (batch_size, input_channels, segment_samples),
                ...,
            }

        Outputs:
            output_dict: dict, e.g., {
                'waveform': (batch_size, output_channels, segment_samples),
                ...,
            }
        """
        mixtures = input_dict['waveform']
        # (batch_size, input_channels, segment_samples)

        x = self.pack_input(mixtures)

        # x = x.transpose(1, 3)
        # x = self.bn0(x)
        # x = x.transpose(1, 3)

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / 2**5))
            * 2**5
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # x: (batch_size, input_channels * subbands_num, padded_time_steps, freq_bins)

        # Let frequency bins be evenly divided by 2, e.g., 257 -> 256
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, input_channels, T, F)
        # x: (batch_size, input_channels * subbands_num, padded_time_steps, freq_bins)

        # UNet
        x1_pool, x1 = self.encoder_block1(x)
        x2_pool, x2 = self.encoder_block2(x1_pool)  
        x3_pool, x3 = self.encoder_block3(x2_pool)  
        x4_pool, x4 = self.encoder_block4(x3_pool)  
        x5_pool, x5 = self.encoder_block5(x4_pool) 
        x6_pool, x6 = self.encoder_block6(x5_pool) 

        x_center = self.conv_block7(x6_pool)  

        x7 = self.decoder_block1(x_center, x6)
        x8 = self.decoder_block2(x7, x5)  
        x9 = self.decoder_block3(x8, x4) 
        x10 = self.decoder_block4(x9, x3)  
        x11 = self.decoder_block5(x10, x2) 
        x12 = self.decoder_block6(x11, x1)  
        x  = self.after_conv_block1(x12) 

        x = self.after_conv2(x)

        # Recover shape
        x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257.

        x = x[:, :, 0:origin_len, :]

        separated_audio = self.render_output(x, raw_waveform=mixtures)
        output_dict = {'waveform': separated_audio}
        return output_dict
