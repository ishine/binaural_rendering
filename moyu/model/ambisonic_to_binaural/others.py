from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from moyu.model.ambisonic_to_binaural.base import Base


class DNN(Base):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        # self.bn0 = nn.BatchNorm2d(1024)

        self.fc0 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # self.fc2 = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.LeakyReLU(0.01, inplace=True)
        # )

        # self.fc_m1 = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.LeakyReLU(0.01, inplace=True)
        # )

        # self.fc_m2 = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.LeakyReLU(0.01, inplace=True)
        # )

        # self.fc_m3 = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.LeakyReLU(0.01, inplace=True)
        # )


        # input: complex + relative phase
        self.fc3 = nn.Sequential(
            nn.Linear(20, 128),
            nn.LeakyReLU(0.01)
        )

        # self.fc4 = nn.Sequential(
        #     nn.Linear(64, 64),s
        #     nn.LeakyReLU(0.01)
        # )

        # output: mask + relative phase
        self.fc5 = nn.Sequential(
            nn.Linear(128, 6)
        )

        self.reset_parameters()

    def forward(self, input_dict):
        mixtures = input_dict['waveform']
        # (batch_size, input_channels, segment_samples)

        x = self.pack_input(mixtures)

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / 2**5))
            * 2**5
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
    
        # Let frequency bins be evenly divided by 2, e.g., 257 -> 256
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, input_channels, T, F)
        # x: (batch_size, input_channels, padded_time_steps, freq_bins)

        x = self.fc0(x)
        x = self.fc1(x)
        # x = self.fc2(x)

        # x = self.fc_m3(self.fc_m2(self.fc_m1(x)))
        # N, 16, 320, 1024
        x = x.permute(0, 2, 3, 1)
        
        x = self.fc3(x)
        # x = self.fc4(x)
        x = self.fc5(x)
        # N, 320, 1024, 6

        x = x.permute(0, 3, 1, 2)
        # (batch_size, target_sources_num * output_channels * self.K * subbands_num, T, F')

        # Recover shape
        x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257.

        x = x[:, :, 0:origin_len, :]
        # (batch_size, target_sources_num * output_channels * self.K * subbands_num, T, F')

        separated_audio = self.render_output(x, raw_waveform=mixtures)
        output_dict = {'waveform': separated_audio}
        return output_dict


class GRU(Base):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        # self.bn0 = nn.BatchNorm2d(1024)

        # input: complex + relative phase 
        self.gru0 =  nn.GRU(
            1024 * self.input_channels, 1024 // 2,
            batch_first=True,
            # dropout=0.1,
            num_layers=3,
            bidirectional=True,
        )

        # self.act0 = nn.LeakyReLU(0.01)
        
        # self.gru1 =  nn.GRU(
        #     1024, 1024 // 2,
        #     batch_first=True,
        #     # dropout=0.1,
        #     bidirectional=True,
        # )

        # self.act1 = nn.LeakyReLU(0.01)

        # self.gru2 = nn.GRU(
        #     1024, 1024 // 2,
        #     batch_first=True,
        #     # dropout=0.1,
        #     bidirectional=True,
        # )

        # self.act2 = nn.LeakyReLU(0.01)
        
        # output: mask + relative phase
        self.fc3 = nn.Linear(
            1024, self.output_channels * 1024 
        )

        self.reset_parameters()

    def forward(self, input_dict):
        mixtures = input_dict['waveform']
        # (batch_size, input_channels, segment_samples)

        x = self.pack_input(mixtures)

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

        x = x.permute(0, 2, 1, 3)
        x = x.flatten(2)

        x, _ = self.gru0(x)
        # x = self.act0(x)
        # x, _ = self.gru1(x)
        # x = self.act1(x)
        # x, _ = self.gru2(x)
        # x = self.act2(x)

        x = self.fc3(x)
        
        x = x.reshape(-1, 320, self.output_channels, 1024)
        x = x.permute(0, 2, 1, 3)
        # (batch_size, target_sources_num * output_channels * self.K * subbands_num, T, F')

        # Recover shape
        x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257.

        x = x[:, :, 0:origin_len, :]
        # (batch_size, target_sources_num * output_channels * self.K * subbands_num, T, F')

        separated_audio = self.render_output(x, raw_waveform=mixtures)
        output_dict = {'waveform': separated_audio}
        return output_dict
