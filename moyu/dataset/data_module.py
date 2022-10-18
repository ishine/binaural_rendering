from typing import Dict, List, Optional

import h5py
import librosa
import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule

from moyu.dataset.augmentor import Augmentor
from moyu.dataset.sampler import DistributedSamplerWrapper
from moyu.utils.calculate import int16_to_float32

def collate_fn(batch: List[Dict]) -> Dict:
    data_dict = {}

    for key in batch[0].keys():
        data_dict[key] = torch.Tensor(
            np.array([data_dict[key] for data_dict in batch])
        )

    return data_dict

class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset,
        sampler,
        num_workers: int,
        distributed: bool,
    ):
        r"""Data module.

        Args:
            dataset: Dataset object
            sampler: Sampler object
            num_workers: int
            distributed: bool
        """
        super().__init__()
        self.dataset = dataset
        self.sampler = sampler
        self.num_workers = num_workers
        self.distributed = distributed

    # @override
    def setup(self, stage: Optional[str] = None) -> None:
        r"""called on every device."""

        # SegmentSampler is used for sampling segment indexes for training.
        # On multiple devices, each SegmentSampler samples a part of mini-batch
        # data.

        if self.distributed:
            self.sampler = DistributedSamplerWrapper(self.sampler)

    def train_dataloader(self) -> DataLoader:
        r"""Get train loader."""
        train_loader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

        return train_loader


class Dataset:
    def __init__(
        self,
        augmentor: Augmentor,
        segment_samples: int,
    ):
        r"""Used for getting data according to a meta.

        Args:
            input_source_types: list of str, e.g., ['vocals', 'accompaniment']
            target_source_types: list of str, e.g., ['vocals']
            input_channels: int
            augmentor: Augmentor
            segment_samples: int
        """
        self.source_types = ["ambisonic", "binaural"]
        self.augmentor = augmentor
        self.segment_samples = segment_samples

    def __getitem__(self, meta: Dict) -> Dict:
        r"""Return data according to a meta. E.g., an input meta looks like: {
            'vocals': [['song_A.h5', 6332760, 6465060], ['song_B.h5', 198450, 330750]],
            'accompaniment': [['song_C.h5', 24232920, 24365250], ['song_D.h5', 1569960, 1702260]]}.
        }

        Then, vocals segments of song_A and song_B will be mixed (mix-audio augmentation).
        Accompaniment segments of song_C and song_B will be mixed (mix-audio augmentation).
        Finally, mixture is created by summing vocals and accompaniment.

        Args:
            meta: dict, e.g., {
                'vocals': [['song_A.h5', 6332760, 6465060], ['song_B.h5', 198450, 330750]],
                'accompaniment': [['song_C.h5', 24232920, 24365250], ['song_D.h5', 1569960, 1702260]]}
            }

        Returns:
            data_dict: dict, e.g., {
                'vocals': (channels, segments_num),
                'accompaniment': (channels, segments_num),
            }
        """
        data_dict = {}
        
        for source_type in self.source_types:
            waveforms = []  # Audio segments to be mix-audio augmented.
            for m in meta[source_type]:
                # E.g., {
                #     'hdf5_path': '.../song_A.h5',
                #     'key_in_hdf5': 'vocals',
                #     'begin_sample': '13406400',
                # }

                hdf5_path = m['hdf5_path']
                key_in_hdf5 = m['key_in_hdf5']
                bgn_sample = m['begin_sample']
                end_sample = bgn_sample + self.segment_samples

                with h5py.File(hdf5_path, 'r') as hf:
                    waveform = int16_to_float32(
                        hf[key_in_hdf5][:, bgn_sample:end_sample]
                    )

                if self.augmentor:
                    waveform = self.augmentor(waveform, source_type)
                
                waveforms.append(waveform)
                # E.g., waveforms: [(input_channels, audio_samples), (input_channels, audio_samples)]
            # mix-audio augmentation
            data_dict[source_type] = np.sum(waveforms, axis=0)
        return data_dict
