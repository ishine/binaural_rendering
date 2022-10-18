import pickle
from typing import Dict, List
import numpy as np
import torch.distributed as dist


class SegmentSampler:
    def __init__(
        self,
        indexes_dict_path: str,
        segment_samples: int,
        mixaudio_dict: Dict,
        mixaudio_prob: float,
        batch_size: int,
        steps_per_epoch: int,
        random_seed=1234,
    ):
        r"""Sample training indexes of sources.

        Args:
            indexes_path: str, path of indexes dict
            input_source_types: list of str, e.g., ['vocals', 'accompaniment']
            target_source_types: list of str, e.g., ['vocals']
            segment_samplers: int
            remixing_sources: bool, remix different sources from different songs.
            mixaudio_dict, dict, mix-audio data augmentation parameters,
                e.g., {'voclas': 2, 'accompaniment': 2}
            mixaudio_prob, float, the ratio of mixed audios
            batch_size: int
            steps_per_epoch: int, #steps_per_epoch is called an `epoch`
            random_seed: int
        """
        self.segment_samples = segment_samples
        self.mixaudio_dict = mixaudio_dict
        self.mixaudio_prob = mixaudio_prob
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.meta_dict = pickle.load(open(indexes_dict_path, "rb"))
        
        self.source_types = ["ambisonic", "binaural"]

        self.pointers_dict = {source_type: 0 for source_type in self.source_types}

        self.indexes_dict = {
            source_type: np.arange(len(self.meta_dict[source_type]))
            for source_type in self.source_types
        }

        self.random_state_dict = {}

        for source_type in self.source_types:
            source_random_seed = random_seed

            self.random_state_dict[source_type] = np.random.RandomState(
                source_random_seed
            )

            self.random_state_dict[source_type].shuffle(self.indexes_dict[source_type])
            # E.g., [198036, 196736, ..., 103408]

            print("{}: {}".format(source_type, len(self.indexes_dict[source_type])))

    def __iter__(self) -> List[Dict]:
        r"""Yield a batch of meta info.

        Returns:
            batch_meta_list: (batch_size,) e.g., when mix-audio is 2, looks like [
                {'vocals': [
                    {'hdf5_path': 'songA.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 13406400, 'end_sample': 13538700},
                    {'hdf5_path': 'songB.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 4440870, 'end_sample': 4573170}]
                'accompaniment': [
                    {'hdf5_path': 'songE.h5', 'key_in_hdf5': 'accompaniment', 'begin_sample': 14579460, 'end_sample': 14711760},
                    {'hdf5_path': 'songF.h5', 'key_in_hdf5': 'accompaniment', 'begin_sample': 3995460, 'end_sample': 4127760}]
                },
                ...
            ]
        """
        batch_size = self.batch_size

        while True:
            batch_meta_dict = {source_type: [] for source_type in self.source_types}
            if_mixaudio = np.random.uniform() < self.mixaudio_prob 
            
            for source_type in self.source_types:
                # Loop until get a mini-batch.
                while len(batch_meta_dict[source_type]) != batch_size:
                    if source_type in self.mixaudio_dict.keys() and if_mixaudio:
                        mix_audios_num = self.mixaudio_dict[source_type]
                    else:
                        mix_audios_num = 1

                    largest_index = len(self.indexes_dict[source_type]) - mix_audios_num
                    # E.g., 225750 = 225752 - 2

                    if self.pointers_dict[source_type] > largest_index:
                        # Reset pointer, and shuffle indexes.
                        self.pointers_dict[source_type] = 0
                        self.random_state_dict[source_type].shuffle(
                            self.indexes_dict[source_type]
                        )

                    source_metas = []

                    for _ in range(mix_audios_num):

                        pointer = self.pointers_dict[source_type]
                        # E.g., 1

                        index = self.indexes_dict[source_type][pointer]
                        # E.g., 12231

                        self.pointers_dict[source_type] += 1

                        source_meta = self.meta_dict[source_type][index]
                        # E.g., {
                        #     'hdf5_path': 'xx/song_A.h5',
                        #     'key_in_hdf5': 'vocals',
                        #     'begin_sample': 13406400,
                        # }
                        source_metas.append(source_meta)

                    batch_meta_dict[source_type].append(source_metas)

            # When mix-audio is 2, batch_meta_dict looks like: {
            #     'vocals': [
            #         [{'hdf5_path': 'songA.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 13406400, 'end_sample': 13538700},
            #          {'hdf5_path': 'songB.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 4440870, 'end_sample': 4573170}
            #         ],
            #         ... (batch_size)
            #     ]
            #     'accompaniment': [
            #         [{'hdf5_path': 'songG.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 24232950, 'end_sample': 24365250},
            #          {'hdf5_path': 'songH.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 1569960, 'end_sample': 1702260}
            #         ],
            #         ... (batch_size)
            #     ]
            # }

            batch_meta_list = [
                {
                    source_type: batch_meta_dict[source_type][i]
                    for source_type in self.source_types
                }
                for i in range(batch_size)
            ]

            yield batch_meta_list

    def __len__(self) -> int:
        return self.steps_per_epoch

    def state_dict(self) -> Dict:
        state = {'pointers_dict': self.pointers_dict, 'indexes_dict': self.indexes_dict}
        return state

    def load_state_dict(self, state) -> None:
        self.pointers_dict = state['pointers_dict']
        self.indexes_dict = state['indexes_dict']


class DistributedSamplerWrapper:
    def __init__(self, sampler):
        r"""Distributed wrapper of sampler."""
        self.sampler = sampler

    def __iter__(self) -> List[Dict]:
        num_replicas = dist.get_world_size()  # number of GPUs.
        rank = dist.get_rank()  # rank of current GPU

        for batch_meta_list in self.sampler:
            # Yield a subset of batch_meta_list on one GPU.
            yield batch_meta_list[rank::num_replicas]

    def __len__(self) -> int:
        return len(self.sampler)
