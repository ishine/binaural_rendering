import argparse
import os
import pickle
from pathlib import Path

import h5py

from moyu.utils.yaml import read_yaml


def create_indexes(args):
    r"""Create and write out training indexes into disk. The indexes may contain information from multiple datasets. 
    During training, training indexes will be shuffled and iterated for selecting segments to be mixed. 
    E.g., the training indexes_dict looks like: {
        'source_type1': [
            {'hdf5_path': '.../songA.h5', 'key_in_hdf5': 'source_type1', 'begin_sample': 0}
            {'hdf5_path': '.../songB.h5', 'key_in_hdf5': 'source_type1', 'begin_sample': samplerate * duration}
            ...
        ]
        'source_type2': [
            {'hdf5_path': '.../songA.h5', 'key_in_hdf5': 'source_type2, 'begin_sample': 0}
            {'hdf5_path': '.../songB.h5', 'key_in_hdf5': 'source_type2', 'begin_sample': samplerate * duration}
            ...
        ]
    }
    """

    # Arugments & parameters
    workspace = Path(args.workspace)
    config_yaml = Path(args.config_yaml)

    # Read config file.
    configs = read_yaml(config_yaml)

    sample_rate = configs["sample_rate"]
    segment_samples = int(configs["segment_seconds"] * sample_rate)

    # Only create indexes for training, because evalution is on entire pieces.
    split = "train"

    # Path to write out index.
    indexes_path: Path = workspace/configs[split]["indexes"]
    if indexes_path.exists():
        print("{} already exists!".format(indexes_path))
        indexes_path.unlink()
    indexes_path.parent.mkdir(parents=True, exist_ok=True)
    # Only create indexes for training, because evalution is on entire pieces.
    split = "train"

    source_types = configs[split]["source_types"].keys()
    # E.g., ['ambisonic', 'binaural']

    indexes_dict = {source_type: [] for source_type in source_types}
    # E.g., indexes_dict will looks like: {
    #     'binaural': [
    #         {'hdf5_path': '.../songA.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 0}
    #         {'hdf5_path': '.../songB.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 4410}
    #         ...
    #     ]
    #     'ambisonic': [
    #         {'hdf5_path': '.../songA.h5', 'key_in_hdf5': 'accompaniment', 'begin_sample': 0}
    #         {'hdf5_path': '.../songB.h5', 'key_in_hdf5': 'accompaniment', 'begin_sample': 4410}
    #         ...
    #     ]
    # }

    # Get training indexes for each source type.
    for source_type in source_types:
        # E.g., ['vocals', 'bass', ...]

        print("--- {} ---".format(source_type))

        hdf5_config = configs[split]["source_types"][source_type]
        
        hdf5_dir: Path = workspace/hdf5_config["hdf5s_directory"]

        hop_samples = int(hdf5_config["hop_seconds"] * sample_rate)

        key_in_hdf5 = hdf5_config["key_in_hdf5"]
        # E.g., 'vocals'

        hdf5_paths = sorted(hdf5_dir.iterdir())
        print("Hdf5 files num: {}".format(len(hdf5_paths)))

        count = 0

        # Traverse all packed hdf5 files of a dataset.
        for n, hdf5_path in enumerate(hdf5_paths):
            # print(n, hdf5_path.stem)

            with h5py.File(hdf5_path, "r") as hf:
                # print(hf[key_in_hdf5].shape[-1])
                bgn_sample = 0
                while bgn_sample + segment_samples < hf[key_in_hdf5].shape[-1]:
                    meta = {
                        'hdf5_path': hdf5_path,
                        'key_in_hdf5': key_in_hdf5,
                        'begin_sample': bgn_sample,
                    }
                    indexes_dict[source_type].append(meta)

                    bgn_sample += hop_samples
                    count += 1
                # If the audio length is shorter than the segment length, discard the entire audio.

        print("{} indexes: {}".format(source_type, count))

    pickle.dump(indexes_dict, open(indexes_path, "wb"))
    print("Write index dict to {}".format(indexes_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser.add_argument(
        "--config_yaml", type=str, required=True, help="User defined config file."
    )

    # Parse arguments.
    args = parser.parse_args()

    # Create training indexes.
    create_indexes(args)
