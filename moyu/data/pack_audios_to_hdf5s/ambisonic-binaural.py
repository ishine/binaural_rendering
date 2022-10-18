import argparse
import os
print(os.getcwd())
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List

import h5py
import numpy as np

from moyu.utils.calculate import float32_to_int16
from moyu.utils.audio import load_audio

def pack_audios_to_hdf5s(args) -> None:
    r"""Pack (resampled) audio files into hdf5 files to speed up loading.

    Args:
        dataset_dir: str
        split: str, 'train' | 'test'
        source_type: str
        hdf5s_dir: str, directory to write out hdf5 files
        sample_rate: int
        channels_num: int
        mono: bool

    Returns:
        None
    """

    # arguments & parameters
    audios_dir = args.audios_dir
    hdf5s_dir = args.hdf5s_dir
    sample_rate = args.sample_rate
    mono = False

    os.makedirs(hdf5s_dir, exist_ok=True)

    audio_names = sorted(os.listdir(audios_dir))

    params = []

    for audio_index, audio_name in enumerate(audio_names):

        audio_path = os.path.join(audios_dir, audio_name)

        hdf5_path = os.path.join(
            hdf5s_dir, "{}.h5".format(pathlib.Path(audio_name).stem)
        )

        source_type = "waveform"

        param = (
            audio_index,
            audio_name,
            source_type,
            audio_path,
            mono,
            sample_rate,
            hdf5_path,
        )
        params.append(param)

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=None) as pool:
        # Maximum works on the machine
        pool.map(write_single_audio_to_hdf5, params)

    print("Pack hdf5 time: {:.3f} s".format(time.time() - start_time))


def write_single_audio_to_hdf5(param: List):
    r"""Write single audio into hdf5 file."""

    (
        audio_index,
        audio_name,
        source_type,
        audio_path,
        mono,
        sample_rate,
        hdf5_path,
    ) = param

    with h5py.File(hdf5_path, "w") as hf:

        hf.attrs.create("audio_name", data=audio_name, dtype="S100")
        hf.attrs.create("sample_rate", data=sample_rate, dtype=np.int32)

        audio = load_audio(audio_path=audio_path, mono=mono, sample_rate=sample_rate)
        # audio: (channels_num, audio_samples)

        hf.create_dataset(
            name=source_type, data=float32_to_int16(audio), dtype=np.int16
        )

    print('{} Write hdf5 to {}'.format(audio_index, hdf5_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--audios_dir",
        type=str,
        required=True,
        help="Directory of the instruments solo dataset.",
    )
    parser.add_argument(
        "--hdf5s_dir",
        type=str,
        required=True,
        help="Directory to write out hdf5 files.",
    )
    parser.add_argument(
        "--sample_rate", type=int, required=True, help="Sample rate."
    )

    args = parser.parse_args()

    pack_audios_to_hdf5s(args)
