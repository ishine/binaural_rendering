import logging
import os
import time
from typing import List

import librosa
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from moyu.callback.utils import SaveCheckpointsCallback
from moyu.model.ambisonic_to_binaural.separator import Separator
from moyu.utils.calculate import calculate_lsd, calculate_sdr
from moyu.utils.yaml import read_yaml


class EvaluationCallback(pl.Callback):
    def __init__(
        self,
        evaluation_audios_dir: str,
        split: str,
        model: nn.Module,
        sample_rate: int,
        segment_samples: int,
        batch_size: int,
        device: str,
        evaluate_step_frequency: int,
        logger: TensorBoardLogger,
    ):
        r"""Callback to evaluate every #save_step_frequency steps.

        Args:
            evaluation_audios_dir: str, directory containing audios for evaluation
            split: ["train", "test"]
            model: nn.Module
            sample_rate: int
            segment_samples: int, length of segments to be input to a model, e.g., 44100*30
            batch_size, int, e.g., 12
            device: str, e.g., 'cuda'
            evaluate_step_frequency: int, evaluate every #save_step_frequency steps
            logger: pl.loggers.TensorBoardLogger
            statistics_container: StatisticsContainer
        """
        self.evaluation_audios_dir = evaluation_audios_dir
        self.split = split
        self.model = model
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.evaluate_step_frequency = evaluate_step_frequency
        self.logger = logger

        # separator
        self.separator = Separator(model, self.segment_samples, batch_size, device)

    @rank_zero_only
    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        r"""Evaluate losses on a few mini-batches. Losses are only used for
        observing training, and are not final F1 metrics.
        """

        global_step = trainer.global_step

        if global_step % self.evaluate_step_frequency == 0:

            ambisonic_audios_dir = os.path.join(self.evaluation_audios_dir, "ambisonic")
            binaural_audios_dir = os.path.join(self.evaluation_audios_dir, "binaural")

            ambisonic_audio_names = sorted(os.listdir(ambisonic_audios_dir))
            binaural_audio_names = sorted(os.listdir(binaural_audios_dir))
            audios_num = len(ambisonic_audio_names)

            error_str = "Directory {} does not contain audios for evaluation!".format(
                self.evaluation_audios_dir
            )
            assert len(ambisonic_audio_names) > 0, error_str

            logging.info(
                "--- Step {}, {} statistics: ---".format(global_step, self.split)
            )
            # logging.info("Total {} pieces for evaluation:".format(audios_num))

            eval_time = time.time()

            sdrs = []

            lsds = []

            for n in range(audios_num):

                ambisonic_path = os.path.join(
                    ambisonic_audios_dir, ambisonic_audio_names[n]
                )
                binaural_path = os.path.join(
                    binaural_audios_dir, binaural_audio_names[n]
                )

                # Load audio.
                ambisonic_audio, _ = librosa.core.load(
                    ambisonic_path, sr=self.sample_rate, mono=False
                )
                binaural_audio, _ = librosa.core.load(
                    binaural_path, sr=self.sample_rate, mono=False
                )

                input_dict = {'waveform': ambisonic_audio}

                # separate
                sep_wav = self.separator.separate(input_dict)
                # (channels_num, audio_length)

                sdr = calculate_sdr(ref=binaural_audio, est=sep_wav)

                lsd = calculate_lsd(binaural_audio, sep_wav)

                logging.info("{} SDR: {:.3f}, LSD: {:.3f}".format(binaural_audio_names[n], sdr, lsd))
               
                sdrs.append(sdr)
                lsds.append(lsd)

            logging.info("-----------------------------")
            logging.info('Avg SDR: {:.3f} LSD: {:.3f}'.format(np.mean(sdrs), np.mean(lsds)))

            logging.info("Evlauation time: {:.3f}".format(time.time() - eval_time))

            pl_module.log("{}/sdr".format(self.split), np.mean(sdrs), on_step=True)
            pl_module.log("{}/lsd".format(self.split), np.mean(lsds), on_step=True)


def get_callbacks(
    config_yaml: str,
    workspace: str,
    checkpoints_dir: str,
    logger: TensorBoardLogger,
    model: nn.Module,
    evaluate_device: str,
) -> List[pl.Callback]:
    """Get callbacks for pytorch_lightning.Trainer.

    Args:
        config_yaml: str
        workspace: str
        checkpoints_dir: str, directory to save checkpoints
        logger: pl.loggers.TensorBoardLogger
        model: nn.Module
        evaluate_device: str

    Return:
        callbacks: List[pl.Callback]
    """
    configs = read_yaml(config_yaml)
 
    sample_rate = configs['train']['sample_rate']
    evaluate_step_frequency = configs['train']['evaluate_step_frequency']
    save_step_frequency = configs['train']['save_step_frequency']

    test_batch_size = configs['evaluate']['batch_size']
    test_segment_samples = int(configs['evaluate']['segment_seconds'] * sample_rate)

    # save checkpoint callback
    save_checkpoints_callback = SaveCheckpointsCallback(
        model=model,
        checkpoints_dir=checkpoints_dir,
        save_step_frequency=save_step_frequency,
    )

    train_audios_dir = os.path.join(workspace, "evaluation_audio", "ambisonic-binaural", "train")
    test_audios_dir = os.path.join(workspace, "evaluation_audio", "ambisonic-binaural", "test")
    # evaluation callback
    evaluate_train_callback = EvaluationCallback(
        model=model,
        split="train",
        sample_rate=sample_rate,
        evaluation_audios_dir=train_audios_dir,
        segment_samples=test_segment_samples,
        batch_size=test_batch_size,
        device=evaluate_device,
        evaluate_step_frequency=evaluate_step_frequency,
        logger=logger,
    )

    evaluate_test_callback = EvaluationCallback(
        model=model,
        split="test",
        sample_rate=sample_rate,
        evaluation_audios_dir=test_audios_dir,
        segment_samples=test_segment_samples,
        batch_size=test_batch_size,
        device=evaluate_device,
        evaluate_step_frequency=evaluate_step_frequency,
        logger=logger,
    )

    callbacks = [
        save_checkpoints_callback,
        evaluate_train_callback,
        evaluate_test_callback,
    ]

    return callbacks
