from typing import List

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn as nn


def get_callbacks(
    dataset_name: str,
    config_yaml: str,
    workspace: str,
    checkpoints_dir: str,
    logger: TensorBoardLogger,
    model: nn.Module,
    evaluate_device: str,
    ) -> List[pl.Callback]:
    r"""Get callbacks of a task and config yaml file.

    Args:
        dataset_name: str
        config_yaml: str
        dataset_dir: str
        workspace: str, containing useful files such as audios for evaluation
        checkpoints_dir: str, directory to save checkpoints
        statistics_dir: str, directory to save statistics
        logger: pl.loggers.TensorBoardLogger
        model: nn.Module
        evaluate_device: str

    Return:
        callbacks: List[pl.Callback]
    """
    
    if dataset_name == "ambisonic-binaural":
        from moyu.callback.ambisonic_binaural import get_ambisonic_binaural_callbacks

        return get_ambisonic_binaural_callbacks(
            config_yaml=config_yaml,
            workspace=workspace,
            checkpoints_dir=checkpoints_dir,
            logger=logger,
            model=model,
            evaluate_device=evaluate_device,
        )

    else:
        raise NotImplementedError
