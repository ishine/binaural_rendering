import argparse
import logging
import os
from functools import partial
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers
import torch

from moyu.callback.ambisonic_to_binaural import get_callbacks
from moyu.dataset.augmentor import Augmentor
from moyu.dataset.data_module import DataModule, Dataset
from moyu.dataset.preprocessor import AmbisonicBinauralPreprocessor
from moyu.dataset.sampler import SegmentSampler
from moyu.loss import get_loss_function
from moyu.model.ambisonic_to_binaural import get_model_class
from moyu.model.ambisonic_to_binaural.pl_module import LitModule
from moyu.optimizer.scheduler import get_warmup_lr_lambda
from moyu.utils.logging import get_dirs
from moyu.utils.yaml import read_yaml


def get_data_module(
    workspace: str,
    config_yaml: str,
    num_workers: int,
    distributed: bool,
) -> DataModule:
    r"""Create data_module. Here is an example to fetch a mini-batch:
    ```
        data_module.setup()
        for batch_data_dict in data_module.train_dataloader():
            print(batch_data_dict.keys())
            break
    ```

    Args:
        workspace: str
        config_yaml: str
        num_workers: int, e.g., 0 for non-parallel and 8 for using cpu cores
            for preparing data in parallel
        distributed: bool

    Returns:
        data_module: DataModule
    """
    config = read_yaml(config_yaml)
    indexes_dict_path = os.path.join(workspace, config['train']['indexes_dict_path'])
    sample_rate = config['train']['sample_rate']
    segment_seconds = config['train']['segment_seconds']
    augmentations = config['train']['augmentations']
    mixaudio_prob = config['train'].get('mixaudio_prob', 1.)
    batch_size = config['train']['batch_size'] 
    steps_per_epoch = config['train']['steps_per_epoch']

    segment_samples = int(segment_seconds * sample_rate)

    # sampler
    train_sampler = SegmentSampler(
        indexes_dict_path=indexes_dict_path,
        segment_samples=segment_samples,
        mixaudio_dict=augmentations['mixaudio'],
        mixaudio_prob=mixaudio_prob,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )

    # augmentor
    augmentor = Augmentor(augmentations=augmentations)

    # dataset
    train_dataset = Dataset(
        augmentor=augmentor,
        segment_samples=segment_samples,
    )

    # data module
    data_module = DataModule(
        dataset=train_dataset,
        sampler=train_sampler,
        num_workers=num_workers,
        distributed=distributed,
    )

    return data_module


def train(args) -> None:
    r"""Train & evaluate and save checkpoints.

    Args:
        workspace: str, directory of workspace
        gpus: int
        config_yaml: str, path of config file for training
    """

    # arugments & parameters
    workspace = args.workspace
    gpus = args.gpus
    num_workers = 8
    distributed = gpus > 1 

    config_yaml = args.config_yaml

    evaluate_device = "cuda" if gpus > 0 else "cpu"

    # Read config file.
    config: dict = read_yaml(config_yaml)

    model_type = config['train']['model_type']
    loss_type = config['train']['loss_type']
    optimizer_type = config['train']['optimizer_type']
    learning_rate = float(config['train']['learning_rate'])
    precision = config['train']['precision']
    max_steps = config['train']['max_steps']
    warm_up_steps = config['train']['warm_up_steps']
    reduce_lr_steps = config['train']['reduce_lr_steps']
    resume_checkpoint_path = config['train']['resume_checkpoint_path']

    # paths
    checkpoints_dir, logger, tb_logger = get_dirs(workspace, config_yaml)

    # training data module
    data_module = get_data_module(workspace, config_yaml, num_workers, distributed)

    # batch data preprocessor
    batch_data_preprocessor = AmbisonicBinauralPreprocessor()

    # model
    model_kwds = {}
    if "model" in config and "kwds" in config["model"]:
        model_kwds = config["model"]["kwds"]

    Model = get_model_class(model_type=model_type)
    model = Model(**model_kwds)
    
    if resume_checkpoint_path:
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        logging.info(
            "Load pretrained checkpoint from {}".format(resume_checkpoint_path)
        )

    # loss function
    loss_function = get_loss_function(loss_type=loss_type)

    # callbacks
    callbacks = get_callbacks(
        config_yaml=config_yaml,
        workspace=workspace,
        checkpoints_dir=checkpoints_dir,
        logger=tb_logger, # Choose logger
        model=model,
        evaluate_device=evaluate_device,
    )
    # callbacks = []

    # learning rate reduce function
    lr_lambda = partial(
        get_warmup_lr_lambda, warm_up_steps=warm_up_steps, reduce_lr_steps=reduce_lr_steps
    )

    # pytorch-lightning model
    pl_model = LitModule(
        batch_data_preprocessor=batch_data_preprocessor,
        model=model,
        optimizer_type=optimizer_type,
        loss_function=loss_function,
        learning_rate=learning_rate,
        lr_lambda=lr_lambda,
    )

    # trainer
    trainer = pl.Trainer(
        logger=tb_logger,
        gpus=gpus,
        callbacks=callbacks,
        max_steps=max_steps,
        precision=precision,
    )

    # Fit, evaluate, and save checkpoints.
    trainer.fit(pl_model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )

    args = parser.parse_args()

    train(args)
