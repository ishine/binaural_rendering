import logging
import os
from typing import Tuple

import pytorch_lightning as pl


def create_logging(logs_dir: str, filemode: str) -> logging.Logger:
    r"""Create logging to write out log files.

    Args:
        logs_dir, str, directory to write out logs
        filemode: str, e.g., "w"

    Returns:
        logging
    """
    os.makedirs(logs_dir, exist_ok=True)
    i1 = 0

    while os.path.isfile(os.path.join(logs_dir, "{:04d}.log".format(i1))):
        i1 += 1

    log_path = os.path.join(logs_dir, "{:04d}.log".format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=log_path,
        filemode=filemode,
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logger = logging.getLogger("")
    logger.addHandler(console)

    return logger


def get_dirs(
    workspace,
    config_yaml: str,
) -> Tuple[str, logging.Logger, pl.loggers.TensorBoardLogger]:
    r"""Get directory paths.

    Args:
        workspace: str
        config_yaml: str

    Returns:
        checkpoints_dir: str
        logs_dir: str
        logger: pl.loggers.TensorBoardLogger
    """
    from moyu.utils.yaml import read_yaml
    configs: dict = read_yaml(config_yaml)
    config_name = configs["name"]
    
    model_type = configs['train']['model_type'].lower()
    # save checkpoints dir
    checkpoints_dir = os.path.join(
        workspace,
        "checkpoint",
        "{}".format(config_name),
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    # logs dir
    logs_dir = os.path.join(
        workspace,
        "log",
        "{}".format(config_name),
    )   
    logger = create_logging(logs_dir, "w+")

    # tensorboard logs dir
    tb_logger_dir = os.path.join(workspace, "tensorboard_log")
    os.makedirs(tb_logger_dir, exist_ok=True)

    experiment_name = os.path.join(model_type, config_name)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=tb_logger_dir, name=experiment_name)
    return checkpoints_dir, logger, tb_logger