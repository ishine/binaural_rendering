import math
from typing import Callable
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT


def l1(input: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    r"""L1 loss.

    Args:
        output: torch.Tensor
        target: torch.Tensor

    Returns:
        loss: torch.float
    """
    return torch.mean(torch.abs(input - target))


def l2(input: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    r"""L2 loss.

    Args:
        output: torch.Tensor
        target: torch.Tensor

    Returns:
        loss: torch.float
    """
    return torch.mean((input - target)**2)
