from typing import Any, Callable, Dict

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class LitModule(pl.LightningModule):
    def __init__(
        self,
        batch_data_preprocessor,
        model: nn.Module,
        loss_function: Callable,
        optimizer_type: str,
        learning_rate: float,
        lr_lambda: Callable,
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            batch_data_preprocessor: object, used for preparing inputs and
                targets for training. E.g., BasicBatchDataPreprocessor is used
                for preparing data in dictionary into tensor.
            model: nn.Module
            loss_function: function
            learning_rate: float
            lr_lambda: function
        """
        super().__init__()

        self.batch_data_preprocessor = batch_data_preprocessor
        self.model = model
        self.optimizer_type = optimizer_type
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.lr_lambda = lr_lambda

    def training_step(self, batch_data_dict: Dict, batch_idx: int) -> Dict:
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: e.g. {
                'vocals': (batch_size, channels_num, segment_samples),
                'accompaniment': (batch_size, channels_num, segment_samples),
                'mixture': (batch_size, channels_num, segment_samples)
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        input_dict, target_dict = self.batch_data_preprocessor(batch_data_dict)
        # input_dict: {
        #     'waveform': (batch_size, channels_num, segment_samples),
        #     (if_exist) 'condition': (batch_size, channels_num),
        # }
        # target_dict: {
        #     'waveform': (batch_size, target_sources_num * channels_num, segment_samples),
        # }

        # Forward.
        self.model.train()

        output_dict = self.model(input_dict)
        # output_dict: {
        #     'waveform': (batch_size, target_sources_num * channels_num, segment_samples),
        # }

        outputs = output_dict['waveform']
        # outputs:, e.g, (batch_size, target_sources_num * channels_num, segment_samples)

        # Calculate loss.
        # if isinstance(self.loss_function, nn.Module):
        #     self.loss_function = self.loss_function.to(self.device)

        loss = self.loss_function(
            input=outputs,
            target=target_dict['waveform'],
        )

        self.log("loss", loss, logger=True, on_step=True)
        return loss

    def configure_optimizers(self) -> Any:
        r"""Configure optimizer."""

        if self.optimizer_type == "Adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0,
                amsgrad=True,
            )

        elif self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0,
                amsgrad=True,
            )

        else:
            raise NotImplementedError

        scheduler = {
            'scheduler': LambdaLR(optimizer, self.lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]