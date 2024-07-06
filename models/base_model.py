import sys

import torch
import torch.nn as nn
from torch.optim import SGD


class BaseModel(nn.Module):
    """
    Base model.
    """
    NAME: str

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 optimizer, lr_scheduler, args, device, transform=None) -> None:
        super(BaseModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.opt = optimizer
        self.lr_scheduler = lr_scheduler
        self.transform = transform
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)


    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        raise NotImplementedError

