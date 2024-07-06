import torch
import torch.nn as nn

import copy

from models.base_model import BaseModel
from sam import SAM
from utils.sam_utils import *

class Naive(BaseModel):
    NAME = 'Naive'
    def __init__(self, backbone: nn.Module, loss: nn.Module, optimizer, lr_scheduler, args, device, transform=None) -> None:
        super(Naive, self).__init__(backbone, loss, optimizer, lr_scheduler, args, device, transform)

    
    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, is_poisoned: torch.Tensor) -> float:
        if self.args.opt_mode != 'normal' and isinstance(self.opt, SAM):
            # first forward-backward step
            enable_running_stats(self.net)  # <- this is the important line
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels) 

            loss.backward()
            self.opt.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(self.net)  # <- this is the important line
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.opt.second_step(zero_grad=True)
        
        else:
            self.opt.zero_grad()
            outputs = self.net(inputs)

            loss = self.loss(outputs, labels)
            assert not torch.isnan(loss)
            loss.backward()
            self.opt.step()

        return loss.item()

