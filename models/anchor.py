import torch
import torch.nn as nn

import copy

from models.base_model import BaseModel
from utils.sam_utils import *
from sam import SAM


class Anchor(BaseModel):
    NAME = 'ANCHORING'
    def __init__(self, backbone: nn.Module, loss: nn.Module, optimizer, lr_scheduler, args, device, transform=None) -> None:
        super(Anchor, self).__init__(backbone, loss, optimizer, lr_scheduler, args, device, transform)
        self.anchor_model = None
    
    def end_task(self, dataloader):
        self.anchor_model = copy.deepcopy(self.net)
        self.anchor_model.eval()

    def penalty(self, inputs, outputs, is_poisoned):
        if self.anchor_model is None:
            return torch.tensor(0.0).to(self.device)
        else:
            outputs = torch.softmax(outputs, dim=1)

            anchor_outputs = self.anchor_model(inputs)
            anchor_outputs = torch.softmax(anchor_outputs, dim=1)
            # print(f'anchor_outputs shape: {anchor_outputs.shape}')
            anchor_loss = torch.sum((anchor_outputs - outputs)**2, dim=1) # return sum of all classes for each sample
            # print(f'anchor_loss prev shape: {anchor_loss.shape}')
            anchor_loss = torch.mean(anchor_loss * (1-is_poisoned)) # only consider clean sample
            # print(f'anchor_loss after shape: {anchor_loss.shape}')
            # print(f'anchor loss is {anchor_loss.item()}')
            return anchor_loss

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, is_poisoned: torch.Tensor) -> float:
        if self.args.opt_mode == 'sam' and isinstance(self.opt, SAM):
            # first forward-backward step
            enable_running_stats(self.net)  # <- this is the important line
            outputs = self.net(inputs)
            penalty = self.penalty(inputs, outputs, is_poisoned)
            loss = self.loss(outputs, labels) + self.args.lambd * penalty

            loss.backward()
            self.opt.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(self.net)  # <- this is the important line
            outputs = self.net(inputs)
            penalty = self.penalty(inputs, outputs, is_poisoned)
            loss = self.loss(outputs, labels) + self.args.lambd * penalty
            loss.backward()
            self.opt.second_step(zero_grad=True)
        else:

            self.opt.zero_grad()

            outputs = self.net(inputs)
            penalty = self.penalty(inputs, outputs, is_poisoned)
            loss = self.loss(outputs, labels) + self.args.lambd * penalty

            assert not torch.isnan(loss)
            loss.backward()
            self.opt.step()


        # self.opt.zero_grad()
        # outputs = self.net(inputs)

        # self.penalty_loss = self.penalty(inputs, outputs, is_poisoned)

        # loss = self.loss(outputs, labels) + self.args.lambd * self.penalty_loss
        # assert not torch.isnan(loss)
        # loss.backward()
        # self.opt.step()

        return loss.item()
