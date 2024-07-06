import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from utils.sam_utils import *
from sam import *

class EWC(BaseModel):
    NAME = 'EWC'

    def __init__(self, backbone: nn.Module, loss: nn.Module, optimizer, lr_scheduler, args, device, transform=None) -> None:
        super(EWC, self).__init__(backbone, loss, optimizer, lr_scheduler, args, device, transform)
        
        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = None
        self.gamma = 1.0

    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataloader):

        fish = torch.zeros_like(self.net.get_params())

        for j, data in enumerate(dataloader):
            if self.args.debug_mode and j > 2:
                break
            inputs, labels, _ = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad()
                output = self.net(ex.unsqueeze(0))
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * self.net.get_grads() ** 2

        fish /= (len(dataloader) * self.args.batch_size)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.gamma
            self.fish += fish

        self.checkpoint = self.net.get_params().data.clone()

    
    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor) -> float:
        
        if self.args.opt_mode != 'normal' and isinstance(self.opt, SAM):
            # first forward-backward step
            enable_running_stats(self.net)  # <- this is the important line
            outputs = self.net(inputs)
            penalty = self.penalty()
            loss = self.loss(outputs, labels) + self.args.lambd * penalty

            loss.backward()
            self.opt.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(self.net)  # <- this is the important line
            outputs = self.net(inputs)
            penalty = self.penalty()
            loss = self.loss(outputs, labels) + self.args.lambd * penalty
            loss.backward()
            self.opt.second_step(zero_grad=True)
        else:

            self.opt.zero_grad()

            outputs = self.net(inputs)
            penalty = self.penalty()
            loss = self.loss(outputs, labels) + self.args.lambd * penalty

            assert not torch.isnan(loss)
            loss.backward()
            self.opt.step()

        return loss.item()
        
    

