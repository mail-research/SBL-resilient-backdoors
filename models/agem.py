import torch
import torch.nn as nn
import numpy as np

import copy

from models.base_model import BaseModel
from sam import SAM
from utils.sam_utils import *


def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def store_grad( params, grads, grad_dims):
        """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        """
        # store the gradients
        grads.fill_(0.0)
        count = 0
        for param in params():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(grad_dims[:count])
                end = np.sum(grad_dims[:count + 1])
                grads[begin: end].copy_(param.grad.data.view(-1))
            count += 1


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger

class AGem(BaseModel):
    def __init__(self, backbone: nn.Module, loss: nn.Module, optimizer, lr_scheduler, args, device, transform=None) -> None:
        super(AGem, self).__init__(backbone, loss, optimizer, lr_scheduler, args, device, transform)

        self.buffer = {
            'images': [],
            'labels': []
        }
        self.buffer_size = args.buffer_size

        self.grad_dims = []
        for param in self.net.parameters():
            self.grad_dims.append(param.data.numel())

        self.device = device
        
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)


    def observe(self, inputs, labels, not_aug_inputs):
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
            self.zero_grad()
            p = self.net.forward(inputs)
            loss = self.loss(p, labels)
            loss.backward()

            if len(self.buffer['labels']) > 0:
                store_grad(self.parameters, self.grad_xy, self.grad_dims)
                sample_size = self.args.batch_size if self.args.batch_size < self.buffer_size else self.buffer_size
                buf_inputs, buf_labels = self.get_buffer_data(size=sample_size)
                self.net.zero_grad()
                buf_outputs = self.net.forward(buf_inputs)
                penalty = self.loss(buf_outputs, buf_labels)
                penalty.backward()
                store_grad(self.parameters, self.grad_er, self.grad_dims)

                dot_prod = torch.dot(self.grad_xy, self.grad_er)
                if dot_prod.item() < 0:
                    g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                    overwrite_grad(self.parameters, g_tilde, self.grad_dims)
                else:
                    overwrite_grad(self.parameters, self.grad_xy, self.grad_dims)

            self.opt.step()

        return loss.item()


    def get_buffer_data(self, size=32):
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :return:
        """
        ids = np.random.choice(range(self.buffer_size), size, replace=False)
        return self.buffer['images'][ids], self.buffer['labels'][ids] 


    def end_task(self, dataloader):

        for images, labels, _ in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            if len(self.buffer) + len(images) <= self.buffer_size:
                self.buffer['images'].append(images)
                self.buffer['labels'].append(labels)

            else:
                n_data = self.buffer_size - len(self.buffer['labels'])
                self.buffer['images'].append(images[:n_data])
                self.buffer['labels'].append(labels[:n_data])
        
        # Will be error if the number of using end_task() > 1 since model tensor has no append
        self.buffer['images'] = torch.concat(self.buffer['images'])
        self.buffer['labels'] = torch.concat(self.buffer['labels'])

        # self.buffer_loader = torch.data.utils.DataLoader(list(zip(self.buffer['images'], self.buffer['labels'])), bu)

