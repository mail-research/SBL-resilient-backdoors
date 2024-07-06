# Based on code taken from https://github.com/facebookresearch/open_lth

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F



class Block(nn.Module):
    """A ResNet block."""

    def __init__(self, f_in: int, f_out: int, downsample=False):
        super(Block, self).__init__()

        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(f_out)
        self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(f_out)
        self.ind = None

        # No parameters for shortcut connections.
        if downsample or f_in != f_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(f_out)
            )
        else:
            # self.shortcut = layers.Identity2d(f_in)
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        if self.ind is not None:
            out += self.shortcut(x)[:,self.ind,:,:]
        else:
            out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""
    
    def __init__(self, plan, num_classes, dense_classifier):
        super(ResNet, self).__init__()

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):  # plan = [(16,20), (32,20), (64,20)]
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        self.fc = nn.Linear(plan[-1][0], num_classes)
        if dense_classifier:
            self.fc = nn.Linear(plan[-1][0], num_classes)

        self._initialize_weights()


    def forward(self, x, returnt='out'):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        if returnt == 'features':
            return out
        out = self.fc(out)
        return out
    
    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, returnt='features')

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        return torch.cat(self.get_grads_list())

    def get_grads_list(self):
        """
        Returns a list containing the gradients (a tensor for each layer).
        :return: gradients list
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return grads

    def _initialize_weights(self, init_method='kaiming_normal'):

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Linear, nn.Conv2d)):
                if init_method == 'kaiming_normal':   
                    nn.init.kaiming_normal_(m.weight)
                elif init_method == 'normal':
                    nn.init.normal_(m.weight)
                elif init_method == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif init_method == 'uniform':
                    nn.init.uniform_(m.weight)
                elif init_method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight)
                elif init_method == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def _plan(D, W):
    """The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.

    The ResNet is structured as an initial convolutional layer followed by three "segments"
    and a linear output layer. Each segment consists of D blocks. Each block is two
    convolutional layers surrounded by a residual connection. Each layer in the first segment
    has W filters, each layer in the second segment has 32W filters, and each layer in the
    third segment has 64W filters.

    The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
    N is the total number of layers in the network: 2 + 6D.
    The default value of W is 16 if it isn't provided.

    For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
    linear layer, there are 18 convolutional layers in the blocks. That means there are nine
    blocks, meaning there are three blocks per segment. Hence, D = 3.
    The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
    """
    if (D - 2) % 3 != 0:
        raise ValueError('Invalid ResNet depth: {}'.format(D))
    D = (D - 2) // 6
    plan = [(W, D), (2*W, D), (4*W, D)]

    return plan

def _resnet(arch, plan, num_classes, dense_classifier, pretrained):
    model = ResNet(plan, num_classes, dense_classifier)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-lottery.pt'.format(arch)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


# ResNet Models
def resnet20(num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(20, 16)
    return _resnet('resnet20', plan, num_classes, dense_classifier, pretrained)

# ResNet Models
def narrow_resnet20(num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(20, 8)
    return _resnet('resnet20', plan, num_classes, dense_classifier, pretrained)

def resnet32(num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(32, 16)
    return _resnet('resnet32', plan, num_classes, dense_classifier, pretrained)

def resnet44(num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(44, 16)
    return _resnet('resnet44', plan, num_classes, dense_classifier, pretrained)

def resnet56(num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(56, 16)
    return _resnet('resnet56', plan, num_classes, dense_classifier, pretrained)

def resnet110(num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(110, 16)
    return _resnet('resnet110', plan, num_classes, dense_classifier, pretrained)

def resnet1202(num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(1202, 16)
    return _resnet('resnet1202', plan, num_classes, dense_classifier, pretrained)

# Wide ResNet Models
def wide_resnet20(num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(20, 32)
    return _resnet('wide_resnet20', plan, num_classes, dense_classifier, pretrained)

def wide_resnet32(num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(32, 32)
    return _resnet('wide_resnet32', plan, num_classes, dense_classifier, pretrained)

def wide_resnet44(num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(44, 32)
    return _resnet('wide_resnet44', plan, num_classes, dense_classifier, pretrained)

def wide_resnet56(num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(56, 32)
    return _resnet('wide_resnet56', plan, num_classes, dense_classifier, pretrained)

def wide_resnet110(num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(110, 32)
    return _resnet('wide_resnet110', plan, num_classes, dense_classifier, pretrained)

def wide_resnet1202(num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(1202, 32)
    return _resnet('wide_resnet1202', plan, num_classes, dense_classifier, pretrained)