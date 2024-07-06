"""
This code is modified from BackdoorBox and Wanet source code
"""

import torch
import torch.nn as nn 
import tqdm 

import copy

from models.base_model import BaseModel
from backbones.resnet import *
from backbones.preact_resnet import *

# from agem import *
from utils.logger import *


# Define model pruning
class MaskedLayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedLayer, self).__init__()
        self.base = base
        self.mask = mask

    def forward(self, input):
        return self.base(input) * self.mask

def pruning(net, dataloader, optimizer, lr_scheduler, criterion, testloaders, epochs, device, args):
    if args.backbone == 'vgg16':
        list_clean_acc = pruning_vgg(net, dataloader, optimizer, lr_scheduler, criterion, testloaders, epochs, device, args)
    elif args.backbone == 'resnet20':
        list_clean_acc = pruning_resnet20(net, dataloader, optimizer, lr_scheduler, criterion, testloaders, epochs, device, args)
    elif args.backbone == 'resnet18':
        list_clean_acc = pruning_resnet18(net, dataloader, optimizer, lr_scheduler, criterion, testloaders, epochs, device, args)
    else:
        raise NameError('Wrong pruning backbone name')

    return list_clean_acc


def pruning_vgg(net, dataloader, optimizer, lr_scheduler, criterion, testloaders, epochs, device, args):
    if not hasattr(args, "layer"):
        args.layer = 'features_layer'
    # if not hasattr(args, "prune_rate"):
    #     args.prune_rate = 0.5

    # prune_rate = args.prune_rate
    layer_to_prune = args.layer

    print_and_log(args.logger, 'PRUNING ...')

    net.eval()
    net.requires_grad_(False)

    with torch.no_grad():
        container = []
        def forward_hook(module, input, output):
            container.append(output)
        
        hook = getattr(net, layer_to_prune).register_forward_hook(forward_hook)
        print_and_log(args.logger, "Fowarding all finetuning set")

        for data, _, _ in dataloader:
            net(data.to(device))
        hook.remove()

    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0,2,3])
    seq_sort = torch.argsort(activation)
    num_channels = len(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)


    # Pruning times - no-tuning after pruning a channel!!!
    list_clean_acc = []
    list_poison_acc = []

    for index in range(pruning_mask.shape[0]):
        if args.debug_mode and index > 2:
            break
        net_pruned = copy.deepcopy(net)
        num_pruned = index
        if index:
            channel = seq_sort[index - 1]
            pruning_mask[channel] = False
        # print("Pruned {} filters".format(num_pruned))

        # net_pruned.layer4[1].conv2 = nn.Conv2d(
        #     pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
        # )
        # net_pruned.linear = nn.Linear(pruning_mask.shape[0] - num_pruned, 10)
        # net_pruned.eval()
        # # Re-assigning weight to the pruned net
        # for name, module in net_pruned._modules.items():
        #     if "layer4" in name:
        #         module[1].conv2.weight.data = net.layer4[1].conv2.weight.data[pruning_mask]
        #         module[1].ind = pruning_mask
        #     elif "linear" == name:
        #         module.weight.data = net.linear.weight.data[:, pruning_mask]
        #         module.bias.data = net.linear.bias.data
        #     else:
        #         continue
        
        net_pruned.features_layer[-4] = nn.Conv2d(
            pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
        )
        net_pruned.classifier = nn.Linear((pruning_mask.shape[0] - num_pruned), 10)
        # assert isinstance(net, ResNet), 'Model should be ResNet-18 for CELEBATTR or T-IMNET'
        for name, module in net_pruned._modules.items():
            if "features_layer" in name:
                module[-4].weight.data = net.features_layer[-4].weight.data[pruning_mask]
                module[-3].running_mean = net.features_layer[-3].running_mean[pruning_mask]
                module[-3].running_var = net.features_layer[-3].running_var[pruning_mask]
                module[-3].weight.data = net.features_layer[-3].weight.data[pruning_mask]
                module[-3].bias.data = net.features_layer[-3].bias.data[pruning_mask]

                # module[1].ind = pruning_mask

            elif "classifier" == name:
                converted_mask = pruning_mask #convert(pruning_mask)
                module.weight.data = net.classifier.weight.data[:, converted_mask]
                module.bias.data = net.classifier.bias.data
            else:
                continue
        net_pruned.to(device)


        clean_acc = test_model(net_pruned, testloaders[0], device, args)
        poison_acc = test_model(net_pruned, testloaders[1], device, args)

        print_and_log(args.logger, f'Pruned {num_pruned} filters \t Clean Acc: {clean_acc} \t ASR: {poison_acc}')

        list_clean_acc.append(clean_acc)
        list_poison_acc.append(poison_acc)

        if args.wandb is not None:  # Log results with wandb
            args.wandb.log(
                {
                    f'Pruning Clean Accuracy': clean_acc,
                    f'Pruning Attack Success Rate': poison_acc,
                }, 
                # step=epoch,
            )
    return list_clean_acc

def pruning_resnet20(net, dataloader, optimizer, lr_scheduler, criterion, testloaders, epochs, device, args):
    print_and_log(args.logger, 'PRUNING ...')

    net.eval()
    net.requires_grad_(False)

    with torch.no_grad():
        container = []
        def forward_hook(module, input, output):
            container.append(output)
        
        hook = getattr(net, 'blocks')[-1].register_forward_hook(forward_hook)
        print_and_log(args.logger, "Fowarding all finetuning set")

        for data, _, _ in dataloader:
            net(data.to(device))
        hook.remove()

    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0,2,3])
    seq_sort = torch.argsort(activation)
    num_channels = len(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)


    # Pruning times - no-tuning after pruning a channel!!!
    list_clean_acc = []
    list_poison_acc = []

    for index in range(pruning_mask.shape[0]):
        if args.debug_mode and index > 2:
            break
        net_pruned = copy.deepcopy(net)
        num_pruned = index
        if index:
            channel = seq_sort[index - 1]
            pruning_mask[channel] = False
        # print("Pruned {} filters".format(num_pruned))

        # net_pruned.layer4[1].conv2 = nn.Conv2d(
        #     pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
        # )
        # net_pruned.linear = nn.Linear(pruning_mask.shape[0] - num_pruned, 10)
        # net_pruned.eval()
        # # Re-assigning weight to the pruned net
        # for name, module in net_pruned._modules.items():
        #     if "layer4" in name:
        #         module[1].conv2.weight.data = net.layer4[1].conv2.weight.data[pruning_mask]
        #         module[1].ind = pruning_mask
        #     elif "linear" == name:
        #         module.weight.data = net.linear.weight.data[:, pruning_mask]
        #         module.bias.data = net.linear.bias.data
        #     else:
        #         continue
        
        net_pruned.blocks[-1].conv2 = nn.Conv2d(
            pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
        )
        net_pruned.fc = nn.Linear((pruning_mask.shape[0] - num_pruned), 10)
        # assert isinstance(net, ResNet), 'Model should be ResNet-18 for CELEBATTR or T-IMNET'
        for name, module in net_pruned._modules.items():
            if "blocks" in name:
                module[-1].conv2.weight.data = net.blocks[-1].conv2.weight.data[pruning_mask]
                module[-1].bn2.running_mean = net.blocks[-1].bn2.running_mean[pruning_mask]
                module[-1].bn2.running_var = net.blocks[-1].bn2.running_var[pruning_mask]
                module[-1].bn2.weight.data = net.blocks[-1].bn2.weight.data[pruning_mask]
                module[-1].bn2.bias.data = net.blocks[-1].bn2.bias.data[pruning_mask]

                module[-1].ind = pruning_mask

            elif "fc" == name:
                converted_mask = pruning_mask #convert(pruning_mask)
                module.weight.data = net.fc.weight.data[:, converted_mask]
                module.bias.data = net.fc.bias.data
            else:
                continue
        net_pruned.to(device)


        clean_acc = test_model(net_pruned, testloaders[0], device, args)
        poison_acc = test_model(net_pruned, testloaders[1], device, args)

        print_and_log(args.logger, f'Pruned {num_pruned} filters \t Clean Acc: {clean_acc} \t ASR: {poison_acc}')

        list_clean_acc.append(clean_acc)
        list_poison_acc.append(poison_acc)

        if args.wandb is not None:  # Log results with wandb
            args.wandb.log(
                {
                    f'Pruning Clean Accuracy': clean_acc,
                    f'Pruning Attack Success Rate': poison_acc,
                }, 
                # step=epoch,
            )
    return list_clean_acc


def pruning_resnet18(net, dataloader, optimizer, lr_scheduler, criterion, testloaders, epochs, device, args):
    if not hasattr(args, "layer"):
        args.layer = 'layer4'
    # if not hasattr(args, "prune_rate"):
    #     args.prune_rate = 0.5

    # prune_rate = args.prune_rate
    layer_to_prune = args.layer

    print_and_log(args.logger, 'PRUNING ...')

    net.eval()
    net.requires_grad_(False)

    with torch.no_grad():
        container = []
        def forward_hook(module, input, output):
            container.append(output)
        
        hook = getattr(net, layer_to_prune).register_forward_hook(forward_hook)
        print_and_log(args.logger, "Fowarding all finetuning set")

        for data, _, _ in dataloader:
            net(data.to(device))
        hook.remove()

    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0,2,3])
    seq_sort = torch.argsort(activation)
    num_channels = len(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)


    # Pruning times - no-tuning after pruning a channel!!!
    list_clean_acc = []
    list_poison_acc = []

    for index in range(pruning_mask.shape[0]):
        if args.debug_mode and index > 2:
            break
        net_pruned = copy.deepcopy(net)
        num_pruned = index
        if index:
            channel = seq_sort[index - 1]
            pruning_mask[channel] = False
        # print("Pruned {} filters".format(num_pruned))

        # net_pruned.layer4[1].conv2 = nn.Conv2d(
        #     pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
        # )
        # net_pruned.linear = nn.Linear(pruning_mask.shape[0] - num_pruned, 10)
        # net_pruned.eval()
        # # Re-assigning weight to the pruned net
        # for name, module in net_pruned._modules.items():
        #     if "layer4" in name:
        #         module[1].conv2.weight.data = net.layer4[1].conv2.weight.data[pruning_mask]
        #         module[1].ind = pruning_mask
        #     elif "linear" == name:
        #         module.weight.data = net.linear.weight.data[:, pruning_mask]
        #         module.bias.data = net.linear.bias.data
        #     else:
        #         continue
        
        net_pruned.layer4[1].conv2 = nn.Conv2d(
            pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
        )
        net_pruned.linear = nn.Linear((pruning_mask.shape[0] - num_pruned), 10)
        # assert isinstance(net, ResNet), 'Model should be ResNet-18 for CELEBATTR or T-IMNET'
        for name, module in net_pruned._modules.items():
            if "layer4" in name:
                module[1].conv2.weight.data = net.layer4[1].conv2.weight.data[pruning_mask]
                module[1].bn2.running_mean = net.layer4[1].bn2.running_mean[pruning_mask]
                module[1].bn2.running_var = net.layer4[1].bn2.running_var[pruning_mask]
                module[1].bn2.weight.data = net.layer4[1].bn2.weight.data[pruning_mask]
                module[1].bn2.bias.data = net.layer4[1].bn2.bias.data[pruning_mask]

                module[1].ind = pruning_mask

            elif "linear" == name:
                converted_mask = pruning_mask #convert(pruning_mask)
                module.weight.data = net.linear.weight.data[:, converted_mask]
                module.bias.data = net.linear.bias.data
            else:
                continue
        net_pruned.to(device)


        clean_acc = test_model(net_pruned, testloaders[0], device, args)
        poison_acc = test_model(net_pruned, testloaders[1], device, args)

        print_and_log(args.logger, f'Pruned {num_pruned} filters \t Clean Acc: {clean_acc} \t ASR: {poison_acc}')

        list_clean_acc.append(clean_acc)
        list_poison_acc.append(poison_acc)

        if args.wandb is not None:  # Log results with wandb
            args.wandb.log(
                {
                    f'Pruning Clean Accuracy': clean_acc,
                    f'Pruning Attack Success Rate': poison_acc,
                }, 
                # step=epoch,
            )
    return list_clean_acc

    


def test_model(net, dataloader, device, args):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if args.debug_mode:
                break

    accuracy = 100 * correct / total
    return accuracy