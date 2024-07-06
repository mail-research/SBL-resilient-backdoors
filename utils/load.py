import os
import sys
import torch
import numpy as np

from backbones.mlp import *
from backbones.resnet import *
from backbones.vgg import * 
from backbones.preact_resnet import *
from backbones.light_resnet import resnet20

from models.ewc import *
from models.anchor import *
from models.agem import *
from models.joint import * 
from models.si import *
from models.mas import *
from models.naive import *




def load_backbone(backbone_name, num_classes, input_size=32):
    net = None
    if backbone_name == 'simpleMLP':
        net = SimpleMLP()
    elif backbone_name == 'resnet18':
        net = ResNet(num=18, num_classes=num_classes, input_size=input_size)        
    elif backbone_name == 'resnet34':
        net = ResNet(num=34, num_classes=num_classes, input_size=input_size)
    elif backbone_name == 'preact_resnet18':
        net = PreActResNet(num=18, num_classes=num_classes)        
    elif backbone_name == 'preact_resnet34':
        net = PreActResNet(num=34, num_classes=num_classes)
    elif backbone_name == 'vgg11':
        net = vgg11_bn(num_classes=num_classes)
    elif backbone_name == 'vgg16':
        net = vgg16_bn(num_classes=num_classes)
    elif backbone_name == 'resnet20':
        net = resnet20(num_classes=num_classes)
    else:
        raise NotImplementedError(f'This {backbone_name} backbone is not implemented!')
    return net
    

def load_model(backbone, criterion, optimizer, lr_scheduler, args, device, transform=None):

    if args.cl_method == 'ewc':
        model = EWC(backbone=backbone, loss=criterion, optimizer=optimizer, \
                        lr_scheduler=lr_scheduler, args=args, device=device, transform=transform)
    elif args.cl_method == 'anchoring':
        model = Anchor(backbone=backbone, loss=criterion, optimizer=optimizer, \
                        lr_scheduler=lr_scheduler, args=args, device=device, transform=transform)
    elif args.cl_method == 'agem':
        model = AGem(backbone=backbone, loss=criterion, optimizer=optimizer, \
                        lr_scheduler=lr_scheduler, args=args, device=device, transform=transform)
    elif args.cl_method == 'si':
        model = SI(backbone=backbone, loss=criterion, optimizer=optimizer, \
                        lr_scheduler=lr_scheduler, args=args, device=device, transform=transform)
    elif args.cl_method == 'mas':
        model = MAS(backbone=backbone, loss=criterion, optimizer=optimizer, \
                        lr_scheduler=lr_scheduler, args=args, device=device, transform=transform)
    elif args.cl_method == 'naive':
        model = Naive(backbone=backbone, loss=criterion, optimizer=optimizer, \
                        lr_scheduler=lr_scheduler, args=args, device=device, transform=transform)
    elif args.cl_method == 'joint':
        model = Joint(backbone=backbone, loss=criterion, optimizer=optimizer, \
                        lr_scheduler=lr_scheduler, args=args, device=device, transform=transform)
    else:
        raise ValueError(f'Continual Learning method {args.cl_method} is not supported!')

    return model



def load_optimizer_and_scheduler(net, args):
    # idea: given model and args, return the optimizer and scheduler you choose to use

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,  # 0.9
                                    weight_decay=args.weight_decay,  # 5e-4
                                    )
    elif args.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=args.lr,
            rho=0.95,  # 0.95,
            eps=1e-07,  # 1e-07,
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    amsgrad=True)


    if args.lr_scheduler == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      cycle_momentum=False)
    elif args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=1,  # 1
                                                    gamma=0.92)  # 0.92
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # elif args.lr_scheduler == 'MultiStepLR':
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, , args.steplr_gamma)
    elif args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
        )
    else:
        scheduler = None

    return optimizer, scheduler

def save_backbone_only(net, filename="checkpoint.pth.tar"):
    """Save model and optimizer states."""
    checkpoint = {
        'model_state_dict': net.state_dict(),
    }
    torch.save(checkpoint, filename)

def save_checkpoint(model, epoch, clean_acc, poison_acc, filename="checkpoint.pth.tar"):
    """Save model and optimizer states."""
    checkpoint = {
        'model_state_dict': model.net.state_dict(),
        'optimizer_state_dict': model.opt.state_dict(),
        'lr_scheduler_state_dict': model.lr_scheduler.state_dict(),
        # Add any other necessary metadata if needed
        'epoch': epoch,
        'clean_acc': clean_acc,
        'poison_acc': poison_acc,
    }
    torch.save(checkpoint, filename)

# Example usage:
# save_checkpoint(model, optimizer, "my_checkpoint.pth.tar")


def load_checkpoint(model, filename="checkpoint.pth.tar"):
    """Load model and optimizer states."""
    checkpoint = torch.load(filename)
    model.net.load_state_dict(checkpoint['model_state_dict'])
    model.opt.load_state_dict(checkpoint['optimizer_state_dict'])
    model.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    # If you have saved any other metadata, you can also load them here
    epoch = checkpoint['epoch']
    clean_acc = checkpoint['clean_acc']
    poison_acc = checkpoint['poison_acc']
    others = {
        'epoch': epoch,
        'clean_acc': clean_acc,
        'poison_acc': poison_acc,
    }
    return model, others

# Example usage:
# model, optimizer = load_checkpoint(model, optimizer, "my_checkpoint.pth.tar")

import datetime
def load_all_path(args):
    if args.mixed_first:
        args.task_order = 'bd_first'
    else:
        args.task_order = 'clean_first'

    logging_path = f'./logging/{args.wandb_note}/{args.dataset}/{args.backbone}/{args.poisoning_method}/seed_{args.seed}/'
    if args.opt_mode != 'normal':
        logging_path = f'./logging/{args.wandb_note}/{args.dataset}/{args.backbone}/{args.opt_mode}/{args.poisoning_method}/seed_{args.seed}/'

    if args.is_dat:
        logging_path += f'DPD/{args.cl_method}/data_mode_{args.data_mode}/'
    else:
        logging_path += f'NormalTraining/{args.cl_method}/data_mode_{args.data_mode}/' 

    os.makedirs(logging_path, exist_ok=True)

    task_portion =  f'{args.task_portion[0]}_{args.task_portion[1]}_{args.task_portion[2]}'
    logging_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    logging_name += f'{args.task_order}_pr_{args.poisoning_rate}_tp_{task_portion}_lr_{args.lr}_sec_lr_{args.sec_lr}'
    logging_name += f'_finetunelr_{args.finetune_lr}_lambda_{args.lambd}'

    args.log_file = f'{logging_path}/{logging_name}.log'
    args.logging_name = logging_name


    if args.is_saved or args.is_load:
        save_path = f'./checkpoints/{args.dataset}/{args.backbone}/{args.poisoning_method}/seed_{args.seed}/'
        if args.opt_mode != 'normal':
            save_path = f'./checkpoints/{args.dataset}/{args.backbone}/{args.opt_mode}/{args.poisoning_method}/seed_{args.seed}/'
        if args.is_dat:
            save_path += f'DPD/{args.cl_method}/data_mode_{args.data_mode}/'
        else:
            save_path += f'NormalTraining/{args.cl_method}/data_mode_{args.data_mode}/' 
        
        os.makedirs(save_path, exist_ok=True)
        save_name = f'{args.task_order}_pr_{args.poisoning_rate}_tp_{task_portion}_lr_{args.lr}_sec_lr_{args.sec_lr}'
        save_name += f'_lambda_{args.lambd}_epochs_{args.epochs}'
        save_name += '.pth.tar'

        args.ckpt_file = f'{save_path}/{save_name}'


        save_first_task_path = f'./checkpoints/{args.dataset}/{args.backbone}/{args.poisoning_method}/seed_{args.seed}/'
        if args.opt_mode != 'normal':
            save_first_task_path = f'./checkpoints/{args.dataset}/{args.backbone}/{args.opt_mode}/{args.poisoning_method}/seed_{args.seed}/'
        if args.is_dat:
            save_first_task_path += f'DPD/first_task/data_mode_{args.data_mode}/'
        else:
            save_first_task_path += f'NormalTraining/first_task/data_mode_{args.data_mode}/' 
        os.makedirs(save_first_task_path, exist_ok=True)
        save_first_task_name = f'{args.task_order}_pr_{args.poisoning_rate}_tp_{task_portion}_lr_{args.lr}_epochs_{args.epochs}.pth.tar'
        args.save_first_task_file = f'{save_first_task_path}/{save_first_task_name}'

    