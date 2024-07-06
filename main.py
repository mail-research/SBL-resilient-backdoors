import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import tqdm
import numpy as np
import random

import argparse
import logging

from backbones.mlp import SimpleMLP
from backbones.resnet import *
from backbones.vgg import *

from data.dataset import *

from models.training import *
from models.ewc import *
from models.anchor import *
from models.agem import *
from models.joint import * 

from utils.logger import *
from utils.load import *
from utils.arguments import *


from sam import *
from utils.sam_utils import *


import datetime


def main(args):

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    task_portion =  f'{args.task_portion[0]}_{args.task_portion[1]}_{args.task_portion[2]}'

    load_all_path(args)

    logger = setup_logger(log_file=args.log_file)
    args.logger = logger

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    args.wandb = None

    print_and_log(logger, args)
    print_and_log(logger, '='*100)
    print_and_log(logger, f'Loading {args.dataset} dataset ...')

    ############################
    ####### LOADING DATA #######
    ############################

    if args.is_dat:   # Use Dynamically Adding Trigger
        num_classes, mixed_loader, clean_loader, finetune_loader, clean_test_loader, poisoned_test_loader, trigger = get_dat_dataloader(args)
    else:   # Use Normal training
        num_classes, mixed_loader, clean_loader, finetune_loader, clean_test_loader, poisoned_test_loader, trigger = get_dataloader(args)
    
    
    if args.mixed_first:
        dataloaders = [mixed_loader, clean_loader]
        args.is_clean_task = [False, True]
    else:
        dataloaders = [clean_loader, mixed_loader]
        args.is_clean_task = [True, False]

    testloaders = [clean_test_loader, poisoned_test_loader]

    #######################
    ###### LOAD BACKBONE
    
    net = load_backbone(args.backbone, num_classes, args.input_size)
    net.to(device)
    #######################

    #################################
    ####### LOADING OPTIMIZER #######
    #################################

    optimizer, lr_scheduler = load_optimizer_and_scheduler(net, args)
    if args.opt_mode == 'sam':
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer = SAM(net.parameters(), base_optimizer, lr=args.lr, momentum=0.9)

    args.sec_opimitzer = None
    if args.sec_lr != args.lr:    # Apply different optimizer's hyper for the second task
        # Similar base optimizer with the first task but different epochs and lr
        args.lr = args.sec_lr
        epochs = args.epochs
        args.epochs = args.sec_epochs
        sec_optimizer, sec_lr_scheduler = load_optimizer_and_scheduler(net, args)
        args.sec_optimizer = sec_optimizer
        args.sec_lr_scheduler = sec_lr_scheduler
        args.epochs = epochs    # re-assign args.epochs for training the first task

    criterion = nn.CrossEntropyLoss()

    model =  load_model(backbone=net, 
                        criterion=criterion,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        args=args,
                        device=device,
                        transform=None,
                        )
        
    ############################
    #### LOADING CHECKPOINT ####
    ############################

    load_succeed = False
    if args.is_load:
        print_and_log(logger, f'==> {args.ckpt_file}')
        if os.path.exists(args.ckpt_file):
            print_and_log(logger, 'LOADING MODEL FROM CHECKPOINTS')
            try:
                model, _ = load_checkpoint(model, filename=args.ckpt_file)
                print_and_log(logger, 'LOADING MODEL FROM CHECKPOINTS SUCCEED!')
                load_succeed = True
                results = {}    # Used for defense
            except:
                load_succeed = False
                print_and_log(logger, 'FAIL TO LOAD MODEL FROM CHECKPOINTS')
        else:
            print_and_log(logger, 'CHECKPOINT DOES NOT EXIST')
            # print(load_succeed)
    
    if not load_succeed:  
        ############################
        ######### TRAINING #########
        ############################


        if args.cl_method == 'joint':
            dataloaders = [mixed_loader]
        
        results = train(model, dataloaders, testloaders, args.epochs, device, args)

        #######################################

        if args.is_saved:
            print_and_log(logger, f'=========> Save checkpoint to {args.ckpt_file}')
            if args.cl_method != 'joint':
                save_checkpoint(model, args.epochs, 
                                clean_acc=results['task_1']['clean_acc'], 
                                poison_acc=results['task_1']['poison_acc'],
                                filename=args.ckpt_file)
            else:
                save_checkpoint(model, args.epochs, 
                                clean_acc=results['task_0']['clean_acc'], 
                                poison_acc=results['task_0']['poison_acc'],
                                filename=args.ckpt_file)
                

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)

    args = parser.parse_args()
    if args.debug_mode:
        args.epochs = 1
        args.sec_epochs = 1
        args.finetune_epochs = 1
    main(args)

