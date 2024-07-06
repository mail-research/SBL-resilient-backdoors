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

from defenses.finetuning import finetuning
from defenses.nad import NAD
from defenses.pruning import pruning

from defenses.ft_sam import finetuning_sam

from sam import *
from utils.sam_utils import *



import datetime

def get_LP_dis(net1, net2):
    L0_dis, L1_dis, L2_dis, Linf_dis = 0, 0, 0, 0

    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(net1.named_parameters(), net2.named_parameters()):
            dw = p1 - p2
            L0_dis += dw.ne(0).float().sum()
            L1_dis += dw.abs().sum()
            L2_dis += (dw ** 2).sum()
            if Linf_dis == 0:
                Linf_dis = dw.abs().max()
            else:
                Linf_dis = max(Linf_dis, dw.abs().max())
        L2_dis = L2_dis.sqrt()
    return L0_dis.item(), L1_dis.item(), L2_dis.item(), Linf_dis.item()


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
        print_and_log(logger, 'NO BACKDOORED MODEL => EXIT ...')
        exit()


    
    ######### Defend model on Clean data ###########
    assert len(args.defenses) > 0, f'No DEFENSE'

    for defense_method in args.defenses:

        net = copy.deepcopy(model.net)
        args.lr = args.finetune_lr
        args.optimizer = args.ft_optimizer
        args.epochs = args.finetune_epochs
        finetune_optimizer, finetune_lr_scheduler = load_optimizer_and_scheduler(net, args)


        #############################################################
        ######################## FINE TUNING ########################
        #############################################################
        if defense_method == 'ft':
            print_and_log(logger, f"Defense by Finetuning (SGD - {args.lr}) backdoored model on clean data")
            print_and_log(logger, "="*50)
            try:
                print_and_log(args.logger, finetune_optimizer)
                print_and_log(args.logger, finetune_lr_scheduler)
            except:
                print(finetune_optimizer)
                print(finetune_lr_scheduler)
            loss_hist_3 = finetuning(net, finetune_loader, finetune_optimizer, finetune_lr_scheduler, 
                                    criterion, testloaders, epochs=args.finetune_epochs, device=device, 
                                    args=args)
            
            if args.is_saved:
                ft_backbone_saved_file = args.ckpt_file.replace('.pth.tar', f'ft_sgd_{args.lr}.pth.tar')
                save_backbone_only(net,ft_backbone_saved_file)

            clean_acc = test_model(net, clean_test_loader, device, args)
            poison_acc = test_model(net, poisoned_test_loader, device, args)
            
            # Compute the L_p distances between backdoored model and defended model
            L0, L1, L2, Linf = get_LP_dis(model.net, net)

            results[defense_method + f' w. SGD-{args.lr}'] = {
                'clean_acc' : clean_acc,
                'poison_acc' : poison_acc,
                'lr' : args.lr,
                'L0 distance': L0,
                'L1 distance': L1,
                'L2 distance': L2,
                'Linf distance': Linf,
            }
            print_and_log(logger, f'L0 distance: \t {L0}')
            print_and_log(logger, f'L1 distance: \t {L1}')
            print_and_log(logger, f'L2 distance: \t {L2}')
            print_and_log(logger, f'Linf distance: \t {Linf}')

            print_and_log(logger, "="*50)
            print_and_log(logger, f"Test accuracy on clean testing data: {clean_acc:.2f}")
            print_and_log(logger, f"Test accuracy on poisoned testing data: {poison_acc:.2f}%")
            print_and_log(logger, "="*50)
            


        #############################################################
        ############# NEURAL ATTENTIVE DISTILLATION #################
        #############################################################

        elif defense_method == 'nad':
            print_and_log(logger, "Defense by NAD backdoored model on clean data")
            print_and_log(logger, "="*50)
            try:
                print_and_log(args.logger, finetune_optimizer)
                print_and_log(args.logger, finetune_lr_scheduler)
            except:
                print(finetune_optimizer)
                print(finetune_lr_scheduler)
            nad_epochs = 20    # Set finetune epochs on teacher and student to 20 
            loss_hist_3 = NAD(net, finetune_loader, finetune_optimizer, finetune_lr_scheduler, 
                                    criterion, testloaders, epochs=nad_epochs, device=device, 
                                    args=args)
            
            clean_acc = test_model(net, clean_test_loader, device, args)
            poison_acc = test_model(net, poisoned_test_loader, device, args)

            # Compute the L_p distances between backdoored model and defended model
            L0, L1, L2, Linf = get_LP_dis(model.net, net)

            results[defense_method] = {
                'clean_acc' : clean_acc,
                'poison_acc' : poison_acc,
                'lr' : args.lr,
                'L0 distance': L0,
                'L1 distance': L1,
                'L2 distance': L2,
                'Linf distance': Linf,
            }
            print_and_log(logger, f'L0 distance: \t {L0}')
            print_and_log(logger, f'L1 distance: \t {L1}')
            print_and_log(logger, f'L2 distance: \t {L2}')
            print_and_log(logger, f'Linf distance: \t {Linf}')

            print_and_log(logger, "="*50)
            print_and_log(logger, f"Test accuracy on clean testing data: {clean_acc:.2f}")
            print_and_log(logger, f"Test accuracy on poisoned testing data: {poison_acc:.2f}%")
            print_and_log(logger, "="*50)

        
        
        #############################################################
        ########################## PRUNING #########################
        #############################################################

        elif defense_method == 'pruning':
            print_and_log(logger, "Defense by Pruning backdoored model on clean data")
            print_and_log(logger, "="*50)
            try:
                print_and_log(args.logger, finetune_optimizer)
                print_and_log(args.logger, finetune_lr_scheduler)
            except:
                print(finetune_optimizer)
                print(finetune_lr_scheduler)
            loss_hist_3 = pruning(net, finetune_loader, finetune_optimizer, finetune_lr_scheduler, 
                                    criterion, testloaders, epochs=args.finetune_epochs, device=device, 
                                    args=args)
            
            clean_acc = test_model(net, clean_test_loader, device, args)
            poison_acc = test_model(net, poisoned_test_loader, device, args)

            results[defense_method] = {
                'clean_acc' : clean_acc,
                'poison_acc' : poison_acc,
                'lr' : args.lr,
            }
            
            print_and_log(logger, "="*50)
            print_and_log(logger, f"Test accuracy on clean testing data: {clean_acc:.2f}")
            print_and_log(logger, f"Test accuracy on poisoned testing data: {poison_acc:.2f}%")
            print_and_log(logger, "="*50)

        
        #############################################################
        ####################### SAM FINE-TUNING #####################
        #############################################################

        elif defense_method == 'sam_ft':
            print_and_log(logger, "Defense by SAM Finetuning backdoored model on clean data")
            print_and_log(logger, "="*50)
            args.lr = args.finetune_lr / 2
            finetune_optimizer = SAM(net.parameters(), torch.optim.SGD, lr=args.lr, momentum=0.9)
            finetune_lr_scheduler = None
            try:
                print_and_log(args.logger, finetune_optimizer)
                print_and_log(args.logger, finetune_lr_scheduler)
            except:
                print(finetune_optimizer)
                print(finetune_lr_scheduler)

            loss_hist_3 = finetuning_sam(net, finetune_loader, finetune_optimizer, finetune_lr_scheduler, 
                                    criterion, testloaders, epochs=args.finetune_epochs, device=device, 
                                    args=args)
            
            clean_acc = test_model(net, clean_test_loader, device, args)
            poison_acc = test_model(net, poisoned_test_loader, device, args)

            # Compute the L_p distances between backdoored model and defended model
            L0, L1, L2, Linf = get_LP_dis(model.net, net)

            
            results[defense_method] = {
                'clean_acc' : clean_acc,
                'poison_acc' : poison_acc,
                'lr' : args.lr,
                'L0 distance': L0,
                'L1 distance': L1,
                'L2 distance': L2,
                'Linf distance': Linf,
            }
            print_and_log(logger, f'L0 distance: \t {L0}')
            print_and_log(logger, f'L1 distance: \t {L1}')
            print_and_log(logger, f'L2 distance: \t {L2}')
            print_and_log(logger, f'Linf distance: \t {Linf}')

            print_and_log(logger, "="*50)
            print_and_log(logger, f"Test accuracy on clean testing data: {clean_acc:.2f}")
            print_and_log(logger, f"Test accuracy on poisoned testing data: {poison_acc:.2f}%")
            print_and_log(logger, "="*50)



        else:
            print_and_log(logger, "This defense method is not supported!")
            print_and_log(logger, "======> NO DEFENSE!")
            print_and_log(logger, "="*50)

            clean_acc = test_model(net, clean_test_loader, device, args)
            poison_acc = test_model(net, poisoned_test_loader, device, args)

            print_and_log(logger, "="*50)
            print_and_log(logger, f"Final Test accuracy on clean testing data: {clean_acc:.2f}")
            print_and_log(logger, f"Final Test accuracy on poisoned testing data: {poison_acc:.2f}%")
            print_and_log(logger, "="*50)


    log_final_results(results, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)

    args = parser.parse_args()
    if args.debug_mode:
        args.epochs = 1
        args.sec_epochs = 1
        args.finetune_epochs = 1
    args.wandb_note = 'Defense'
    main(args)

