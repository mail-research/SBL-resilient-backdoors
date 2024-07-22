import os
import argparse
from argparse import ArgumentParser

def add_arguments(parser: ArgumentParser):

    
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 
                                                                        'cifar10', 
                                                                        'gtsrb',
                                                                        'celebA',
                                                                        'imagenet10',
                                                                        ])
    parser.add_argument('--backbone', type=str, default='simpleMLP')

    parser.add_argument('--cl_method', type=str, default='ewc')
    parser.add_argument('--lambd', type=float, default=1)
    parser.add_argument('--buffer_size', type=int, default=256)
    parser.add_argument('--xi', type=float, default=1.0, help='Used in SI')

    parser.add_argument('--opt_mode', type=str, default='normal', choices=['normal', 'sam', 'gam', 'none'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--sec_lr', type=float, default=0.001)
    parser.add_argument('--finetune_lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--ft_optimizer', type=str, default='sgd')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--sec_epochs', type=int, default=100)
    parser.add_argument('--finetune_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--p_intervals', type=int, default=10)

    parser.add_argument('--defenses', nargs='+', type=str, default=['ft', 'nad'], \
                        help='list defense methods')

    parser.add_argument('--poisoning_method', type=str, default='badnet',\
                        help='poisoning methods: badnet, blended, sig')
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--poisoning_rate', type=float, default=0.1)
    parser.add_argument('--trigger_size', type=int, default=3)
    parser.add_argument('--task_portion', nargs='+', type=float, default=[0.05, 0.1, 0.85], \
                        help="The portion of data for finetuning clean data (Step 1), unseen clean data (Defense), and mixed data (Step 0)")
    parser.add_argument('--mixed_first', action='store_true', default=False, help='Training the mixed of clean and poison first')
    parser.add_argument('--is_dat', action='store_true', default=False, \
                        help="Dynamically adding triggers during training")
    parser.add_argument('--data_mode', type=int, default=0, \
                        help='0: all separate, 1: attack defense separate, 2: fine-tune on seen data')
    parser.add_argument('--input_size', type=int, default=32)

    parser.add_argument('--is_wandb', action='store_true', default=False, help='report to wandb')
    parser.add_argument('--wandb_note', type=str, default='', help='note for wandb and logging name')
    parser.add_argument('--is_saved', action='store_true', default=False, help='save model')
    parser.add_argument('--is_load', action='store_true', default=False, help='load model')
    parser.add_argument('--debug_mode', action='store_true', default=False)
