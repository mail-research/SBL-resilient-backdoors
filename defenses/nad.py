"""
This is the implement of NAD [1]. 
This source is modified from BackdoorBox codebase

Reference:
[1] Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks. ICLR 2021.
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F
import tqdm 
import copy

import copy

from models.base_model import BaseModel

# from agem import *
from utils.logger import *
from utils.load import *



class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks via Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am



def NAD(net, dataloader, optimizer, lr_scheduler, criterion, testloaders, epochs, device, args):
    
    if not hasattr(args, 'power'):
        args.power = 2.0
    if args.backbone in ['resnet18', 'preact_resnet18']:
        if not hasattr(args, 'beta'):
            args.beta = [500,500,500]
        if not hasattr(args, "target_layers"):
            args.target_layers=['layer2', 'layer3', 'layer4']

    elif args.backbone in ['resnet20']:   
        if not hasattr(args, 'beta'):
            args.beta = [500]
        if not hasattr(args, "target_layers"):
            args.target_layers=['blocks']

    elif args.backbone in ['vgg16', 'vgg11']:   
        if not hasattr(args, 'beta'):
            args.beta = [500]
        if not hasattr(args, "target_layers"):
            args.target_layers=['features_layer']
    else:
        raise NameError('Add layer name in NAD')

    # Finetune and get the teacher model
    teacher_model = copy.deepcopy(net)
    teacher_model = teacher_model.to(device)
    teacher_model.train()

    t_optimizer, t_lr_scheduler = load_optimizer_and_scheduler(teacher_model, args)

    print_and_log(args.logger, "="*50)
    print_and_log(args.logger, "Finetune teacher model")
    print_and_log(args.logger, "="*50)

    for epoch in range(epochs):
        loss = finetune_epoch(teacher_model, dataloader, t_optimizer,
                            t_lr_scheduler, criterion, epoch, device, args)
    
        clean_acc = test_model(teacher_model, testloaders[0], device, args)
        poison_acc = test_model(teacher_model, testloaders[1], device, args)
        if args.wandb is not None:  # Log results with wandb
            args.wandb.log(
                {
                    f'Finetuning Teacher model (NAD) Training Loss': loss,
                    f'Finetuning Teacher model (NAD) Clean Accuracy': clean_acc,
                    f'Finetuning Teacher model (NAD) Attack Success Rate': poison_acc,
                }, 
                # step=epoch,
            )
        if epoch % args.p_intervals == 0:
            # print_and_log(args.logger, f'Fine-tuning epoch {epoch}: Loss: {loss}')
            print_and_log(args.logger, f'NAD Fine-tuning teacher epoch {epoch} Clean Accuracy: {clean_acc}')
            print_and_log(args.logger, f'NAD Fine-tuning teacher epoch {epoch} Poison Accuracy: {poison_acc}')

    # Perform NAD and get the repaired model
    for param in teacher_model.parameters():
            param.requires_grad = False
    net = net.to(device)
    net.train()

    criterionAT = AT(args.power)

    print_and_log(args.logger, "="*50)
    print_and_log(args.logger, "Performing NAD ...")
    print_and_log(args.logger, "="*50)

    for epoch in range(epochs):
        loss = perform_NAD_epoch(net, teacher_model, dataloader, optimizer, 
                                lr_scheduler, criterion, criterionAT, epoch, 
                                device, args)
        
        clean_acc = test_model(net, testloaders[0], device, args)
        poison_acc = test_model(net, testloaders[1], device, args)
        if args.wandb is not None:  # Log results with wandb
            args.wandb.log(
                {
                    f'NAD Training Loss': loss,
                    f'NAD Clean Accuracy': clean_acc,
                    f'NAD Attack Success Rate': poison_acc,
                }, 
                # step=epoch,
            )
        if epoch % args.p_intervals == 0:
            print_and_log(args.logger, f'NAD Fine-tuning student epoch {epoch} Clean Accuracy: {clean_acc}')
            print_and_log(args.logger, f'NAD Fine-tuning student epoch {epoch} Poison Accuracy: {poison_acc}')





# NAD loop
def perform_NAD_epoch(net, teacher_model, dataloader, optimizer, lr_scheduler, criterion, criterionAT, epoch, device, args):
    net.train()
    avg_loss = 0
    count = 0
    pbar = tqdm.tqdm(dataloader, desc=f'Finetuning Epoch: {epoch}')
    for inputs, targets, _ in pbar:
        if args.debug_mode and count > 2:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        container = []
        def forward_hook(module, input, output):
            container.append(output)

        hook_list = []
        for name, module in net._modules.items():
            if name in args.target_layers:
                hk = module.register_forward_hook(forward_hook)
                hook_list.append(hk)
        
        for name, module in teacher_model._modules.items():
            if name in args.target_layers:
                hk = module.register_forward_hook(forward_hook)
                hook_list.append(hk)

        # forward to add intermediate features into containers 
        outputs = net(inputs)
        _ = teacher_model(inputs)

        for hk in hook_list:
            hk.remove()

        loss = criterion(outputs, targets)
        AT_loss = 0
        for idx in range(len(args.beta)):
            AT_loss = AT_loss + criterionAT(container[idx], container[idx+len(args.beta)]) * args.beta[idx]  
        
        pbar.set_postfix({'loss': loss.item(), 'AT_loss': AT_loss.item()})
        
        loss = loss + AT_loss
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        count += 1

    avg_loss = avg_loss/count

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(avg_loss)
        else:
            lr_scheduler.step()
        
    return avg_loss



# Training loop
def finetuning(net, dataloader, optimizer, lr_scheduler, criterion, testloaders, epochs, device, args):
    loss_hist = []
    try:
        print_and_log(args.logger, optimizer)
        print_and_log(args.logger, lr_scheduler)
    except:
        print(optimizer)
        print(lr_scheduler)

    for epoch in range(epochs):
        
        loss = finetune_epoch(net, dataloader, optimizer, lr_scheduler,
                            criterion, epoch, device, args)    
        clean_acc = test_model(net, testloaders[0], device, args)
        poison_acc = test_model(net, testloaders[1], device, args)
        if args.wandb is not None:  # Log results with wandb
            args.wandb.log(
                {
                    f'Finetuning Training Loss': loss,
                    f'Finetuning Clean Accuracy': clean_acc,
                    f'Finetuning Attack Success Rate': poison_acc,
                }, 
                # step=epoch,
            )
        
        loss_hist.append(loss)

        if epoch % 10 == 0:

            # print_and_log(args.logger, f'Fine-tuning epoch {epoch}: Loss: {loss}')
            print_and_log(args.logger, f'Fine-tuning epoch {epoch} Clean Accuracy: {clean_acc}')
            print_and_log(args.logger, f'Fine-tuning epoch {epoch} Poison Accuracy: {poison_acc}')

    return loss_hist


# Finetuning loop
def finetune_epoch(net, dataloader, optimizer, lr_scheduler, criterion, epoch, device, args):
    net.train()
    avg_loss = 0
    count = 0
    # pbar = tqdm.tqdm(dataloader, desc=f'Finetuning Epoch: {epoch}')
    print_and_log(args.logger, f'Finetuning Epoch: {epoch}')
    for inputs, targets, _ in dataloader:
        if args.debug_mode and count > 2:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        count += 1
        # pbar.set_postfix({'loss': loss.item()})

    avg_loss = avg_loss/count

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(avg_loss)
        else:
            lr_scheduler.step()
        
    return avg_loss



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