import torch
import torch.nn as nn 
import tqdm 

import copy

from models.base_model import BaseModel

# from agem import *
from utils.logger import *


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
                    f'Finetuning (SGD - {args.lr}) Training Loss': loss,
                    f'Finetuning (SGD - {args.lr}) Clean Accuracy': clean_acc,
                    f'Finetuning (SGD - {args.lr}) Attack Success Rate': poison_acc,
                }, 
                # step=epoch,
            )
        
        loss_hist.append(loss)

        if epoch % args.p_intervals == 0:

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