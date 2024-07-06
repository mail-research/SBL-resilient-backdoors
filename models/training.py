import os
import torch
import torch.nn as nn 
import tqdm 

from models.base_model import BaseModel

# from agem import *
from utils.logger import *
from utils.load import *
from data.dataset import *


def train(model: BaseModel, dataloaders, testloaders, epochs, device, args):
    n_task = len(dataloaders)
    results = {}
    # training_results = {}

    for task_id in range(len(dataloaders)): # loop over tasks
        # tr_result = {}
        # loss_hist = []
        # clean_acc_hist = []
        # poisoned_acc_hist = []
        args.clean_training = args.is_clean_task[task_id]
        if args.cl_method == 'joint':
            args.clean_training = False
            
        print_and_log(args.logger, f'Training on task {task_id}')
        print_and_log(args.logger, '='*50)

        dataloader = dataloaders[task_id]
        task_res = {}

        if task_id > 0: # Change epochs to epochs of second task
            epochs = args.sec_epochs

        ############################
        load_succeed = False
        if args.is_load and n_task > 1 and task_id == 0:
            if os.path.exists(args.save_first_task_file):
                print_and_log(args.logger, 'LOADING FIRST TASK MODEL FROM CHECKPOINTS')
                try:
                    model, others = load_checkpoint(model, filename=args.save_first_task_file)
                    clean_acc = others['clean_acc']
                    poison_acc = others['poison_acc']
                    print_and_log(args.logger, 'LOADING FIRST TASK MODEL FROM CHECKPOINTS SUCCEED!')
                    load_succeed = True
                except:
                    load_succeed = False
                    print_and_log(args.logger, 'FAIL TO LOAD FIRST TASK MODEL FROM CHECKPOINTS')
            else:
                print_and_log(args.logger, 'FIRST TASK MODEL CHECKPOINT DOES NOT EXIST')
        
        try:
            print_and_log(args.logger, model.opt)
            print_and_log(args.logger, model.lr_scheduler)
        except:
            print(model.opt)
            print(model.lr_scheduler)

        if not load_succeed:
            for epoch in range(epochs):

                loss = train_epoch(model, dataloader, epoch, device, args)
                clean_acc = test_model(model.net, testloaders[0], device, args)
                poison_acc = test_model(model.net, testloaders[1], device, args)
                
                if args.wandb is not None:  # Log results with wandb
                    args.wandb.log(
                        {
                            f'Task {task_id} Training Loss': loss,
                            f'Task {task_id} Clean Accuracy': clean_acc,
                            f'Task {task_id} Attack Success Rate': poison_acc,
                        }, 
                        # step=epoch,
                    )
                # loss_hist.append(loss)
                # clean_acc_hist.append(clean_acc)
                # poisoned_acc_hist.append(poison_acc)

                if epoch % args.p_intervals == 0: # Print and log the result
                    print_and_log(args.logger, f'Training task {task_id} epoch {epoch}\tClean Accuracy is {clean_acc}')
                    print_and_log(args.logger, f'Training task {task_id} epoch {epoch}\tPoison Accuracy is {poison_acc}')

            # clean_acc = test_model(model.net, testloaders[0], device, args)
            # poison_acc = test_model(model.net, testloaders[1], device, args)

            if args.is_saved and n_task > 1 and task_id == 0: # only save the first model
                save_checkpoint(model, args.epochs, clean_acc, poison_acc, filename=args.save_first_task_file)


        #### IMPORTANT 
        if hasattr(model, 'end_task') and (task_id + 1) < n_task:  # update CL
            model.end_task(dataloader)  

        if args.sec_optimizer is not None:    # Update optimizer for the second task
            args.opt_mode = 'normal'
            model.opt = args.sec_optimizer
            model.lr_scheduler = args.sec_lr_scheduler
            try:
                print_and_log(args.logger, model.opt)
                print_and_log(args.logger, model.lr_scheduler)
            except:
                print(model.opt)
                print(model.lr_scheduler)
        
        print_and_log(args.logger, f'Clean Accuracy after task {task_id} is {clean_acc}')
        print_and_log(args.logger, f'Poison Accuracy after task {task_id} is {poison_acc}')

        task_res['clean_acc'] = clean_acc
        task_res['poison_acc'] = poison_acc
    
        results[f'task_{task_id}'] = task_res
    
    return results
    




def train_epoch(model: BaseModel, dataloader, epoch, device, args):
    model.net.train()
    avg_loss = 0
    count = 0 
    # pbar = tqdm.tqdm(dataloader, desc=f'Training Epoch: {epoch}')
    print_and_log(args.logger, f'Training Epoch: {epoch}')
    for inputs, targets, is_poisoned in dataloader:
        if args.debug_mode and count > 2:
            break
        
        if args.is_dat and not args.clean_training: # Dynamically adding trigger during training mixed set
            inputs, targets, is_poisoned = dynamically_add_trigger(inputs, targets, args)

        inputs, targets = inputs.to(device), targets.to(device)
        if is_poisoned is not None:
            is_poisoned = is_poisoned.to(device)
        
        loss = model.observe(inputs, targets, is_poisoned)
        avg_loss += loss 
        count += 1
        # pbar.set_postfix({'loss': loss})
    
    avg_loss = avg_loss/count

    if model.lr_scheduler is not None and args.opt_mode == 'none':
        if isinstance(model.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            model.lr_scheduler.step(avg_loss)
        else:
            model.lr_scheduler.step()
        
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