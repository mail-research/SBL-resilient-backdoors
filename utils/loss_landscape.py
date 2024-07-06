import copy
import re

import numpy as np
import torch




def test_model(net, dataloader, criterion, device, args):
    net.eval()
    correct = 0
    total = 0
    avg_loss = 0
    count = 0
    with torch.no_grad():
        
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            avg_loss += loss.item()
            count += 1
            if args.debug_mode:
                break

    accuracy = 100 * correct / total
    avg_loss = avg_loss / count if count != 0 else 0
    return accuracy, avg_loss

def cal_l1(model, gpu=False):
    l1_norm = torch.tensor(0.)
    l1_norm = l1_norm.cuda() if gpu else l1_norm

    for param in model.parameters():
        l1_norm += torch.norm(param, 1)

    return l1_norm


def cal_l2(model, gpu=False):
    l2_norm = torch.tensor(0.)
    l2_norm = l2_norm.cuda() if gpu else l2_norm

    for param in model.parameters():
        l2_norm += torch.norm(param)

    return l2_norm


def normalize_filter(bs, ws):
    bs = {k: v.float() for k, v in bs.items()}
    ws = {k: v.float() for k, v in ws.items()}

    norm_bs = {}
    for k in bs:
        ws_norm = torch.norm(ws[k], dim=0, keepdim=True)
        bs_norm = torch.norm(bs[k], dim=0, keepdim=True)
        norm_bs[k] = ws_norm / (bs_norm + 1e-7) * bs[k]

    return norm_bs


def ignore_bn(ws):
    ignored_ws = {}
    for k in ws:
        if len(ws[k].size()) < 2:
            ignored_ws[k] = torch.zeros(size=ws[k].size(), device=ws[k].device)
        else:
            ignored_ws[k] = ws[k]
    return ignored_ws


def ignore_running_stats(ws):
    return ignore_kw(ws, ["num_batches_tracked"])


def ignore_kw(ws, kws=None):
    kws = [] if kws is None else kws

    ignored_ws = {}
    for k in ws:
        if any([re.search(kw, k) for kw in kws]):
            ignored_ws[k] = torch.zeros(size=ws[k].size(), device=ws[k].device)
        else:
            ignored_ws[k] = ws[k]
    return ignored_ws


def rand_basis(ws, gpu=True):
    return {k: torch.randn(size=v.shape, device="cuda" if gpu else None) for k, v in ws.items()}


def create_bases(model, kws=None, gpu=True):
    kws = [] if kws is None else kws
    ws0 = copy.deepcopy(model.state_dict())
    bases = [rand_basis(ws0, gpu) for _ in range(2)]  # Use two bases
    bases = [normalize_filter(bs, ws0) for bs in bases]
    bases = [ignore_bn(bs) for bs in bases]
    bases = [ignore_kw(bs, kws) for bs in bases]

    return bases


def get_loss_landscape(model, clean_test_loader, poisoned_test_loader, criterion, device, args,
                    bases=None, kws=None, x_min=-1.0, x_max=1.0, n_x=11, y_min=-1.0, y_max=1.0, n_y=11):
    if device != 'cpu':
        gpu = True
    else:
        gpu = False
    model.to(device)
    model = copy.deepcopy(model)
    ws0 = copy.deepcopy(model.state_dict())
    kws = [] if kws is None else kws
    bases = create_bases(model, kws, gpu) if bases is None else bases
    xs = np.linspace(x_min, x_max, n_x)
    ys = np.linspace(y_min, y_max, n_y)
    ratio_grid = np.stack(np.meshgrid(xs, ys), axis=0).transpose((1, 2, 0))

    metrics_grid = {}
    for ratio in ratio_grid.reshape([-1, 2]):
        ws = copy.deepcopy(ws0)
        gs = [{k: r * bs[k] for k in bs} for r, bs in zip(ratio, bases)]
        gs = {k: torch.sum(torch.stack([g[k] for g in gs]), dim=0) + ws[k] for k in gs[0]}
        model.load_state_dict(gs)

        print("Grid: ", ratio, end=", ")
        # *metrics, cal_diag = tests.test(model, n_ff, dataset, transform=transform,
        #                                 cutoffs=cutoffs, bins=bins, verbose=verbose, period=period, gpu=gpu)
        clean_acc, clean_loss = test_model(model, clean_test_loader, criterion, device, args)
        poison_acc, poison_loss = test_model(model, poisoned_test_loader, criterion, device, args)
        metrics = (clean_acc, clean_loss, poison_acc, poison_loss)

        l1, l2 = cal_l1(model, gpu).item(), cal_l2(model, gpu).item()
        metrics_grid[tuple(ratio)] = (l1, l2, *metrics)

    return metrics_grid