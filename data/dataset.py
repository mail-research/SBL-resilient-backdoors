"""
This source code is adapted from https://github.com/RJ-T/NIPS2022_EP_BNP/blob/main/data.py
"""

import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image

from data.utils import *

class CelebA_attr(torch.utils.data.Dataset):
    def __init__(self, root, split, transform):
        self.dataset = torchvision.datasets.CelebA(root=root, split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        self.transforms = transform
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)


class ImageNet(torch.utils.data.Dataset):
    def __init__(self, root, split, transform):
        self.split = split
        self.dataset = torchvision.datasets.ImageFolder(root=os.path.join(root, "imagenet10", self.split))
        self.transforms = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        return (input, target)
    

class TensorsDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor=None, transforms=None, target_transforms=None, poisoned_indices=None):
        if target_tensor is not None:
            assert data_tensor.size(0) == target_tensor.size(0)
        self.data = data_tensor
        self.labels = target_tensor

        if transforms is None:
            transforms = []
        if target_transforms is None:
            target_transforms = []

        self.is_poisoned = torch.zeros_like(self.labels)
        if poisoned_indices is not None:
            self.is_poisoned[poisoned_indices] = 1

        if not isinstance(transforms, list):
            transforms = [transforms]
        if not isinstance(target_transforms, list):
            target_transforms = [target_transforms]

        self.transform = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):

        data_tensor = self.data[index]
        for transform in self.transform:
            data_tensor = transform(data_tensor)

        if self.labels is None:
            return data_tensor

        target_tensor = self.labels[index]
        for transform in self.target_transforms:
            target_tensor = transform(target_tensor)

        return data_tensor, target_tensor, self.is_poisoned[index]

    def __len__(self):
        return self.data.size(0)

def get_data_tensor(dataset):
    # Shuffle the dataset with manual seed
    dataset = torch.utils.data.random_split(dataset, [len(dataset), 0], generator=torch.Generator().manual_seed(0))[0]

    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    img_list = []
    tgt_list = []
    for img, tgt in loader:
        img_list.append(img)
        tgt_list.append(tgt)
    return torch.cat(img_list), torch.cat(tgt_list).long()


def get_random_indices(num_indices, length):
    random_number = torch.randn(length)
    return random_number.topk(num_indices)[1], (-random_number).topk(length-num_indices)[1]


def get_poisoned_indices(labels, num_poisoned, args):
    if args.poisoning_method in ['badnet', 'blended', 'wanet', 'sig']:
        all_indices = torch.arange(labels.shape[0])
        poisoned_indices, clean_indices = get_random_indices(num_poisoned, labels.shape[0])
        return poisoned_indices, clean_indices
    elif args.poisoning_method in ['cla']:
        class_indices = torch.where(labels == args.target_label)[0]
        random_indices, other_indices = get_random_indices(num_poisoned, class_indices.shape[0])
        clean_indices = torch.cat([torch.where(labels != args.target_label)[0], class_indices[other_indices]])
        poisoned_indices = class_indices[random_indices]
        return poisoned_indices, clean_indices


def add_trigger(images, labels, num_classes, trigger, train, args):
    if args.poisoning_method == 'badnet':
        trigger_size = trigger.shape[-1]
        trigger = trigger.reshape(1, 3, trigger_size, trigger_size)
        trigger_images = images.clone()
        trigger_images[:, :, -trigger_size:, -trigger_size:] = trigger
        return trigger_images

    elif args.poisoning_method == 'blended':
        trigger_size = trigger.shape[-1]

        trigger = Image.open('./data/hello_kitty.jpeg')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((args.input_size, args.input_size))
        ])
        trigger = transform(trigger)

        trigger_images = images.clone()
        trigger_images = trigger_images * 0.8 + trigger * 0.2
        return trigger_images
    
    elif args.poisoning_method == 'wanet':
        s = 0.5
        grid_rescale = 1
        noise_rescale = 2
        h = args.identity_grid.shape[2]
        grid = args.identity_grid + s * args.noise_grid / h
        grid = torch.clamp(grid * grid_rescale, -1, 1)
        if args.noise:
            ins = torch.rand(1, h, h, 2) * noise_rescale - 1  # [-1, 1]
            grid = grid + ins / h
            grid = torch.clamp(grid + ins / h, -1, 1)

        trigger_images = images.clone()
        bs = trigger_images.shape[0]
        trigger_images = nn.functional.grid_sample(trigger_images, grid.repeat((bs, 1, 1, 1)), align_corners=True) 
        return trigger_images
    
    elif args.poisoning_method == 'sig':
        input_size = args.input_size
        delta = 20
        f = 6
        pattern = torch.zeros((3, input_size, input_size))
        m = pattern.shape[1]
        for i in range(int(input_size)):
            for j in range(int(input_size)):
                pattern[:, i, j] = delta * np.sin(2 * np.pi * j * f / m) / 255

        trigger_images = images.clone()
        trigger_images = (trigger_images + pattern).clamp(0, 1)
        return trigger_images
        
    else:
        raise



def get_dataloader(args, trigger=None):
    if args.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root='../dat', train=True, download=True, transform=transforms.ToTensor())
        test_set = torchvision.datasets.CIFAR10(root='../dat', train=False, download=True, transform=transforms.ToTensor())
    elif args.dataset == 'stl10':
        train_set = torchvision.datasets.STL10(root='../dat', split='train', download=True, transform=transforms.ToTensor())
        test_set = torchvision.datasets.STL10(root='../dat', split='test', download=True, transform=transforms.ToTensor())
    elif args.dataset == 'gtsrb':
        train_set = torchvision.datasets.GTSRB(root='../dat', split='train', download=True, 
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(), 
                                                    transforms.Resize((32,32)),
                                                    ])
                                                )
        test_set = torchvision.datasets.GTSRB(root='../dat', split='test', download=True, 
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(), 
                                                    transforms.Resize((32,32)),
                                                    ])
                                                )
    elif args.dataset == 'celebA':
        train_set = CelebA_attr(root='../dat', split='train', transform=transforms.Compose([
                                                                    transforms.ToTensor(), 
                                                                    transforms.Resize((64,64)),
                                                                    ])
                                )
        test_set = CelebA_attr(root='../dat', split='test', transform=transforms.Compose([
                                                                    transforms.ToTensor(), 
                                                                    transforms.Resize((64,64)),
                                                                    ])
                                )
        print(f'Train size {len(train_set)}')
        print(f'Test size: {len(test_set)}')
    
    elif args.dataset == 'imagenet10':
        img_size = args.input_size
        train_set = ImageNet(root='../dat', split='train', transform=transforms.Compose([
                                                                    transforms.ToTensor(), 
                                                                    transforms.Resize((img_size,img_size)),
                                                                    ])
                            )
        test_set = ImageNet(root='../dat', split='val', transform=transforms.Compose([
                                                                    transforms.ToTensor(), 
                                                                    transforms.Resize((img_size,img_size)),
                                                                    ])
                            )
        print(f'Train size {len(train_set)}')
        print(f'Test size: {len(test_set)}')
    else:
        raise

    train_img, train_tgt = get_data_tensor(train_set)
    test_img, test_tgt = get_data_tensor(test_set)

    args.input_size = train_img.shape[2]

    if args.dataset == 'gtsrb':
        transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop((train_img.shape[2], train_img.shape[3]), padding=train_img.shape[3]//8),
        transforms.ToTensor(),
        # transforms.Normalize(train_img.mean([0, 2, 3]), train_img.std([0, 2, 3])),
    ])
    elif args.dataset == 'imagenet10':
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((train_img.shape[2], train_img.shape[3]), padding=8),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            # transforms.Normalize(train_img.mean([0, 2, 3]), train_img.std([0, 2, 3])),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((train_img.shape[2], train_img.shape[3]), padding=train_img.shape[3]//8),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            # transforms.Normalize(train_img.mean([0, 2, 3]), train_img.std([0, 2, 3])),
        ])

    # transform_test = transforms.Normalize(train_img.mean([0, 2, 3]), train_img.std([0, 2, 3]))
    transform_test = None

    if trigger is None:
        trigger = torch.randn(3, args.trigger_size, args.trigger_size).sigmoid()
        args.trigger = trigger
    
    if args.poisoning_method == 'wanet':
        args.identity_grid, args.noise_grid = gen_grid(train_img.shape[2], 4)
        args.noise = False

    if args.cl_method == 'joint':
        task_portion = [args.task_portion[0], 1 - args.task_portion[0]] # [0.05 0.95]
        num_poisoned = int(args.poisoning_rate * len(train_set))
        num_finetune = int(task_portion[0] * len(train_set)) # 0.05
        num_mixed = int(task_portion[1] * len(train_set)) # 0.95

        #                       |---------------------------------mixed----------------------------------|
        # Cut training set into |----poisoned----|--------------------------clean------------------------|--finetune--|
        poisoned_indices, clean_indices = get_poisoned_indices(train_tgt[:num_mixed], num_poisoned, args)
        poisoned_images = add_trigger(train_img[poisoned_indices], train_tgt[poisoned_indices], train_tgt.max()+1, trigger, True, args)
        poisoned_targets = torch.ones_like(train_tgt[poisoned_indices]) * args.target_label
        mixed_images = torch.cat((poisoned_images, train_img[clean_indices]))
        mixed_targets = torch.cat((poisoned_targets, train_tgt[clean_indices]))
        mixed_dataset = TensorsDataset(mixed_images, mixed_targets, transforms=transform_train, poisoned_indices=poisoned_indices)
        mixed_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        clean_loader = None
        clean_dataset = None

        finetune_images = train_img[len(train_set)-num_finetune:]
        finetune_targets = train_tgt[len(train_set)-num_finetune:]
        finetune_dataset = TensorsDataset(finetune_images, finetune_targets, transforms=transform_train)
        finetune_loader = torch.utils.data.DataLoader(finetune_dataset, batch_size=int(args.batch_size/2), shuffle=True, num_workers=args.num_workers)

    else:
        num_poisoned = int(args.poisoning_rate * len(train_set))
        num_mixed = int(args.task_portion[2] * len(train_set))
        num_finetune = int(args.task_portion[0] * len(train_set))
        num_clean = int(args.task_portion[1] * len(train_set))

        #                       |---------mixed----------|
        # Cut training set into |----poisoned----|-clean-|----clean----|--finetune--|
        poisoned_indices, clean_indices = get_poisoned_indices(train_tgt[:num_mixed], num_poisoned, args)
        poisoned_images = add_trigger(train_img[poisoned_indices], train_tgt[poisoned_indices], train_tgt.max()+1, trigger, True, args)
        poisoned_targets = torch.ones_like(train_tgt[poisoned_indices]) * args.target_label
        mixed_images = torch.cat((poisoned_images, train_img[clean_indices]))
        mixed_targets = torch.cat((poisoned_targets, train_tgt[clean_indices]))
        mixed_dataset = TensorsDataset(mixed_images, mixed_targets, transforms=transform_train, poisoned_indices=poisoned_indices)
        mixed_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=int(args.batch_size/2), shuffle=True, num_workers=args.num_workers)

        clean_images = train_img[num_mixed:len(train_set)-num_finetune]
        clean_targets = train_tgt[num_mixed:len(train_set)-num_finetune]
        clean_dataset = TensorsDataset(clean_images, clean_targets, transforms=transform_train)
        clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        finetune_images = train_img[len(train_set)-num_finetune:]
        finetune_targets = train_tgt[len(train_set)-num_finetune:]
        finetune_dataset = TensorsDataset(finetune_images, finetune_targets, transforms=transform_train)
        finetune_loader = torch.utils.data.DataLoader(finetune_dataset, batch_size=int(args.batch_size/2), shuffle=True, num_workers=args.num_workers)


    # # Save batch to check
    # save_batches(mixed_loader, f'./checkpoints/visualized_poisoned_data/{args.dataset}_{args.poisoning_method}/')
    # exit()


    test_poisoned_images = add_trigger(test_img, None, train_tgt.max()+1, trigger, False, args)
    test_poisoned_images = test_poisoned_images[test_tgt!=args.target_label]
    test_poisoned_targets = test_tgt[test_tgt!=args.target_label]
    test_poisoned_targets = torch.ones_like(test_poisoned_targets) * args.target_label
    test_poisoned_dataset = TensorsDataset(test_poisoned_images, test_poisoned_targets, transforms=transform_test)
    test_poisoned_loader = torch.utils.data.DataLoader(test_poisoned_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    test_clean_dataset = TensorsDataset(test_img, test_tgt, transforms=transform_test)
    test_clean_loader = torch.utils.data.DataLoader(test_clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    
    return train_tgt.max()+1, mixed_loader, clean_loader, finetune_loader, test_clean_loader, test_poisoned_loader, trigger




### Dynamically Adding Trigger (DAT) Case

def dynamically_add_trigger(images, labels, args):
    B, C, H, W = images.size()
    assert(C == 3)
    # print("Dynamically adding trigger")
    
    index = []
    init_index = []
    is_poisoned = (np.random.random(B) <= args.poisoning_rate)
    for i in range(B):
        if bool(is_poisoned[i]):
            index.append(i)
        else:
            init_index.append(i)
    labels[index] = args.target_label

    if args.poisoning_method == 'badnet':
        trigger_size = args.trigger.shape[-1]
        args.trigger = args.trigger.reshape(1, 3, trigger_size, trigger_size)
        # trigger_images = images.clone()
        images[index, :, -trigger_size:, -trigger_size:] = args.trigger
        return images, labels, torch.tensor(is_poisoned, dtype=torch.float)

    elif args.poisoning_method == 'blended':
        # print("Blended")
        trigger = Image.open('./data/hello_kitty.jpeg')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((H, W))
        ])
        trigger = transform(trigger)

        images[index] = images[index] * 0.8 + trigger * 0.2
        return images, labels, torch.tensor(is_poisoned, dtype=torch.float)
    
    elif args.poisoning_method == 'wanet':
        s = 0.5
        grid_rescale = 1
        noise_rescale = 2
        h = args.identity_grid.shape[2]
        grid = args.identity_grid + s * args.noise_grid / h
        grid = torch.clamp(grid * grid_rescale, -1, 1)
        if args.noise:
            ins = torch.rand(1, h, h, 2) * noise_rescale - 1  # [-1, 1]
            grid = grid + ins / h
            grid = torch.clamp(grid + ins / h, -1, 1)

        images[index] = nn.functional.grid_sample(images[index], grid.repeat((len(index), 1, 1, 1)), align_corners=True)  

        return images, labels, torch.tensor(is_poisoned, dtype=torch.float)

    elif args.poisoning_method == 'sig':
        input_size = H
        delta = 20
        f = 6
        pattern = torch.zeros((3, input_size, input_size))
        m = pattern.shape[1]
        for i in range(int(input_size)):
            for j in range(int(input_size)):
                pattern[:, i, j] = delta * np.sin(2 * np.pi * j * f / m) / 255

        images[index] = (images[index] + pattern).clamp(0, 1)

        return images, labels, torch.tensor(is_poisoned, dtype=torch.float)

    

    else:
        raise



def get_dat_dataloader(args, trigger=None):
    if args.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root='../dat', train=True, download=True, transform=transforms.ToTensor())
        test_set = torchvision.datasets.CIFAR10(root='../dat', train=False, download=True, transform=transforms.ToTensor())
    elif args.dataset == 'stl10':
        train_set = torchvision.datasets.STL10(root='../dat', split='train', download=True, transform=transforms.ToTensor())
        test_set = torchvision.datasets.STL10(root='../dat', split='test', download=True, transform=transforms.ToTensor())
    elif args.dataset == 'gtsrb':
        train_set = torchvision.datasets.GTSRB(root='../dat', split='train', download=True, 
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(), 
                                                    transforms.Resize((32,32)),
                                                    ])
                                                )
        test_set = torchvision.datasets.GTSRB(root='../dat', split='test', download=True, 
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(), 
                                                    transforms.Resize((32,32)),
                                                    ])
                                                )
    elif args.dataset == 'celebA':
        train_set = CelebA_attr(root='../dat', split='train', transform=transforms.Compose([
                                                                    transforms.ToTensor(), 
                                                                    transforms.Resize((64,64)),
                                                                    ])
                                )
        test_set = CelebA_attr(root='../dat', split='test', transform=transforms.Compose([
                                                                    transforms.ToTensor(), 
                                                                    transforms.Resize((64,64)),
                                                                    ])
                                )
        print(f'Train size {len(train_set)}')
        print(f'Test size: {len(test_set)}')
        
    elif args.dataset == 'imagenet10':
        img_size = args.input_size
        train_set = ImageNet(root='../dat', split='train', transform=transforms.Compose([
                                                                    transforms.ToTensor(), 
                                                                    transforms.Resize((img_size,img_size)),
                                                                    ])
                            )
        test_set = ImageNet(root='../dat', split='val', transform=transforms.Compose([
                                                                    transforms.ToTensor(), 
                                                                    transforms.Resize((img_size,img_size)),
                                                                    ])
                            )
        print(f'Train size {len(train_set)}')
        print(f'Test size: {len(test_set)}')

    else:
        raise

    train_img, train_tgt = get_data_tensor(train_set)
    test_img, test_tgt = get_data_tensor(test_set)

    args.input_size = train_img.shape[2]

    if args.dataset == 'gtsrb':
        transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop((train_img.shape[2], train_img.shape[3]), padding=train_img.shape[3]//8),
        transforms.ToTensor(),
        # transforms.Normalize(train_img.mean([0, 2, 3]), train_img.std([0, 2, 3])),
    ])
    elif args.dataset == 'imagenet10':
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((train_img.shape[2], train_img.shape[3]), padding=8),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            # transforms.Normalize(train_img.mean([0, 2, 3]), train_img.std([0, 2, 3])),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((train_img.shape[2], train_img.shape[3]), padding=train_img.shape[3]//8),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            # transforms.Normalize(train_img.mean([0, 2, 3]), train_img.std([0, 2, 3])),
        ])


    # transform_test = transforms.Normalize(train_img.mean([0, 2, 3]), train_img.std([0, 2, 3]))
    transform_test = None

    if trigger is None:
        trigger = torch.randn(3, args.trigger_size, args.trigger_size).sigmoid()
        args.trigger = trigger
    
    if args.poisoning_method == 'wanet':
        args.identity_grid, args.noise_grid = gen_grid(train_img.shape[2], 4)
        args.noise = False

    num_data = len(train_set)
    if args.cl_method == 'joint':
        task_portion = [args.task_portion[0], 1 - args.task_portion[0]] # [0.05 0.95]
        num_poisoned = int(args.poisoning_rate * len(train_set))
        num_finetune = int(task_portion[0] * len(train_set)) # 0.05
        num_mixed = int(task_portion[1] * len(train_set)) # 0.95

        #                       |---------------------------------mixed----------------------------------|
        # Cut training set into |----poisoned----|--------------------------clean------------------------|--finetune--|
        n_data = num_mixed
        if args.data_mode == 0:
            n_data = num_mixed
        elif args.data_mode == 1:
            n_data = num_mixed 
        elif args.data_mode == 2:
            n_data = num_data
        else:
            print('Use separate sets')
            
        mixed_images = train_img[:n_data]
        mixed_targets = train_tgt[:n_data]

        # mixed_images = train_img[:num_mixed]
        # mixed_targets = train_tgt[:num_mixed]
        mixed_dataset = TensorsDataset(mixed_images, mixed_targets, transforms=transform_train)
        mixed_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        clean_loader = None
        clean_dataset = None

        finetune_images = train_img[len(train_set)-num_finetune:]
        finetune_targets = train_tgt[len(train_set)-num_finetune:]
        finetune_dataset = TensorsDataset(finetune_images, finetune_targets, transforms=transform_train)
        finetune_loader = torch.utils.data.DataLoader(finetune_dataset, batch_size=int(args.batch_size/2), shuffle=True, num_workers=args.num_workers)

    else:
        num_poisoned = int(args.poisoning_rate * len(train_set))
        num_mixed = int(args.task_portion[2] * len(train_set))
        num_finetune = int(args.task_portion[0] * len(train_set))
        num_clean = int(args.task_portion[1] * len(train_set))

        #                       |---------mixed----------|
        # Cut training set into |----poisoned----|-clean-|-----------------------clean-----------------------|--finetune--|
        n_data = num_mixed
        if args.data_mode == 0:
            n_data = num_mixed
        elif args.data_mode == 1:
            n_data = num_mixed + num_clean
        elif args.data_mode == 2:
            n_data = num_data
        else:
            print('Use separate sets')
            
        mixed_images = train_img[:n_data]
        mixed_targets = train_tgt[:n_data]
        # mixed_images = train_img[:num_mixed]
        # mixed_targets = train_tgt[:num_mixed]
        mixed_dataset = TensorsDataset(mixed_images, mixed_targets, transforms=transform_train)
        mixed_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=int(args.batch_size/2), shuffle=True, num_workers=args.num_workers)

        clean_images = train_img[num_mixed:len(train_set)-num_finetune]
        clean_targets = train_tgt[num_mixed:len(train_set)-num_finetune]
        clean_dataset = TensorsDataset(clean_images, clean_targets, transforms=transform_train)
        clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        finetune_images = train_img[len(train_set)-num_finetune:]
        finetune_targets = train_tgt[len(train_set)-num_finetune:]
        finetune_dataset = TensorsDataset(finetune_images, finetune_targets, transforms=transform_train)
        finetune_loader = torch.utils.data.DataLoader(finetune_dataset, batch_size=int(args.batch_size/2), shuffle=True, num_workers=args.num_workers)

    test_poisoned_images = add_trigger(test_img, None, train_tgt.max()+1, trigger, False, args)
    test_poisoned_images = test_poisoned_images[test_tgt!=args.target_label]
    test_poisoned_targets = test_tgt[test_tgt!=args.target_label]
    test_poisoned_targets = torch.ones_like(test_poisoned_targets) * args.target_label
    test_poisoned_dataset = TensorsDataset(test_poisoned_images, test_poisoned_targets, transforms=transform_test)
    test_poisoned_loader = torch.utils.data.DataLoader(test_poisoned_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    test_clean_dataset = TensorsDataset(test_img, test_tgt, transforms=transform_test)
    test_clean_loader = torch.utils.data.DataLoader(test_clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_tgt.max()+1, mixed_loader, clean_loader, finetune_loader, test_clean_loader, test_poisoned_loader, trigger



