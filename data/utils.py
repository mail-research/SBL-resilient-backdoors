import torch
import torch.nn as nn


# FROM BackdoorBox WaNet code
def gen_grid(height=32, k=4):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid

import torchvision.utils as vutils
import os 
def save_batches(dataloader, directory):
    os.makedirs(directory, exist_ok=True)
    for i, (batch, _, _) in enumerate(dataloader):
        # 'batch' will be a Tensor of shape [batch_size, channels, height, width]
        # We'll save each batch as a grid for visualization purposes.
        # However, you can also save individual images from each batch if you want.

        grid = vutils.make_grid(batch, padding=2, normalize=True)
        vutils.save_image(grid, f"{directory}/batch_{i}.png")
        if i > 2: 
            exit()

