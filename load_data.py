from torchvision import datasets as dset
from torchvision import transforms, utils as vutils
import torch
import matplotlib.pyplot as plt
import numpy as np


image_size = 64

def load_data(path: str):
    """
    Load the image

    Parameters
    ----------
    path: str
        The path to the image

    Returns
    -------
    dataloader: torch.utils.data.DataLoader
        The dataloader for the image
    """

    dataset = dset.ImageFolder(root=path,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                               ]))
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")    
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True), (1,2,0)))