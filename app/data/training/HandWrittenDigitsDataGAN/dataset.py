import torch
from torch import nn, optim
from torchvision import transforms, datasets

def mnist_data():
    compose = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Normalize([0.5],[0.5])
        ]
    )
    out_dir = "./dataset"
    
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
