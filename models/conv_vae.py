from .vae import VAE, Stack, CustomTransform

import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.optim import Adam

import os
from typing import Optional

from collections import namedtuple


#Stride 2 by default
def ConvBlock(in_channels, out_channels, kernel_size):
        return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

#Stride 2 by default
def DeconvBlock(in_channels, out_channels, kernel_size, last=False):
        if not last:   
            return nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
        return nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2),
                    nn.Tanh(),
                )

class Convolutional_VAE(VAE):
    def __init__(self,
                 latent_dim:int,
                 input_height:int,
                 input_width:int,
                 input_channels:int,
                 lr: float,
                 batch_size: int,
                 save_path: Optional[str] = None, **kwargs):

        super().__init__(latent_dim, input_height, input_width, input_channels, lr, batch_size, save_path, **kwargs)
        

        #Sorry user, you don't have control over this for now
        LATENT_DIM = 1024
        self.latent_dim = LATENT_DIM

        self.save_hyperparameters()
        self.lr = lr

        self.batch_size = batch_size
 
        assert input_height == input_width
        assert input_height==64 or input_height==96

        final_height = None
        if input_height == 64:
            final_height = 2
        else:
            final_height = 4

        self.encoder = nn.Sequential(
               ConvBlock(input_channels, 32, 4), 
               ConvBlock(32, 64, 4),
               ConvBlock(64, 128, 4),
               ConvBlock(128, 256, 4),
               nn.Flatten()
            )
        self.decoder = nn.Sequential(
               Stack(1024, 1, 1),
               DeconvBlock(1024, 128, 5),
               DeconvBlock(128, 64, 5),
               DeconvBlock(64, 32, 6),
               DeconvBlock(32, input_channels, 6, last=True)
            )

        self.hidden2mu = nn.Linear(256*final_height**2, 256*final_height**2)
        self.hidden2log_var = nn.Linear(256*final_height**2, 256*final_height**2)

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    @staticmethod
    def custom_transform(normalization):

        RESIZE = 64
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(RESIZE),
            transforms.Normalize(
                mean=normalization.mean,
                std=normalization.std
            )
        ]), CustomTransform(custom_resize=RESIZE, custom_normalize=None)
