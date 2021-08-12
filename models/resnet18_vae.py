import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
from typing import Optional

from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)

from .vae import VAE


class Resnet18_VAE(VAE):
    def __init__(self,
                 enc_out_dim:int, 
                 latent_dim:int,
                 input_height:int,
                 input_width:int,
                 input_channels:int,
                 lr: float,
                 batch_size: int,
                 save_path: Optional[str] = None, **kwargs):

        super().__init__(enc_out_dim, latent_dim, input_height, input_width, input_channels, lr, batch_size, save_path, **kwargs)
     
        self.latent_dim = latent_dim


        self.save_hyperparameters()
        self.lr = lr

        self.batch_size = batch_size

        assert input_height == input_width
        self.encoder = resnet18_encoder(first_conv=False, maxpool1=False)
        self.decoder = resnet18_decoder(latent_dim=latent_dim, input_height=input_height, first_conv=False, maxpool1=False)

        self.hidden2mu = nn.Linear(enc_out_dim, latent_dim)
        self.hidden2log_var = nn.Linear(enc_out_dim, latent_dim)

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))