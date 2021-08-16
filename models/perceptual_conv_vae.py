
from .perceptual_networks import AlexNet
from .conv_vae import Convolutional_VAE
from typing import Optional


import torch
import torch.nn as nn
import pytorch_lightning as pl
import random

class Perceptual_VAE(Convolutional_VAE):
	def __init__(self,
				latent_dim:int,
				input_height:int,
				input_width:int,
				input_channels:int,
				lr: float,
				batch_size: int,
				save_path: Optional[str] = None, **kwargs):

		super().__init__(latent_dim, input_height, input_width, input_channels, lr, batch_size, save_path, **kwargs)
		
		self.perceptual_net = AlexNet()

	def gaussian_likelihood(self, x_hat, logscale, x):

		x = self.perceptual_net(x)
		x_hat = self.perceptual_net(x_hat)

		scale = torch.exp(logscale)
		mean = x_hat
		dist = torch.distributions.Normal(mean, scale)

		# measure prob of seeing image under p(x|z)
		log_pxz = dist.log_prob(x)

		return log_pxz.sum(dim=(1, 2, 3))
