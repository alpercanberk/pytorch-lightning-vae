from .vae import VAE, Stack 
from .conv_vae import Conv_VAE 
from .resnet18_vae import Resnet18_VAE

__all__ = [
    'VAE', 'Stack'
    'Conv_VAE', 'Resnet18_VAE'
]

vae_models = {
    "conv-vae": Conv_VAE,
    "vae": VAE,
    'resnet18-vae': Resnet18_VAE
}