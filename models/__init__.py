from .vae import VAE, Stack 
from .resnet18_vae import Resnet18_VAE

__all__ = [
    'VAE', 'Stack'
    'Resnet18_VAE'
]

vae_models = {
    "vae": VAE,
    'resnet18-vae': Resnet18_VAE
}