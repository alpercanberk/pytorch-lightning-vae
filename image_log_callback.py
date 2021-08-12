from matplotlib.pyplot import imshow, figure
import numpy as np
from torchvision.utils import make_grid
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization
import pytorch_lightning as pl
import torch
from collections import namedtuple


custom_normalization = namedtuple('custom_normalization', ['mean', 'std'])

def binary_normalization():
    normalization = custom_normalization(mean=[.5], std=[.5])
    return normalization

class ImageSampler(pl.Callback):
    def __init__(self, dataset):
        super().__init__()

        self.img_size = None
        self.num_preds = 16
        self.normalization = None        

        if(dataset == 'mnist' or dataset == 'fashion_mnist'):
            self.normalization = binary_normalization()
        elif(dataset == 'cifar10'):
            self.normalization = cifar10_normalization()
        elif(dataset == 'imagenet'):
            self.normalization = imagenet_normalization()


    def on_train_epoch_end(self, trainer, pl_module):
        figure(figsize=(8, 3), dpi=300)

        # Z COMES FROM NORMAL(0, 1)
        rand_v = torch.rand((self.num_preds, pl_module.hparams.latent_dim), device=pl_module.device)
        
        sample_shape = (self.num_preds, pl_module.hparams.latent_dim)

        p = torch.distributions.Normal(torch.zeros(sample_shape), torch.ones(sample_shape))
        z = p.rsample()

        # SAMPLE IMAGES
        with torch.no_grad():
            pred = pl_module.decoder(z.to(pl_module.device)).cpu()

        # UNDO DATA NORMALIZATION
        mean, std = np.array(self.normalization.mean), np.array(self.normalization.std)
        img = make_grid(pred).permute(1, 2, 0).numpy() * std + mean


        # PLOT IMAGES
        trainer.logger.experiment.add_image('Renormalized Sample Images',torch.tensor(img).permute(2, 0, 1), global_step=trainer.global_step)