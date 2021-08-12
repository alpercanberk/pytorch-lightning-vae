
from pytorch_lightning import Trainer
from models import vae_models
from config import config
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os
from argparse import ArgumentParser
from image_log_callback import ImageSampler
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, MNISTDataModule, FashionMNISTDataModule

def make_model(config, input_shape):

    model_type = config.model_type
    model_config = config.model_config

    if model_type not in vae_models.keys():
        raise NotImplementedError("Model Architecture not implemented")
    else:
        return vae_models[model_type](**model_config.dict(), **input_shape)


if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=config.train_config.gpus)
    parser.add_argument('--dataset', type=str, default=config.model_config.dataset)
    args = parser.parse_args()

    dataset_config = {
        'data_dir':'./data',
        'normalize': True,
        'num_workers':12,
        'batch_size':config.model_config.batch_size
    }

    dataset_name = args.dataset
    if(dataset_name == 'mnist'):
        dataset = MNISTDataModule(**dataset_config)
    if(dataset_name == 'fashion_mnist'):
        dataset = FashionMNISTDataModule(**dataset_config)
    if(dataset_name == 'cifar10'):
        dataset = CIFAR10DataModule(**dataset_config)
    if(dataset_name == 'imagenet'):
        dataset = ImagenetDataModule(**dataset_config)

    print(">>>>>>")
    print("Training with dataset", dataset_name)
    print("Images from this dataset have dimension", dataset.dims)
    print(">>>>>>")

    input_shape = {'input_channels':dataset.dims[0], 'input_width':dataset.dims[1], 'input_height':dataset.dims[2]}

    model = make_model(config, input_shape)

    train_config = config.train_config
    logger = TensorBoardLogger(**config.log_config.dict())


    image_sampler = ImageSampler(dataset=dataset_name)
    lr_logger = LearningRateMonitor(logging_interval='epoch')

    train_config.gpus = args.gpus
    trainer = Trainer(**train_config.dict(), logger=logger,
                      callbacks=[lr_logger, image_sampler])
    

    trainer.fit(model, dataset)

    if not os.path.isdir("./saved_models"):
        os.mkdir("./saved_models")
    # trainer.save_checkpoint(
    #     f"saved_models/{config.model_type}_latent_{config.model_config.latent_dim}.ckpt")
    trainer.save_checkpoint(
        f"_latent_{config.model_config.latent_dim}.ckpt")


