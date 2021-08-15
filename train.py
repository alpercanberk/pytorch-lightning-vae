
from pytorch_lightning import Trainer
from models import vae_models
from config import config
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os
from argparse import ArgumentParser
from image_log_callback import ImageSampler
import torchvision.transforms as transforms

from get_dataset import get_dataset

"""
TODO:
- Add a feature to load model from checkpoint
- Add dataset options to config file
"""

def make_model(config, input_shape):

    model_type = config.model_config.model_type
    model_config = config.model_config

    if model_type not in vae_models.keys():
        raise NotImplementedError("Model Architecture not implemented")
    else:
        return vae_models[model_type](**model_config.dict(), **input_shape)


if __name__ == "__main__":

    parser = ArgumentParser()

    config_dict = config.dict()

    #Assuming no NoneTypes in config.yaml, this part creates
    #optional command arguments for every config parameter
    for config_key, config_value in config_dict.items():
        for key, value in config_value.items():
            parser.add_argument(f'--{key}', type=type(value), default=value)

    parser_args = parser.parse_args()

    for config_key, config_value in config_dict.items():
        for key, value in config_value.items():
            config.__dict__[config_key].__dict__[key] = parser_args.__dict__[key] 



    #Configure data modules
    #TODO: Add custom datamodule options
    dataset_options = {
        'data_dir':'./data',
        'normalize': True,
        'num_workers':12, #Feel free to chnage this value
        'batch_size':config.model_config.batch_size
    }
    dataset_name = config.model_config.dataset

    dataset, dataset_normalization = get_dataset(dataset_name, dataset_options)

    input_shape = {'input_channels':dataset.dims[0], 
                'input_width':dataset.dims[1], 'input_height':dataset.dims[2]}

    print(">>>>>>")
    print("Training with dataset", dataset_name)
    print("This dataset has normalization\n\tmean:", dataset_normalization.mean, "\n\tstd:", dataset_normalization.std)


    #Some models require a custom resize, normalization, or dataset augmentation,
    #which is accounted for in this section. custom transforms are implemented
    #as static methods inside models currently with only parameter "dataset_normalization"
    #in order to keep the previous normalization determined for the dataset while making 
    #the necessary customizations. For more info, check out get_dataset.py.

    custom_transform, transform_info = vae_models[config.model_config.model_type].custom_transform(dataset_normalization)

    if custom_transform != None:
        print("Custom transform detected", transform_info)

        dataset.train_transforms = custom_transform
        dataset.val_transforms = custom_transform
        dataset.test_transforms = custom_transform
 
        if transform_info.custom_resize != None:
            print(" Custom resize", transform_info.custom_resize)
            input_shape['input_width'] = transform_info.custom_resize
            input_shape['input_height'] = transform_info.custom_resize

        if transform_info.custom_normalize != None:
            #Haven't implemented this part yet
            pass

    print("Images for this training run have dimension", input_shape)

    print(">>>>>>")

    #Create the type of model specified with config params
    model = make_model(config, input_shape)

    logger = TensorBoardLogger(**config.log_config.dict())

    #Logs lr to Tensorboard every epoch
    lr_logger = LearningRateMonitor(logging_interval='epoch')

    #At the end of every training epoch, ImageSampler samples randomly from
    #VAE outputs and logs them to Tensorboard.
    image_sampler = ImageSampler(dataset_normalization)

    trainer = Trainer(**config.train_config.dict(), logger=logger,
                      callbacks=[lr_logger, image_sampler])

    #Pytorch lightning's automatic learning rate finder feature
    if config.train_config.auto_lr_find:
        lr_finder = trainer.tuner.lr_find(model, dataset)
        new_lr = lr_finder.suggestion()
        print(">>>>>>>>>")
        print("Learning Rate Chosen:", new_lr)
        print(">>>>>>>>>")
        model.lr = new_lr  

    trainer.fit(model, dataset)


    #Save model at the end of training
    if not os.path.isdir("./saved_models"):
        os.mkdir("./saved_models")
    trainer.save_checkpoint(
        f"saved_models/{config.model_type}_latent_{config.model_config.latent_dim}_{dataset_name}.ckpt")


