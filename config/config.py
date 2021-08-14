from pydantic import BaseModel
from typing import Optional, Union
import yaml


class TrainConfig(BaseModel):
    max_epochs: int
    auto_lr_find: Union[bool, int]
    gpus: int

class VAEConfig(BaseModel):
    model_type: str
    latent_dim:int
    dataset: str
    batch_size: int
    lr: float
    
class LoggerConfig(BaseModel):
    name: str
    save_dir: str

class Config(BaseModel):
    model_config: Union[VAEConfig]
    train_config: TrainConfig
    log_config: LoggerConfig


def load_config(path="config.yaml"):
    config = yaml.load(open(path), yaml.SafeLoader)
    model_type = config['model_params']['model_type']
    if model_type == "conv-vae" or model_type == "resnet18-vae" or model_type=="vae":
        model_config = VAEConfig(**config["model_params"])
    else:
        raise NotImplementedError(f"Model {model_type} is not implemented")
    train_config = TrainConfig(**config["training_params"])
    log_config = LoggerConfig(**config["logger_params"])
    config = Config(model_config=model_config, train_config=train_config,
                   log_config=log_config)

    return config


config = load_config()