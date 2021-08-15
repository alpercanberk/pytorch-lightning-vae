from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, MNISTDataModule, FashionMNISTDataModule
from collections import namedtuple

custom_normalization = namedtuple('custom_normalization', ['mean', 'std'])

def binary_normalization():
    normalization = custom_normalization(mean=[.5], std=[.5])
    return normalization


dataset_dict = {
	'mnist':(MNISTDataModule, binary_normalization),
	'fashion_mnist':(FashionMNISTDataModule, binary_normalization),
	'cifar10':(CIFAR10DataModule, cifar10_normalization),
	'imagenet':(ImagenetDataModule, imagenet_normalization)
}

def get_dataset(dataset_name, dataset_options):
	dataset = dataset_dict[dataset_name][0]
	normalization = dataset_dict[dataset_name][1]
	return dataset(**dataset_options), normalization()
