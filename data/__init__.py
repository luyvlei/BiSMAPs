"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
from random import sample
import torch.utils.data
from torch.utils.data.sampler import Sampler
from data.base_dataset import BaseDataset
from augmentations import get_composed_augmentations
from data.DataProvider import DataProvider
import numbers


def find_dataset_using_name(name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = name + '_loader'
    for _name, cls in datasetlib.__dict__.items():
        if _name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(cfg):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(cfg)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, cfg):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.cfg = cfg

        cfg_source = cfg.data.source
        cfg_target = cfg.data.target

        source_train = find_dataset_using_name(cfg_source.name)
        augmentations = cfg.data.source.augmentations
        data_aug = get_composed_augmentations(augmentations)
        self.source_train = source_train(cfg_source, augmentations=data_aug)

        augmentations = cfg.data.target.augmentations
        data_aug = get_composed_augmentations(augmentations)
        target_train = find_dataset_using_name(cfg_target.name)
        self.target_train = target_train(cfg_target, augmentations=data_aug)

        source_train_sampler, target_train_sampler = None, None
        if cfg.training.ddp.status:
            source_train_sampler = torch.utils.data.distributed.DistributedSampler(self.source_train)
            target_train_sampler = torch.utils.data.distributed.DistributedSampler(self.target_train)

        ## create train loader
        self.source_train_loader = DataProvider(
            dataset=self.source_train,
            batch_size=cfg_source.batch_size,
            shuffle=cfg_source.shuffle if not source_train_sampler else False,
            num_workers=int(cfg.data.source.num_workers),
            sampler = source_train_sampler,
            drop_last=True,
            pin_memory=True,
        )
        self.target_train_loader = DataProvider(
            self.target_train,
            batch_size=cfg_target.batch_size,
            shuffle=cfg_target.shuffle if not target_train_sampler else False,
            num_workers=int(cfg.data.target.num_workers),
            sampler = target_train_sampler,
            drop_last=True,
            pin_memory=True,
        )

        # status == valid
        cfg_source_valid = cfg.data.source_valid
        self.source_valid = None
        self.source_valid_loader = None
        if cfg_source_valid != None:
            source_valid = find_dataset_using_name(cfg_source_valid.name)
            self.source_valid = source_valid(cfg_source_valid, augmentations=None)

            self.source_valid_loader = torch.utils.data.DataLoader(
                self.source_valid,
                batch_size=cfg_source_valid.batch_size,
                shuffle=cfg_source_valid.shuffle,
                num_workers=int(cfg.data.source_valid.num_workers),
                drop_last=True,
                pin_memory=True,
            )

        self.target_valid = None
        self.target_valid_loader = None
        cfg_target_valid = cfg.data.target_valid
        if cfg_target_valid != None:
            target_valid = find_dataset_using_name(cfg_target_valid.name)
            self.target_valid = target_valid(cfg_target_valid, augmentations=None)

            self.target_valid_loader = torch.utils.data.DataLoader(
                self.target_valid,
                batch_size=cfg_target_valid.batch_size,
                shuffle=cfg_target_valid.shuffle,
                num_workers=int(cfg.data.target_valid.num_workers),
                drop_last=False,
                pin_memory=True,
            )


    def load_data(self):
        return self

