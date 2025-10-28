import os
import time
from abc import abstractmethod

import torch
from os import path
from torch.utils.data import DataLoader, Dataset


class BaseModel():
    # def __init__(self, config, dataloader):
    #     """init model with basic input, which are from __init__(**kwargs) function in inherited class."""
    #     self.config = config
    #     self.phase = config['phase']
    #     self.device = config[self.phase]['device']
    #     self.batch_size = config[self.phase]['dataloader']["args"]['batch_size']
    #     self.epoch = config['train']['n_epoch']
    #     self.lr = config['train']['lr']
    #     self.is_dataset_paired = config['test']['dataset']['is_paired']
    #     self.dataloader = dataloader
    #     self.apply_post_processing = config['test']['apply_post_processing']
    #     self.model_path = config[self.phase]['model_path']
    #     self.model_name = config[self.phase]['model_name']
    #     self.output_images_path = config['test']['output_images_path']
    def __init__(
        self,
        train_phase: bool, 
        device: str,
        is_dataset_paired: bool,
        dataloader: DataLoader,
        apply_post_processing: bool,
        output_images_path: path,
        epoch: int,
        learning_rate: float,
        dataset: Dataset
    ):
        """init model with basic input, which are from __init__(**kwargs) function in inherited class."""
        self.phase: str = "train" if train_phase else "test"
        self.device: str = device
        self.is_dataset_paired: bool = is_dataset_paired
        self.dataloader: DataLoader = dataloader
        self.apply_post_processing: bool = apply_post_processing
        self.output_images_path: path = output_images_path
        self.epoch: int = epoch
        self.lr: float = learning_rate
        # Use dataset to access image and patch size for the 'generate_output_images', so they don't
        # have to be defined there again.
        self.dataset: Dataset = dataset

    def train(self):
        since = time.time()        
        self.train_step()
        time_elapsed = time.time() - since
        print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def test(self):
        self.test_step()

    @abstractmethod
    def train_step(self):
        raise NotImplementedError('You must specify how to train your model.')

    @abstractmethod
    def val_step(self):
        raise NotImplementedError('You must specify how to do validation on your model.')

    def save_model(self, model):
        """Saves the model's state dictionary."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        path = os.path.join(self.model_path, self.model_name)
        torch.save(model.state_dict(), path)