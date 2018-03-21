# -*- coding: utf-8 -*-
""" Pose dataset indexing. """
import pickle
import os

from PIL import Image
import torch
from torch.utils import data
import numpy as np


class PoseDataset(data.Dataset):
    """ Pose dataset indexing.

    Args:
        path (str): A path to dataset.
        input_transform (Transform): Transform to input.
        output_transform (Transform): Transform to output.
        transform (Transform): Transform to both input and target.
    """

    def __init__(self, pickle_path, input_transform=None, output_transform=None, transform=None):
        self.path = pickle_path
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.transform = transform
        # load dataset.
        self.images, self.target = self._load_dataset()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """ Returns the i-th example. """
        image = self._read_image(self.images[index])
        target = self.target[index]
        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.transform is not None:
            image, target = self.transform(image, target)
        if self.output_transform is not None:
            target = self.output_transform(target)
        return image, target

    def _load_dataset(self):
        images = []
        target = []
        with open(self.path, 'rb') as f:
            data_dict = pickle.load(f)
        cnt = 0
        for dir_path, info_list in data_dict.items():
            cnt += 1
            if cnt > 10:
                break
            if 'agejo' in dir_path:
                continue
            for info_dict in info_list:
                for key, val in info_dict.items():
                    images.append(
                            os.path.join(
                                dir_path, key 
                                )
                            )
                    target.append(val['target'])
        return images, target


    @staticmethod
    def _read_image(path):
        return Image.open(path).convert('HSV').resize((256, 256))
