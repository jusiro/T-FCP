import os
import collections.abc
import copy
import json
import random

import numpy as np

from PIL import Image
from torch.utils.data import Subset
from torch.utils.data import Dataset as _TorchDataset

from local_data.constants import PATH_DATASETS

template = ['a photo of a {}.']


class Dataset(_TorchDataset):

    def __init__(self, transform=None, partition=None, shots="all", seed=0):

        # Transform to apply
        self.transform = transform
        self.partition = partition

        # Set image directories
        root = os.path.abspath(PATH_DATASETS)
        self.dataset_dir = os.path.join(root, "caltech-101")
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")

        # Read images paths
        self.data = self.read_split(self.split_path, self.image_dir, partition=self.partition)

        # Produce classnames
        self.classnames = []
        for i in range(len(np.unique([idata['label'] for idata in self.data]))):
            idx = np.argwhere([idata['label'] == i for idata in self.data])[0][0]
            self.classnames.append(self.data[idx]["classname"])

        # Creating a few-shot dataset for training
        random.seed(seed)
        if (shots != "all") and (partition == "train"):
            data_fs = []
            labels = np.array([sample["label"] for sample in self.data])
            for ilabel in range(len(self.classnames)):
                indexes = list(np.squeeze(np.argwhere(labels == ilabel)))
                indexes_fs = random.sample(indexes, int(shots))
                [data_fs.append(self.data[i]) for i in indexes_fs]
            self.data = data_fs

        # Prepare class templates
        self.templates = template
        self.indexes = list(range(len(self.classnames)))
        print("Caltech101 Num of samples: " + str(len(self.data)), end="\n")

    def __len__(self):
        return len(self.data)

    def _transform(self, index):
        # Retrieve sample
        data_index = copy.deepcopy(self.data[index])
        # Read image
        img = Image.open(data_index["impath"])
        # Transform image
        img = self.transform(img)
        # Include image
        data_index["img"] = img
        return data_index

    def __getitem__(self, index):
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)

        return self._transform(index)

    @staticmethod
    def read_split(filepath, path_prefix, partition):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                out.append({"impath": impath, "label": int(label), "classname": classname})
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        if partition is None:
            train = _convert(split["train"])
            val = _convert(split["val"])
            test = _convert(split["test"])
            return train, val, test
        else:
            return _convert(split[partition])


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj