import os
import torch.utils.data
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class CHMNIST(torch.utils.data.Dataset):
    def __init__(self, root ='data/CHMNIST',train=True, download=True, transform = None):
        self.images = []
        self.root = root
        self.targets = []
        self.train = train
        self.download = download
        self.transform = transform

        x_train, x_test, y_train, y_test = self._train_test_split()

        if self.train:
            self._setup_dataset(x_train, y_train)
        else:
            self._setup_dataset(x_test, y_test)

    def _train_test_split(self):
        img_names = []
        img_label = []
        for i, folder_name in enumerate(os.listdir(self.root)):

            for j, img_name in enumerate(os.listdir(self.root + '/' +folder_name)):
                img_names.append(os.path.join(self.root+'/', folder_name, img_name))
                img_label.append(int(folder_name[0:2])-1)

        x_train,x_test, y_train, y_test = train_test_split(img_names, img_label, train_size=0.9,
                                                            random_state=1)

        return x_train, x_test, y_train, y_test

    def _setup_dataset(self, x, y):
            self.images = x
            self.targets = y

    def __getitem__(self, item):
        img_fn = self.images[item]
        label = self.targets[item]
        img = Image.open(img_fn)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)


class Purchase(torch.utils.data.Dataset):
    def __init__(self, root ='data/purchase/dataset_purchase',train=True, download=True, transform = None):
        self.images = []
        self.root = root
        self.targets = []
        self.train = train
        self.download = download
        self.transform = transform

        x_train, x_test, y_train, y_test = self._train_test_split()

        if self.train:
            self._setup_dataset(x_train, y_train)
        else:
            self._setup_dataset(x_test, y_test)

    def _train_test_split(self):
        df = pd.read_csv(self.root)

        img_names = df.iloc[:, 1:].to_numpy(dtype='f')
        img_label = df.iloc[:, 0].to_numpy()-1
        x_train,x_test, y_train, y_test = train_test_split(img_names, img_label, train_size=0.8,
                                                            random_state=1)


        return x_train, x_test, y_train, y_test

    def _setup_dataset(self, x, y):
            self.images = x
            self.targets = y

    def __getitem__(self, item):
        img = self.images[item]
        label = self.targets[item]
        return img, label
