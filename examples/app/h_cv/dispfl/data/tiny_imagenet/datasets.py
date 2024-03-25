# MIT License
#
# Copyright (c) 2022 Rong Dai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch.utils.data as data
import torchvision
from PIL import Image


class tiny(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(tiny, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.train = train  # training set or test set
        self.data: Any = []
        self.targets = []

        # if self.train:
        #     downloaded_list = self.train_list
        # else:
        #     downloaded_list = self.test_list

        # now load the picked numpy arrays

        root_dir = root + '/tiny-imagenet-200'
        self.identity = root_dir + "/tiny" + str(self.train) + ".pkl"
        if os.path.exists(self.identity):
            tmp_file = open(self.identity, 'rb')
            read_data = pickle.load(tmp_file)
            self.data = read_data[0]
            self.targets = read_data[1]
            tmp_file.close()
        else:
            trn_img_list, trn_lbl_list, tst_img_list, tst_lbl_list = [], [], [], []
            trn_file = os.path.join(root_dir, 'train_list.txt')
            tst_file = os.path.join(root_dir, 'val_list.txt')
            with open(trn_file) as f:
                line_list = f.readlines()
                for line in line_list:
                    img, lbl = line.strip().split()
                    trn_img_list.append(img)
                    trn_lbl_list.append(int(lbl))
            with open(tst_file) as f:
                line_list = f.readlines()
                for line in line_list:
                    img, lbl = line.strip().split()
                    tst_img_list.append(img)
                    tst_lbl_list.append(int(lbl))
            self.root_dir = root_dir
            if self.train:
                self.img_list = trn_img_list
                self.label_list = trn_lbl_list

            else:
                self.img_list = tst_img_list
                self.label_list = tst_lbl_list

            self.size = len(self.img_list)

            self.transform = transform
            # if self.train:
            #     tmp = DatasetFromDir(img_root=root_dir, img_list=trn_img_list, label_list=trn_lbl_list,
            #                           transformer=transform)
            # else:
            #     tmp = DatasetFromDir(img_root=root_dir, img_list=tst_img_list, label_list=tst_lbl_list,
            #                          transformer=transform)
            # self.data = tmp.img
            # self.img_id = tmp.img_id
            for index in range(self.size):
                img_name = self.img_list[index % self.size]
                # ********************
                img_path = os.path.join(self.root_dir, img_name)
                img_id = self.label_list[index % self.size]
                img_raw = Image.open(img_path).convert('RGB')
                self.data.append(np.asarray(img_raw))
                self.targets.append(img_id)
                if index % 1000 == 999:
                    print('Load PIL images ' + str(self.train) + ': No.', index)

            self.data = np.vstack(self.data).reshape(-1, 3, 64, 64)
            self.data = self.data.transpose((0, 2, 3, 1))
            tmp_file = open(self.identity, 'wb')
            pickle.dump([self.data, self.targets], tmp_file)
            tmp_file.close()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], self.targets[index]
        #
        # # doing this so that it is consistent with all other datasets
        # # to return a PIL Image
        # img = Image.fromarray(img)
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        #
        # return img, target
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


# class Dataset_myy(torch.utils.data.Dataset):
#
#     def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
#         self.name = dataset_name
#
#         if self.name == 'CIFAR10' or self.name == 'CIFAR100' or self.name == "tinyimagenet":
#             self.train = train
#             self.transform = transforms.Compose([transforms.ToTensor()])
#             # self.transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),transforms.ToTensor(), transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])
#
#             self.X_data = data_x
#             self.y_data = data_y
#             if not isinstance(data_y, bool):
#                 self.y_data = data_y.astype('float32')
#
#     def __len__(self):
#         return len(self.X_data)
#
#     def __getitem__(self, idx):
#
#         if self.name == 'tinyimagenet':
#             img = self.X_data[idx]
#             if self.train:
#                 img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img  # Horizontal flip
#                 if np.random.rand() > .5:
#                     # Random cropping
#                     pad = 8
#                     extended_img = np.zeros((3, 64 + pad * 2, 64 + pad * 2)).astype(np.float32)
#                     extended_img[:, pad:-pad, pad:-pad] = img
#                     dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
#                     img = extended_img[:, dim_1:dim_1 + 64, dim_2:dim_2 + 64]
#             img = np.moveaxis(img, 0, -1)
#             img = self.transform(img)
#             if isinstance(self.y_data, bool):
#                 return img
#             else:
#                 y = self.y_data[idx]
#                 return img, y


# class DatasetFromDir(data.Dataset):
#
#     def __init__(self, img_root, img_list, label_list, transformer):
#         super(DatasetFromDir, self).__init__()
#         self.root_dir = img_root
#         self.img_list = img_list
#         self.label_list = label_list
#         self.size = len(self.img_list)
#         self.transform = transformer
#
#     def __getitem__(self, index):
#         img_name = self.img_list[index % self.size]
#         # ********************
#         img_path = os.path.join(self.root_dir, img_name)
#         img_id = self.label_list[index % self.size]
#
#         img_raw = Image.open(img_path).convert('RGB')
#         img = self.transform(img_raw)
#         return img, img_id
#
#     def __len__(self):
#         return len(self.img_list)


class tiny_truncated(data.Dataset):
    def __init__(
        self,
        root,
        cache_data_set=None,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__(cache_data_set)

    def __build_truncated_dataset__(self, cache_data_set):
        # print("download = " + str(self.download))
        if cache_data_set == None:
            cifar_dataobj = tiny(
                self.root,
                self.train,
                self.transform,
                self.target_transform,
                self.download,
            )
        else:
            cifar_dataobj = cache_data_set

            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    # def truncate_channel(self, index):
    #     for i in range(index.shape[0]):
    #         gs_index = index[i]
    #         self.data[gs_index, :, :, 1] = 0.0
    #         self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
