# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Callable, List, Optional, Union

import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchmetrics import AUROC, Accuracy, Precision
from torchvision import datasets, transforms

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.image.mnist.mnist_base import MnistBase
from benchmark_examples.autoattack.utils.dataset_utils import (
    create_custom_dataset_builder,
)
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.applications.sl_vgg_torch import VGGBase, VGGFuse
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.ml.nn.utils import TorchModel, metric_wrapper, optim_wrapper
from secretflow.utils.simulation.datasets import _CACHE_DIR

vgg_resize = 112
half_vgg_resize = vgg_resize // 2


def vgg_transform():
    """for better metrics."""
    return transforms.Compose(
        [
            transforms.Resize(
                (vgg_resize, vgg_resize)
            ),  # resice the pic into 112 * 112
            transforms.Grayscale(
                num_output_channels=3
            ),  # Convert single channel grayscale images to 3 channels
            transforms.ToTensor(),
        ]
    )


def root_dir():
    return _CACHE_DIR + "/mnist"


class MyMnistDataset(datasets.MNIST):
    def __init__(
        self,
        x,
        is_left: bool = True,
        has_label: int = 0,
        list_return: bool = False,
        indexes=None,
    ):
        """
        MNIST dataset for VGG16.
        Args:
            is_left: left part (0, 56), right part (56, 112).
            has_label: whether return label, '0' for return label, 1 for do not return label, -1 for return all label as -1.
            list_return: whether return data as a list.
        """
        self.is_left = is_left
        self.has_label = has_label
        self.list_return = list_return
        train = True if x[0] == 'train' else False
        super().__init__(
            root_dir(), train=train, transform=vgg_transform(), download=True
        )
        if indexes is not None:
            self.data = self.data[indexes]
            self.targets = np.array(self.targets)[indexes]

    def __getitem__(self, item):
        data, label = super().__getitem__(item)
        data = (
            data[..., :half_vgg_resize] if self.is_left else data[..., half_vgg_resize:]
        )
        data = [data] if self.list_return else data
        label = label if self.has_label == 0 else -1
        if self.has_label != 1:
            return data, label
        else:
            return data


class AliceDataset(MyMnistDataset):
    def __init__(self, x):
        super().__init__(x, is_left=True, has_label=1, list_return=False)


class BobDataset(MyMnistDataset):
    def __init__(self, x):
        super().__init__(x, is_left=False, has_label=0, list_return=False)


class MnistVGG16(MnistBase):
    def __init__(self, config, alice, bob):
        super().__init__(
            config,
            alice,
            bob,
            hidden_size=4608,
            dnn_fuse_units_size=[512 * 3 * 3 * 2, 4096, 4096],
        )

    def prepare_data(self, **kwargs):
        self.alice_train_dataset = MyMnistDataset(
            ['train'], is_left=True, has_label=1, list_return=True
        )
        self.bob_train_dataset = MyMnistDataset(
            ['train'], is_left=False, has_label=1, list_return=True
        )
        sample_len = int(0.4 * len(self.alice_train_dataset))
        self.sample_alice_dataset, _ = random_split(
            self.alice_train_dataset,
            [sample_len, len(self.alice_train_dataset) - sample_len],
        )
        self.sample_bob_dataset, _ = random_split(
            self.bob_train_dataset,
            [sample_len, len(self.alice_train_dataset) - sample_len],
        )
        self.plain_train_label = datasets.MNIST(
            root_dir(), train=True, download=True
        ).targets
        self.plain_test_label = datasets.MNIST(
            root_dir(), train=False, download=True
        ).targets

    def train(
        self, callbacks: Optional[Union[List[Callback], Callback]] = None, **kwargs
    ):
        """VGG16 use dataset builder as the input, so we do not need to pass VDF into fit.
        So we rewrite the train function here."""
        base_model_dict = {
            self.alice: self.create_base_model_alice(),
            self.bob: self.create_base_model_bob(),
        }
        dataset_builder_dict = {
            self.alice: self.create_dataset_builder_alice(),
            self.bob: self.create_dataset_builder_bob(),
        }

        self.sl_model = SLModel(
            base_model_dict=base_model_dict,
            device_y=self.device_y,
            model_fuse=self.create_fuse_model(),
            backend='torch',
            num_gpus=0.001 if global_config.is_use_gpu() else 0,
        )
        shuffle = kwargs.get('shuffle', False)
        history = self.sl_model.fit(
            {self.alice: "train", self.bob: "train"},
            "train",
            validation_data=({self.alice: "test", self.bob: "test"}, "test"),
            epochs=self.epoch,
            batch_size=self.train_batch_size,
            shuffle=shuffle,
            verbose=1,
            validation_freq=1,
            dataset_builder=dataset_builder_dict,
            callbacks=callbacks,
        )
        logging.warning(
            f"RESULT: {type(self).__name__} {type(callbacks).__name__} training history = {history}"
        )
        return history

    def _predict(
        self, callbacks: Optional[Union[List[Callback], Callback]] = None, **kwargs
    ):
        dataset_builder_dict = {
            self.alice: self.create_predict_dataset_builder_alice(),
            self.bob: self.create_predict_dataset_builder_bob(),
        }
        if dataset_builder_dict[self.alice] is None:
            dataset_builder_dict = None
        return self.sl_model.predict(
            {self.alice: 'test', self.bob: 'test'},
            self.train_batch_size,
            dataset_builder=dataset_builder_dict,
            callbacks=callbacks,
        )

    def _create_base_model(self):
        return TorchModel(
            model_fn=VGGBase,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(AUROC, task="multiclass", num_classes=10),
            ],
            input_channels=3,
        )

    def create_base_model_alice(self):
        return self._create_base_model()

    def create_base_model_bob(self):
        return self._create_base_model()

    def create_fuse_model(self):
        return TorchModel(
            model_fn=VGGFuse,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(AUROC, task="multiclass", num_classes=10),
            ],
            dnn_units_size=self.dnn_fuse_units_size,
        )

    def create_dataset_builder_alice(self, *args, **kwargs) -> Optional[Callable]:
        return create_custom_dataset_builder(AliceDataset, self.train_batch_size)

    def create_dataset_builder_bob(self, *args, **kwargs) -> Optional[Callable]:
        return create_custom_dataset_builder(BobDataset, self.train_batch_size)

    def create_predict_dataset_builder_alice(
        self, *args, **kwargs
    ) -> Optional[Callable]:
        return create_custom_dataset_builder(AliceDataset, self.train_batch_size)

    def create_predict_dataset_builder_bob(self, *args, **kwargs) -> Optional[Callable]:
        return create_custom_dataset_builder(
            MyMnistDataset, self.train_batch_size, is_left=False, has_label=1
        )

    def alice_feature_nums_range(self) -> list:
        return [3 * vgg_resize * vgg_resize // 2]

    def hidden_size_range(self) -> list:
        return [4608]

    def dnn_fuse_units_size_range(self):
        return [
            [512 * 3 * 3 * 2, 4096],
            [512 * 3 * 3 * 2, 4096, 4096],
            [512 * 3 * 3 * 2, 4096, 4096, 4096],
            [512 * 3 * 3 * 2, 4096, 4096, 4096, 4096],
        ]

    def support_attacks(self):
        return ['lia', 'fia', 'replay', 'replace']

    def lia_auxiliary_model(self, ema=False):
        from benchmark_examples.autoattack.attacks.lia import BottomModelPlus

        bottom_model = VGGBase(input_channels=3)
        model = BottomModelPlus(bottom_model, size_bottom_out=self.hidden_size)

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    def fia_victim_mean_attr(self):
        loader = DataLoader(
            self.sample_alice_dataset, batch_size=len(self.sample_alice_dataset)
        )
        # self.sample_alice_dataset return a list (since list_return=True), so we need to get the [0].
        np_data = next(iter(loader))[0].numpy()
        return np_data.reshape((np_data.shape[0], -1)).mean(axis=0)

    def fia_victim_input_shape(self):
        return [3, vgg_resize, half_vgg_resize]

    def fia_attack_input_shape(self):
        return [3, vgg_resize, half_vgg_resize]

    def fia_auxiliary_data_builder(self):
        alice_train = self.sample_alice_dataset
        bob_train = self.sample_bob_dataset
        batch_size = self.train_batch_size

        def _prepare_data():
            alice_dataloader = DataLoader(
                dataset=alice_train, shuffle=False, batch_size=batch_size
            )
            bob_dataloader = DataLoader(
                dataset=bob_train, shuffle=False, batch_size=batch_size
            )
            dataloader_dict = {'alice': alice_dataloader, 'bob': bob_dataloader}
            return dataloader_dict, dataloader_dict

        return _prepare_data

    def lia_auxiliary_data_builder(self, batch_size=16, file_path=None):
        train_labeled_completed = BobDataset(['train'])
        train_labeled, _ = random_split(
            train_labeled_completed, [50, len(train_labeled_completed) - 50]
        )
        train_unlabeled = MyMnistDataset(['train'], is_left=False, has_label=-1)
        train_unlabeled, _ = random_split(
            train_unlabeled, [50, len(train_unlabeled) - 50]
        )
        test_labeled = BobDataset(['test'])

        def prepare_data():
            train_complete_trainloader = DataLoader(
                train_labeled_completed,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            train_labeled_dataloader = DataLoader(
                train_labeled,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            train_unlabeled_dataloader = DataLoader(
                train_unlabeled,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            test_loader = DataLoader(
                test_labeled,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            return (
                train_labeled_dataloader,
                train_unlabeled_dataloader,
                test_loader,
                train_complete_trainloader,
            )

        return prepare_data

    def replace_auxiliary_attack_configs(self, target_nums: int = 15):
        target_class = 8
        target_indexes = np.where(np.array(self.plain_train_label) == target_class)[0]
        target_set = np.random.choice(target_indexes, target_nums, replace=False)

        train_poison_set = np.random.choice(
            range(len(self.plain_train_label)), 100, replace=False
        )
        train_poison_dataset = MyMnistDataset(
            ['train'], is_left=True, has_label=1, indexes=train_poison_set
        )
        train_poison_dataloader = DataLoader(
            train_poison_dataset, batch_size=len(train_poison_set)
        )
        # train_poison_dataset return a tensor but not list, so do not use [0] to convert.
        train_poison_np = next(iter(train_poison_dataloader)).numpy()
        eval_poison_set = np.random.choice(
            range(len(self.plain_test_label)), 100, replace=False
        )
        return (
            target_class,
            target_set,
            train_poison_set,
            train_poison_np,
            eval_poison_set,
        )
