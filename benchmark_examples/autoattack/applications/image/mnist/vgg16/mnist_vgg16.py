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
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, Precision
from torchvision import datasets, transforms

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import ModelType
from benchmark_examples.autoattack.applications.image.mnist.mnist_base import MnistBase
from benchmark_examples.autoattack.utils.data_utils import (
    create_custom_dataset_builder,
    get_sample_indexes,
)
from benchmark_examples.autoattack.utils.resources import ResourceDict, ResourcesPack
from secretflow import reveal
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.applications.sl_vgg_torch import VGGBase, VGGFuse
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow.utils.simulation.datasets import _CACHE_DIR

vgg_resize = 112
half_vgg_resize = vgg_resize // 2
simple_sample_nums = 1000


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
        enable_label: int = 0,
        list_return: bool = False,
        indexes=None,
        **kwargs,
    ):
        """
        MNIST dataset for VGG16.
        Args:
            is_left: left part (0, 56), right part (56, 112).
            enable_label: whether return label, '0' for return label, 1 for do not return label, -1 for return all label as -1.
            list_return: whether return data as a list.
        """
        self.is_left = is_left
        self.enable_label = enable_label
        self.list_return = list_return
        train = True if x[0] == 'train' else False
        super().__init__(
            root_dir(), train=train, transform=vgg_transform(), download=True
        )
        if indexes is not None:
            self.data = self.data[indexes]
            self.targets = np.array(self.targets)[indexes]
        if global_config.is_simple_test():
            self.data = self.data[0:simple_sample_nums]
            self.targets = self.targets[0:simple_sample_nums]

    def __getitem__(self, item):
        data, label = super().__getitem__(item)
        data = (
            data[..., :half_vgg_resize] if self.is_left else data[..., half_vgg_resize:]
        )
        data = [data] if self.list_return else data
        label = label if self.enable_label == 0 else -1
        if self.enable_label != 1:
            return data, label
        else:
            return data


class AliceDataset(MyMnistDataset):
    def __init__(self, x):
        super().__init__(x, is_left=True, enable_label=1, list_return=False)


class BobDataset(MyMnistDataset):
    def __init__(self, x):
        super().__init__(x, is_left=False, enable_label=0, list_return=False)


class MnistVGG16(MnistBase):
    def __init__(self, alice, bob):
        super().__init__(
            alice,
            bob,
            has_custom_dataset=True,
            total_fea_nums=3 * 112 * 112,
            alice_fea_nums=3 * 112 * 56,
            hidden_size=4608,
            dnn_fuse_units_size=[512 * 3 * 3 * 2, 4096, 4096],
            epoch=1,
        )
        self.metrics = [
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average='micro'
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average='micro'
            ),
            metric_wrapper(AUROC, task="multiclass", num_classes=10),
        ]

    def prepare_data(self, **kwargs):
        raise RuntimeError("Mnist Vgg16 does not need to prepare data, please check.")

    def _train(
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

    def get_plain_train_alice_data(self):
        if self._plain_train_alice_data is not None:
            return self._plain_train_alice_data
        alice_train_dataset = MyMnistDataset(
            ['train'], is_left=True, enable_label=1, list_return=True
        )
        plain_alice_train_data_loader = DataLoader(
            alice_train_dataset,
            batch_size=len(alice_train_dataset),
        )
        self._plain_train_alice_data = next(iter(plain_alice_train_data_loader))[
            0
        ].numpy()
        return self.get_plain_train_alice_data()

    def get_plain_train_bob_data(self):
        if self._plain_train_bob_data is not None:
            return self._plain_train_bob_data
        bob_train_dataset = MyMnistDataset(
            ['train'], is_left=False, enable_label=1, list_return=True
        )
        plain_bob_train_data_loader = DataLoader(
            bob_train_dataset,
            batch_size=len(bob_train_dataset),
        )
        self._plain_train_bob_data = next(iter(plain_bob_train_data_loader))[0].numpy()
        return self.get_plain_train_bob_data()

    def get_plain_test_alice_data(self):
        if self._plain_test_alice_data is not None:
            return self._plain_test_alice_data
        alice_test_dataset = MyMnistDataset(
            ['test'], is_left=True, enable_label=1, list_return=True
        )
        plain_alice_test_data_loader = DataLoader(
            alice_test_dataset,
            batch_size=len(alice_test_dataset),
        )
        self._plain_test_alice_data = next(iter(plain_alice_test_data_loader))[
            0
        ].numpy()
        return self.get_plain_train_alice_data()

    def get_plain_test_bob_data(self):
        if self._plain_test_bob_data is not None:
            return self._plain_test_bob_data
        bob_test_dataset = MyMnistDataset(
            ['test'], is_left=False, enable_label=1, list_return=True
        )
        plain_bob_test_data_loader = DataLoader(
            bob_test_dataset,
            batch_size=len(bob_test_dataset),
        )
        self._plain_test_bob_data = next(iter(plain_bob_test_data_loader))[0].numpy()
        return self.get_plain_train_bob_data()

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

    def model_type(self) -> ModelType:
        return ModelType.VGG16

    def _create_base_model(self):
        return TorchModel(
            model_fn=VGGBase,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
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
            metrics=self.metrics,
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
            MyMnistDataset, self.train_batch_size, is_left=False, enable_label=1
        )

    def get_plain_train_label(self):
        if self._plain_train_label is not None:
            return self._plain_train_label
        _, train_label, _, _ = super().prepare_data()
        if global_config.is_simple_test():
            train_label = train_label[0:simple_sample_nums]
        self._plain_train_label = reveal(train_label.partitions[self.bob].data)
        return self.get_plain_train_label()

    def get_plain_test_label(self):
        if self._plain_test_label is not None:
            return self._plain_test_label
        _, _, _, test_label = super().prepare_data()
        if global_config.is_simple_test():
            test_label = test_label[0:simple_sample_nums]
        self._plain_test_label = reveal(test_label.partitions[self.bob].data)
        return self.get_plain_test_label()

    def get_device_f_train_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 1,
        **kwargs,
    ):
        indexes = get_sample_indexes(self.train_dataset_len, sample_size, frac, indexes)
        # bob
        return MyMnistDataset(
            ['train'],
            is_left=False,
            indexes=indexes,
            enable_label=enable_label,
            **kwargs,
        )

    def get_device_y_train_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 0,
        **kwargs,
    ):
        indexes = get_sample_indexes(self.train_dataset_len, sample_size, frac, indexes)
        # bob
        return MyMnistDataset(
            ['train'],
            is_left=True,
            indexes=indexes,
            enable_label=enable_label,
            **kwargs,
        )

    def get_device_f_test_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 1,
        **kwargs,
    ):
        indexes = get_sample_indexes(self.test_dataset_len, sample_size, frac, indexes)
        # bob
        return MyMnistDataset(
            ['test'],
            is_left=False,
            list_return=False,
            enable_label=enable_label,
            indexes=indexes,
            **kwargs,
        )

    def get_device_y_test_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 0,
        **kwargs,
    ):
        indexes = get_sample_indexes(self.test_dataset_len, sample_size, frac, indexes)
        # bob
        return MyMnistDataset(
            ['test'],
            is_left=True,
            list_return=False,
            indexes=indexes,
            enable_label=enable_label,
            **kwargs,
        )

    def get_device_y_input_shape(self):
        return [self.train_dataset_len, 3, vgg_resize, half_vgg_resize]

    def get_device_f_input_shape(self):
        return [self.train_dataset_len, 3, vgg_resize, half_vgg_resize]

    def resources_consumption(self) -> ResourcesPack:
        # 6414MiB
        return (
            ResourcesPack()
            .with_debug_resources(
                ResourceDict(
                    gpu_mem=7 * 1024 * 1024 * 1024, CPU=1, memory=3 * 1024 * 1024 * 1024
                )
            )
            .with_sim_resources(
                self.device_y.party,
                ResourceDict(
                    gpu_mem=7 * 1024 * 1024 * 1024, CPU=1, memory=3 * 1024 * 1024 * 1024
                ),
            )
            .with_sim_resources(
                self.device_f.party,
                ResourceDict(
                    gpu_mem=6 * 1024 * 1024 * 1024, CPU=1, memory=3 * 1024 * 1024 * 1024
                ),
            )
        )
