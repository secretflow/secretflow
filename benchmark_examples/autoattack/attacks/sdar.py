# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file references code of paper SDAR: Passive Inference Attacks on Split Learning via Adversarial Regularization (https://arxiv.org/abs/2310.10483)
"""
from typing import Dict
import logging
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10
import os
import torch.utils.data as torch_data
from torchvision import datasets, transforms
from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    DatasetType,
    InputMode,
    ModelType,
)
from torchmetrics import AUROC, Accuracy, Precision
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow_fl.ml.nn import SLModel
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow_fl.ml.nn.callbacks.attack import AttackCallback
from secretflow_fl.ml.nn.core.torch import TorchModel
from secretflow_fl.ml.nn.sl.attacks.sdar_torch import SDARAttack
from secretflow_fl.ml.nn.utils import optim_wrapper, loss_wrapper, metric_wrapper
from secretflow_fl.utils.simulation import datasets_fl
from secretflow.utils.errors import NotSupportedError

from secretflow.utils.simulation.datasets import (
    _DATASETS,
    get_dataset,
    _CACHE_DIR,
    unzip,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

INTERMIDIATE_SHAPE = lambda level: (
    (16, 32, 32) if level == 3 else (32, 16, 16) if level < 7 else (64, 8, 8)
)
LEVEL = 4  # depth of split learning


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.downsample + 1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)

        if downsample:
            self.downsampleconv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=2, bias=False
            )
            self.downsamplebn = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsampleconv(identity)
            identity = self.downsamplebn(identity)
        out += identity
        out = self.relu(out)
        return out


class FModel(nn.Module):
    def __init__(self, level, input_shape, width="standard"):
        super(FModel, self).__init__()

        if width not in ["narrow", "standard", "wide"]:
            raise ValueError(f'Width {width} is not supported.')
        model_width = {
            "narrow": [8, 16, 32],
            "standard": [16, 32, 64],
            "wide": [32, 64, 128],
        }
        self.widths = model_width[width]

        layer_config = {
            1: [self.widths[0], self.widths[0]],
            2: [self.widths[0], self.widths[0]],
            3: [self.widths[0], self.widths[0]],
            4: [self.widths[0], self.widths[1]],
            5: [self.widths[1], self.widths[1]],
            6: [self.widths[1], self.widths[1]],
            7: [self.widths[1], self.widths[2]],
            8: [self.widths[2], self.widths[2]],
            9: [self.widths[2], self.widths[2]],
        }

        if level < 3 or level > 9:
            raise NotImplementedError(f'Level {level} is not supported.')

        self.conv1 = nn.Conv2d(
            input_shape[0],
            self.widths[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.widths[0], momentum=0.9, eps=1e-5, affine=False)
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        for i in range(level, 0, -1):
            in_c, out_c = layer_config[i]
            if i == 0:
                in_c = input_shape[0]
            self.layers.insert(
                0, BasicBlock(in_c, out_c, True if i in [4, 7] else False)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for layer in self.layers:
            x = layer(x)

        return x

    def output_num(self):
        return 1


class GModel(nn.Module):
    def __init__(
        self, level, input_shape, num_classes=10, dropout=0.0, width="standard"
    ):
        super(GModel, self).__init__()

        if width not in ["narrow", "standard", "wide"]:
            raise ValueError(f'Width {width} is not supported.')
        model_width = {
            "narrow": [8, 16, 32],
            "standard": [16, 32, 64],
            "wide": [32, 64, 128],
        }
        if level < 3 or level > 9:
            raise NotImplementedError(f'Level {level} is not supported.')
        self.widths = model_width[width]
        layer_config = {
            3: [self.widths[0], self.widths[1]],
            4: [self.widths[1], self.widths[1]],
            5: [self.widths[1], self.widths[1]],
            6: [self.widths[1], self.widths[2]],
            7: [self.widths[2], self.widths[2]],
            8: [self.widths[2], self.widths[2]],
            9: [self.widths[2], self.widths[2]],
        }

        self.layers = nn.ModuleList()
        for i in range(8, level - 1, -1):
            in_c, out_c = layer_config[i]
            self.layers.insert(
                0,
                BasicBlock(
                    in_c if i != level else input_shape[0],
                    out_c,
                    True if i in [3, 6] else False,
                ),
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.widths[2], num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, inference=False):

        # if not inference:
        # x = x[0]
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x).squeeze(2).squeeze(2)
        x = self.fc(x)
        x = self.dropout(x)
        return x


class Decoder(nn.Module):
    def __init__(self, level, input_shape, num_classes, width="standard"):
        super(Decoder, self).__init__()

        model_width = {
            "narrow": [8, 16, 32],
            "standard": [16, 32, 64],
            "wide": [32, 64, 128],
        }

        if width not in {"narrow", "standard", "wide"}:
            raise ValueError("width must be one of {'narrow', 'standard', 'wide'}")

        self.widths = model_width[width]

        self.embedding = nn.Embedding(num_classes, 50)
        self.fc = nn.Linear(50, input_shape[1] * input_shape[2])
        self.fc_out_channels = input_shape[1] * input_shape[2]
        self.in_c = input_shape[0] + 1
        self.level = level
        self.layers = nn.ModuleList()
        layer_config = {
            1: (self._build_conv_block, [self.widths[0], self.widths[0]]),
            2: (self._build_conv_block, [self.widths[0], self.widths[0]]),
            3: (self._build_conv_block, [self.widths[1], self.widths[0]]),
            4: (self._upsample_block, [self.widths[1], self.widths[1]]),
            5: (self._build_conv_block, [self.widths[1], self.widths[1]]),
            6: (self._build_conv_block, [self.widths[2], self.widths[1]]),
            7: (self._upsample_block, [self.widths[2], self.widths[2]]),
            8: (self._build_conv_block, [self.widths[2], self.widths[2]]),
            9: (self._build_conv_block, [self.widths[2], self.widths[2]]),
        }
        for i in range(1, level + 1):
            mehtod, [in_c, out_c] = layer_config[i]
            self.layers = mehtod(self.in_c if i == level else in_c, out_c) + self.layers
        self.final_conv = nn.Conv2d(self.widths[0], 3, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def _build_conv_block(self, in_channels, out_channels):
        """Creates a Conv2D -> BatchNorm -> LeakyReLU block."""
        layers = nn.ModuleList()
        layers.append(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        layers.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5))
        layers.append(nn.ReLU())
        return layers

    def _upsample_block(self, in_channels, out_channels):
        """Creates an UpSampling2D -> Conv2D block."""
        layers = nn.ModuleList()
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        layers.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5))
        layers.append(nn.ReLU())
        return layers

    def forward(self, xin, yin=None):
        yin_embedding = self.embedding(yin)
        yin_embedding = self.fc(yin_embedding)
        yin_embedding = yin_embedding.view(-1, 1, xin.size(2), xin.size(3))
        xin = torch.cat([xin, yin_embedding], dim=1)
        x = xin

        for layer in self.layers:
            x = layer(x)

        x = self.final_conv(x)
        return self.sigmoid(x)


class SimulatorDiscriminator(nn.Module):
    def __init__(self, level, input_shape, num_classes, width="standard", bn=True):
        super(SimulatorDiscriminator, self).__init__()

        # Define the model width variations
        model_width = {
            "narrow": [16, 32, 64, 128],
            "standard": [32, 64, 128, 256],
            "wide": [64, 128, 256, 512],
        }

        if width not in {"narrow", "standard", "wide"}:
            raise ValueError("width must be one of {'narrow', 'standard', 'wide'}")

        self.widths = model_width[width]

        self.in_channels = input_shape[0] + 1

        self.embedding = nn.Embedding(num_classes, 50)
        self.fc = nn.Linear(50, input_shape[1] * input_shape[2])

        self.bn = bn

        self.conv_layers = nn.ModuleList()
        scale_fc = 32
        if level == 3:  # input_shape = (32, 32, 16)
            self.conv_layers.extend(self._build_level_3())
        elif level <= 6:
            self.conv_layers.extend(self._build_level_6())
        elif level <= 9:
            self.conv_layers.extend(self._build_level_9())

        self.fc1 = nn.Linear(
            self.widths[3] * input_shape[0] * input_shape[1] // scale_fc, 1
        )
        self.dropout = nn.Dropout(0.4)

    def _build_level_3(self):
        layers = nn.ModuleList()
        layers.extend(
            self._build_conv_block(self.in_channels, self.widths[0], stride=1)
        )
        layers.extend(
            self._build_conv_block(
                self.widths[0], self.widths[1], check_bn=True, stride=2
            )
        )
        layers.extend(
            self._build_conv_block(
                self.widths[1], self.widths[2], check_bn=True, stride=2
            )
        )
        layers.extend(
            self._build_conv_block(
                self.widths[2], self.widths[3], check_bn=True, stride=1
            )
        )
        layers.extend(
            self._build_conv_block(
                self.widths[3], self.widths[3], check_bn=True, stride=1
            )
        )
        layers.extend(
            self._build_conv_block(
                self.widths[3], self.widths[3], check_bn=True, stride=1
            )
        )
        layers.append(
            nn.Conv2d(
                self.widths[3], self.widths[3], kernel_size=3, stride=2, padding=1
            )
        )
        return layers

    def _build_level_6(self):
        layers = nn.ModuleList()
        layers.extend(
            self._build_conv_block(self.in_channels, self.widths[1], stride=1)
        )
        layers.extend(
            self._build_conv_block(
                self.widths[1], self.widths[2], check_bn=True, stride=2
            )
        )
        layers.extend(
            self._build_conv_block(
                self.widths[2], self.widths[3], check_bn=True, stride=1
            )
        )
        layers.extend(
            self._build_conv_block(
                self.widths[3], self.widths[3], check_bn=True, stride=1
            )
        )
        layers.extend(
            self._build_conv_block(
                self.widths[3], self.widths[3], check_bn=True, stride=1
            )
        )
        layers.append(
            nn.Conv2d(
                self.widths[3], self.widths[3], kernel_size=3, stride=2, padding=1
            )
        )
        return layers

    def _build_level_9(self):
        layers = nn.ModuleList()
        layers.extend(
            self._build_conv_block(
                self.in_channels, self.widths[2], check_bn=False, stride=1
            )
        )
        layers.extend(
            self._build_conv_block(
                self.widths[2], self.widths[3], check_bn=True, stride=1
            )
        )
        layers.extend(
            self._build_conv_block(
                self.widths[3], self.widths[3], check_bn=True, stride=1
            )
        )
        layers.extend(
            self._build_conv_block(
                self.widths[3], self.widths[3], check_bn=True, stride=1
            )
        )
        layers.append(
            nn.Conv2d(
                self.widths[3], self.widths[3], kernel_size=3, stride=2, padding=1
            )
        )
        return layers

    def _build_conv_block(self, in_channels, out_channels, check_bn=False, stride=1):
        """Creates a Conv2D -> BatchNorm -> LeakyReLU block."""
        layers = nn.ModuleList()
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            )
        )
        if check_bn and self.bn:
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5))
        layers.append(nn.LeakyReLU(0.2))
        return layers

    def forward(self, xin, yin=None):

        yin_embedding = self.embedding(yin)
        yin_embedding = self.fc(yin_embedding)
        yin_embedding = yin_embedding.view(-1, 1, xin.size(2), xin.size(3))
        xin = torch.cat([xin, yin_embedding], dim=1)

        x = xin

        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


class DecoderDiscriminator(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(DecoderDiscriminator, self).__init__()
        self.input_shape = input_shape

        self.embedding = nn.Embedding(num_classes, 50)
        self.fc = nn.Linear(50, input_shape[1] * input_shape[2])

        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0] + 1,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(256 * input_shape[1] // 8 * input_shape[2] // 8, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, xin, yin=None):
        yin_embedding = self.embedding(yin)
        yin_embedding = self.fc(yin_embedding)
        yin_embedding = yin_embedding.view(-1, 1, xin.size(2), xin.size(3))
        xin = torch.cat([xin, yin_embedding], dim=1)
        x = F.leaky_relu(self.conv1(xin), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)

        return x


def get_model(app: ApplicationBase):

    e_optim_fn = optim_wrapper(optim.Adam, lr=0.001, eps=1e-07)
    decoder_optim_fn = optim_wrapper(optim.Adam, lr=0.0005, eps=1e-07)
    simulator_d_optim_fn = optim_wrapper(optim.Adam, lr=4e-05, eps=1e-07)
    decoder_d_optim_fn = optim_wrapper(optim.Adam, lr=5e-09, eps=1e-07)

    if app.model_type() == ModelType.RESNET20 and app.dataset_name() in [
        "cifar10",
        "cifar100",
        "stl10",
    ]:
        num_classes = 100 if app.dataset_name() == "cifar100" else 10
        return (
            TorchModel(
                model_fn=FModel,
                loss_fn=None,
                optim_fn=e_optim_fn,
                metrics=None,
                level=LEVEL,
                input_shape=(3, 32, 32),
            ),
            TorchModel(
                model_fn=Decoder,
                loss_fn=None,
                optim_fn=decoder_optim_fn,
                metrics=None,
                level=LEVEL,
                input_shape=INTERMIDIATE_SHAPE(4),
                num_classes=num_classes,
            ),
            TorchModel(
                model_fn=SimulatorDiscriminator,
                loss_fn=None,
                optim_fn=simulator_d_optim_fn,
                metrics=None,
                level=LEVEL,
                input_shape=INTERMIDIATE_SHAPE(4),
                num_classes=num_classes,
            ),
            TorchModel(
                model_fn=DecoderDiscriminator,
                loss_fn=None,
                optim_fn=decoder_d_optim_fn,
                metrics=None,
                input_shape=(3, 32, 32),
                num_classes=num_classes,
            ),
        )
    else:
        raise NotSupportedError(
            f"SDAR attack not supported in dataset {app.dataset_name()} app {app.model_type()}! "
        )


def data_builder(device_f_dataset, batch_size):
    def prepare_data():
        len_aux_ds = len(device_f_dataset) // 2
        target_loader = DataLoader(
            dataset=device_f_dataset[:len_aux_ds], shuffle=False, batch_size=batch_size
        )
        aux_dataloader = DataLoader(
            dataset=device_f_dataset[len_aux_ds + 1 :],
            shuffle=False,
            batch_size=batch_size,
        )
        return target_loader, aux_dataloader

    return prepare_data


def inject_create_fuse_model(dataset_name):
    def create_fuse_model():
        g_optim_fn = optim_wrapper(optim.Adam, lr=0.001, eps=1e-07)
        loss_fn = nn.CrossEntropyLoss
        return TorchModel(
            model_fn=GModel,
            loss_fn=loss_fn,
            optim_fn=g_optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(AUROC, task="multiclass", num_classes=10),
            ],
            level=LEVEL,
            input_shape=INTERMIDIATE_SHAPE(LEVEL),
            num_classes=10 if dataset_name in ["cifar10", "stl10"] else 100,
        )

    return create_fuse_model


def inject_create_base_model():
    def create_base_model():
        f_optim_fn = optim_wrapper(optim.Adam, lr=0.001, eps=1e-07)
        return TorchModel(
            model_fn=FModel,
            optim_fn=f_optim_fn,
            level=LEVEL,
            input_shape=(3, 32, 32),
        )

    return create_base_model


def inject__train(instance):
    def _train(callbacks, **kwargs):
        base_model_dict = {
            instance.alice: instance.create_base_model_alice(),
        }
        instance.sl_model = SLModel(
            base_model_dict=base_model_dict,
            device_y=instance.device_y,
            model_fuse=instance.create_fuse_model(),
            dp_strategy_dict=None,
            compressor=None,
            simulation=True,
            random_seed=1234,
            backend='torch',
            strategy='split_nn',
            num_gpus=0.001 if global_config.is_use_gpu() else 0,
        )
        history = instance.sl_model.fit(
            instance.get_train_data(),
            instance.get_train_label(),
            validation_data=(instance.get_test_data(), instance.get_test_label()),
            epochs=2,
            batch_size=128,
            shuffle=False,
            random_seed=1234,
            dataset_builder=None,
            callbacks=callbacks,
        )

        pred_bs = 128
        result = instance.sl_model.predict(
            instance.get_train_data(), batch_size=pred_bs, verbose=1
        )
        logging.warning(
            f"RESULT: {type(instance).__name__} {type(callbacks).__name__} training history = {history}"
        )
        return history

    return _train


get_loader = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "stl10": STL10,
}


def inject_get_train_data(dataset_name, instance):
    def get_train_data():
        loader = get_loader[dataset_name]
        data_dir = os.path.join(_CACHE_DIR, dataset_name)
        train_dataset = loader(
            data_dir, True, transform=transforms.ToTensor(), download=True
        )
        train_loader = torch_data.DataLoader(
            dataset=train_dataset, batch_size=len(train_dataset), shuffle=False
        )
        train_data, train_labels = next(iter(train_loader))
        len_train_ds = len(train_data) // 2
        train_plain_data = train_data.numpy()[:len_train_ds]
        train_data = FedNdarray(
            partitions={
                instance.alice: instance.alice(lambda x: x)(train_plain_data),
                instance.bob: instance.bob(lambda x: x)(train_plain_data),
            },
            partition_way=PartitionWay.VERTICAL,
        )
        if global_config.is_simple_test():
            sample_nums = 4000
            train_data = train_data[0:sample_nums]
        return train_data

    return get_train_data


def inject_get_train_label(dataset_name, instance):
    def get_train_label():
        loader = get_loader[dataset_name]
        data_dir = os.path.join(_CACHE_DIR, dataset_name)
        train_dataset = loader(
            data_dir, True, transform=transforms.ToTensor(), download=True
        )
        train_loader = torch_data.DataLoader(
            dataset=train_dataset, batch_size=len(train_dataset), shuffle=False
        )
        train_data, train_labels = next(iter(train_loader))
        len_train_ds = len(train_labels) // 2
        train_plain_label = train_labels.numpy()[:len_train_ds]
        train_label = instance.bob(lambda x: x)(train_plain_label)
        if global_config.is_simple_test():
            sample_nums = 4000
            train_label = train_label.device(lambda df: df[0:sample_nums])(train_label)
        return train_label

    return get_train_label


def inject_get_test_data(dataset_name, instance):
    def get_test_data():
        loader = get_loader[dataset_name]
        data_dir = os.path.join(_CACHE_DIR, dataset_name)
        test_dataset = loader(
            data_dir, False, transform=transforms.ToTensor(), download=True
        )
        test_loader = torch_data.DataLoader(
            dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
        )
        test_data, test_labels = next(iter(test_loader))
        len_test_ds = len(test_data)
        test_plain_data = test_data.numpy()
        test_data = FedNdarray(
            partitions={
                instance.alice: instance.alice(lambda x: x)(test_plain_data),
                instance.bob: instance.bob(lambda x: x)(test_plain_data),
            },
            partition_way=PartitionWay.VERTICAL,
        )
        if global_config.is_simple_test():
            sample_nums = 4000
            test_data = test_data[0:sample_nums]
        return test_data

    return get_test_data


def inject_get_test_label(dataset_name, instance):
    def get_test_label():
        loader = get_loader[dataset_name]
        data_dir = os.path.join(_CACHE_DIR, dataset_name)
        test_dataset = loader(
            data_dir, False, transform=transforms.ToTensor(), download=True
        )
        test_loader = torch_data.DataLoader(
            dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
        )
        test_data, test_labels = next(iter(test_loader))
        len_test_ds = len(test_labels)
        test_plain_label = test_labels.numpy()
        test_label = instance.bob(lambda x: x)(test_plain_label)
        if global_config.is_simple_test():
            sample_nums = 4000
            test_label = test_label.device(lambda df: df[0:sample_nums])(test_label)
        return test_label

    return get_test_label


class RepeatedDataset(torch_data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.len = len(x)
        self.transform = transform

    def __len__(self):
        return int(2**23)

    def __getitem__(self, idx):
        img = self.x[idx % self.len]
        label = self.y[idx % self.len]
        if self.transform:
            img = self.transform(img)
        return img, label


class OriginalDataset(torch_data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_data_builder(dataset_name):
    def data_builder():
        loader = get_loader[dataset_name]
        data_dir = os.path.join(_CACHE_DIR, dataset_name)
        train_dataset = loader(
            data_dir, True, transform=transforms.ToTensor(), download=True
        )
        train_loader = torch_data.DataLoader(
            dataset=train_dataset, batch_size=len(train_dataset), shuffle=False
        )
        train_data, train_labels = next(iter(train_loader))
        len_train_ds = len(train_data) // 2
        # evaluate with client data
        evaluate_data = train_data.numpy()[:len_train_ds]
        evaluate_label = train_labels.numpy()[:len_train_ds]

        train_plain_data = torch.tensor(train_data.numpy()[len_train_ds:])
        train_plain_label = torch.tensor(train_labels.numpy()[len_train_ds:])
        train_dataset = RepeatedDataset(train_plain_data, train_plain_label)
        train_loader = torch_data.DataLoader(
            train_dataset, batch_size=128, shuffle=False
        )
        evaluate_dataset = OriginalDataset(evaluate_data, evaluate_label)
        evaluate_loader = torch_data.DataLoader(
            evaluate_dataset, batch_size=128, shuffle=False
        )
        return train_loader, evaluate_loader

    return data_builder


class SdarAttackCase(AttackBase):
    """
    Fsha attack needs:
    - A databuilder which returns victim dataloader, need impl in app.
    """

    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)

    def __str__(self):
        return 'sdar'

    def build_attack_callback(self, app: ApplicationBase) -> AttackCallback:
        e_model, decoder, simulator_d, decoder_d = get_model(app)

        app.create_fuse_model = inject_create_fuse_model(app.dataset_name())
        try:
            app.create_base_model = inject_create_base_model()
        except AttributeError:
            if app.device_f.party == 'alice':
                app.create_base_model_alice = inject_create_base_model()
            else:
                app.create_base_model_bob = inject_create_base_model()

        app.get_train_data = inject_get_train_data(app.dataset_name(), app)
        app.get_train_label = inject_get_train_label(app.dataset_name(), app)

        app.get_test_data = inject_get_test_data(app.dataset_name(), app)
        app.get_test_label = inject_get_test_label(app.dataset_name(), app)
        app._train = inject__train(app)

        return SDARAttack(
            attack_party=app.device_y,
            victim_party=app.device_f,
            base_model_list=[self.alice],
            e_model_wrapper=e_model,
            decoder_model_wrapper=decoder,
            simulator_d_model_wrapper=simulator_d,
            decoder_d_model_wrapper=decoder_d,
            reconstruct_loss_builder=torch.nn.MSELoss,
            data_builder=get_data_builder(app.dataset_name()),
            exec_device='cuda' if global_config.is_use_gpu() else 'cpu',
        )

    def attack_type(self) -> AttackType:
        return AttackType.OTHER  # MIA and FIA

    def check_app_valid(self, app: ApplicationBase) -> bool:
        return app.base_input_mode() in [InputMode.SINGLE]

    def tune_metrics(self) -> Dict[str, str]:
        return {'mean_model_loss': 'min', 'mean_guess_loss': 'min'}

    def update_resources_consumptions(
        self, cluster_resources_pack: ResourcesPack, app: ApplicationBase
    ) -> ResourcesPack:
        update_gpu = lambda x: x * 3
        update_mem = lambda x: x * 3
        return (
            cluster_resources_pack.apply_debug_resources('gpu_mem', update_gpu)
            .apply_debug_resources('memory', update_mem)
            .apply_sim_resources(app.device_y.party, 'gpu_mem', update_gpu)
            .apply_sim_resources(app.device_y.party, 'memory', update_mem)
        )
