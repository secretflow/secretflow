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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torchmetrics import Accuracy, Precision

import secretflow as sf
from benchmark_examples.autoattack.applications.image.cifar10.cifar10_base import (
    Cifar10TrainBase,
)
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
from secretflow.ml.nn.utils import TorchModel


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=kernel_size, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, kernel_size, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=kernel_size, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], kernel_size, stride=2)
        self.linear = nn.Linear(64, num_classes, bias=False)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # [bs,3,32,16]
        out = F.relu(self.bn1(self.conv1(x)))
        # [bs,16,32,16]
        out = self.layer1(out)
        # [bs,16,32,16]
        out = self.layer2(out)
        # [bs,32,16,8]
        out = self.layer3(out)
        # [bs,64,8,4]
        out = F.avg_pool2d(out, out.size()[2:])
        # [bs,64,1,1]
        out = out.view(out.size(0), -1)
        # [bs,64]
        out = self.linear(out)
        # [bs,10]
        return out


def resnet20(kernel_size=(3, 3), num_classes=10):
    return ResNet(
        block=BasicBlock,
        num_blocks=[3, 3, 3],
        kernel_size=kernel_size,
        num_classes=num_classes,
    )


# base model
class BottomModelForCifar10(nn.Module):
    def __init__(self):
        super(BottomModelForCifar10, self).__init__()
        self.resnet20 = resnet20(num_classes=10)

    def forward(self, x):
        x = self.resnet20(x)
        return x

    def output_num(self):
        return 1


# fuse model
class TopModelForCifar10(nn.Module):
    def __init__(self):
        super(TopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(20, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        print('batch norm: ', self.bn0top)
        self.apply(weights_init)

    def forward(self, input_tensor):
        output_bottom_models = torch.cat(input_tensor, dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.bn1top(x)
        x = self.fc2top(F.relu(x))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)


def correct_counter(output, target, batch_size, topk=(1, 5)):
    tensor_target = torch.Tensor(target)
    dataset = torch.utils.data.TensorDataset(tensor_target)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    correct_counts = [0] * len(topk)
    for idx, tt in enumerate(dataloader):
        for i, k in enumerate(topk):
            _, pred = output[idx].topk(k, 1, True, True)
            correct_k = torch.eq(pred, tt[0].view(-1, 1)).sum().float().item()
            correct_counts[i] += correct_k

    print('correct_counts: ', correct_counts)
    return correct_counts


class Cifar10Resnet20(Cifar10TrainBase):
    def __init__(self, config, alice, bob):
        super().__init__(config, alice, bob, bob)

    def _prepare_data(self):
        from secretflow.utils.simulation import datasets

        (train_data, train_label), (test_data, test_label) = datasets.load_cifar10(
            [self.alice, self.bob],
        )

        return train_data, train_label, test_data, test_label

    def create_base_model(self):
        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=5e-4)
        return TorchModel(
            model_fn=BottomModelForCifar10,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average='micro'
                ),
            ],
        )

    def _create_base_model_alice(self):
        return self.create_base_model()

    def _create_base_model_bob(self):
        return self.create_base_model()

    def _create_fuse_model(self):
        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=5e-4)
        return TorchModel(
            model_fn=TopModelForCifar10,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average='micro'
                ),
            ],
        )

    def train(self, callbacks=None):
        base_model_dict = {
            self.alice: self.alice_base_model,
            self.bob: self.bob_base_model,
        }
        sl_model = SLModel(
            base_model_dict=base_model_dict,
            device_y=self.device_y,
            model_fuse=self.fuse_model,
            dp_strategy_dict=None,
            compressor=None,
            simulation=True,
            random_seed=1234,
            backend='torch',
            strategy='split_nn',
        )
        history = sl_model.fit(
            self.train_data,
            self.train_label,
            validation_data=(self.test_data, self.test_label),
            epochs=2,
            batch_size=128,
            shuffle=False,
            random_seed=1234,
            dataset_builder=None,
            callbacks=callbacks,
        )

        pred_bs = 128
        result = sl_model.predict(self.train_data, batch_size=pred_bs, verbose=1)
        cor_count = self.bob(correct_counter)(
            result, self.train_label, batch_size=pred_bs, topk=(1, 4)
        )
        sf.wait(cor_count)
        logging.warning(history)

    def support_attacks(self):
        return ['lia']

    def lia_auxiliary_model(self, ema=False):
        from benchmark_examples.autoattack.attacks.lia import BottomModelPlus

        bottom_model = BottomModelForCifar10()
        model = BottomModelPlus(bottom_model)

        if ema:
            for param in model.parameters():
                param.detach_()

        return model
