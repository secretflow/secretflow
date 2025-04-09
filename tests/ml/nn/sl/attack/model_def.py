# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


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


def weights_init_ones(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.ones_(m.weight)


# for attacker
class BottomModelPlus(nn.Module):
    def __init__(
        self,
        bottom_model,
        size_bottom_out=10,
        num_classes=10,
        num_layer=1,
        activation_func_type='ReLU',
        use_bn=True,
    ):
        super(BottomModelPlus, self).__init__()
        self.bottom_model = bottom_model

        dict_activation_func_type = {'ReLU': F.relu, 'Sigmoid': F.sigmoid, 'None': None}
        self.activation_func = dict_activation_func_type[activation_func_type]
        self.num_layer = num_layer
        self.use_bn = use_bn

        self.fc_1 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_1 = nn.BatchNorm1d(size_bottom_out)
        self.fc_1.apply(weights_init_ones)

        self.fc_2 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_2 = nn.BatchNorm1d(size_bottom_out)
        self.fc_2.apply(weights_init_ones)

        self.fc_3 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_3 = nn.BatchNorm1d(size_bottom_out)
        self.fc_3.apply(weights_init_ones)

        self.fc_4 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_4 = nn.BatchNorm1d(size_bottom_out)
        self.fc_4.apply(weights_init_ones)

        self.fc_final = nn.Linear(size_bottom_out, num_classes, bias=True)
        self.bn_final = nn.BatchNorm1d(size_bottom_out)
        self.fc_final.apply(weights_init_ones)

    def forward(self, x):
        x = self.bottom_model(x)

        if self.num_layer >= 2:
            if self.use_bn:
                x = self.bn_1(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_1(x)

        if self.num_layer >= 3:
            if self.use_bn:
                x = self.bn_2(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_2(x)

        if self.num_layer >= 4:
            if self.use_bn:
                x = self.bn_3(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_3(x)

        if self.num_layer >= 5:
            if self.use_bn:
                x = self.bn_4(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_4(x)
        if self.use_bn:
            x = self.bn_final(x)
        if self.activation_func:
            x = self.activation_func(x)
        x = self.fc_final(x)

        return x


class WideDeepBase(nn.Module):
    def __init__(
        self,
        inputs_dim,
        hidden_units,
        dropout_rate,
    ):
        super(WideDeepBase, self).__init__()
        self.inputs_dim = inputs_dim
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate)

        self.hidden_units = [inputs_dim] + list(self.hidden_units)
        self.linear = nn.ModuleList(
            [
                nn.Linear(self.hidden_units[i], self.hidden_units[i + 1])
                for i in range(len(self.hidden_units) - 1)
            ]
        )
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.activation = nn.ReLU()

    def forward(self, X):
        inputs = X
        for i in range(len(self.linear)):
            fc = self.linear[i](inputs)
            fc = self.activation(fc)
            fc = self.dropout(fc)
            inputs = fc
        return inputs


class WideDeepBottomAlice(nn.Module):
    def __init__(
        self,
        feat_size,
        embedding_size,
        dnn_feature_columns,
        dnn_hidden_units=(256, 128),
    ):
        super(WideDeepBottomAlice, self).__init__()
        self.sparse_feature_columns = list(
            filter(lambda x: x[1] == 'sparse', dnn_feature_columns)
        )
        self.embedding_dic = nn.ModuleDict(
            {
                feat[0]: nn.Embedding(feat_size[feat[0]], embedding_size, sparse=False)
                for feat in self.sparse_feature_columns
            }
        )
        self.dense_feature_columns = list(
            filter(lambda x: x[1] == 'dense', dnn_feature_columns)
        )

        self.feature_index = defaultdict(int)
        start = 0
        for feat in feat_size:
            self.feature_index[feat] = start
            start += 1

        self.dnn = WideDeepBase(
            len(self.dense_feature_columns) + embedding_size * len(self.embedding_dic),
            dnn_hidden_units,
            0.5,
        )

    def forward(self, X):

        sparse_embedding = [
            self.embedding_dic[feat[0]](
                torch.clamp(X[:, self.feature_index[feat[0]]].long(), min=0)
            ).reshape(X.shape[0], 1, -1)
            for feat in self.sparse_feature_columns
        ]
        sparse_input = torch.cat(sparse_embedding, dim=1)
        sparse_input = torch.flatten(sparse_input, start_dim=1)
        dense_values = [
            X[:, self.feature_index[feat[0]]].reshape(-1, 1)
            for feat in self.dense_feature_columns
        ]
        dense_input = torch.cat(dense_values, dim=1)
        dnn_input = torch.cat((sparse_input, dense_input), dim=1)
        dnn_out = self.dnn(dnn_input)
        return dnn_out

    def output_num(self):
        return 1


class WideDeepBottomBob(nn.Module):
    def __init__(
        self,
        feat_size,
        dnn_hidden_units=(256, 128),
    ):
        super(WideDeepBottomBob, self).__init__()
        dnn_hidden_units = [len(feat_size), 1]
        self.linear = nn.ModuleList(
            [
                nn.Linear(dnn_hidden_units[i], dnn_hidden_units[i + 1])
                for i in range(len(dnn_hidden_units) - 1)
            ]
        )
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.00001)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        # wide
        X = X.float()
        logit = X
        for i in range(len(self.linear)):
            fc = self.linear[i](logit)
            fc = self.act(fc)
            fc = self.dropout(fc)
            logit = fc
        return logit

    def output_num(self):
        return 1


class WideDeepFuse(nn.Module):
    # Only dnn_linear
    def __init__(
        self,
        dnn_hidden_units=(256, 128),
    ):
        super(WideDeepFuse, self).__init__()

        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)

    def forward(self, X):
        logit = X[1]
        dnn_logit = self.dnn_linear(X[0])
        _logit = dnn_logit + logit
        y_pred = torch.sigmoid(_logit)
        return y_pred


class LocalEmbedding(nn.Module):
    def __init__(self, seed=1):
        super(LocalEmbedding, self).__init__()
        torch.manual_seed(seed)

        # Convolutional layer 1
        self.c1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=5, padding='same'
        )
        # Max pooling 1
        self.s1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layer 2
        self.c2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, padding='same'
        )
        # Max pooling 2
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(4 * 4 * 128, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x, cafe=False):
        # Input size should be 14x14x1
        # print(x)
        x = x.view(-1, 1, 14, 14)

        x = self.c1(x)
        x = F.relu(x)
        x = self.s1(x)

        x = self.c2(x)
        x = F.relu(x)
        x = self.s2(x)

        middle_input = x.view(-1, 4 * 4 * 128)  # 2304
        # print(x.shape)
        middle_output = self.fc1(middle_input)
        x = F.relu(middle_output)
        # x should be 256

        x = self.fc2(x)
        x = F.relu(x)
        # x should be 64

        x = self.fc3(x)
        x = F.relu(x)
        # return x
        if cafe:
            return middle_input, x, middle_output
        return x

    def output_num(self):
        return 1


class CafeServer(nn.Module):
    def __init__(self, seed=0, clients_num=4):
        super(CafeServer, self).__init__()
        torch.manual_seed(seed)

        # Define the last fully connected layer with softmax activation
        self.last = nn.Linear(clients_num * 10, 10)

    def forward(self, x):
        if isinstance(x, list):
            tmp_x = x
            # tmp_x = [x[i * 3 + 1] for i in range(len(x) // 3)]
            x = torch.cat(tmp_x, dim=1)

        x = self.last(x)
        output = F.softmax(
            x, dim=1
        )  # Apply softmax activation along the class dimension
        # The size of output is 10
        return output
