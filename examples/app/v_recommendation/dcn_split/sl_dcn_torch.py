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
import torch
import torch.nn as nn
from layer import Deep, Cross


class DeepCrossbase(nn.Module):
    def __init__(self, config, dense_features_cols, sparse_features_cols):
        super(DeepCrossbase, self).__init__()
        self._config = config

        # 稠密和稀疏特征的数量
        self._num_of_dense_feature = len(dense_features_cols)
        self._num_of_sparse_feature = len(sparse_features_cols)
        # print("self._num_of_dense_feature: ",self._num_of_dense_feature)
        # print("self._num_of_sparse_feature: ",self._num_of_sparse_feature)

        # 对于稀疏特征，会根据其类别数来计算嵌入维度（通常公式是 6 * 类别数的1/4次方），可以有效减少类别特征的维度，同时保持足够的信息表达能力。
        self.embedding_dims = list(
            map(lambda x: int(6 * pow(x, 0.25)), sparse_features_cols)
        )

        # 对每个稀疏特征列构建嵌入层，nn.Embedding 将类别型特征映射为稠密的嵌入向量，
        # scale_grad_by_freq=True 会根据出现频率缩放梯度，处理稀疏特征时有助于加快收敛。

        self.embedding_layers = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=e[0], embedding_dim=e[1], scale_grad_by_freq=True
                )
                for e in list(zip(sparse_features_cols, self.embedding_dims))
            ]
        )

        self._input_dim = self._num_of_dense_feature + sum(self.embedding_dims)

        # 初始化
        self._deepNet = Deep(self._input_dim, self._config['deep_layers'])
        self._crossNet = Cross(self._input_dim, self._config['num_cross_layers'])

        self._final_dim = self._input_dim + self._config['deep_layers'][-1]

    def forward(self, x):
        # print("Model input shape:", x.shape)  # 打印输入形状
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # 区分稠密和稀疏特征
        dense_input, sparse_inputs = (
            x[:, : self._num_of_dense_feature],
            x[:, self._num_of_dense_feature :],
        )
        sparse_inputs = sparse_inputs.long()

        # 稀疏特征的嵌入处理
        sparse_embeds = [
            self.embedding_layers[i](sparse_inputs[:, i])
            for i in range(sparse_inputs.shape[1])
        ]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)

        # 将稠密特征和稀疏特征拼接起来
        input = torch.cat([sparse_embeds, dense_input], axis=-1)

        # Deep 和 Cross 网络的输出
        deep_out = self._deepNet(input)
        cross_out, xTw = self._crossNet(input)
        return deep_out, cross_out, xTw

    def saveModel(self):
        """保存模型权重"""
        torch.save(self.state_dict(), self._config['model_name'])

    def loadModel(self, map_location):
        """加载模型权重"""
        state_dict = torch.load(self._config['model_name'], map_location=map_location)
        self.load_state_dict(state_dict, strict=False)


class DeepCrossfuse(nn.Module):
    def __init__(
        self,
        config,
        dense_features_cols_alice,
        dense_features_cols_bob,
        sparse_features_cols_alice,
        sparse_features_cols_bob,
    ):
        super(DeepCrossfuse, self).__init__()
        self._config = config

        self.dnn_input_dim = self._config['dnn_input_dim']
        # 稠密和稀疏特征的数量
        self._num_of_dense_feature = len(dense_features_cols_alice) + len(
            dense_features_cols_bob
        )

        # 对于稀疏特征，会根据其类别数来计算嵌入维度（通常公式是 6 * 类别数的1/4次方），可以有效减少类别特征的维度，同时保持足够的信息表达能力。
        self.embedding_dim_alice = list(
            map(lambda x: int(6 * pow(x, 0.25)), sparse_features_cols_alice)
        )
        self.embedding_dim_bob = list(
            map(lambda x: int(6 * pow(x, 0.25)), sparse_features_cols_bob)
        )

        self.cross_out_num = (
            self._num_of_dense_feature
            + sum(self.embedding_dim_alice)
            + sum(self.embedding_dim_bob)
        )
        # 初始化
        self._deepNet = Deep(self.dnn_input_dim, self._config['deep_layers'])

        self._final_dim = self.cross_out_num + self._config['deep_layers'][-1]
        self._final_linear = nn.Linear(self._final_dim, 1)

    def forward(self, deep_outs, cross_outs, xTws):

        # 将deep_out的结果拼接起来
        deep_input = torch.cat([deep_outs[0], deep_outs[1]], axis=-1)

        # Deep 和 Cross 网络的输出
        deep_out = self._deepNet(deep_input)
        cross_out_0, cross_out_1 = cross_outs

        xTws_0, xTws_1 = xTws
        # print("cross_out_0.shape: ",cross_out_0.shape)
        # print("xTws_1.shape: ",xTws_1.shape)
        xTws_0 = xTws_0.squeeze(-1)  # 去掉最后一个维度，变为 [32, 1]
        xTws_1 = xTws_1.squeeze(-1)  # 去掉最后一个维度，变为 [32, 1]
        xTws_0 = xTws_0.expand(
            -1, cross_out_1.shape[1]
        )  # 将 xTws_0 的第二维广播到 D 维
        xTws_1 = xTws_1.expand(
            -1, cross_out_0.shape[1]
        )  # 将 xTws_1 的第二维广播到 D 维
        cross_out_1 = cross_out_1 + xTws_0  # 相加
        cross_out_0 = cross_out_0 + xTws_1  # 相加
        # 将cross_out的结果拼接起来
        cross_out = torch.cat([cross_out_0, cross_out_1], axis=-1)
        # 最终输出
        final_input = torch.cat([deep_out, cross_out], dim=1)
        output = self._final_linear(final_input)
        output = torch.sigmoid(output)

        return output

    def saveModel(self):
        """保存模型权重"""
        torch.save(self.state_dict(), self._config['model_name'])

    def loadModel(self, map_location):
        """加载模型权重"""
        state_dict = torch.load(self._config['model_name'], map_location=map_location)
        self.load_state_dict(state_dict, strict=False)
