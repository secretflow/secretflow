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
from secretflow import PYUObject, proxy


@proxy(PYUObject)
class Server:
    def __init__(self, model, config):
        self._config = config
        self._model = model
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=config['lr'],
            weight_decay=config['l2_regularization'],
        )
        self._loss_func = torch.nn.BCELoss()

    def _train_single_batch(self, client_models, labels):
        """
        对单个小批量数据进行训练
        """

        self._optimizer.zero_grad()
        deep_outs = []
        cross_outs = []
        xTws = []
        for weight in client_models:
            deep_out, cross_out, xTw = weight
            deep_outs.append(deep_out)
            cross_outs.append(cross_out)
            xTws.append(xTw)
        y_predict = self._model(deep_outs, cross_outs, xTws)
        loss = self._loss_func(y_predict.view(-1), labels)
        loss.backward()
        self._optimizer.step()

        return loss, y_predict

    def get_model(self):
        return self._model

    def get_weights(self):
        """获取当前模型的权重"""
        return self._model.state_dict()

    def set_weights(self, weights):
        """更新服务器模型的权重"""
        self._model.load_state_dict(weights)
