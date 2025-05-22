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
import logging
from secretflow import PYUObject, proxy


##########################################
# 加一行 proxy 代码，表示隐语中的一个 worker
##########################################
@proxy(PYUObject)
class Client:
    def __init__(self, model, config, client_id, client_num):
        self._config = config
        self.client_id = client_id
        self.client_num = client_num
        self._model = model
        self._config = config
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=config['lr'],
            weight_decay=config['l2_regularization'],
        )
        self._loss_func = torch.nn.BCELoss()

    def _train_single_batch(self, x):
        self._optimizer.zero_grad()

        deep_out, cross_out, xTw = self._model(x)

        return deep_out, cross_out, xTw

    def get_client_id(self):
        return self.client_id

    def update_gradients(self, server_loss):
        #    根据从 server 传来的 loss 进行反向传播和梯度更新
        server_loss.backward()  # 对 loss 反向传播
        self._optimizer.step()  # 更新模型参数

    def get_model(self):
        return self._model

    def use_cuda(self):
        if self._config['use_cuda'] is True:
            assert torch.cuda.is_available(), 'CUDA is not available'
            torch.cuda.set_device(self._config['device_id'])
            self._model.cuda()

    def save(self):
        self._model.saveModel()

    def get_weights(self):
        return self._model.state_dict()

    def set_weights(self, weights):
        self._model.load_state_dict(weights)


logging.basicConfig(level=logging.INFO)
