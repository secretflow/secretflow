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

from secretflow import PYUObject, proxy
import copy


@proxy(PYUObject)
class Server(object):
    def __init__(self, args):
        self.args = args

    def aggregate_reps(self, client_models):
        # 初始化累加结果为第一个客户端的参数副本
        weights_sum = copy.deepcopy(client_models[0])

        # 遍历每个参数（列表中每个张量的位置）
        for i in range(len(weights_sum)):
            for client_index in range(1, len(client_models)):
                # 累加对应位置的张量
                weights_sum[i] += client_models[client_index][i]

        return weights_sum
