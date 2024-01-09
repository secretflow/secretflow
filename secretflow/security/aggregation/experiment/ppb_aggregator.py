# Copyright 2022 Ant Group Co., Ltd.
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

import copy
import math
import re
from typing import Any, List, Union

import numpy as np
import torch
from heu import phe

import secretflow as sf
from secretflow.device import PYU, DeviceObject, PYUObject, proxy
from secretflow.security.aggregation.aggregator import Aggregator


@proxy(PYUObject)
class _WorkerOperator:
    def __init__(
        self,
        device,
        public_key,
        paillier_key_size=2048,
        len_after_encode=55,
        bias_of_weight=20,
        expand_of_weight=10e12,
    ):
        self._device = device
        self._PAILLIER_KEY_SIZE = paillier_key_size
        self._len_after_encode = len_after_encode
        self._bias_of_weight = bias_of_weight
        self._expand_of_weight = expand_of_weight
        self._amount_of_encoded_per_row = int(
            math.floor(self._PAILLIER_KEY_SIZE / len_after_encode)
        )
        self._public_key = public_key
        self._kit = phe.setup(self._public_key)
        self._encryptor = self._kit.encryptor()
        self._is_weights = None

    def is_weights(self, data):
        if isinstance(data, (list, tuple)):
            self._is_weights = True
            return self._is_weights
        else:
            self._is_weights = False
            return self._is_weights

    @staticmethod
    def _get_dtype(arr):
        if isinstance(arr, np.ndarray):
            return arr.dtype
        else:
            try:
                import tensorflow as tf

                if isinstance(arr, tf.Tensor):
                    return arr.numpy().dtype
            except ImportError:
                return None

    def add_weights(self, data, weight, total_weights):
        return (np.multiply(np.array(data), weight) / total_weights).tolist()

    def paillier_enc_local_weights(self, local_weights):  # 加密每个参与方的weights
        if self._is_weights:  # weights mode
            for index in range(len(local_weights)):
                local_weights[index] = self.binary_encode(
                    np.array(local_weights[index])
                )
                for i in range(len(local_weights[index])):
                    local_weights[index][i] = self.per_binary_enc(
                        local_weights[index][i]
                    )
            return local_weights
        else:  # single mode
            local_weights = self.binary_encode(np.array(local_weights))
            for i in range(len(local_weights)):
                local_weights[i] = self.per_binary_enc(local_weights[i])
            return local_weights

    def binary_encode(self, local_weight):  # 多个二进制数编码到一起
        local_weight_to_binary = self.torch_float_to_binary(local_weight)
        return [
            local_weight_to_binary[i : i + self._amount_of_encoded_per_row]
            for i in range(
                0, len(local_weight_to_binary), self._amount_of_encoded_per_row
            )
        ]

    def torch_float_to_binary(self, local_weight):  # torch转成二进制
        return [
            bin(int((abs(i + self._bias_of_weight) * self._expand_of_weight)))[
                2:
            ].zfill(self._len_after_encode)
            for i in local_weight.flatten().tolist()
        ]

    def per_binary_enc(self, per_binary):  # bits级别的加密
        a = "".join(per_binary)
        return self._encryptor.encrypt_raw(int(a, 2))

    def get_shape(self, local_weights):  # 得到权重的shape
        if self._is_weights:
            shape_of_weight = []
            for index in range(len(local_weights)):
                shape_of_weight.append(local_weights[index].shape)
            return shape_of_weight
        else:  # single mode
            return local_weights.shape

    def get_size_of_per_col(self, global_weights):  # 得到编码后的每个weight每列的大小
        if self._is_weights:
            size_of_per_col = []
            for index in range(len(global_weights)):
                size_of_per_col.append(self.binary_encode(global_weights[index]))
                size_of_per_col[index] = [len(i) for i in size_of_per_col[index]]
            return size_of_per_col
        else:  # single mode
            size_of_per_col = self.binary_encode(global_weights)
            size_of_per_col = [len(i) for i in size_of_per_col]
            return size_of_per_col


@proxy(PYUObject)
class _AggregatorOperator:
    def __init__(
        self,
        device,
        amount_of_party,
        paillier_key_size=2048,
        len_after_encode=55,
        bias_of_weight=20,
        expand_of_weight=10e12,
    ):
        self._device = device
        self._PAILLIER_KEY_SIZE = paillier_key_size
        self._kit = phe.setup(phe.SchemaType.ZPaillier, self._PAILLIER_KEY_SIZE)
        self._evaluator = self._kit.evaluator()
        self._decryptor = self._kit.decryptor()
        self._public_key = self._kit.public_key()
        self._bias_of_weight = bias_of_weight
        self._expand_of_weight = expand_of_weight
        self._len_after_encode = len_after_encode
        self._amount_of_party = amount_of_party

        self._is_weights = None
        self._dtype = None

    def set_is_weights_and_dtype(self, _is_weights, dtype):
        self._is_weights = _is_weights
        self._dtype = dtype

    def get_public_key(self):
        return self._public_key

    def homo_add(self, local_weights_after_enc_list):  # 同态聚合
        enc_add_list = copy.deepcopy(local_weights_after_enc_list[0])
        if self._is_weights:
            for i in range(1, len(local_weights_after_enc_list)):
                for m in range(len(local_weights_after_enc_list[i])):
                    for n in range(len(local_weights_after_enc_list[i][m])):
                        self._evaluator.add_inplace(
                            enc_add_list[m][n], local_weights_after_enc_list[i][m][n]
                        )
            return enc_add_list
        else:  # single mode
            for i in range(1, len(local_weights_after_enc_list)):
                for m in range(len(local_weights_after_enc_list[i])):
                    self._evaluator.add_inplace(
                        enc_add_list[m], local_weights_after_enc_list[i][m]
                    )
            return enc_add_list

    def flatten(self, iterable):
        for item in iterable:
            if isinstance(item, (list, tuple)):
                yield from self.flatten(item)
            else:
                yield item

    def homo_dec_list_without_average(
        self, enc_add_list, local_weight_shape, size_of_per_col
    ):  # 解密整个加密聚合后的列表，不取均值
        _size_of_per_col = size_of_per_col
        _local_weight_shape = local_weight_shape
        dec_list = []
        if self._is_weights:
            for index in range(len(enc_add_list)):
                # 解密操作
                dec_list.append([])
                for i in range(len(enc_add_list[index])):
                    dec_list[index].append(
                        re.findall(
                            r'\w{%d}' % self._len_after_encode,
                            bin(self._decryptor.decrypt(enc_add_list[index][i]))[
                                2:
                            ].zfill(
                                self._len_after_encode * _size_of_per_col[index][i]
                            ),
                        )
                    )
                dec_list[index] = list(self.flatten(dec_list[index]))
                dec_list[index] = torch.tensor(
                    [
                        (
                            int(i, 2) / self._expand_of_weight
                            - self._amount_of_party * self._bias_of_weight
                        )
                        for i in dec_list[index]
                    ]
                ).reshape(_local_weight_shape[index])
            return dec_list
        else:  # single mode
            for i in range(len(enc_add_list)):
                dec_list.append(
                    re.findall(
                        r'\w{%d}' % self._len_after_encode,
                        bin(self._decryptor.decrypt(enc_add_list[i]))[2:].zfill(
                            self._len_after_encode * _size_of_per_col[i]
                        ),
                    )
                )
            dec_list = list(self.flatten(dec_list))
            dec_list = torch.tensor(
                [
                    (
                        int(i, 2) / self._expand_of_weight
                        - self._amount_of_party * self._bias_of_weight
                    )
                    for i in dec_list
                ]
            ).reshape(_local_weight_shape)
            return dec_list


class PPBAggregator(Aggregator):
    """phe_paillier_batch_aggregator.

    Warnings:
        PpbAggregator is for experiment purpose only.
        You should not use it in production.
    """

    def __init__(
        self,
        device: PYU,
        participants: List[PYU],
        paillier_key_size=2048,
        len_after_encode=55,
        bias_of_weight=20,
        expand_of_weight=10e12,
    ):
        assert isinstance(device, PYU), f'Accepts PYU only but got {type(device)}.'
        self.device = device
        self._PAILLIER_KEY_SIZE = paillier_key_size
        self._len_after_encode = len_after_encode
        self._bias_of_weight = bias_of_weight
        self._expand_of_weight = expand_of_weight
        self._amount_of_party = len(participants)
        self._is_not_single = None
        self._dtype = None
        self._local_weight_shape = None
        self._size_of_per_col = None
        self._aggregator_operator = _AggregatorOperator(
            self.device, self._amount_of_party, device=self.device
        )
        self._public_key = self._aggregator_operator.get_public_key()
        self._worker_operator = {
            pyu: _WorkerOperator(
                pyu,
                public_key=sf.reveal(self._public_key),
                paillier_key_size=self._PAILLIER_KEY_SIZE,
                len_after_encode=55,
                bias_of_weight=20,
                expand_of_weight=10e12,
                device=pyu,
            )
            for pyu in participants
        }

    def sum(self, data: List[DeviceObject], axis=None) -> PYUObject:
        """Sum of encrypted weights.
        Args:
            data: array of device objects.
            axis: optional. Same as the axis argument of :py:meth:`numpy.mean`, this feature has not been implemented yet.

        Returns:
            a list object: Sum of weights.
        """

        assert data, 'Data to aggregate should not be None or empty!'
        data_operate = copy.deepcopy(data)

        whichmode = [
            self._worker_operator[d.device].is_weights(d).to(self.device)
            for d in data_operate
        ]
        whichdtype = [
            self._worker_operator[d.device]._get_dtype(d).to(self.device)
            for d in data_operate
        ]
        self._is_not_single = whichmode[0]
        self._dtype = whichdtype[0]
        self._aggregator_operator.set_is_weights_and_dtype(whichmode[0], whichdtype[0])
        self._local_weight_shape = (
            self._worker_operator[data_operate[0].device]
            .get_shape(data_operate[0])
            .to(self.device)
        )

        self._size_of_per_col = (
            self._worker_operator[data_operate[0].device]
            .get_size_of_per_col(data_operate[0])
            .to(self.device)
        )

        data_operate = [
            self._worker_operator[d.device]
            .paillier_enc_local_weights(d)
            .to(self.device)
            for d in data_operate
        ]
        added_weight_list = self._aggregator_operator.homo_add(data_operate)
        dec_list = self._aggregator_operator.homo_dec_list_without_average(
            added_weight_list, self._local_weight_shape, self._size_of_per_col
        )
        return dec_list

    def average(
        self, data: List[DeviceObject], axis=None, weights=None
    ) -> List[Union[list, Any]]:
        """Compute the average weight using encrypted weights.

        Args:
            data: array of device objects.
            axis: optional. Same as the axis argument of :py:meth:`numpy.average`, this feature has not been implemented yet.
            weights: optional. Same as the weights argument of :py:meth:`numpy.average`, this feature has not been implemented yet.

        Returns:
            a list object: the averaged weights.
        """

        assert data, 'Data to aggregate should not be None or empty!'
        data_operate = copy.deepcopy(data)

        is_weights = True
        if weights is None:
            weights = np.ones(self._amount_of_party)
            is_weights = False
        weights_operate = copy.deepcopy(weights)
        weights_for_sum = weights_operate
        if is_weights:
            weights_for_sum = [
                w.to(self.device) if isinstance(w, DeviceObject) else w
                for w in weights_for_sum
            ]
            total_weights = self.device(sum)(weights_for_sum)
        else:
            total_weights = sum(weights_for_sum)

        whichmode = [
            self._worker_operator[d.device].is_weights(d).to(self.device)
            for d in data_operate
        ]

        whichdtype = [
            self._worker_operator[d.device]._get_dtype(d).to(self.device)
            for d in data_operate
        ]

        self._is_not_single = whichmode[0]
        self._dtype = whichdtype[0]
        self._aggregator_operator.set_is_weights_and_dtype(whichmode[0], whichdtype[0])
        self._local_weight_shape = (
            self._worker_operator[data_operate[0].device]
            .get_shape(data_operate[0])
            .to(self.device)
        )
        self._size_of_per_col = (
            self._worker_operator[data_operate[0].device]
            .get_size_of_per_col(data_operate[0])
            .to(self.device)
        )

        data_operate = [
            self._worker_operator[data_operate[i].device].add_weights(
                data_operate[i],
                weights[i],
                total_weights.to(data_operate[i].device)
                if isinstance(total_weights, DeviceObject)
                else total_weights,
            )
            for i in range(self._amount_of_party)
        ]
        data_operate = [
            self._worker_operator[d.device]
            .paillier_enc_local_weights(d)
            .to(self.device)
            for d in data_operate
        ]
        added_weight_list = self._aggregator_operator.homo_add(data_operate)
        dec_list = self._aggregator_operator.homo_dec_list_without_average(
            added_weight_list, self._local_weight_shape, self._size_of_per_col
        )
        return dec_list
