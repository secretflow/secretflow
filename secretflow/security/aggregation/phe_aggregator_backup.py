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
import logging
from typing import List

import numpy as np

from secretflow.device import PYU, DeviceObject, PYUObject, proxy
from secretflow.security.aggregation.aggregator import Aggregator

import math
import re
from heu import phe
import time
from more_itertools import flatten
import copy
import torch


@proxy(PYUObject)
class _WorkerOperator:
    def __init__(self, party:PYU, public_key, paillier_key_size=2048, len_after_encode=55, bias_of_weight=5, expand_of_weight=10e12):
        self._party = party

        self._PAILLIER_KEY_SIZE = paillier_key_size
        self._len_after_encode = len_after_encode
        self._bias_of_weight = bias_of_weight
        self._expand_of_weight = expand_of_weight

        self._amount_of_encoded_per_row = int(math.floor(self._PAILLIER_KEY_SIZE / len_after_encode))

        self._kit = phe.setup(public_key)

        self._encryptor = self._kit.encryptor()

        self._amount_of_real_encoded_row = {}

    def paillier_enc_local_weights(self, local_weights):  # 加密每个参与方的weights
            for index in range(len(local_weights)):
                local_weights[index] = self.binary_encode(local_weights[index])
                self._amount_of_real_encoded_row[index] = [len(i) for i in local_weights[index]]
                for i in range(len(local_weights[index])):
                    local_weights[index][i] = self.per_binary_enc(local_weights[index][i])
            return local_weights

    def binary_encode(self,local_weight):  # 多个二进制数编码到一起
        local_weight_to_binary = self.torch_float_to_binary(local_weight)
        return [local_weight_to_binary[i:i + self._amount_of_encoded_per_row] for i in
                range(0, len(local_weight_to_binary), self._amount_of_encoded_per_row)]

    def torch_float_to_binary(self, local_weight):  # torch转成二进制
        return [bin(int(((i + self._bias_of_weight) * self._expand_of_weight)))[2:].zfill(self._len_after_encode) for i in
                local_weight.flatten().tolist()]

    def per_binary_enc(self, per_binary):  # bits级别的加密
        return self._encryptor.encrypt_raw(int("".join(per_binary), 2))

    def get_shape(self, local_weights):  # 得到权重的shape
        shape_of_weight = []
        for index in range(len(local_weights)):
            shape_of_weight.append(local_weights[index].shape)
        return shape_of_weight

    def get_size_of_per_col(self, global_weights):  # 得到编码后的每个weight每列的大小
        size_of_per_col = []
        for index in range(len(global_weights)):
            size_of_per_col.append(
                self.binary_encode(global_weights[index]))
            size_of_per_col[index] = [len(i) for i in size_of_per_col[index]]
        return size_of_per_col


@proxy(PYUObject)
class PheAggregator(Aggregator):

    def __init__(self, device: PYU, participants: List[PYU], paillier_key_size=2048, len_after_encode=55, bias_of_weight=5, expand_of_weight=10e12):

        assert isinstance(device, PYU), f'Accepts PYU only but got {type(device)}.'
        self.device = device

        self._PAILLIER_KEY_SIZE = paillier_key_size

        self._kit = phe.setup(phe.SchemaType.ZPaillier, self._PAILLIER_KEY_SIZE)

        self._evaluator = self._kit.evaluator()
        self._decryptor = self._kit.decryptor()

        self._publick_key = self._kit.public_key()


        self._len_after_encode = len_after_encode
        self._bias_of_weight = bias_of_weight
        self._expand_of_weight = expand_of_weight

        self._amount_of_party = len(participants)

        self._local_weight_shape = None
        self._size_of_per_col = None
        logging.info(f'phe110')
        self._operator = {
            pyu: _WorkerOperator(pyu.party, public_key=self._publick_key, paillier_key_size=self._PAILLIER_KEY_SIZE,
                                 len_after_encode=55, bias_of_weight=5, expand_of_weight=10e12, device=pyu) for pyu in participants
        }

        logging.info(f'phe116')

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

    def sum(self, data: List[DeviceObject], axis=None) -> PYUObject:
        """Sum of array elements over a given axis.

        Args:
            data: array of device objects.
            axis: optional. Same as the axis argument of :py:meth:`numpy.mean`.

        Returns:
            a device object holds the sum.
        """
        assert data, 'Data to aggregate should not be None or empty!'
        pass

    def average(self, data: List[DeviceObject], axis=None, weights=None) -> PYUObject:
        """Compute the weighted average along the specified axis.

        Args:
            data: array of device objects.
            axis: optional. Same as the axis argument of :py:meth:`numpy.average`.
            weights: optional. Same as the weights argument of :py:meth:`numpy.average`.

        Returns:
            a device object holds the weighted average.
        """

        def _homo_add(local_weights_after_enc_list, public_key):  # 同态聚合

            from heu import phe

            _evaluator = phe.setup(public_key).evaluator()

            logging.info(f"The homomorphic aggregation starts")
            time_start_dec = time.time()
            enc_add_list = copy.deepcopy(local_weights_after_enc_list[0])
            for i in range(1, len(local_weights_after_enc_list)):
                for m in range(len(local_weights_after_enc_list[i])):
                    for n in range(len(local_weights_after_enc_list[i][m])):
                        _evaluator.add_inplace(enc_add_list[m][n], local_weights_after_enc_list[i][m][n])
                        # print(enc_add_list[m][n],local_weights_after_enc_list[i][m][n])
            time_end_dec = time.time()
            logging.info(f"The homomorphic aggregation takes{time_end_dec - time_start_dec}")
            return enc_add_list

        assert data, 'Data to aggregate should not be None or empty!'

        def _homo_dec_list(enc_add_list , charlie_kit, expand_of_weight, amount_of_party,
                           bias_of_weight, local_weight_shape, size_of_per_col, len_after_encode ):  # 解密整个加密聚合后的列表

            _decryptor = charlie_kit.decrytor()

            _expand_of_weight = expand_of_weight
            _amount_of_party = amount_of_party
            _bias_of_weight = bias_of_weight
            _local_weight_shape = local_weight_shape
            _size_of_per_col = size_of_per_col
            _len_after_encode = len_after_encode

            logging.info(f"The homomorphic decryption starts")
            time_start_dec = time.time()
            dec_list = []
            for index in range(len(enc_add_list)):
                # 解密操作
                dec_list.append([])
                for i in range(len(enc_add_list[index])):
                    dec_list[index].append(re.findall(r'\w{%d}' % _len_after_encode,
                                                      bin(_decryptor.decrypt(enc_add_list[index][i]))[2:].zfill(
                                                          _len_after_encode * _size_of_per_col[index][i])))
                dec_list[index] = list(flatten(dec_list[index]))
                dec_list[index] = torch.tensor(
                    [(int(i, 2) / _expand_of_weight - _amount_of_party * _bias_of_weight) for i in
                     dec_list[index]]).reshape(_local_weight_shape[index])
                dec_list[index] = dec_list[index].div(_amount_of_party)
            time_end_dec = time.time()
            logging.info(f"The homomorphic decryption takes {time_end_dec - time_start_dec}")
            return dec_list

        data_operate = copy.deepcopy(data)

        self._local_weight_shape = self._operator[data_operate[0].device].get_shape(data_operate[0]).to(self.device)
        self._size_of_per_col = self._operator[data_operate[0].device].get_size_of_per_col(data_operate[0]).to(self.device)

        data_operate = [self._operator[d.device].paillier_enc_local_weights(d).to(self.device) for d in data_operate]

        added_weight_list = self.device(_homo_add)(data_operate, self._publick_key)

        dec_list = self.device(_homo_dec_list)(added_weight_list,self._kit, self._expand_of_weight, self._amount_of_party,
                                               self._bias_of_weight, self._local_weight_shape, self._size_of_per_col,self._len_after_encode)
        return dec_list