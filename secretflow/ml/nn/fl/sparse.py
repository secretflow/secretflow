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


from typing import List

import numpy as np
import jaxlib


class STCSparse:
    """Stc sparser, sample TopK element from original Weights
    TODO: 补充docstring
    """

    def __init__(self, sparse_rate: float):
        self.sparse_rate = sparse_rate
        self.name = 'STC'

    def __call__(
        self,
        weights: List[np.ndarray],
    ) -> List[np.ndarray]:
        compression_weights = []
        mask_arrays = []
        for weight in weights:
            weight_shape = weight.shape
            weight_flat = weight.flatten()
            if isinstance(weight_flat, jaxlib.xla_extension.DeviceArray):
                weight_flat = np.array(weight_flat)
            weight_flat_abs = np.abs(weight_flat)
            weight_len = weight_flat.shape[0]
            mask_num = round(self.sparse_rate * weight_len)
            mask_index = np.sort(np.argsort(weight_flat_abs)[:mask_num])
            mask_array = np.ones(weight_flat.shape)
            if mask_index.shape[0] != 0:
                weight_flat[mask_index] = 0
                mask_array[mask_index] = 0
            if weight_len == mask_num:
                average_value = 0.0
            else:
                average_value = np.sum(np.absolute(weight_flat)) / (
                    weight_len - mask_num
                )
            weight_compress = average_value * np.sign(weight_flat)
            compression_weight = weight_compress.reshape(weight_shape)
            compression_weights.append(compression_weight)
            mask_array = mask_array.reshape(weight_shape)
            mask_arrays.append(mask_array)
        return compression_weights


class SCRSparse:
    """Stc sparser, sample TopK element from original Weights
    TODO: 补充docstring
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.name = 'SCR'

    def __call__(
        self,
        weights: List[np.ndarray],
    ) -> List[np.ndarray]:
        compression_weights = []
        mask_arrays = []
        for weight in weights:
            weight_shape = weight.shape
            if len(weight_shape) == 4:
                # CNN layer
                # Keep the 0th dimension
                sum_0 = np.sum(np.absolute(weight), axis=(1, 2, 3))
                sum_0 = sum_0 / np.max(sum_0)
                index_zero_0 = self.get_dimension(sum_0, self.threshold)
                weight[index_zero_0, :, :, :] = 0.0
                # Keep the 1th dimension
                sum_1 = np.sum(np.absolute(weight), axis=(0, 2, 3))
                sum_1 = sum_1 / np.max(sum_1)
                index_zero_1 = self.get_dimension(sum_1, self.threshold)
                weight[:, index_zero_1, :, :] = 0.0
            if len(weight_shape) == 2:
                # Dense layer
                # Keep the 0th dimension
                sum_0 = np.sum(np.absolute(weight), axis=1)
                sum_0 = sum_0 / np.max(sum_0)
                index_zero_0 = self.get_dimension(sum_0, self.threshold)
                weight[index_zero_0, :] = 0.0
                # Keep the 1th dimension
                sum_1 = np.sum(np.absolute(weight), axis=0)
                sum_1 = sum_1 / np.max(sum_1)
                index_zero_1 = self.get_dimension(sum_1, self.threshold)
                weight[:, index_zero_1] = 0.0
            compression_weight = weight
            compression_weights.append(compression_weight)
            mask_array = np.array(compression_weight, dtype=bool)
            mask_arrays.append(mask_array)
        return compression_weights

    def get_dimension(self, index_value, threshold):
        return np.argwhere(index_value <= threshold)
