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

from typing import List, Optional, Union

import torch
import torch.nn as nn

from secretflow.ml.nn.utils import BaseModule


class DnnBase(BaseModule):
    """
    Dnn Base Model for Troch.
    """

    def __init__(
        self,
        input_dims: Union[int, List[int]],
        dnn_units_size: List[int],
        sparse_feas_indexes: Optional[List[int]] = None,
        embedding_dim=16,
        preprocess_layer=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            input_dims: All input dims, input_dims = 5 when there are 1 input with dim = 5,
                    input_dims = [2,4] when there are 2 inputs with dim = 2 and dim = 4.
            dnn_units_size: List of int of dnn layer size
            sparse_feas_indexes: If some of the inputs are spare features, specify them and embedding.
            embedding_dim: When youhave multi inputs, dnn base will compute embedding first.
                    This is the embedding output dims.
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        if not isinstance(input_dims, List):
            input_dims = [input_dims]
        self._sparse_feas_index = (
            [] if sparse_feas_indexes is None else sparse_feas_indexes
        )
        all_inputs = [i for i in range(len(input_dims))]
        self._dense_feas_index = [
            i for i in all_inputs if i not in self._sparse_feas_index
        ]
        self._has_sparse_feas = not len(self._sparse_feas_index) == 0
        self._has_dense_feas = not len(self._dense_feas_index) == 0
        if self._has_sparse_feas:
            self._embedding_features_layer = [
                nn.Embedding(input_dims[idx], embedding_dim)
                for idx in self._sparse_feas_index
            ]
        input_shape = 0
        for i, input_dim in enumerate(input_dims):
            if i in self._sparse_feas_index:
                input_shape += embedding_dim
            else:
                input_shape += input_dim
        dnn_layer = []
        self.preprocess_layer = preprocess_layer
        for units in dnn_units_size:
            dnn_layer.append(nn.Linear(input_shape, units))
            dnn_layer.append(nn.ReLU())
            input_shape = units
        self._dnn = nn.Sequential(*(dnn_layer[:-1]))

    def forward(self, x):
        if self.preprocess_layer is not None:
            x = self.preprocess_layer(x)
        x = [x] if not isinstance(x, List) else x
        cat_feas = None
        # handle sparse feas
        if self._has_sparse_feas:
            sparse_feas = []
            for i, idx in enumerate(self._sparse_feas_index):
                sparse_feas.append(self._embedding_features_layer[i](x[idx]))
            sparse_res = torch.cat(sparse_feas, dim=1)
            cat_feas = sparse_res
        # handle dense feas
        if self._has_dense_feas:
            dense_feas = [x[i] for i in self._dense_feas_index]
            dense_res = torch.cat(dense_feas, dim=1)
            if cat_feas is None:
                cat_feas = dense_res
            else:
                cat_feas = torch.cat([cat_feas, dense_res], dim=1)
        res = self._dnn(cat_feas)
        return res

    def output_num(self):
        return 1


class DnnFuse(BaseModule):
    def __init__(
        self,
        input_dims: List[int],
        dnn_units_size: List[int],
        output_func: nn.Module = nn.Sigmoid,
        *args,
        **kwargs,
    ):
        """
        Fuse model of the dnn.
        Args:
            input_dims: All input dims, input_dims = [2,5] when you have 2 parties,
                    and first output is 2 dim and 5 the other.
            dnn_units_size: List of int of dnn layer output size.
            output_func: the output function, default for Sigmoid.
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        start = sum(input_dims)
        dnn_layer = []
        for units in dnn_units_size:
            dnn_layer.append(nn.Linear(start, units))
            dnn_layer.append(nn.ReLU())
            start = units
        self._dnn = nn.Sequential(*dnn_layer[:-1])
        self._output_func = output_func() if output_func else None

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self._dnn(x)
        if self._output_func:
            x = self._output_func(x)
        return x
