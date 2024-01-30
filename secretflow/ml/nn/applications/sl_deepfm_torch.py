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


class DeepFMBase(BaseModule):
    def __init__(
        self,
        input_dims: Union[List[int], int],
        dnn_units_size: List[int],
        continuous_feas_index: Optional[List[int]] = None,
        fm_embedding_dim: int = 4,
        fm_embedding_init_range: float = 0.1,
        fm_w_init_range: float = 0.1,
        *args,
        **kwargs,
    ):
        """
        DeepFM Base Model.
        Args:
            input_dims: A list of inputs dims, e.g. 4 if there are 1 input with dim = 4,
                        e.g. [1,2] if there are 2 inputs with dim=1 and dim=2.
            dnn_units_size: List of int of dnn layer size.
            continuous_feas_index: If your inputs has continuous features, set this list to show their indexs.
            fm_embedding_dim: embedding output dims.
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        input_dims = [input_dims] if not isinstance(input_dims, List) else input_dims
        self._input_size = len(input_dims)
        self._dnn_units_size = dnn_units_size
        self._continuous_feas_index = (
            [] if continuous_feas_index is None else continuous_feas_index
        )
        self._has_cat_feas = (
            False if len(self._continuous_feas_index) == self._input_size else True
        )
        self._has_cont_feas = False if len(self._continuous_feas_index) == 0 else True
        self._cat_feas_nums = self._input_size - len(self._continuous_feas_index)
        self._cont_feas_nums = len(self._continuous_feas_index)

        dnn_input_shape = 0
        if self._has_cat_feas:
            self._fm_1st_order_sparse_emb_layers = nn.ModuleDict()
            self._fm_2nd_order_sparse_emb_layers = nn.ModuleDict()
            for index, input_dim in enumerate(input_dims):
                if index not in self._continuous_feas_index:
                    dnn_input_shape += fm_embedding_dim
                    self._fm_1st_order_sparse_emb_layers[str(index)] = nn.Embedding(
                        input_dim, 1
                    )
                    self._fm_2nd_order_sparse_emb_layers[str(index)] = nn.Embedding(
                        input_dim, fm_embedding_dim
                    )
                    self._fm_1st_order_sparse_emb_layers[str(index)].weight.data.copy_(
                        torch.FloatTensor(input_dim, 1).uniform_(
                            -fm_w_init_range, fm_w_init_range
                        )
                    )
                    self._fm_2nd_order_sparse_emb_layers[str(index)].weight.data.copy_(
                        torch.FloatTensor(input_dim, fm_embedding_dim).uniform_(
                            -fm_embedding_init_range, fm_embedding_init_range
                        )
                    )

        if self._has_cont_feas:
            continuous_dim = 0
            for index in self._continuous_feas_index:
                continuous_dim += input_dims[index]
            self._fm_1st_order_dense_emb_layers = nn.Linear(continuous_dim, 1)
            dnn_input_shape += continuous_dim

        dnn_layer = []
        for units in dnn_units_size:
            dnn_layer.append(nn.Linear(dnn_input_shape, units))
            dnn_layer.append(nn.ReLU())
            dnn_input_shape = units
        self._dnn_layer = nn.Sequential(*dnn_layer[:-1])

    def forward(self, x):
        x = [x] if not isinstance(x, List) else x
        assert len(x) == self._input_size

        # do fm 1st
        fm_1st_part = None
        if self._has_cat_feas:
            fm_1st_sparse_res = [
                self._fm_1st_order_sparse_emb_layers[idx](x[int(idx)])
                for idx in self._fm_1st_order_sparse_emb_layers
            ]
            fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)
            fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, dim=1, keepdim=True)
            fm_1st_part = fm_1st_sparse_res

        if self._has_cont_feas:
            dense_feas = [x[i] for i in self._continuous_feas_index]
            dense_cat = torch.cat(dense_feas, dim=1)
            fm_1st_dense_res = self._fm_1st_order_dense_emb_layers(dense_cat)
            if fm_1st_part is not None:
                fm_1st_part = fm_1st_part + fm_1st_dense_res
            else:
                fm_1st_part = fm_1st_dense_res

        dnn_input = None
        sum_emb = None
        square_and_sum_emb = None
        if self._has_cat_feas:
            fm_2nd_order_res = [
                self._fm_2nd_order_sparse_emb_layers[idx](x[int(idx)])
                for idx in self._fm_2nd_order_sparse_emb_layers
            ]
            dnn_input = torch.cat(fm_2nd_order_res, dim=1)
            # do fm 2nd
            with torch.no_grad():
                fm_2nd_stack_1d = torch.stack(fm_2nd_order_res, dim=1)
                # need sum and suqare, but compute sum first, suqare act in fuse.
                sum_emb = torch.sum(fm_2nd_stack_1d, dim=1)
                # suare and sum, this can be directly sum again in fuse.
                square_and_sum_emb = torch.sum(torch.pow(fm_2nd_stack_1d, 2), dim=1)

        if self._has_cont_feas:
            if dnn_input is None:
                dnn_input = dense_cat
            else:
                dnn_input = torch.cat([dnn_input, dense_cat], dim=1)

        dnn_output = self._dnn_layer(dnn_input)

        if self._has_cat_feas:
            # shape:
            # dnn_output: batch_size * dnn_units_size[-1] (hidden_size)
            # fm_1st_part: batch_size * 1
            # sum_emb: batch_size * fm_embedding_dim
            # square_and_sum_emb: batch_size * fm_embedding_dim
            return [dnn_output, fm_1st_part, sum_emb, square_and_sum_emb]
        else:
            return [
                dnn_output,
                fm_1st_part,
                torch.zeros((dnn_output.shape[0], 1)).to(dnn_output.device),
                torch.zeros((dnn_output.shape[0], 1)).to(dnn_output.device),
            ]

    def output_num(self):
        return 4

    def get_config(self):
        config = {
            "dnn_units_size": self._dnn_units_size,
        }
        base_config = super(DeepFMBase, self).get_config()
        return {**base_config, **config}


class DeepFMFuse(BaseModule):
    def __init__(
        self,
        input_dims: List[int],
        dnn_units_size: List[int],
        *args,
        **kwargs,
    ):
        """

        Args:
            input_dims: All input dims, input_dims = [2,5] when there are 2 parties,
                    and first output is 2 dim and 5 the other.
            dnn_units_size: List of int of dnn layer output size.
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self._dnn_units_size = dnn_units_size
        dnn_layer = []

        start = sum(input_dims)
        for units in self._dnn_units_size:
            dnn_layer.append(nn.Linear(start, units))
            dnn_layer.append(nn.ReLU())
            start = units
        self._dnn = nn.Sequential(*(dnn_layer + [nn.Linear(start, 1)]))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        dnn_outputs = torch.cat(x[::4], dim=1)
        fm_1st_part = torch.stack(x[1::4], dim=1)
        fm_1st_part = torch.sum(fm_1st_part, dim=1)

        _zero_aux_tensor = torch.zeros((dnn_outputs.shape[0], 1)).to(dnn_outputs.device)
        if torch.equal(x[2], _zero_aux_tensor):
            # aux tensor isnot used, but will need grad in fuse net, so set a grad.
            # backward base model will not use grad anymore, it is safe.
            x[2].grad = _zero_aux_tensor
            sum_embs = x[6]
            sum_embs = torch.unsqueeze(sum_embs, dim=1)
        elif torch.equal(x[6], _zero_aux_tensor):
            x[6].grad = _zero_aux_tensor
            sum_embs = x[2]
            sum_embs = torch.unsqueeze(sum_embs, dim=1)
        else:
            sum_embs = torch.stack(x[2::4], dim=1)
        sum_and_square_emb = torch.pow(torch.sum(sum_embs, dim=1), 2)

        if torch.equal(x[3], _zero_aux_tensor):
            x[3].grad = _zero_aux_tensor
            square_and_sum_emb = x[7]
        elif torch.equal(x[7], _zero_aux_tensor):
            x[7].grad = _zero_aux_tensor
            square_and_sum_emb = x[3]
        else:
            square_and_sum_embs = torch.stack(x[3::4], dim=1)
            square_and_sum_emb = torch.sum(square_and_sum_embs, dim=1)

        sub = 0.5 * (sum_and_square_emb - square_and_sum_emb)
        fm_2nd_part = torch.sum(sub, dim=1, keepdim=True)

        outputs = fm_1st_part + fm_2nd_part + self._dnn(dnn_outputs)
        preds = self.sigmoid(outputs)
        return preds
