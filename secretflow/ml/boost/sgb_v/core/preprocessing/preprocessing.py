# Copyright 2023 Ant Group Co., Ltd.
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

import math
from typing import Tuple, Union

from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.vertical import VDataFrame

from .params import RegType, SGBTrainParams


def prepare_dataset(
    ds: Union[FedNdarray, VDataFrame]
) -> Tuple[FedNdarray, Tuple[int, int]]:
    """
    check data setting and get total shape.

    Args:
        ds: input dataset

    Return:
        First: dataset in unified type
        Second: shape concat all partition.
    """
    assert isinstance(
        ds, (FedNdarray, VDataFrame)
    ), f"ds should be FedNdarray or VDataFrame, got {type(ds)}"

    ds = ds if isinstance(ds, FedNdarray) else ds.values

    assert ds.partition_way == PartitionWay.VERTICAL, (
        "SGB Only support vertical dataset, "
        "for horizontal dataset please use secreflow.ml.boost.homo_boost"
    )

    shape = ds.shape
    assert math.prod(shape), f"not support empty dataset, shape {shape}"

    return ds, shape


def validate_sgb_params_dict(params: dict) -> SGBTrainParams:
    trees = int(params.pop('num_boost_round', 10))
    assert 1 <= trees <= 1024, f"num_boost_round should in [1, 1024], got {trees}"

    depth = int(params.pop('max_depth', 5))
    assert depth > 0 and depth <= 16, f"max_depth should in [1, 16], got {depth}"

    lr = float(params.pop('learning_rate', 0.3))
    assert lr > 0 and lr <= 1, f"learning_rate should in (0, 1], got {lr}"

    obj = params.pop('objective', 'logistic')
    assert obj in [
        e.value for e in RegType
    ], f"objective should in {[e.value for e in RegType]}, got {obj}"
    obj = RegType(obj)

    reg_lambda = float(params.pop('reg_lambda', 0.1))
    assert (
        reg_lambda >= 0 and reg_lambda <= 10000
    ), f"reg_lambda should in [0, 10000], got {reg_lambda}"

    gamma = float(params.pop('gamma', 0))
    assert gamma >= 0 and gamma <= 10000, f"gamma should in [0, 10000], got {gamma}"

    subsample = float(params.pop('subsample', 1))
    assert (
        subsample > 0 and subsample <= 1
    ), f"subsample should in (0, 1], got {subsample}"

    colsample = float(params.pop('colsample_by_tree', 1))
    assert (
        colsample > 0 and colsample <= 1
    ), f"colsample_bytree should in (0, 1], got {colsample}"

    base = float(params.pop('base_score', 0))

    sketch = params.pop('sketch_eps', 0.1)
    assert sketch > 0 and sketch <= 1, f"sketch_eps should in (0, 1], got {sketch}"
    seed = int(params.pop('seed', 42))

    # heu batch encoding setting
    fxp_r = params.pop('fixed_point_parameter', 20)

    return SGBTrainParams(
        trees,
        depth,
        lr,
        obj,
        reg_lambda,
        gamma,
        subsample,
        colsample,
        base,
        sketch,
        seed,
        fxp_r,
    )
