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

from dataclasses import dataclass
from typing import Union


@dataclass()
class TreeParam:
    """Param class, externally exposed interface

    Attributes:
        max_depth :  the max depth of a decision tree.
        eta : learning rate, same as xgb's "eta"
        verbosity : int level of log printing. Valid values are 0 (silent) - 3 (debug).
        objective : Optional[callable , str] objective function, default 'squareloss'
        tree_method: Optional[str] tree type, only support hist
        criterion_method: str split criterion method, default xgboost
        gamma : Optional[float] same as min_impurity_split,minimum gain
        min_child_weight : Optional[float] sum of hessian needed in child nodes
        subsample : Optional[float] subsample rate for rows
        colsample_bytree : Optional[float] subsample rate for columns(by tree)
        colsample_bylevel : Optional[float] subsample rate for columns(by level)
        reg_alpha : Optional[float] L1 regularization term on weights (xgb's alpha).
        reg_lambda : Optional[float] L2 regularization term on weights (xgb's lambda).
        base_score : Optional[float] base score, global bias.
        random_state : Optional[Union[numpy.random.RandomState, int]] Random number seed.
        num_parallel: Optional[int] num of parallel when built tree
        importance_type: Optional[str] importance type, in ['gain','split']
        use_missing: bool whether missing value participate in train
        min_sample_split: minimum sample split of splitting, default to 2
        max_split_nodes: max_split_nodes to parallel finding their splits in a batch
        min_leaf_node: minimum samples on node to split
        decimal: decimal reserved of gain
        num_class: num of class
    """

    max_depth: int = 3
    eta: float = 0.3
    verbosity: int = 0
    objective: Union[callable, str] = None
    tree_method: str = 'hist'
    criterion_method: str = 'xgboost'
    gamma: float = 1e-4
    min_child_weight: float = 1
    subsample: float = 1
    colsample_bytree: float = 1
    colsample_byleval: float = 1
    reg_alpha: float = 0.0
    reg_lambda: float = 0.1
    base_score: float = 0.5
    random_state: int = 1234
    num_parallel: int = None
    importance_type: str = 'split'  # 'split', 'gain'
    use_missing: bool = False
    min_sample_split: int = 2
    max_split_nodes: int = 20
    min_leaf_node: int = 1
    decimal: int = 10
    num_class: int = 0
