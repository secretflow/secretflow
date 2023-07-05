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
from dataclasses import dataclass, field
from enum import Enum, unique


@unique
class RegType(Enum):
    Linear = 'linear'
    Logistic = 'logistic'


class TreeGrowingMethod(Enum):
    LEVEL = "level"
    LEAF = "leaf"


# all params and range check here: unify default values and range checking condition
@dataclass
class SGBParams:
    """
    'label_holder_feature_only': bool. where to use label holder's feature only. DO NOT SET IT, auto managed.
        default: False
        if turned on, gh won't be sent to workers in anyway.
    'fixed_point_parameter': int. Any floating point number encoded by heu,
             will multiply a scale and take the round,
             scale = 2 ** fixed_point_parameter.
             larger value may mean more numerical accurate,
             but too large will lead to overflow problem.
             See HEU's document for more details.
        default: 20
        range: [1, 100]
    'batch_encoding_enabled': bool. if use batch encoding optimization.
        default: True.
    'audit_paths': dict. {device : path to save log for audit}
    'reg_lambda': float. L2 regularization term on weights.
        default: 0.1
        range: [0, 10000]
    'learning_rate': float, step size shrinkage used in update to prevent overfitting.
        default: 0.3
        range: (0, 1]
    'max_leaf': int, maximum leaf of a tree. only effective if train leaf wise.
        default: 15
        range: [1, 2**15]
    'max_depth': int, maximum depth of a tree.only effective if train level wise.
        default: 5
        range: [1, 16]
    'gamma': float. Greater than 0 means pre-pruning enabled.
        Gain less than it will not induce split node.
        default: 0.1
        range: [0, 10000]
    'seed': Pseudorandom number generator seed.
        default: 1212
    'row_sample_rate': float. Row sub sample ratio of the training instances.
        default: 1
        range: (0, 1]
    'col_sample_rate': float. Col sub sample ratio of columns when constructing each tree.
        default: 1
        range: (0, 1]
    'enable_goss': bool. whether enable GOSS, see lightGBM's paper for more understanding in GOSS.
        default: False
    'top_rate': float. GOSS-specific parameter. The fraction of large gradients to sample.
        default: 0.3
        range: (0, 1), but top_rate + bottom_rate < 1
    'bottom_rate': float. GOSS-specific parameter. The fraction of small gradients to sample.
        default: 0.5
        range: (0, 1), but top_rate + bottom_rate < 1
    'sketch_eps': This roughly translates into O(1 / sketch_eps) number of bins.
        default: 0.1
        range: (0, 1]
    'objective': Specify the learning objective.
        default: 'logistic'
        range: ['linear', 'logistic']
    'base_score': The initial prediction score of all instances, global bias.
        default: 0
    'enable_quantization': Whether enable quantization of g and h.
        only recommended for encryption schemes with small plaintext range, like elgamal.
        default: False
        range: [True, False]
    'quantization_scale': only effective if quanization enabled. Scale the sum of g to the specified value.
        default: 10000.0
        range: [0, 10000000.0]
    'early_stop_criterion_g_abs_sum': if sum(abs(g)) is lower than or equal to this threadshold, training will stop.
        default: 0.0
        range [0.0, inf)
    'early_stop_criterion_g_abs_sum_change_ratio': if absolute g sum change ratio is lower than or equal to this threadshold, training will stop.
        default: 0.0
        range [0, 1]
    'tree_growing_method': how to grow tree?
        default: level-wise
    """

    label_holder_feature_only: bool = False
    fixed_point_parameter: int = 20
    batch_encoding_enabled: bool = True
    audit_paths: dict = field(default_factory=dict)
    reg_lambda: float = 0.1
    learning_rate: float = 0.3
    max_leaf: int = 15
    max_depth: int = 5
    gamma: float = 0
    seed: int = 1212
    row_sample_rate: float = 1
    col_sample_rate: float = 1
    enable_goss: bool = False
    top_rate: float = 0.3
    bottom_rate: float = 0.5
    sketch_eps: float = 0.1
    objective: RegType = RegType('logistic')
    base_score: float = 0
    enable_quantization: bool = False
    quantization_scale: float = 10000.0
    early_stop_criterion_g_abs_sum: float = 0.0
    early_stop_criterion_g_abs_sum_change_ratio: float = 0.0
    tree_growing_method = TreeGrowingMethod.LEVEL


default_params = SGBParams()
