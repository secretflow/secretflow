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
from dataclasses import asdict, dataclass, field
from enum import Enum, unique

import numpy as np


@unique
class RegType(Enum):
    Linear = 'linear'
    Logistic = 'logistic'
    Tweedie = 'tweedie'


class TreeGrowingMethod(Enum):
    LEVEL = "level"
    LEAF = "leaf"


# the params is used define all possible parameters for SGB.
# user usually use dictionary to set params.
# note that SGB also supports
# some profiling-related settings for developers
# in logging params (which are not listed here).
@dataclass
class SGBParams:
    """
    The first part is security or encryption related params.
    'first_tree_with_label_holder_feature': bool, default=False
                Whether to train the first tree with label holder's own features.
                Can increase training speed and label security.
                The training loss may increase.
                If label holder has no feature, set this to False.
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
    'enable_quantization': Whether enable quantization of g and h.
        only recommended for encryption schemes with small plaintext range, like elgamal.
        default: False
        range: [True, False]
    'quantization_scale': only effective if quanization enabled. Scale the sum of g to the specified value.
        default: 10000.0
        range: [0, 10000000.0]

    The second part is tree boosting params. Some of them align with XGB params, others may not be.
    'num_boost_round' : int, default=10
                Number of boosting iterations. Same as number of trees.
                range: [1, 1024]
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
        default: 1.0
        range: [0, 10000]
    'seed': Pseudorandom number generator seed.
        default: 1212
    'rowsample_by_tree': float. Row sub sample ratio of the training instances.
        default: 1
        range: (0, 1]
    'colsample_by_tree': float. Col sub sample ratio of columns when constructing each tree.
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
        range: ['linear', 'logistic', 'tweedie']
    'base_score': The initial prediction score of all instances, global bias.
        default: 0
    'tree_growing_method': how to grow tree?
        default: level-wise
    'enable_packbits': bool. if true, turn on packbits transmission.
        default: False
    'eval_metric': str. evaluation metric name, must be one of 'roc_auc', 'tweedie_deviance', 'tweedie_nll', 'mse' or 'rmse'.
        'tweedie_nll' means tweedie negative log likelihood.
        Note if objective is not logistic, auc may not work.
        default: 'roc_auc'
    'enable_monitor': bool. whether enable model monitor call back.
        default: False
    'enable_early_stop': bool. whether enable early stop call back.
        default: False
    'validation_fraction': float. the fraction to use as the validation set.
    Only effective if early stop enabled.
        default: 0.1
        range: (0, 1)
    'stopping_rounds': int. if more than `stopping_rounds` consecutive rounds without improvement, training will stop.
    Only effective if early stop enabled.
        default: 1
    'stopping_tolerance': float. if the difference between current score and past best score is smaller than this threshold,
    then model is considered not inproving. Only effective if early stop enabled.
        default: 0.001
    'save_best_model': bool. whether save best model on validation set during training, only effective if early stop enabled.
        default: False
    'tweedie_variance_power': Parameter that controls the variance of the Tweedie distribution.
        var(y) ~ E(y)^tweedie_variance_power
        default: 1.5
        range: (1, 2)
        Set closer to 2 to shift towards a gamma distribution.
        Set closer to 1 to shift towards a Poisson distribution.
    """

    # security or encryption related params
    first_tree_with_label_holder_feature: bool = True
    fixed_point_parameter: int = 20
    batch_encoding_enabled: bool = True
    audit_paths: dict = field(default_factory=dict)
    enable_quantization: bool = False
    quantization_scale: float = 10000.0

    # tree boosting params
    num_boost_round: int = 10
    reg_lambda: float = 1.0
    learning_rate: float = 0.3
    # only effective if tree growing method is leaf wise
    max_leaf: int = 15
    # only effective if tree growing method is level wise
    max_depth: int = 5
    gamma: float = 0.0
    seed: int = 1212
    rowsample_by_tree: float = 1.0
    colsample_by_tree: float = 1.0
    enable_goss: bool = False
    top_rate: float = 0.3
    bottom_rate: float = 0.5
    sketch_eps: float = 0.1
    objective: RegType = RegType('logistic')
    base_score: float = 0.0
    tree_growing_method: TreeGrowingMethod = TreeGrowingMethod.LEVEL
    enable_packbits: bool = False

    # callback params
    eval_metric: str = 'roc_auc'
    enable_monitor: bool = False
    enable_early_stop: bool = False
    validation_fraction: float = 0.1
    stopping_rounds: int = 1
    stopping_tolerance: float = 0.0
    save_best_model: bool = False

    # objective-related
    # tweedie specific parameter, only effective if objective is tweedie
    tweedie_variance_power: float = 1.5


default_params = SGBParams()


def get_classic_lightGBM_params() -> dict:
    """Return a param dictionary that is typical for a lightGBM style training"""
    classic_lightGBM_params = SGBParams()
    classic_lightGBM_params.enable_goss = True
    classic_lightGBM_params.tree_growing_method = TreeGrowingMethod.LEAF
    result = asdict(classic_lightGBM_params)
    result['objective'] = classic_lightGBM_params.objective.value
    result['tree_growing_method'] = classic_lightGBM_params.tree_growing_method.value
    return result


def get_classic_XGB_params() -> dict:
    """Return a param dictionary that is typical for a XGB style training"""
    result = asdict(default_params)
    result['objective'] = default_params.objective.value
    result['tree_growing_method'] = default_params.tree_growing_method.value
    return result


param_names = set(asdict(default_params).keys())
numeric_params_range = {
    # param_name : (lower, higher, lower_inclusive, higher_inclusive)
    'fixed_point_parameter': (1, 100, True, True),
    'quantization_scale': (0, 10000000.0, True, True),
    'num_boost_round': (1, 1024, True, True),
    'reg_lambda': (0, 10000, True, True),
    'learning_rate': (0, 1, False, True),
    'max_leaf': (1, 32768, True, True),
    'max_depth': (1, 16, True, True),
    'gamma': (0, 10000, True, True),
    'rowsample_by_tree': (0, 1, False, True),
    'colsample_by_tree': (0, 1, False, True),
    'top_rate': (0, 1, False, False),
    'bottom_rate': (0, 1, False, False),
    'sketch_eps': (0, 1, False, True),
    'validation_fraction': (0, 1, False, False),
    'stopping_rounds': (1, 1024, True, True),
    'stopping_tolerance': (0, np.inf, True, False),
    'tweedie_variance_power': (1, 2, False, False),
    'base_score': (-10, 10, True, True),
}

categorical_params_options = {
    'objective': [e.value for e in RegType],
    'tree_growing_method': [e.value for e in TreeGrowingMethod],
    'eval_metric': ['roc_auc', 'tweedie_deviance', 'tweedie_nll', 'mse', 'rmse'],
}


def is_numeric_parameter_in_range(param_name, value) -> bool:
    min_value, max_value, lower_inclusive, higher_inclusive = numeric_params_range[
        param_name
    ]
    if lower_inclusive:
        if higher_inclusive:
            return min_value <= value <= max_value
        else:
            return min_value <= value < max_value
    else:
        if higher_inclusive:
            return min_value < value <= max_value
        else:
            return min_value < value < max_value


def is_categorical_parameter_valid_option(param_name, value) -> bool:
    return value in categorical_params_options[param_name]


def assert_numeric_parameter_in_range(param_name, value):
    if param_name not in numeric_params_range:
        return
    assert is_numeric_parameter_in_range(
        param_name, value
    ), f"{param_name} out of range, its range is  \
    (lower, higher, lower_inclusive, higher_inclusive):\
    {numeric_params_range[param_name]}, but got {value}"


def assert_categorical_parameter_valid_option(param_name, value):
    if param_name not in categorical_params_options:
        return
    assert is_categorical_parameter_valid_option(
        param_name, value
    ), f"{param_name} is not in valid options, \
    its valid options are {categorical_params_options[param_name]},\
    but got {value}"


def assert_parameter_combination_valid(params_dict):
    if (
        params_dict.get('enable_monitor', False)
        or params_dict.get('enable_early_stop', False)
    ) and 'eval_metric' in params_dict:
        objective = params_dict.get('objective', 'logistic')
        if objective == 'logistic':
            assert (
                params_dict['eval_metric'] == 'roc_auc'
            ), f"when objective is logistic, eval_metric must be auc, got {params_dict['eval_metric']}"
        if objective == 'tweedie':
            assert params_dict['eval_metric'] in [
                'tweedie_nll',
                'tweedie_deviance',
            ], f"when objective is tweedie, eval_metric must be tweedie_nll or tweedie_deviance, got {params_dict['eval_metric']}"
        if objective == 'linear':
            assert params_dict['eval_metric'] in [
                'mse',
                'rmse',
            ], f"when objective is linear, eval_metric must be mse or rmse, got {params_dict['eval_metric']}"


def get_unused_params(params) -> set:
    if isinstance(params, set):
        return params - param_names
    elif isinstance(params, dict):
        return set(params.keys()) - param_names
    else:
        raise NotImplementedError


def is_type_matched(param_name, value):
    if param_name in categorical_params_options:
        return isinstance(value, str)
    true_type = type(getattr(default_params, param_name))
    our_type = type(value)
    if true_type == float:
        return our_type == int or our_type == float
    return isinstance(value, true_type)


def type_and_range_check(params_dict):
    for param_name, value in params_dict.items():
        if param_name in param_names:
            assert is_type_matched(
                param_name, value
            ), f"type not correct for {param_name}, \
            expect {type(getattr(default_params, param_name))}, got {type(value)}, value is {value}"
            assert_numeric_parameter_in_range(param_name, value)
            assert_categorical_parameter_valid_option(param_name, value)

    assert_parameter_combination_valid(params_dict)


XGB_COMMON_PARAMS = [
    'reg_lambda',
    'seed',
    'learning_rate',
    'max_depth',
    'gamma',
    'base_score',
    'tweedie_variance_power',
]


OBJ_CONVERSION_DICT = {
    'binary:logistic': RegType.Logistic.value,
    'reg:logistic': RegType.Logistic.value,
    'reg:squarederror': RegType.Linear.value,
    'reg:tweedie': RegType.Tweedie.value,
}


def objective_conversion_function(xgb_obj_str: str) -> RegType:
    if xgb_obj_str in OBJ_CONVERSION_DICT:
        return OBJ_CONVERSION_DICT[xgb_obj_str]
    else:
        raise ValueError("Unsupported objective")


XGB_TO_SGB_PARAMS = {
    'max_leaves': ('max_leaf', lambda x: x),
    'max_bin': ('sketch_eps', lambda x: 1 / x),
    'subsample': ('rowsample_by_tree', lambda x: x),
    'colsample_bytree': ('colsample_by_tree', lambda x: x),
    'lambda': ('reg_lambda', lambda x: x),
    'eta': ('learning_rate', lambda x: x),
    'min_split_loss': ('gamma', lambda x: x),
    'n_estimators': ('num_boost_round', lambda x: x),
    'random_state': ('seed', lambda x: x),
    'early_stopping_rounds': ('stopping_rounds', lambda x: x),
    'objective': ('objective', objective_conversion_function),
}


# eval metric conversion is not supported for now, maybe added later
def xgb_params_converter(xgb_params: dict) -> dict:
    """Convert params from xgboost to sgb params

    Args:
        xgb_params (dict): xgb_params in dictionary. suppose clf is a fitted xgb model, this can be obtained by clf.get_params()

    Returns:
        dict: dictionary that can be used for sgb
    """
    sgb_params = get_classic_XGB_params()
    for k, v in xgb_params.items():
        if v is None:
            continue
        if k in XGB_COMMON_PARAMS:
            sgb_params[k] = v
            continue
        if k in XGB_TO_SGB_PARAMS:
            sgb_key, func = XGB_TO_SGB_PARAMS[k]
            sgb_params[sgb_key] = func(v)
            if k == "early_stopping_rounds" and v > 0:
                sgb_params['enable_early_stop'] = True
                sgb_params['save_best_model'] = True
            continue
    sgb_params.update(
        {
            'wait_execution': False,
            'first_tree_with_label_holder_feature': False,
            'enable_quantization': False,
            'enable_packbits': False,
            'enable_monitor': True,
        }
    )
    return sgb_params


def apply_new_params(old_params: dict, new_params: dict) -> dict:
    """Apply new params to old params"""
    result_params = old_params
    for k, v in new_params.items():
        result_params[k] = v
    return result_params


if __name__ == '__main__':
    print(param_names)
