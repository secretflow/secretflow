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
import json
from typing import List

from secretflow.component.checkpoint import CompCheckpoint
from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import (
    DistDataType,
    SimpleVerticalBatchReader,
    get_model_public_info,
    load_table,
    model_dumps,
    model_loads,
    save_prediction_dd,
)
from secretflow.device.device.heu import heu_from_base_config
from secretflow.device.device.pyu import PYU
from secretflow.ml.boost.core.callback import TrainingCallback
from secretflow.ml.boost.core.metric import ALL_METRICS_NAMES
from secretflow.ml.boost.sgb_v import Sgb, SgbModel
from secretflow.ml.boost.sgb_v.checkpoint import (
    SGBCheckpointData,
    SGBSnapshot,
    build_sgb_model,
    sgb_model_to_snapshot,
)
from secretflow.ml.boost.sgb_v.core.params import RegType
from secretflow.ml.boost.sgb_v.factory.booster.global_ordermap_booster import (
    GlobalOrdermapBooster,
    build_checkpoint,
)
from secretflow.spec.v1.data_pb2 import DistData

DEFAULT_PREDICT_BATCH_SIZE = 10000

sgb_train_comp = Component(
    "sgb_train",
    domain="ml.train",
    version="0.0.4",
    desc="""Provides both classification and regression tree boosting (also known as GBDT, GBM)
    for vertical split dataset setting by using secure boost.

    - SGB is short for SecureBoost. Compared to its safer counterpart SS-XGB, SecureBoost focused on protecting label holder.

    - Check https://arxiv.org/abs/1901.08755.
    """,
)
sgb_train_comp.int_attr(
    name="num_boost_round",
    desc="Number of boosting iterations.",
    is_list=False,
    is_optional=True,
    default_value=10,
    allowed_values=None,
    lower_bound=1,
    upper_bound=None,
    lower_bound_inclusive=True,
)
sgb_train_comp.int_attr(
    name="max_depth",
    desc="Maximum depth of a tree.",
    is_list=False,
    is_optional=True,
    default_value=5,
    allowed_values=None,
    lower_bound=1,
    upper_bound=16,
    lower_bound_inclusive=True,
    upper_bound_inclusive=True,
)
sgb_train_comp.float_attr(
    name="learning_rate",
    desc="Step size shrinkage used in update to prevent overfitting.",
    is_list=False,
    is_optional=True,
    default_value=0.1,
    lower_bound=0,
    upper_bound=1,
    lower_bound_inclusive=False,
    upper_bound_inclusive=True,
)
sgb_train_comp.str_attr(
    name="objective",
    desc="Specify the learning objective.",
    is_list=False,
    is_optional=True,
    default_value="logistic",
    allowed_values=[reg_type.value for reg_type in RegType],
)
sgb_train_comp.float_attr(
    name="reg_lambda",
    desc="L2 regularization term on weights.",
    is_list=False,
    is_optional=True,
    default_value=0.1,
    lower_bound=0,
    upper_bound=10000,
    lower_bound_inclusive=True,
    upper_bound_inclusive=True,
)
sgb_train_comp.float_attr(
    name="gamma",
    desc="Greater than 0 means pre-pruning enabled. If gain of a node is less than this value, it would be pruned.",
    is_list=False,
    is_optional=True,
    default_value=1,
    lower_bound=0,
    upper_bound=10000,
    lower_bound_inclusive=True,
    upper_bound_inclusive=True,
)
sgb_train_comp.float_attr(
    name="colsample_by_tree",
    desc="Subsample ratio of columns when constructing each tree.",
    is_list=False,
    is_optional=True,
    default_value=1,
    lower_bound=0,
    upper_bound=1,
    lower_bound_inclusive=False,
    upper_bound_inclusive=True,
)
sgb_train_comp.float_attr(
    name="sketch_eps",
    desc="This roughly translates into O(1 / sketch_eps) number of bins.",
    is_list=False,
    is_optional=True,
    default_value=0.1,
    lower_bound=0,
    upper_bound=1,
    lower_bound_inclusive=False,
    upper_bound_inclusive=True,
)
sgb_train_comp.float_attr(
    name="base_score",
    desc="The initial prediction score of all instances, global bias.",
    is_list=False,
    is_optional=True,
    default_value=0,
    lower_bound=-10,
    lower_bound_inclusive=True,
    upper_bound=10,
    upper_bound_inclusive=True,
)
sgb_train_comp.int_attr(
    name="seed",
    desc="Pseudorandom number generator seed.",
    is_list=False,
    is_optional=True,
    default_value=42,
    lower_bound=0,
    lower_bound_inclusive=True,
)

sgb_train_comp.int_attr(
    name="fixed_point_parameter",
    desc="""Any floating point number encoded by heu,
            will multiply a scale and take the round,
            scale = 2 ** fixed_point_parameter.
            larger value may mean more numerical accuracy,
            but too large will lead to overflow problem.""",
    is_list=False,
    is_optional=True,
    default_value=20,
    lower_bound=1,
    upper_bound=100,
    lower_bound_inclusive=True,
    upper_bound_inclusive=True,
)
sgb_train_comp.bool_attr(
    name="first_tree_with_label_holder_feature",
    desc="Whether to train the first tree with label holder's own features.",
    is_list=False,
    is_optional=True,
    default_value=False,
)

sgb_train_comp.bool_attr(
    name="batch_encoding_enabled",
    desc="If use batch encoding optimization.",
    is_list=False,
    is_optional=True,
    default_value=True,
)
sgb_train_comp.bool_attr(
    name="enable_quantization",
    desc="Whether enable quantization of g and h.",
    is_list=False,
    is_optional=True,
    default_value=False,
)
sgb_train_comp.float_attr(
    name="quantization_scale",
    desc="Scale the sum of g to the specified value.",
    is_list=False,
    is_optional=True,
    default_value=10000.0,
    lower_bound=0,
    upper_bound=10000000.0,
    lower_bound_inclusive=True,
    upper_bound_inclusive=True,
)

sgb_train_comp.int_attr(
    name="max_leaf",
    desc="Maximum leaf of a tree. Only effective if train leaf wise.",
    is_list=False,
    is_optional=True,
    default_value=15,
    lower_bound=1,
    upper_bound=2**15,
    lower_bound_inclusive=True,
    upper_bound_inclusive=True,
)

sgb_train_comp.float_attr(
    name="rowsample_by_tree",
    desc="Row sub sample ratio of the training instances.",
    is_list=False,
    is_optional=True,
    default_value=1,
    lower_bound=0,
    upper_bound=1,
    lower_bound_inclusive=False,
    upper_bound_inclusive=True,
)

sgb_train_comp.bool_attr(
    name="enable_goss",
    desc="Whether to enable GOSS.",
    is_list=False,
    is_optional=True,
    default_value=False,
)
sgb_train_comp.float_attr(
    name="top_rate",
    desc="GOSS-specific parameter. The fraction of large gradients to sample.",
    is_list=False,
    is_optional=True,
    default_value=0.3,
    lower_bound=0,
    upper_bound=1,
    lower_bound_inclusive=False,
    upper_bound_inclusive=True,
)
sgb_train_comp.float_attr(
    name="bottom_rate",
    desc="GOSS-specific parameter. The fraction of small gradients to sample.",
    is_list=False,
    is_optional=True,
    default_value=0.5,
    lower_bound=0,
    upper_bound=1,
    lower_bound_inclusive=False,
    upper_bound_inclusive=True,
)
sgb_train_comp.str_attr(
    name="tree_growing_method",
    desc="How to grow tree?",
    is_list=False,
    is_optional=True,
    default_value="level",
)

sgb_train_comp.bool_attr(
    name="enable_early_stop",
    desc="Whether to enable early stop during training.",
    is_list=False,
    is_optional=True,
    default_value=False,
)

sgb_train_comp.bool_attr(
    name="enable_monitor",
    desc="Whether to enable monitoring performance during training.",
    is_list=False,
    is_optional=True,
    default_value=False,
)

sgb_train_comp.str_attr(
    name="eval_metric",
    desc=f"Use what metric for monitoring and early stop? Currently support {ALL_METRICS_NAMES}",
    is_list=False,
    is_optional=True,
    default_value="roc_auc",
    allowed_values=ALL_METRICS_NAMES,
)

sgb_train_comp.float_attr(
    name="validation_fraction",
    desc="Early stop specific parameter. Only effective if early stop enabled. The fraction of samples to use as validation set.",
    is_list=False,
    is_optional=True,
    default_value=0.1,
    lower_bound=0,
    upper_bound=1,
    lower_bound_inclusive=False,
    upper_bound_inclusive=False,
)

sgb_train_comp.int_attr(
    name="stopping_rounds",
    desc="""Early stop specific parameter. If more than `stopping_rounds` consecutive rounds without improvement, training will stop.
    Only effective if early stop enabled""",
    is_list=False,
    is_optional=True,
    default_value=1,
    lower_bound=1,
    upper_bound=1024,
    lower_bound_inclusive=True,
    upper_bound_inclusive=True,
)

sgb_train_comp.float_attr(
    name="stopping_tolerance",
    desc="Early stop specific parameter. If metric on validation set is no longer improving by at least this amount, then consider not improving.",
    is_list=False,
    is_optional=True,
    default_value=0.0,
    lower_bound=0,
    lower_bound_inclusive=True,
)

sgb_train_comp.float_attr(
    name="tweedie_variance_power",
    desc="Parameter that controls the variance of the Tweedie distribution.",
    is_list=False,
    is_optional=True,
    default_value=1.5,
    lower_bound=1.0,
    lower_bound_inclusive=False,
    upper_bound=2.0,
    upper_bound_inclusive=False,
)

sgb_train_comp.bool_attr(
    name="save_best_model",
    desc="Whether to save the best model on validation set during training.",
    is_list=False,
    is_optional=True,
    default_value=False,
)

sgb_train_comp.io(
    io_type=IoType.INPUT,
    name="train_dataset",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="feature_selects",
            desc="which features should be used for training.",
            col_min_cnt_inclusive=1,
        ),
        TableColParam(
            name="label",
            desc="Label of train dataset.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1,
        ),
    ],
)
sgb_train_comp.io(
    io_type=IoType.OUTPUT,
    name="output_model",
    desc="Output model.",
    types=[DistDataType.SGB_MODEL],
)

# current version 0.1
MODEL_MAX_MAJOR_VERSION = 0
MODEL_MAX_MINOR_VERSION = 1


@sgb_train_comp.enable_checkpoint
class SGBCheckpoint(CompCheckpoint):
    def associated_arg_names(self) -> List[str]:
        return [
            "num_boost_round",
            "max_depth",
            "learning_rate",
            "objective",
            "reg_lambda",
            "gamma",
            "rowsample_by_tree",
            "colsample_by_tree",
            "bottom_rate",
            "top_rate",
            "max_leaf",
            "quantization_scale",
            "sketch_eps",
            "base_score",
            "seed",
            "fixed_point_parameter",
            "enable_goss",
            "enable_quantization",
            "batch_encoding_enabled",
            "tree_growing_method",
            "first_tree_with_label_holder_feature",
            "enable_monitor",
            "enable_early_stop",
            "eval_metric",
            "validation_fraction",
            "stopping_rounds",
            "stopping_tolerance",
            "save_best_model",
            "train_dataset",
            "train_dataset_label",
            "train_dataset_feature_selects",
            "tweedie_variance_power",
        ]


def dump_sgb_checkpoint(
    ctx,
    uri: str,
    checkpoint: SGBCheckpointData,
    system_info,
) -> DistData:
    return model_dumps(
        ctx,
        "sgb",
        DistDataType.SGB_CHECKPOINT,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        checkpoint.model_objs,
        json.dumps(checkpoint.model_train_state_metas),
        uri,
        system_info,
    )


def load_sgb_checkpoint_data(
    ctx,
    cp: DistData,
    pyus: List[PYU],
) -> SGBCheckpointData:
    pyu_objs, model_train_state_meta_str = model_loads(
        ctx,
        cp,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        DistDataType.SGB_CHECKPOINT,
        pyus=pyus,
    )
    return SGBCheckpointData(pyu_objs, json.loads(model_train_state_meta_str))


# audit path is not supported in this form yet.
@sgb_train_comp.eval_fn
def sgb_train_eval_fn(
    *,
    ctx,
    num_boost_round,
    max_depth,
    learning_rate,
    objective,
    reg_lambda,
    gamma,
    rowsample_by_tree,
    colsample_by_tree,
    bottom_rate,
    top_rate,
    max_leaf,
    quantization_scale,
    sketch_eps,
    base_score,
    seed,
    fixed_point_parameter,
    enable_goss,
    enable_quantization,
    batch_encoding_enabled,
    tree_growing_method,
    first_tree_with_label_holder_feature,
    enable_monitor,
    enable_early_stop,
    eval_metric,
    validation_fraction,
    stopping_rounds,
    stopping_tolerance,
    tweedie_variance_power,
    save_best_model,
    train_dataset,
    train_dataset_label,
    output_model,
    train_dataset_feature_selects,
):
    assert ctx.heu_config is not None, "need heu config in SFClusterDesc"
    assert (
        len(set(train_dataset_label).intersection(set(train_dataset_feature_selects)))
        == 0
    ), f"expect no intersection between label and features, got {train_dataset_label} and {train_dataset_feature_selects}"
    y = load_table(
        ctx,
        train_dataset,
        load_labels=True,
        load_features=True,
        col_selects=train_dataset_label,
    )
    x = load_table(
        ctx,
        train_dataset,
        load_labels=True,
        load_features=True,
        col_selects=train_dataset_feature_selects,
    )
    assert len(x.columns) > 0

    label_party = next(iter(y.partitions.keys())).party
    heu = heu_from_base_config(
        ctx.heu_config,
        label_party,
        [p.party for p in x.partitions if p.party != label_party],
    )
    pyus = {p: PYU(p) for p in ctx.cluster_config.desc.parties}
    checkpoint_data = None
    if ctx.comp_checkpoint:
        cp_dd = ctx.comp_checkpoint.load()
        if cp_dd:
            checkpoint_data = load_sgb_checkpoint_data(ctx, cp_dd, pyus)

    def dump_function(
        model: GlobalOrdermapBooster,
        epoch: int,
        evals_log: TrainingCallback.EvalsLog,
    ):
        cp_uri = f"{output_model}_checkpoint_{epoch}"
        cp_dd = dump_sgb_checkpoint(
            ctx,
            cp_uri,
            build_checkpoint(model, evals_log, x, train_dataset_label),
            train_dataset.system_info,
        )
        ctx.comp_checkpoint.save(epoch, cp_dd)

    with ctx.tracer.trace_running():
        sgb = Sgb(heu)
        model = sgb.train(
            params={
                'num_boost_round': num_boost_round,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'objective': objective,
                'reg_lambda': reg_lambda,
                'gamma': gamma,
                'rowsample_by_tree': rowsample_by_tree,
                'colsample_by_tree': colsample_by_tree,
                'bottom_rate': bottom_rate,
                'top_rate': top_rate,
                'max_leaf': max_leaf,
                'quantization_scale': quantization_scale,
                'sketch_eps': sketch_eps,
                'base_score': base_score,
                'seed': seed,
                'fixed_point_parameter': fixed_point_parameter,
                'enable_goss': enable_goss,
                'enable_quantization': enable_quantization,
                'batch_encoding_enabled': batch_encoding_enabled,
                'tree_growing_method': tree_growing_method,
                'first_tree_with_label_holder_feature': first_tree_with_label_holder_feature,
                "enable_monitor": enable_monitor,
                "enable_early_stop": enable_early_stop,
                "eval_metric": eval_metric,
                "validation_fraction": validation_fraction,
                "stopping_rounds": stopping_rounds,
                "stopping_tolerance": stopping_tolerance,
                "save_best_model": save_best_model,
                "tweedie_variance_power": tweedie_variance_power,
            },
            dtrain=x,
            label=y,
            checkpoint_data=checkpoint_data,
            dump_function=dump_function if ctx.comp_checkpoint else None,
        )

    snapshot = sgb_model_to_snapshot(model, x, train_dataset_label)
    model_db = model_dumps(
        ctx,
        "sgb",
        DistDataType.SGB_MODEL,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        snapshot.model_objs,
        json.dumps(snapshot.model_meta),
        output_model,
        train_dataset.system_info,
    )

    return {"output_model": model_db}


sgb_predict_comp = Component(
    "sgb_predict",
    domain="ml.predict",
    version="0.0.3",
    desc="Predict using SGB model.",
)
sgb_predict_comp.party_attr(
    name="receiver",
    desc="Party of receiver.",
    list_min_length_inclusive=1,
    list_max_length_inclusive=1,
)
sgb_predict_comp.str_attr(
    name="pred_name",
    desc="Name for prediction column",
    is_list=False,
    is_optional=True,
    default_value="pred",
    allowed_values=None,
)
sgb_predict_comp.bool_attr(
    name="save_ids",
    desc=(
        "Whether to save ids columns into output prediction table. "
        "If true, input feature_dataset must contain id columns, and receiver party must be id owner."
    ),
    is_list=False,
    is_optional=True,
    default_value=False,
)
sgb_predict_comp.bool_attr(
    name="save_label",
    desc=(
        "Whether or not to save real label columns into output pred file. "
        "If true, input feature_dataset must contain label columns and receiver party must be label owner."
    ),
    is_list=False,
    is_optional=True,
    default_value=True,
)
sgb_predict_comp.io(
    io_type=IoType.INPUT, name="model", desc="model", types=["sf.model.sgb"]
)
sgb_predict_comp.io(
    io_type=IoType.INPUT,
    name="feature_dataset",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="saved_features",
            desc="which features should be saved with prediction result",
            col_min_cnt_inclusive=0,
        )
    ],
)
sgb_predict_comp.io(
    io_type=IoType.OUTPUT,
    name="pred",
    desc="Output prediction.",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=None,
)


def load_sgb_model(ctx, pyus, model) -> SgbModel:
    model_objs, model_meta_str = model_loads(
        ctx,
        model,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        DistDataType.SGB_MODEL,
        pyus=pyus,
    )

    return build_sgb_model(SGBSnapshot(model_objs, json.loads(model_meta_str)))


@sgb_predict_comp.eval_fn
def sgb_predict_eval_fn(
    *,
    ctx,
    feature_dataset,
    feature_dataset_saved_features,
    model,
    receiver,
    pred_name,
    pred,
    save_ids,
    save_label,
):
    model_public_info = get_model_public_info(model)
    assert len(model_public_info["feature_names"]) > 0
    feature_reader = SimpleVerticalBatchReader(
        ctx,
        feature_dataset,
        partitions_order=list(model_public_info["party_features_length"].keys()),
        col_selects=model_public_info["feature_names"],
    )

    pyus = {p: PYU(p) for p in ctx.cluster_config.desc.parties}

    sgb_model = load_sgb_model(ctx, pyus, model)

    receiver_pyu = PYU(receiver[0])

    def batch_pred():
        with ctx.tracer.trace_running():
            for batch in feature_reader:
                yield sgb_model.predict(batch, receiver_pyu)

    y_db = save_prediction_dd(
        ctx,
        pred,
        receiver_pyu,
        batch_pred(),
        pred_name,
        feature_dataset,
        feature_dataset_saved_features,
        model_public_info['label_col'] if save_label else [],
        save_ids,
    )

    return {"pred": y_db}
