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
import os

from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import (
    DistDataType,
    extract_table_header,
    gen_prediction_csv_meta,
    load_table,
    model_dumps,
    model_loads,
    save_prediction_csv,
)
from secretflow.device.device.heu import heu_from_base_config
from secretflow.device.device.pyu import PYU
from secretflow.device.driver import wait
from secretflow.ml.boost.sgb_v import Sgb, SgbModel
from secretflow.ml.boost.sgb_v.model import from_dict
from secretflow.spec.v1.data_pb2 import DistData

sgb_train_comp = Component(
    "sgb_train",
    domain="ml.train",
    version="0.0.1",
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
    allowed_values=["linear", "logistic"],
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
    default_value=0.1,
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
    lower_bound=0,
    lower_bound_inclusive=True,
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
sgb_train_comp.float_attr(
    name="early_stop_criterion_g_abs_sum",
    desc="If sum(abs(g)) is lower than or equal to this threshold, training will stop.",
    is_list=False,
    is_optional=True,
    default_value=0.0,
    lower_bound=0.0,
    lower_bound_inclusive=True,
)
sgb_train_comp.float_attr(
    name="early_stop_criterion_g_abs_sum_change_ratio",
    desc="If absolute g sum change ratio is lower than or equal to this threshold, training will stop.",
    is_list=False,
    is_optional=True,
    default_value=0.0,
    lower_bound=0,
    upper_bound=1,
    lower_bound_inclusive=True,
    upper_bound_inclusive=True,
)
sgb_train_comp.str_attr(
    name="tree_growing_method",
    desc="How to grow tree?",
    is_list=False,
    is_optional=True,
    default_value="level",
)

sgb_train_comp.io(
    io_type=IoType.INPUT,
    name="train_dataset",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="label",
            desc="Label of train dataset.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1,
        )
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
    early_stop_criterion_g_abs_sum_change_ratio,
    early_stop_criterion_g_abs_sum,
    tree_growing_method,
    first_tree_with_label_holder_feature,
    train_dataset,
    train_dataset_label,
    output_model,
):
    assert ctx.heu_config is not None, "need heu config in SFClusterDesc"

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
        col_excludes=train_dataset_label,
    )
    label_party = next(iter(y.partitions.keys())).party
    heu = heu_from_base_config(
        ctx.heu_config,
        label_party,
        [p.party for p in x.partitions if p.party != label_party],
    )

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
                'early_stop_criterion_g_abs_sum_change_ratio': early_stop_criterion_g_abs_sum_change_ratio,
                'early_stop_criterion_g_abs_sum': early_stop_criterion_g_abs_sum,
                'tree_growing_method': tree_growing_method,
                'first_tree_with_label_holder_feature': first_tree_with_label_holder_feature,
            },
            dtrain=x,
            label=y,
        )

    m_dict = model.to_dict()
    leaf_weights = m_dict.pop("leaf_weights")
    split_trees = m_dict.pop("split_trees")
    m_dict["label_holder"] = m_dict["label_holder"].party

    m_objs = sum([leaf_weights, *split_trees.values()], [])

    model_db = model_dumps(
        "sgb",
        DistDataType.SGB_MODEL,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        m_objs,
        json.dumps(m_dict),
        ctx.local_fs_wd,
        output_model,
        train_dataset.system_info,
    )

    return {"output_model": model_db}


sgb_predict_comp = Component(
    "sgb_predict",
    domain="ml.predict",
    version="0.0.1",
    desc="Predict using SGB model.",
)
sgb_predict_comp.str_attr(
    name="receiver",
    desc="Party of receiver.",
    is_list=False,
    is_optional=False,
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
    default_value=False,
)
sgb_predict_comp.io(
    io_type=IoType.INPUT, name="model", desc="model", types=["sf.model.sgb"]
)
sgb_predict_comp.io(
    io_type=IoType.INPUT,
    name="feature_dataset",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
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
        model,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        DistDataType.SGB_MODEL,
        # only local fs is supported at this moment.
        ctx.local_fs_wd,
        pyus=pyus,
    )

    model_meta = json.loads(model_meta_str)
    assert (
        isinstance(model_meta, dict)
        and "common" in model_meta
        and "label_holder" in model_meta
        and "tree_num" in model_meta["common"]
        and model_meta["label_holder"] in pyus
    )
    tree_num = model_meta["common"]["tree_num"]
    assert (
        tree_num > 0 and len(model_objs) % tree_num == 0
    ), f"model_objs {model_objs}, model_meta_str {model_meta_str}"
    leaf_weights = model_objs[:tree_num]
    split_trees = {}
    for pos in range(1, int(len(model_objs) / tree_num)):
        splits = model_objs[tree_num * pos : tree_num * (pos + 1)]
        assert splits[0].device not in split_trees
        split_trees[splits[0].device] = splits
    model_meta["leaf_weights"] = leaf_weights
    model_meta["split_trees"] = split_trees
    model_meta["label_holder"] = pyus[model_meta["label_holder"]]

    model = from_dict(model_meta)
    return model


@sgb_predict_comp.eval_fn
def sgb_predict_eval_fn(
    *,
    ctx,
    feature_dataset,
    model,
    receiver,
    pred_name,
    pred,
    save_ids,
    save_label,
):
    x = load_table(ctx, feature_dataset, load_features=True)
    pyus = {p.party: p for p in x.partitions.keys()}

    model = load_sgb_model(ctx, pyus, model)

    with ctx.tracer.trace_running():
        pyu = PYU(receiver)
        pyu_y = model.predict(x, pyu)

        y_path = os.path.join(ctx.local_fs_wd, pred)

        if save_ids:
            id_df = load_table(ctx, feature_dataset, load_ids=True)
            assert pyu in id_df.partitions
            id_header_map = extract_table_header(feature_dataset, load_ids=True)
            assert receiver in id_header_map
            id_header = list(id_header_map[receiver].keys())
            id_data = id_df.partitions[pyu].data
        else:
            id_header_map = None
            id_header = None
            id_data = None

        if save_label:
            label_df = load_table(ctx, feature_dataset, load_labels=True)
            assert pyu in label_df.partitions
            label_header_map = extract_table_header(feature_dataset, load_labels=True)
            assert receiver in label_header_map
            label_header = list(label_header_map[receiver].keys())
            label_data = label_df.partitions[pyu].data
        else:
            label_header_map = None
            label_header = None
            label_data = None

        wait(
            pyu(save_prediction_csv)(
                pyu_y.partitions[pyu],
                pred_name,
                y_path,
                label_data,
                label_header,
                id_data,
                id_header,
            )
        )

    y_db = DistData(
        name=pred_name,
        type=str(DistDataType.INDIVIDUAL_TABLE),
        data_refs=[DistData.DataRef(uri=pred, party=receiver, format="csv")],
    )

    meta = gen_prediction_csv_meta(
        id_header=id_header_map,
        label_header=label_header_map,
        party=receiver,
        pred_name=pred_name,
        line_count=x.shape[0],
        id_keys=id_header,
        label_keys=label_header,
    )

    y_db.meta.Pack(meta)

    return {"pred": y_db}
