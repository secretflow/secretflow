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

from secretflow.component.component import (
    CompEvalError,
    Component,
    IoType,
    TableColParam,
)
from secretflow.component.data_utils import (
    DistDataType,
    extract_table_header,
    gen_prediction_csv_meta,
    load_table,
    model_dumps,
    model_loads,
    save_prediction_csv,
)
from secretflow.device.device.pyu import PYU
from secretflow.device.device.spu import SPU
from secretflow.device.driver import wait
from secretflow.ml.boost.ss_xgb_v import Xgb, XgbModel
from secretflow.ml.boost.ss_xgb_v.core.node_split import RegType
from secretflow.spec.v1.data_pb2 import DistData

ss_xgb_train_comp = Component(
    "ss_xgb_train",
    domain="ml.train",
    version="0.0.1",
    desc="""This method provides both classification and regression tree boosting (also known as GBDT, GBM)
    for vertical partitioning dataset setting by using secret sharing.

    - SS-XGB is short for secret sharing XGB.
    - More details: https://arxiv.org/pdf/2005.08479.pdf
    """,
)
ss_xgb_train_comp.int_attr(
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
ss_xgb_train_comp.int_attr(
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
ss_xgb_train_comp.float_attr(
    name="learning_rate",
    desc="Step size shrinkage used in updates to prevent overfitting.",
    is_list=False,
    is_optional=True,
    default_value=0.1,
    lower_bound=0,
    upper_bound=1,
    lower_bound_inclusive=False,
    upper_bound_inclusive=True,
)
ss_xgb_train_comp.str_attr(
    name="objective",
    desc="Specify the learning objective.",
    is_list=False,
    is_optional=True,
    default_value="logistic",
    allowed_values=["linear", "logistic"],
)
ss_xgb_train_comp.float_attr(
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
ss_xgb_train_comp.float_attr(
    name="subsample",
    desc="Subsample ratio of the training instances.",
    is_list=False,
    is_optional=True,
    default_value=0.1,
    lower_bound=0,
    upper_bound=1,
    lower_bound_inclusive=False,
    upper_bound_inclusive=True,
)
ss_xgb_train_comp.float_attr(
    name="colsample_by_tree",
    desc="Subsample ratio of columns when constructing each tree.",
    is_list=False,
    is_optional=True,
    default_value=0.1,
    lower_bound=0,
    upper_bound=1,
    lower_bound_inclusive=False,
    upper_bound_inclusive=True,
)
ss_xgb_train_comp.float_attr(
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
ss_xgb_train_comp.float_attr(
    name="base_score",
    desc="The initial prediction score of all instances, global bias.",
    is_list=False,
    is_optional=True,
    default_value=0,
    lower_bound=0,
    lower_bound_inclusive=True,
)
ss_xgb_train_comp.int_attr(
    name="seed",
    desc="Pseudorandom number generator seed.",
    is_list=False,
    is_optional=True,
    default_value=42,
    lower_bound=0,
    lower_bound_inclusive=True,
)

ss_xgb_train_comp.io(
    io_type=IoType.INPUT,
    name="train_dataset",
    desc="Input vertical table.",
    types=["sf.table.vertical_table"],
    col_params=[
        TableColParam(
            name="label",
            desc="Label of train dataset.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1,
        )
    ],
)
ss_xgb_train_comp.io(
    io_type=IoType.OUTPUT,
    name="output_model",
    desc="Output model.",
    types=[DistDataType.SS_XGB_MODEL],
)

# current version 0.1
MODEL_MAX_MAJOR_VERSION = 0
MODEL_MAX_MINOR_VERSION = 1


@ss_xgb_train_comp.eval_fn
def ss_xgb_train_eval_fn(
    *,
    ctx,
    num_boost_round,
    max_depth,
    learning_rate,
    objective,
    reg_lambda,
    subsample,
    colsample_by_tree,
    sketch_eps,
    base_score,
    seed,
    train_dataset,
    train_dataset_label,
    output_model,
):
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])

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

    with ctx.tracer.trace_running():
        sgb = Xgb(spu)
        model = sgb.train(
            params={
                "num_boost_round": num_boost_round,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "objective": objective,
                "reg_lambda": reg_lambda,
                "subsample": subsample,
                "colsample_by_tree": colsample_by_tree,
                "sketch_eps": sketch_eps,
                "base_score": base_score,
                "seed": seed,
            },
            dtrain=x,
            label=y,
        )

    m_dict = {
        "objective": model.objective.value,
        "base": model.base,
        "tree_num": len(model.weights),
    }
    split_trees = []
    for p in x.partitions.keys():
        split_trees.extend([t[p] for t in model.trees])

    model_db = model_dumps(
        "sgb",
        DistDataType.SS_XGB_MODEL,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        [*model.weights, *split_trees],
        json.dumps(m_dict),
        ctx.local_fs_wd,
        output_model,
        train_dataset.system_info,
    )

    return {"output_model": model_db}


ss_xgb_predict_comp = Component(
    "ss_xgb_predict",
    domain="ml.predict",
    version="0.0.1",
    desc="Predict using the SS-XGB model.",
)
ss_xgb_predict_comp.str_attr(
    name="receiver",
    desc="Party of receiver.",
    is_list=False,
    is_optional=False,
)
ss_xgb_predict_comp.str_attr(
    name="pred_name",
    desc="Column name for predictions.",
    is_list=False,
    is_optional=True,
    default_value="pred",
    allowed_values=None,
)
ss_xgb_predict_comp.bool_attr(
    name="save_ids",
    desc=(
        "Whether to save ids columns into output prediction table. "
        "If true, input feature_dataset must contain id columns, and receiver party must be id owner."
    ),
    is_list=False,
    is_optional=True,
    default_value=False,
)
ss_xgb_predict_comp.bool_attr(
    name="save_label",
    desc=(
        "Whether or not to save real label columns into output pred file. "
        "If true, input feature_dataset must contain label columns and receiver party must be label owner."
    ),
    is_list=False,
    is_optional=True,
    default_value=False,
)
ss_xgb_predict_comp.io(
    io_type=IoType.INPUT,
    name="model",
    desc="Input model.",
    types=[DistDataType.SS_XGB_MODEL],
)
ss_xgb_predict_comp.io(
    io_type=IoType.INPUT,
    name="feature_dataset",
    desc="Input vertical table.",
    types=["sf.table.vertical_table"],
    col_params=None,
)
ss_xgb_predict_comp.io(
    io_type=IoType.OUTPUT,
    name="pred",
    desc="Output prediction.",
    types=["sf.table.individual"],
    col_params=None,
)


def load_ss_xgb_model(ctx, spu, pyus, model) -> XgbModel:
    model_objs, model_meta_str = model_loads(
        model,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        DistDataType.SS_XGB_MODEL,
        # only local fs is supported at this moment.
        ctx.local_fs_wd,
        pyus=pyus,
        spu=spu,
    )

    model_meta = json.loads(model_meta_str)
    assert (
        isinstance(model_meta, dict)
        and "objective" in model_meta
        and "base" in model_meta
        and "tree_num" in model_meta
    )
    tree_num = model_meta["tree_num"]
    assert (
        tree_num > 0 and len(model_objs) % tree_num == 0
    ), f"model_objs {model_objs}, model_meta_str {model_meta_str}"
    weights = model_objs[:tree_num]
    trees = []
    parties_num = int(len(model_objs) / tree_num) - 1
    for pos in range(tree_num):
        tree = {}
        for p in range(parties_num):
            obj = model_objs[tree_num * (p + 1) + pos]
            tree[obj.device] = obj
        trees.append(tree)

    model = XgbModel(spu, RegType(model_meta["objective"]), model_meta["base"])
    model.weights = weights
    model.trees = trees

    return model


@ss_xgb_predict_comp.eval_fn
def ss_xgb_predict_eval_fn(
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
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])
    x = load_table(ctx, feature_dataset, load_features=True)
    pyus = {p.party: p for p in x.partitions.keys()}

    model = load_ss_xgb_model(ctx, spu, pyus, model)

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
