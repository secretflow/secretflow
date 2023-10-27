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
from typing import Tuple

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
from secretflow.device.device.spu import SPU, SPUObject
from secretflow.device.driver import wait
from secretflow.ml.linear import SSGLM
from secretflow.ml.linear.ss_glm.core import Linker, get_link
from secretflow.spec.v1.data_pb2 import DistData

ss_glm_train_comp = Component(
    "ss_glm_train",
    domain="ml.train",
    version="0.0.1",
    desc="""generalized linear model (GLM) is a flexible generalization of ordinary linear regression.
    The GLM generalizes linear regression by allowing the linear model to be related to the response
    variable via a link function and by allowing the magnitude of the variance of each measurement to
    be a function of its predicted value.""",
)
ss_glm_train_comp.int_attr(
    name="epochs",
    desc="The number of complete pass through the training data.",
    is_list=False,
    is_optional=True,
    default_value=10,
    allowed_values=None,
    lower_bound=1,
    upper_bound=None,
    lower_bound_inclusive=True,
)
ss_glm_train_comp.float_attr(
    name="learning_rate",
    desc="The step size at each iteration in one iteration.",
    is_list=False,
    is_optional=True,
    default_value=0.1,
    lower_bound=0,
    lower_bound_inclusive=False,
)
ss_glm_train_comp.int_attr(
    name="batch_size",
    desc="The number of training examples utilized in one iteration.",
    is_list=False,
    is_optional=True,
    default_value=1024,
    lower_bound=0,
    lower_bound_inclusive=False,
)
ss_glm_train_comp.str_attr(
    name="link_type",
    desc="link function type",
    is_list=False,
    is_optional=False,
    allowed_values=["Logit", "Log", "Reciprocal", "Indentity"],
)
ss_glm_train_comp.str_attr(
    name="label_dist_type",
    desc="label distribution type",
    is_list=False,
    is_optional=False,
    allowed_values=["Bernoulli", "Poisson", "Gamma", "Tweedie"],
)
ss_glm_train_comp.float_attr(
    name="tweedie_power",
    desc="Tweedie distribution power parameter",
    is_list=False,
    is_optional=True,
    default_value=1,
    lower_bound=0,
    lower_bound_inclusive=True,
    upper_bound=2,
    upper_bound_inclusive=True,
)
ss_glm_train_comp.float_attr(
    name="dist_scale",
    desc="A guess value for distribution's scale",
    is_list=False,
    is_optional=True,
    default_value=1,
    lower_bound=1,
    lower_bound_inclusive=True,
)
ss_glm_train_comp.float_attr(
    name="eps",
    desc="""If the change rate of weights is less than this threshold,
            the model is considered to be converged,
            and the training stops early. 0 to disable.""",
    is_list=False,
    is_optional=True,
    default_value=0.0001,
    lower_bound=0,
    lower_bound_inclusive=True,
)
ss_glm_train_comp.int_attr(
    name="iter_start_irls",
    desc="""run a few rounds of IRLS training as the initialization of w,
    0 disable""",
    is_list=False,
    is_optional=True,
    default_value=0,
    lower_bound=0,
    lower_bound_inclusive=True,
)
ss_glm_train_comp.int_attr(
    name="decay_epoch",
    desc="""decay learning interval""",
    is_list=False,
    is_optional=True,
    default_value=0,
    lower_bound=0,
    lower_bound_inclusive=True,
)
ss_glm_train_comp.float_attr(
    name="decay_rate",
    desc="""decay learning rate""",
    is_list=False,
    is_optional=True,
    default_value=0,
    lower_bound=0,
    lower_bound_inclusive=True,
    upper_bound=1,
    upper_bound_inclusive=False,
)
ss_glm_train_comp.str_attr(
    name="optimizer",
    desc="which optimizer to use: IRLS(Iteratively Reweighted Least Squares) or SGD(Stochastic Gradient Descent)",
    is_list=False,
    is_optional=False,
    allowed_values=["SGD", "IRLS"],
)
ss_glm_train_comp.str_attr(
    name="offset_col",
    desc="Specify a column to use as the offset",
    is_list=False,
    is_optional=True,
    default_value="",
)
ss_glm_train_comp.str_attr(
    name="weight_col",
    desc="Specify a column to use for the observation weights",
    is_list=False,
    is_optional=True,
    default_value="",
)
ss_glm_train_comp.io(
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
ss_glm_train_comp.io(
    io_type=IoType.OUTPUT,
    name="output_model",
    desc="Output model.",
    types=[DistDataType.SS_GLM_MODEL],
)

# current version 0.1
MODEL_MAX_MAJOR_VERSION = 0
MODEL_MAX_MINOR_VERSION = 1


@ss_glm_train_comp.eval_fn
def ss_glm_train_eval_fn(
    *,
    ctx,
    epochs,
    learning_rate,
    batch_size,
    link_type,
    label_dist_type,
    tweedie_power,
    dist_scale,
    eps,
    iter_start_irls,
    optimizer,
    offset_col,
    weight_col,
    decay_epoch,
    decay_rate,
    train_dataset,
    train_dataset_label,
    output_model,
):
    # only local fs is supported at this moment.
    local_fs_wd = ctx.local_fs_wd

    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])

    glm = SSGLM(spu)

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
    if offset_col:
        assert (
            offset_col in x.columns
        ), f"can't find offset_col {offset_col} in train_dataset"
        offset = x[offset_col]
        x.drop(columns=offset_col, inplace=True)
    else:
        offset = None

    if weight_col:
        assert (
            weight_col in x.columns
        ), f"can't find weight_col {weight_col} in train_dataset"
        weight = x[weight_col]
        x.drop(columns=weight_col, inplace=True)
    else:
        weight = None

    with ctx.tracer.trace_running():
        if optimizer == "SGD":
            if decay_epoch == 0 or decay_rate == 0:
                decay_rate = None
                decay_epoch = None

            glm.fit_sgd(
                x=x,
                y=y,
                offset=offset,
                weight=weight,
                epochs=epochs,
                link=link_type,
                dist=label_dist_type,
                tweedie_power=tweedie_power,
                scale=dist_scale,
                learning_rate=learning_rate,
                batch_size=batch_size,
                iter_start_irls=iter_start_irls,
                eps=eps,
                decay_epoch=decay_epoch,
                decay_rate=decay_rate,
            )
        elif optimizer == "IRLS":
            glm.fit_irls(
                x=x,
                y=y,
                offset=offset,
                weight=weight,
                epochs=epochs,
                link=link_type,
                dist=label_dist_type,
                tweedie_power=tweedie_power,
                scale=dist_scale,
                eps=eps,
            )
        else:
            raise CompEvalError(f"Unknown optimizer {optimizer}")

    model_meta = {"link": glm.link.link_type().value, "y_scale": glm.y_scale}

    model_db = model_dumps(
        "ss_glm",
        DistDataType.SS_GLM_MODEL,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        [glm.spu_w],
        json.dumps(model_meta),
        local_fs_wd,
        output_model,
        train_dataset.system_info,
    )

    return {"output_model": model_db}


ss_glm_predict_comp = Component(
    "ss_glm_predict",
    domain="ml.predict",
    version="0.0.1",
    desc="Predict using the SSGLM model.",
)
ss_glm_predict_comp.str_attr(
    name="receiver",
    desc="Party of receiver.",
    is_list=False,
    is_optional=False,
)
ss_glm_predict_comp.str_attr(
    name="pred_name",
    desc="Column name for predictions.",
    is_list=False,
    is_optional=True,
    default_value="pred",
    allowed_values=None,
)
ss_glm_predict_comp.bool_attr(
    name="save_ids",
    desc=(
        "Whether to save ids columns into output prediction table. "
        "If true, input feature_dataset must contain id columns, and receiver party must be id owner."
    ),
    is_list=False,
    is_optional=True,
    default_value=False,
)
ss_glm_predict_comp.bool_attr(
    name="save_label",
    desc=(
        "Whether or not to save real label columns into output pred file. "
        "If true, input feature_dataset must contain label columns and receiver party must be label owner."
    ),
    is_list=False,
    is_optional=True,
    default_value=False,
)
ss_glm_predict_comp.str_attr(
    name="offset_col",
    desc="Specify a column to use as the offset",
    is_list=False,
    is_optional=True,
    default_value="",
)
ss_glm_predict_comp.io(
    io_type=IoType.INPUT,
    name="model",
    desc="Input model.",
    types=[DistDataType.SS_GLM_MODEL],
)
ss_glm_predict_comp.io(
    io_type=IoType.INPUT,
    name="feature_dataset",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)
ss_glm_predict_comp.io(
    io_type=IoType.OUTPUT,
    name="pred",
    desc="Output prediction.",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=None,
)


def load_ss_glm_model(ctx, spu, model) -> Tuple[SPUObject, Linker, float]:
    model_objs, model_meta_str = model_loads(
        model,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        DistDataType.SS_GLM_MODEL,
        # only local fs is supported at this moment.
        ctx.local_fs_wd,
        spu=spu,
    )
    assert len(model_objs) == 1 and isinstance(
        model_objs[0], SPUObject
    ), f"model_objs {model_objs}, model_meta_str {model_meta_str}"

    model_meta = json.loads(model_meta_str)
    assert (
        isinstance(model_meta, dict)
        and "link" in model_meta
        and "y_scale" in model_meta
    ), f"model meta format err {model_meta}"

    return model_objs[0], get_link(model_meta["link"]), float(model_meta["y_scale"])


@ss_glm_predict_comp.eval_fn
def ss_glm_predict_eval_fn(
    *,
    ctx,
    feature_dataset,
    offset_col,
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

    model = load_ss_glm_model(ctx, spu, model)

    glm = SSGLM(spu)
    glm.spu_w, glm.link, glm.y_scale = model

    x = load_table(ctx, feature_dataset, load_features=True)

    if offset_col:
        assert (
            offset_col in x.columns
        ), f"can't find offset_col {offset_col} in train_dataset"
        offset = x[offset_col]
        x.drop(columns=offset_col, inplace=True)
    else:
        offset = None

    with ctx.tracer.trace_running():
        pyu = PYU(receiver)
        pyu_y = glm.predict(
            x=x,
            o=offset,
            to_pyu=pyu,
        )

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
