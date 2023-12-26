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
    generate_random_string,
    get_model_public_info,
    load_table,
    model_dumps,
    model_loads,
    save_prediction_csv,
)
from secretflow.device.device.pyu import PYU
from secretflow.device.device.spu import SPU, SPUObject
from secretflow.device.driver import reveal, wait
from secretflow.ml.linear import SSGLM
from secretflow.ml.linear.ss_glm.core import Linker, get_link
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.report_pb2 import Descriptions, Div, Report, Tab

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
ss_glm_train_comp.float_attr(
    name="l2_lambda",
    desc="L2 regularization term",
    is_list=False,
    is_optional=True,
    default_value=0.1,
    lower_bound=0,
    lower_bound_inclusive=True,
)
ss_glm_train_comp.bool_attr(
    name="report_weights",
    desc="If this option is set to true, model will be revealed and model details are visible to all parties",
    is_list=False,
    is_optional=True,
    default_value=False,
)
ss_glm_train_comp.io(
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
            name="offset",
            desc="Specify a column to use as the offset",
            col_min_cnt_inclusive=0,
            col_max_cnt_inclusive=1,
        ),
        TableColParam(
            name="weight",
            desc="Specify a column to use for the observation weights",
            col_min_cnt_inclusive=0,
            col_max_cnt_inclusive=1,
        ),
        TableColParam(
            name="label",
            desc="Label of train dataset.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1,
        ),
    ],
)
ss_glm_train_comp.io(
    io_type=IoType.OUTPUT,
    name="output_model",
    desc="Output model.",
    types=[DistDataType.SS_GLM_MODEL],
)
ss_glm_train_comp.io(
    io_type=IoType.OUTPUT,
    name="report",
    desc="If report_weights is true, report model details",
    types=[DistDataType.REPORT],
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
    l2_lambda,
    report_weights,
    train_dataset_offset,
    train_dataset_weight,
    decay_epoch,
    decay_rate,
    train_dataset,
    train_dataset_label,
    output_model,
    train_dataset_feature_selects,
    report,
):
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    cluster_def = spu_config["cluster_def"].copy()

    # forced to use 128 ring size & 40 fxp
    cluster_def["runtime_config"]["field"] = "FM128"
    cluster_def["runtime_config"]["fxp_fraction_bits"] = 40

    spu = SPU(cluster_def, spu_config["link_desc"])

    glm = SSGLM(spu)

    assert (
        train_dataset_label[0] not in train_dataset_feature_selects
    ), f"col {train_dataset_label[0]} used in both label and features"

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

    col_selects = x.columns
    if train_dataset_offset:
        assert (
            train_dataset_offset[0] not in train_dataset_feature_selects
        ), f"col {train_dataset_offset[0]} used in both offset and features"
        offset = load_table(
            ctx,
            train_dataset,
            load_labels=True,
            load_features=True,
            col_selects=train_dataset_offset,
        )
        offset_col = train_dataset_offset[0]
    else:
        offset = None
        offset_col = ""

    if train_dataset_weight:
        assert (
            train_dataset_weight[0] not in train_dataset_feature_selects
        ), f"col {train_dataset_weight[0]} used in both weight and features"
        weight = load_table(
            ctx,
            train_dataset,
            load_labels=True,
            load_features=True,
            col_selects=train_dataset_weight,
        )
    else:
        weight = None

    l2_lambda = l2_lambda if l2_lambda > 0 else None

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
                l2_lambda=l2_lambda,
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
                l2_lambda=l2_lambda,
            )
        else:
            raise CompEvalError(f"Unknown optimizer {optimizer}")

    feature_names = x.columns
    party_features_length = {
        device.party: len(columns) for device, columns in x.partition_columns.items()
    }

    model_meta = {
        "link": glm.link.link_type().value,
        "y_scale": glm.y_scale,
        "col_selects": col_selects,
        "offset_col": offset_col,
        "label_col": train_dataset_label,
        "feature_names": feature_names,
        "party_features_length": party_features_length,
        "model_hash": generate_random_string(next(iter(x.partition_columns.keys()))),
    }

    model_db = model_dumps(
        ctx,
        "ss_glm",
        DistDataType.SS_GLM_MODEL,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        [glm.spu_w],
        json.dumps(model_meta),
        output_model,
        train_dataset.system_info,
    )

    if report_weights:
        weights = list(map(float, list(reveal(glm.spu_w))))
        named_weight = {}
        for _, features in x.partition_columns.items():
            party_weight = weights[: len(features)]
            named_weight.update({f: w for f, w in zip(features, party_weight)})
            weights = weights[len(features) :]
        assert len(weights) == 1

        w_desc = Descriptions(
            items=[
                Descriptions.Item(
                    name="_intercept_", type="float", value=Attribute(f=weights[-1])
                ),
                Descriptions.Item(
                    name="_y_scale_", type="float", value=Attribute(f=glm.y_scale)
                ),
            ]
            + [
                Descriptions.Item(name=f, type="float", value=Attribute(f=w))
                for f, w in named_weight.items()
            ],
        )

        report_mate = Report(
            name="weights",
            desc="weights",
            tabs=[
                Tab(
                    divs=[
                        Div(
                            children=[
                                Div.Child(
                                    type="descriptions",
                                    descriptions=w_desc,
                                )
                            ],
                        )
                    ],
                )
            ],
        )
    else:
        report_mate = Report(
            name="weights",
            desc="weights",
        )

    report_dd = DistData(
        name=report,
        type=str(DistDataType.REPORT),
        system_info=train_dataset.system_info,
    )
    report_dd.meta.Pack(report_mate)

    return {"output_model": model_db, "report": report_dd}


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
    default_value=True,
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
        ctx,
        model,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        DistDataType.SS_GLM_MODEL,
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

    return (
        model_objs[0],
        get_link(model_meta["link"]),
        float(model_meta["y_scale"]),
    )


@ss_glm_predict_comp.eval_fn
def ss_glm_predict_eval_fn(
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

    cluster_def = spu_config["cluster_def"].copy()

    # forced to use 128 ring size & 40 fxp
    cluster_def["runtime_config"]["field"] = "FM128"
    cluster_def["runtime_config"]["fxp_fraction_bits"] = 40

    spu = SPU(cluster_def, spu_config["link_desc"])

    model_public_info = get_model_public_info(model)

    glm = SSGLM(spu)
    glm.spu_w, glm.link, glm.y_scale = load_ss_glm_model(ctx, spu, model)

    x = load_table(
        ctx,
        feature_dataset,
        load_features=True,
        col_selects=model_public_info['col_selects'],
    )

    offset_col = model_public_info['offset_col']

    if offset_col:
        offset = load_table(
            ctx,
            feature_dataset,
            load_labels=True,
            load_features=True,
            col_selects=[offset_col],
        )
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
            label_df = load_table(
                ctx,
                feature_dataset,
                load_features=True,
                load_labels=True,
                col_selects=model_public_info['label_col'],
            )
            assert pyu in label_df.partitions
            label_header_map = extract_table_header(
                feature_dataset,
                load_features=True,
                load_labels=True,
                col_selects=model_public_info['label_col'],
            )
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
