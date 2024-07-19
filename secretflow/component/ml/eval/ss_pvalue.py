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

from secretflow.component.component import CompEvalError, Component, IoType
from secretflow.component.data_utils import (
    DistDataType,
    get_model_public_info,
    load_table,
)
from secretflow.component.ml.linear.ss_glm import load_ss_glm_model
from secretflow.component.ml.linear.ss_sgd import load_ss_sgd_model
from secretflow.device.device.spu import SPU
from secretflow.ml.linear import SSGLM, RegType, SSRegression
from secretflow.ml.linear.ss_glm.core.distribution import DistributionType
from secretflow.ml.linear.ss_glm.core.link import LinkType
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.report_pb2 import Descriptions, Div, Report, Tab
from secretflow.stats.ss_pvalue_v import PValue

ss_pvalue_comp = Component(
    "ss_pvalue",
    domain="ml.eval",
    version="0.0.1",
    desc="""Calculate P-Value for LR model training on vertical partitioning dataset by using secret sharing.

    For large dataset(large than 10w samples & 200 features),
    recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
    """,
)
ss_pvalue_comp.io(
    io_type=IoType.INPUT,
    name="model",
    desc="Input model.",
    types=[DistDataType.SS_SGD_MODEL, DistDataType.SS_GLM_MODEL],
)
ss_pvalue_comp.io(
    io_type=IoType.INPUT,
    name="input_data",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
)
ss_pvalue_comp.io(
    io_type=IoType.OUTPUT,
    name="report",
    desc="Output P-Value report.",
    types=[DistDataType.REPORT],
)


def sgd_pvalue(ctx, model_dd, input_data):
    spu_config = next(iter(ctx.spu_configs.values()))
    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])

    saved_model, model_meta = load_ss_sgd_model(ctx, spu, model_dd)
    model = SSRegression(spu)
    model.load_model(saved_model)

    label_col = model_meta["label_col"]
    feature_names = model_meta["feature_names"]
    partitions_order = list(model_meta["party_features_length"].keys())
    x, y, _, _ = load_ds(ctx, input_data, feature_names, partitions_order, label_col)

    with ctx.tracer.trace_running():
        yhat = model.predict(x)

    spu_w = model.spu_w
    with ctx.tracer.trace_running():
        if model.reg_type == RegType.Linear:
            return PValue(spu).t_statistic_p_value(x, y, yhat, spu_w), feature_names
        else:
            link = LinkType.Logit
            dist = DistributionType.Bernoulli
            return (
                PValue(spu).z_statistic_p_value(x, y, yhat, spu_w, link, dist),
                feature_names,
            )


def load_ds(
    ctx,
    input_data,
    x_cols,
    x_partitions_order,
    y_cols,
    offset_cols=None,
    weight_cols=None,
):
    with ctx.tracer.trace_io():
        x = load_table(
            ctx,
            input_data,
            partitions_order=x_partitions_order,
            load_features=True,
            col_selects=x_cols,
        )
        assert x.columns == x_cols

        y = load_table(
            ctx,
            input_data,
            load_features=True,
            load_labels=True,
            col_selects=y_cols,
        )

        if offset_cols:
            o = load_table(
                ctx,
                input_data,
                load_features=True,
                load_labels=True,
                col_selects=offset_cols,
            )
        else:
            o = None

        if weight_cols:
            w = load_table(
                ctx,
                input_data,
                load_features=True,
                load_labels=True,
                col_selects=weight_cols,
            )
        else:
            w = None

    return x, y, o, w


def glm_pvalue(ctx, model_dd, input_data):
    spu_config = next(iter(ctx.spu_configs.values()))
    cluster_def = spu_config["cluster_def"].copy()
    # forced to use 128 ring size & 40 fxp
    cluster_def["runtime_config"]["field"] = "FM128"
    cluster_def["runtime_config"]["fxp_fraction_bits"] = 40
    spu = SPU(cluster_def, spu_config["link_desc"])

    model_meta = get_model_public_info(model_dd)
    model = SSGLM(spu)
    model.spu_w, model.link, model.y_scale = load_ss_glm_model(ctx, spu, model_dd)

    offset_col = model_meta["offset_col"]
    label_col = model_meta["label_col"]
    sample_weight_col = model_meta["sample_weight_col"]
    feature_names = model_meta["feature_names"]
    partitions_order = list(model_meta["party_features_length"].keys())

    x, y, o, w = load_ds(
        ctx,
        input_data,
        feature_names,
        partitions_order,
        label_col,
        offset_col,
        sample_weight_col,
    )

    with ctx.tracer.trace_running():
        yhat = model.predict(x, o)

    spu_w = model.spu_w
    link = LinkType(model_meta["link"])
    dist = DistributionType(model_meta["dist"])
    tweedie_power = model_meta["tweedie_power"]
    y_scale = model.y_scale

    with ctx.tracer.trace_running():
        return (
            PValue(spu).z_statistic_p_value(
                x, y, yhat, spu_w, link, dist, tweedie_power, y_scale, sample_weights=w
            ),
            feature_names,
        )


@ss_pvalue_comp.eval_fn
def ss_pearsonr_eval_fn(
    *,
    ctx,
    model,
    input_data,
    report,
):
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")

    if model.type == DistDataType.SS_SGD_MODEL:
        pv, feature_names = sgd_pvalue(ctx, model, input_data)
    elif model.type == DistDataType.SS_GLM_MODEL:
        pv, feature_names = glm_pvalue(ctx, model, input_data)
    else:
        raise AttributeError(f"not support model.type {model.type}")

    assert pv.shape[0] == len(feature_names) + 1  # last one is bias

    r_desc = Descriptions(
        items=[
            Descriptions.Item(
                name=f'feature/{feature_names[i]}',
                type="float",
                value=Attribute(f=pv[i]),
            )
            for i in range(len(feature_names))
        ]
        + [
            Descriptions.Item(
                name="bias",
                type="float",
                value=Attribute(f=pv[len(feature_names)]),
            )
        ],
    )

    report_mate = Report(
        name="pvalue",
        desc="pvalue list",
        tabs=[
            Tab(
                divs=[
                    Div(
                        children=[
                            Div.Child(
                                type="descriptions",
                                descriptions=r_desc,
                            )
                        ],
                    )
                ],
            )
        ],
    )

    report_dd = DistData(
        name=report,
        type=str(DistDataType.REPORT),
        system_info=input_data.system_info,
    )
    report_dd.meta.Pack(report_mate)

    return {"report": report_dd}
