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
import logging

from google.protobuf.json_format import MessageToJson

from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Output,
    register,
    Reporter,
    SPU_RUNTIME_CONFIG_FM128_FXP40,
    SS_GLM_MODEL_MAX,
    SS_SGD_MODEL_MAX,
    VTable,
    VTableFieldKind,
)
from secretflow.device import SPUObject
from secretflow.ml.linear import RegType, SSGLM, SSRegression
from secretflow.ml.linear.linear_model import LinearModel
from secretflow.ml.linear.ss_glm.core.distribution import DistributionType
from secretflow.ml.linear.ss_glm.core.link import get_link, LinkType
from secretflow.stats.ss_pvalue_v import PValue
from secretflow.utils.sigmoid import SigType


@register(domain="ml.eval", version="1.0.0", name="ss_pvalue")
class SSPValue(Component):
    '''
    Calculate P-Value for LR model training on vertical partitioning dataset by using secret sharing.

    For large dataset(large than 10w samples & 200 features),
    recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
    '''

    input_model: Input = Field.input(
        desc="Input model.",
        types=[DistDataType.SS_SGD_MODEL, DistDataType.SS_GLM_MODEL],
    )
    input_ds: Input = Field.input(
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    report: Output = Field.output(
        desc="Output P-Value report.",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        if self.input_model.type == DistDataType.SS_SGD_MODEL:
            pv, feature_names = self.sgd_pvalue(ctx)
        elif self.input_model.type == DistDataType.SS_GLM_MODEL:
            pv, feature_names = self.glm_pvalue(ctx)
        else:
            raise AttributeError(f"not support model.type {self.input_model.type}")

        assert pv.shape[0] == len(feature_names) + 1  # last one is bias
        feature_names = [f"feature/{name}" for name in feature_names]
        feature_names.append('bias')
        desc = {name: value for name, value in zip(feature_names, pv)}

        system_info = self.input_ds.system_info
        r = Reporter(name="pvalue", desc="pvalue list", system_info=system_info)
        r.add_tab(desc)
        logging.info(f'\n--\n*report* \n\n{MessageToJson(r.report())}\n--\n')
        self.report.data = r.to_distdata()

    def sgd_pvalue(self, ctx: Context) -> tuple[list[float], list[str]]:
        spu = ctx.make_spu()
        model = ctx.load_model(
            self.input_model, DistDataType.SS_SGD_MODEL, SS_SGD_MODEL_MAX, spu=spu
        )
        assert len(model.objs) == 1 and isinstance(
            model.objs[0], SPUObject
        ), f"model_objs {model.objs}, model_meta_str {model.public_info}"
        model_meta = json.loads(model.public_info)
        assert (
            isinstance(model_meta, dict)
            and "reg_type" in model_meta
            and "sig_type" in model_meta
        )
        lr_model = LinearModel(
            weights=model.objs[0],
            reg_type=RegType(model_meta["reg_type"]),
            sig_type=SigType(model_meta["sig_type"]),
        )

        ss_reg = SSRegression(spu)
        ss_reg.load_model(lr_model)

        label_col = model_meta["label_col"]
        feature_names = model_meta["feature_names"]
        partitions_order = list(model_meta["party_features_length"].keys())
        x, y, _, _ = self.load_ds(ctx, feature_names, partitions_order, label_col)

        with ctx.trace_running():
            yhat = ss_reg.predict(x)

        spu_w = ss_reg.spu_w
        with ctx.trace_running():
            if ss_reg.reg_type == RegType.Linear:
                return PValue(spu).t_statistic_p_value(x, y, yhat, spu_w), feature_names
            else:
                link = LinkType.Logit
                dist = DistributionType.Bernoulli
                return (
                    PValue(spu).z_statistic_p_value(x, y, yhat, spu_w, link, dist),
                    feature_names,
                )

    def glm_pvalue(self, ctx: Context) -> tuple[list[float], list[str]]:
        spu = ctx.make_spu(config=SPU_RUNTIME_CONFIG_FM128_FXP40)

        model_dd = ctx.load_model(
            self.input_model, DistDataType.SS_GLM_MODEL, SS_GLM_MODEL_MAX, spu=spu
        )
        model_meta = json.loads(model_dd.public_info)

        model = SSGLM(spu)
        model.spu_w = model_dd.objs[0]
        model.link = get_link(model_meta["link"])
        model.y_scale = float(model_meta["y_scale"])

        offset_col = model_meta["offset_col"]
        label_col = model_meta["label_col"]
        sample_weight_col = model_meta["sample_weight_col"]
        feature_names = model_meta["feature_names"]
        partitions_order = list(model_meta["party_features_length"].keys())

        x, y, o, w = self.load_ds(
            ctx,
            feature_names,
            partitions_order,
            label_col,
            offset_col,
            sample_weight_col,
        )

        with ctx.trace_running():
            yhat = model.predict(x, o)

        spu_w = model.spu_w
        link = LinkType(model_meta["link"])
        dist = DistributionType(model_meta["dist"])
        tweedie_power = model_meta["tweedie_power"]
        y_scale = model.y_scale

        with ctx.trace_running():
            return (
                PValue(spu).z_statistic_p_value(
                    x,
                    y,
                    yhat,
                    spu_w,
                    link,
                    dist,
                    tweedie_power,
                    y_scale,
                    sample_weights=w,
                ),
                feature_names,
            )

    def load_ds(
        self,
        ctx: Context,
        x_cols: list[str],
        x_partitions_order: list[str],
        y_cols: list[str],
        offset_cols=None,
        weight_cols=None,
    ):
        col_selects = (
            x_cols
            + y_cols
            + (offset_cols if offset_cols else [])
            + (weight_cols if weight_cols else [])
        )
        tbl = VTable.from_distdata(self.input_ds, col_selects)
        tbl.check_kinds(VTableFieldKind.FEATURE_LABEL)
        if x_partitions_order is not None:
            for p in tbl.parties.keys():
                if p not in x_partitions_order:
                    x_partitions_order.append(p)
            tbl.sort_partitions(x_partitions_order)

        df = ctx.load_table(tbl)

        if offset_cols:
            o = df[offset_cols].to_pandas()
        else:
            o = None

        if weight_cols:
            w = df[weight_cols].to_pandas()
        else:
            w = None

        return df[x_cols].to_pandas(), df[y_cols].to_pandas(), o, w
