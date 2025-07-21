# Copyright 2024 Ant Group Co., Ltd.
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

from secretflow.component.core import (
    SS_GLM_MODEL_MAX,
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    IServingExporter,
    ObjectFile,
    Output,
    ServingBuilder,
    SPURuntimeConfig,
    register,
    save_prediction,
)
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU
from secretflow.ml.linear import SSGLM
from secretflow.ml.linear.ss_glm.core import get_link

from .ss_glm import SSGLMExportMixin


@register(domain="ml.predict", version="1.1.0", name="ss_glm_predict")
class SSGLMPredict(SSGLMExportMixin, Component, IServingExporter):
    '''
    Predict using the SSGLM model.
    '''

    receiver: str = Field.party_attr(desc="Party of receiver.")
    pred_name: str = Field.attr(desc="Column name for predictions.", default="pred")
    save_ids: bool = Field.attr(
        desc=(
            "Whether to save ids columns into output prediction table. "
            "If true, input feature_dataset must contain id columns, and receiver party must be id owner."
        ),
        default=True,
    )
    save_label: bool = Field.attr(
        desc=(
            "Whether or not to save real label columns into output pred file. "
            "If true, input feature_dataset must contain label columns and receiver party must be label owner."
        ),
        default=False,
    )
    input_model: Input = Field.input(
        desc="Input model.",
        types=[DistDataType.SS_GLM_MODEL],
    )
    saved_features: list[str] = Field.table_column_attr(
        "input_ds",
        desc="which features should be saved with prediction result",
    )
    input_ds: Input = Field.input(
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="Output prediction.",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )

    def evaluate(self, ctx: Context):
        model_info = ObjectFile.from_distdata(self.input_model)
        model_meta = json.loads(model_info.public_info)
        if model_info.version.minor <= 3:
            # model_meta["fxp_exp_mode"] is automatically supporte
            # other configs use default values
            model_meta["experimental_exp_prime_offset"] = 13
            model_meta["experimental_exp_prime_disable_lower_bound"] = False
            model_meta["experimental_exp_prime_enable_upper_bound"] = False
        spu_rt_config = SPURuntimeConfig(
            field="FM128",
            fxp_fraction_bits=40,
            fxp_exp_mode=model_meta["fxp_exp_mode"],
            fxp_exp_iters=model_meta["fxp_exp_iters"],
            experimental_exp_prime_offset=model_meta["experimental_exp_prime_offset"],
            experimental_exp_prime_disable_lower_bound=model_meta[
                "experimental_exp_prime_disable_lower_bound"
            ],
            experimental_exp_prime_enable_upper_bound=model_meta[
                "experimental_exp_prime_enable_upper_bound"
            ],
        )
        spu = ctx.make_spu(config=spu_rt_config)

        model_dd = ctx.load_model(
            model_info, DistDataType.SS_GLM_MODEL, SS_GLM_MODEL_MAX, spu=spu
        )

        glm = SSGLM(spu)
        glm.spu_w = model_dd.objs[0]
        glm.link = get_link(model_meta["link"])
        glm.y_scale = float(model_meta["y_scale"])

        feature_names = model_meta['feature_names']
        offset_col = model_meta['offset_col']
        feature_names.extend(offset_col)
        receiver_pyu = PYU(self.receiver)

        def batch_pred(batch: VDataFrame):
            with ctx.trace_running():
                if offset_col:
                    offset = batch[offset_col]
                    batch = batch.drop(columns=offset_col)
                else:
                    offset = None
                return glm.predict(x=batch, o=offset, to_pyu=receiver_pyu)

        y_db = save_prediction(
            ctx.storage,
            ctx.tracer,
            self.output_ds.uri,
            receiver_pyu,
            batch_pred,
            self.pred_name,
            feature_names,
            list(model_meta["party_features_length"].keys()),
            self.input_ds,
            self.saved_features,
            model_meta['label_col'] if self.save_label else [],
            self.save_ids,
        )
        self.output_ds.data = y_db

    def export(self, ctx: Context, builder: ServingBuilder, he_mode: bool) -> None:
        return self.do_export(ctx, builder, self.input_ds, self.input_model, he_mode)
