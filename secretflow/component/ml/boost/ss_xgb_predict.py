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

from secretflow.component.core import (
    SS_XGB_MODEL_MAX,
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    IServingExporter,
    Output,
    ServingBuilder,
    register,
    save_prediction,
)
from secretflow.device import PYU
from secretflow.ml.boost.ss_xgb_v.checkpoint import (
    SSXGBCheckpointData,
    build_ss_xgb_model,
)

from .ss_xgb import SSXGBExportMixin


@register(domain="ml.predict", version="1.0.0", name="ss_xgb_predict")
class SSXGBPredict(SSXGBExportMixin, Component, IServingExporter):
    '''
    Predict using the SS-XGB model.
    '''

    receiver: str = Field.party_attr(desc="Party of receiver.")
    pred_name: str = Field.attr(desc="Column name for predictions.", default="pred")
    save_ids: bool = Field.attr(
        desc=(
            "Whether to save ids columns into output prediction table. "
            "If true, input feature_dataset must contain id columns, and receiver party must be id owner."
        ),
        default=False,
    )
    save_label: bool = Field.attr(
        desc=(
            "Whether or not to save real label columns into output pred file. "
            "If true, input feature_dataset must contain label columns and receiver party must be label owner."
        ),
        default=True,
    )
    input_model: Input = Field.input(
        desc="model",
        types=[DistDataType.SS_XGB_MODEL],
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
        spu = ctx.make_spu()
        pyus = {p: PYU(p) for p in ctx.parties}
        model_dd = ctx.load_model(
            self.input_model,
            DistDataType.SS_XGB_MODEL,
            SS_XGB_MODEL_MAX,
            pyus=pyus,
            spu=spu,
        )
        model_public_info = json.loads(model_dd.public_info)
        ss_xgb_model = build_ss_xgb_model(
            SSXGBCheckpointData(model_dd.objs, model_public_info), spu=spu
        )
        receiver_pyu = PYU(self.receiver)

        def batch_pred(batch):
            with ctx.trace_running():
                return ss_xgb_model.predict(batch, receiver_pyu)

        y_db = save_prediction(
            ctx.storage,
            ctx.tracer,
            self.output_ds.uri,
            receiver_pyu,
            batch_pred,
            self.pred_name,
            model_public_info["feature_names"],
            list(model_public_info["party_features_length"].keys()),
            self.input_ds,
            self.saved_features,
            model_public_info['label_col'] if self.save_label else [],
            self.save_ids,
        )

        self.output_ds.data = y_db

    def export(
        self, ctx: Context, builder: ServingBuilder, he_mode: bool = False
    ) -> None:
        return self.do_export(ctx, builder, self.input_ds, self.input_model, he_mode)
