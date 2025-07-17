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
import logging

from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    IServingExporter,
    Output,
    register,
    save_prediction,
    ServingBuilder,
    SGB_MODEL_MAX,
)
from secretflow.device import PYU
from secretflow.ml.boost.sgb_v.checkpoint import build_sgb_model, SGBSnapshot

from .sgb import SGBExportMixin


@register(domain="ml.predict", version="1.0.0", name="sgb_predict")
class SGBPredict(SGBExportMixin, Component, IServingExporter):
    '''
    Predict using SGB model.
    '''

    receiver: str = Field.party_attr(desc="Party of receiver.")
    pred_name: str = Field.attr(desc="Name for prediction column", default="pred")
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
        types=[DistDataType.SGB_MODEL],
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
        pyus = {p: PYU(p) for p in ctx.parties}
        model_dd = ctx.load_model(
            self.input_model,
            DistDataType.SGB_MODEL,
            SGB_MODEL_MAX,
            pyus=pyus,
        )
        model_public_info = json.loads(model_dd.public_info)
        assert len(model_public_info["feature_names"]) > 0
        sgb_model = build_sgb_model(SGBSnapshot(model_dd.objs, model_public_info))

        receiver_pyu = PYU(self.receiver)

        def batch_pred(batch):
            with ctx.trace_running():
                return sgb_model.predict(batch, receiver_pyu)

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
