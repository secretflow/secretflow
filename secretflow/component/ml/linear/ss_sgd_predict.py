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
    SS_SGD_MODEL_MAX,
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    IServingExporter,
    Output,
    ServingBuilder,
    register,
    save_prediction,
)
from secretflow.device import PYU, SPUObject
from secretflow.ml.linear import LinearModel, RegType, SSRegression
from secretflow.utils.sigmoid import SigType

from .ss_sgd import SSSGDExportMixin


@register(domain="ml.predict", version="1.0.0", name="ss_sgd_predict")
class SSSGDPredict(SSSGDExportMixin, Component, IServingExporter):
    '''
    Predict using the SS-SGD model.
    '''

    batch_size: int = Field.attr(
        desc="The number of training examples utilized in one iteration.",
        default=1024,
        bound_limit=Interval.open(0, None),
    )
    receiver: str = Field.party_attr(desc="Party of receiver.")
    pred_name: str = Field.attr(
        desc="Column name for predictions.",
        default="pred",
    )
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
        types=[DistDataType.SS_SGD_MODEL],
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
        receiver_pyu = PYU(self.receiver)

        def batch_pred(batch):
            with ctx.trace_running():
                return ss_reg.predict(batch, to_pyu=receiver_pyu)

        y_db = save_prediction(
            ctx.storage,
            ctx.tracer,
            self.output_ds.uri,
            receiver_pyu,
            batch_pred,
            self.pred_name,
            model_meta["feature_names"],
            list(model_meta["party_features_length"].keys()),
            self.input_ds,
            self.saved_features,
            model_meta["label_col"] if self.save_label else [],
            self.save_ids,
        )
        self.output_ds.data = y_db

    def export(self, ctx: Context, builder: ServingBuilder, he_mode: bool) -> None:
        return self.do_export(ctx, builder, self.input_ds, self.input_model, he_mode)
