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
from pathlib import Path

from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Output,
    Version,
    VTable,
    VTableFieldKind,
    register,
    save_prediction,
)
from secretflow.device import PYU, PYUObject

from ..core.utils import check_enabled_or_fail
from .base import MODEL_MAX_MAJOR_VERSION, MODEL_MAX_MINOR_VERSION, ModelMeta, mkdtemp
from .compile.compile import ModelConfig


@register(domain="ml.predict", version="0.0.2", name="slnn_predict")
class SLNNPredict(Component):
    '''
    Predict using the SLNN model.
    This component is not enabled by default, it requires the use of the full version
    of secretflow image and setting the ENABLE_NN environment variable to true.
    '''

    batch_size: int = Field.attr(
        desc="The number of examples per batch.",
        default=8192,
        bound_limit=Interval.open(0, None),
    )
    receiver: str = Field.party_attr(
        desc="Party of receiver.",
    )
    pred_name: str = Field.attr(
        desc="Column name for predictions.",
        default="pred",
    )
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
        default=False,
    )
    model: Input = Field.input(  # type: ignore
        desc="Input model.",
        types=[DistDataType.SL_NN_MODEL],
    )
    saved_features: list[str] = Field.table_column_attr(
        "feature_dataset",
        desc="which features should be saved with prediction result",
    )
    feature_dataset: Input = Field.input(  # type: ignore
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    pred: Output = Field.output(
        desc="Output prediction.",
        types=[DistDataType.INDIVIDUAL_TABLE],
    )

    def evaluate(self, ctx: Context):
        check_enabled_or_fail()

        # import after enabling check to avoid missing dependencies
        from .training import predictor

        receiver_pyu = PYU(self.receiver)

        x_info = VTable.from_distdata(self.feature_dataset)
        x_info = x_info.select_by_kinds(VTableFieldKind.FEATURE)
        pyus = {PYU(p) for p in x_info.parties.keys()}
        pyus.add(receiver_pyu)
        # ensure all parties have save order
        pyus = sorted(list(pyus))
        tmpdirs = mkdtemp(pyus)

        pyus = {str(pyu): pyu for pyu in pyus}
        model, model_meta = self.load_slnn_model(ctx, pyus, tmpdirs)
        feature_names = model_meta.feature_names

        def batch_pred(batch):
            with ctx.trace_running():
                pyu_y = predictor.predict(
                    batch_size=self.batch_size,
                    feature_dataset=batch,
                    model=model,
                    model_input_scheme=model_meta.model_input_scheme,
                )

                assert (
                    receiver_pyu in pyu_y.partitions
                ), f"receiver must be the label provider"

                return pyu_y

        with ctx.trace_io():
            y_db = save_prediction(
                ctx.storage,
                ctx.tracer,
                self.pred.uri,
                receiver_pyu,
                batch_pred,
                self.pred_name,
                feature_names,
                None,
                self.feature_dataset,
                self.saved_features,
                model_meta.label_col if self.save_label else [],
                self.save_ids,
                check_null=False,
            )

        self.pred.data = y_db

    def load_slnn_model(
        self, ctx: Context, pyus: dict[str, PYU], tmpdirs: dict[PYU, Path]
    ) -> tuple[dict[PYU, ModelConfig], ModelMeta]:
        model_dd = ctx.load_model(
            self.model,
            DistDataType.SL_NN_MODEL,
            Version(MODEL_MAX_MAJOR_VERSION, MODEL_MAX_MINOR_VERSION),
            pyus=pyus,
        )
        model_meta_str = model_dd.public_info
        assert len(model_dd.objs) == len(pyus) and isinstance(
            model_dd.objs[0], PYUObject
        ), f"model_objs {model_dd.objs}, model_meta_str {model_meta_str}"

        model_meta_dict = json.loads(model_meta_str)
        assert isinstance(model_meta_dict, dict)

        model_meta = ModelMeta.from_dict(model_meta_dict)
        assert (
            len(model_meta.parts) == 2
            and model_meta.model_input_scheme
            and len(model_meta.feature_names) > 0
        )

        from .training import saver

        model = saver.load(model_dd.objs, model_meta.parts, tmpdirs)

        return model, model_meta
