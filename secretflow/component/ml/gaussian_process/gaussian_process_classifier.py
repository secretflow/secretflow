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
    SS_GPC_MODEL_MAX,
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Model,
    Output,
    VTable,
    VTableFieldKind,
    register,
)
from secretflow.component.core.dataframe import save_prediction
from secretflow.data.mix.dataframe import PartitionWay
from secretflow.data.ndarray.ndarray import FedNdarray
from secretflow.device.device.pyu import PYU
from secretflow.ml.gaussian_process import GPC


@register(
    domain="ml.train",
    version="1.0.0",
    name="gpc_train",
    labels={"experimental": True, "package": "sml"},
)
class GaussianProcessClassifier(Component):
    '''
    Provide gaussian process classifier training. This component is currently experimental.
    '''

    max_iter_predict: int = Field.attr(
        desc="""The maximum number of iterations in Newton's method for approximating
the posterior during predict. Smaller values will reduce computation
time at the cost of worse results.""",
        default=20,
        bound_limit=Interval.closed(1, None),
    )
    n_classes: int = Field.attr(
        desc="The number of classes in the training data, must be preprocessed to 0, 1, 2, ...",
        default=2,
        bound_limit=Interval.closed(1, None),
    )

    feature_selects: list[str] = Field.table_column_attr(
        "input_ds",
        desc="which features should be used for training.",
        limit=Interval.closed(1, None),
        is_checkpoint=True,
    )
    label: str = Field.table_column_attr(
        "input_ds",
        desc="Label of train dataset.",
        is_checkpoint=True,
    )
    input_ds: Input = Field.input(
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
        is_checkpoint=True,
    )
    output_model: Output = Field.output(
        desc="Output model.",
        types=[DistDataType.SS_GPC_MODEL],
    )

    def evaluate(self, ctx: Context):
        spu = ctx.make_spu()

        tbl = VTable.from_distdata(self.input_ds)

        tbl_x = tbl.select(self.feature_selects)
        tbl_x.check_kinds(VTableFieldKind.FEATURE_LABEL)

        tbl_y = tbl.select([self.label])
        tbl_y.check_kinds(VTableFieldKind.FEATURE_LABEL)

        x = ctx.load_table(tbl_x).to_pandas()
        y = ctx.load_table(tbl_y).to_pandas()

        with ctx.trace_running():
            gpc = GPC(spu)
            gpc.fit(
                x,
                y,
                n_classes=self.n_classes,
                max_iter_predict=self.max_iter_predict,
            )
        model_db = Model(
            name="gpc",
            type=DistDataType.SS_GPC_MODEL,
            public_info=json.dumps(
                {
                    "n_classes": self.n_classes,
                    "max_iter_predict": self.max_iter_predict,
                    "feature_selects": self.feature_selects,
                    "label": self.label,
                }
            ),
            objs=[gpc.model],
            version=SS_GPC_MODEL_MAX,
            system_info=self.input_ds.system_info,
        )
        ctx.dump_to(model_db, self.output_model)


@register(
    domain="ml.predict",
    version="1.0.0",
    name="gpc_predict",
    labels={"experimental": True, "package": "sml"},
)
class GaussianProcessClassifierPredict(Component):
    '''
    Predict using the gaussian process classifier model. This component is currently experimental.
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
        types=[DistDataType.SS_GPC_MODEL],
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
            DistDataType.SS_GPC_MODEL,
            SS_GPC_MODEL_MAX,
            pyus=pyus,
            spu=spu,
        )
        model_public_info = json.loads(model_dd.public_info)
        model = GPC(spu)
        model.model = model_dd.objs[0]

        receiver_pyu = PYU(self.receiver)

        def batch_pred(batch) -> FedNdarray:
            with ctx.trace_running():
                return FedNdarray(
                    partitions={
                        receiver_pyu: model.predict(batch).to(receiver_pyu),
                    },
                    partition_way=PartitionWay.VERTICAL,
                )

        y_db = save_prediction(
            storage=ctx.storage,
            tracer=ctx.tracer,
            uri=self.output_ds.uri,
            pyu=receiver_pyu,
            batch_pred=batch_pred,
            pred_name=self.pred_name,
            pred_features=model_public_info["feature_selects"],
            pred_partitions_order=None,
            feature_dataset=self.input_ds,
            saved_features=self.saved_features,
            saved_labels=[model_public_info['label']] if self.save_label else None,
            save_ids=self.save_ids,
        )

        self.output_ds.data = y_db
