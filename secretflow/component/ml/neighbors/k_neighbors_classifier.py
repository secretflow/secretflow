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
from dataclasses import dataclass

from secretflow.component.core import (
    SS_KNN_MODEL_MAX,
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Model,
    Output,
    UnionGroup,
    VTable,
    VTableFieldKind,
    register,
)
from secretflow.component.core.dataframe import save_prediction
from secretflow.data.mix.dataframe import PartitionWay
from secretflow.data.ndarray.ndarray import FedNdarray
from secretflow.device.device.pyu import PYU
from secretflow.device.driver import reveal
from secretflow.ml.neighbors import KNNClassifer


@dataclass
class WeightFunc(UnionGroup):
    uniform: str = Field.selection_attr(
        desc="Uniform weights. All points in each neighborhood are weighted equally."
    )
    distance: str = Field.selection_attr(
        desc="Weight points by the inverse of their distance."
    )


@register(
    domain="ml.train",
    version="1.0.0",
    name="knn_train",
    labels={"experimental": True, "package": "sml"},
)
class KNeighborsClassifier(Component):
    '''
    Provide k neighbors classifier training. This component is currently experimental.
    '''

    weights: WeightFunc = Field.union_attr(
        desc="weights function used in prediction method.",
        default="uniform",
    )
    n_classes: int = Field.attr(
        desc="The number of classes in the training data, must be preprocessed to 0, 1, 2, ...",
        default=2,
        bound_limit=Interval.closed(1, None),
    )
    n_neighbors: int = Field.attr(
        desc="Number of neighbors to use for prediction.",
        default=5,
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
        types=[DistDataType.VERTICAL_TABLE],
        is_checkpoint=True,
    )
    output_model: Output = Field.output(
        desc="Output model.",
        types=[DistDataType.SS_KNN_MODEL],
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
            knn = KNNClassifer(spu)
            knn.fit(
                x,
                y,
                n_classes=self.n_classes,
                n_neighbors=self.n_neighbors,
                weights=self.weights.get_selected(),
            )

        model_db = Model(
            name="knn",
            type=DistDataType.SS_KNN_MODEL,
            public_info=json.dumps(
                {
                    "n_classes": self.n_classes,
                    "n_neighbors": self.n_neighbors,
                    "weights": self.weights.get_selected(),
                    "feature_selects": self.feature_selects,
                    "label": self.label,
                }
            ),
            objs=[knn.model],
            version=SS_KNN_MODEL_MAX,
            system_info=self.input_ds.system_info,
        )
        ctx.dump_to(model_db, self.output_model)


@register(
    domain="ml.predict",
    version="1.0.0",
    name="knn_predict",
    labels={"experimental": True, "package": "sml"},
)
class KNeighborsClassifierPredict(Component):
    '''
    Predict using the K neighbors classifier model. This component is currently experimental.
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
        types=[DistDataType.SS_KNN_MODEL],
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
            DistDataType.SS_KNN_MODEL,
            SS_KNN_MODEL_MAX,
            pyus=pyus,
            spu=spu,
        )
        model_public_info = json.loads(model_dd.public_info)
        model = KNNClassifer(spu)
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
