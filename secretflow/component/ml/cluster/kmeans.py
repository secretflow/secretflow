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
from dataclasses import dataclass

from secretflow.component.core import (
    SS_KMEANS_MODEL_MAX,
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
from secretflow.ml.cluster.kmeans import KMeans


@dataclass
class InitMethod(UnionGroup):
    random: str = Field.selection_attr(desc="random method")
    kmeans_plus_plus: str = Field.selection_attr(desc="k-means++ method")


@register(
    domain="ml.train",
    version="1.0.0",
    name="kmeans_train",
    labels={"experimental": True, "package": "sml"},
)
class KMeansTrain(Component):
    '''
    Provide kmeans training. This component is currently experimental.
    '''

    n_clusters: int = Field.attr(
        desc="Number of clusters.",
        bound_limit=Interval.closed(1, None),
    )
    max_iter: int = Field.attr(
        desc="Number of iterations for kmeans training.",
        default=10,
        bound_limit=Interval.closed(1, None),
    )
    n_init: int = Field.attr(
        desc="Number of groups for initial centers.",
        default=1,
        bound_limit=Interval.closed(1, None),
    )
    init_method: InitMethod = Field.union_attr(
        desc="Params initialization method.",
        default="kmeans_plus_plus",
    )
    feature_selects: list[str] = Field.table_column_attr(
        "input_ds",
        desc="which features should be used for training.",
        limit=Interval.closed(1, None),
        is_checkpoint=True,
    )
    input_ds: Input = Field.input(
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
        is_checkpoint=True,
    )
    output_model: Output = Field.output(
        desc="Output model.",
        types=[DistDataType.SS_KMEANS_MODEL],
    )

    def evaluate(self, ctx: Context):
        spu = ctx.make_spu()

        tbl = VTable.from_distdata(self.input_ds)

        tbl_x = tbl.select(self.feature_selects)
        tbl_x.check_kinds(VTableFieldKind.FEATURE_LABEL)
        x = ctx.load_table(tbl_x).to_pandas()

        with ctx.trace_running():
            kmeans = KMeans(spu)
            kmeans.fit(
                x,
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                n_init=self.n_init,
                init=(
                    "random"
                    if self.init_method.get_selected() == "random"
                    else "k-means++"
                ),
            )

        model_db = Model(
            name="kmeans",
            type=DistDataType.SS_KMEANS_MODEL,
            public_info=json.dumps(
                {
                    "n_clusters": self.n_clusters,
                    "max_iter": self.max_iter,
                    "n_init": self.n_init,
                    'init_method': str(self.init_method),
                    'feature_selects': self.feature_selects,
                }
            ),
            objs=[kmeans.model],
            version=SS_KMEANS_MODEL_MAX,
            system_info=self.input_ds.system_info,
        )
        ctx.dump_to(model_db, self.output_model)


@register(
    domain="ml.predict",
    version="1.0.0",
    name="kmeans_predict",
    labels={"experimental": True, "package": "sml"},
)
class KMeansPredict(Component):
    '''
    Predict using the KMeans model. This component is currently experimental.
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

    input_model: Input = Field.input(
        desc="model",
        types=[DistDataType.SS_KMEANS_MODEL],
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
            DistDataType.SS_KMEANS_MODEL,
            SS_KMEANS_MODEL_MAX,
            pyus=pyus,
            spu=spu,
        )
        model_public_info = json.loads(model_dd.public_info)
        kmeans_model = KMeans(spu)
        kmeans_model.model = model_dd.objs[0]

        receiver_pyu = PYU(self.receiver)

        def batch_pred(batch) -> FedNdarray:
            with ctx.trace_running():
                return FedNdarray(
                    partitions={
                        receiver_pyu: kmeans_model.predict(batch).to(receiver_pyu),
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
            saved_labels=[],
            save_ids=self.save_ids,
        )

        self.output_ds.data = y_db
