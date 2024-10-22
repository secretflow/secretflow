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
    SS_XGB_MODEL_MAX,
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Model,
    Output,
    ServingBuilder,
    VTable,
    VTableFieldKind,
    register,
)
from secretflow.ml.boost.core.callback import TrainingCallback
from secretflow.ml.boost.ss_xgb_v import Xgb
from secretflow.ml.boost.ss_xgb_v.booster import build_checkpoint
from secretflow.ml.boost.ss_xgb_v.checkpoint import (
    SSXGBCheckpointData,
    ss_xgb_model_to_checkpoint_data,
)

from .ss_xgb import SSXGBExportMixin


@register(domain="ml.train", version="1.0.0", name="ss_xgb_train")
class SSXGBTrain(SSXGBExportMixin, Component):
    '''
    This method provides both classification and regression tree boosting (also known as GBDT, GBM)
    for vertical partitioning dataset setting by using secret sharing.

    - SS-XGB is short for secret sharing XGB.
    - More details: https://arxiv.org/pdf/2005.08479.pdf
    '''

    num_boost_round: int = Field.attr(
        desc="Number of boosting iterations.",
        default=10,
        bound_limit=Interval.closed(1, None),
        is_checkpoint=True,
    )
    max_depth: int = Field.attr(
        desc="Maximum depth of a tree.",
        default=5,
        bound_limit=Interval.closed(1, 16),
        is_checkpoint=True,
    )
    learning_rate: float = Field.attr(
        desc="Step size shrinkage used in updates to prevent overfitting.",
        default=0.1,
        bound_limit=Interval.open_closed(0, 1),
        is_checkpoint=True,
    )
    objective: str = Field.attr(
        desc="Specify the learning objective.",
        default="logistic",
        choices=["linear", "logistic"],
        is_checkpoint=True,
    )
    reg_lambda: float = Field.attr(
        desc="L2 regularization term on weights.",
        default=0.1,
        bound_limit=Interval.closed(0, 10000),
        is_checkpoint=True,
    )
    subsample: float = Field.attr(
        desc="Subsample ratio of the training instances.",
        default=0.1,
        bound_limit=Interval.open_closed(0, 1),
        is_checkpoint=True,
    )
    colsample_by_tree: float = Field.attr(
        desc="Subsample ratio of columns when constructing each tree.",
        default=0.1,
        bound_limit=Interval.open_closed(0, 1),
        is_checkpoint=True,
    )
    sketch_eps: float = Field.attr(
        desc="This roughly translates into O(1 / sketch_eps) number of bins.",
        default=0.1,
        bound_limit=Interval.open_closed(0, 1),
        is_checkpoint=True,
    )
    base_score: float = Field.attr(
        desc="The initial prediction score of all instances, global bias.",
        default=0,
        bound_limit=Interval.closed(-10, 10),
        is_checkpoint=True,
    )
    seed: int = Field.attr(
        desc="Pseudorandom number generator seed.",
        default=42,
        bound_limit=Interval.closed(0, None),
        is_checkpoint=True,
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
    input_ds: Input = Field.input(  # type: ignore
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
        is_checkpoint=True,
    )
    output_model: Output = Field.output(
        desc="Output model.",
        types=[DistDataType.SS_XGB_MODEL],
    )

    def evaluate(self, ctx: Context):
        spu = ctx.make_spu()

        assert (
            self.label not in self.feature_selects
        ), f"expect no intersection between label and features, got {self.label} and {self.feature_selects}"
        tbl = VTable.from_distdata(self.input_ds)

        tbl_y = tbl.select([self.label])
        tbl_y.check_kinds(VTableFieldKind.FEATURE_LABEL)
        y = ctx.load_table(tbl_y).to_pandas()

        tbl_x = tbl.select(self.feature_selects)
        tbl_x.check_kinds(VTableFieldKind.FEATURE_LABEL)
        x = ctx.load_table(tbl_x).to_pandas()

        pyus = {p.party: p for p in x.partitions.keys()}
        checkpoint_data = None
        cp_dd = ctx.load_checkpoint(
            DistDataType.SS_XGB_CHECKPOINT, SS_XGB_MODEL_MAX, pyus=pyus, spu=spu
        )
        if cp_dd is not None:
            checkpoint_data = SSXGBCheckpointData(
                cp_dd.objs, json.loads(cp_dd.public_info)
            )

        def save_checkpoint_cb(
            model: Xgb, epoch: int, evals_log: TrainingCallback.EvalsLog
        ):
            cp_uri = f"{self.output_model.uri}_checkpoint_{epoch}"
            cp = build_checkpoint(model, evals_log, x, [self.label])
            cp_dd = Model(
                "sgb",
                DistDataType.SS_XGB_CHECKPOINT,
                SS_XGB_MODEL_MAX,
                objs=cp.model_objs,
                public_info=json.dumps(cp.model_metas),
                system_info=self.input_ds.system_info,
            )
            ctx.dump_checkpoint(epoch, cp_dd, cp_uri)

        with ctx.trace_running():
            ss_xgb = Xgb(spu)
            model = ss_xgb.train(
                params={
                    "num_boost_round": self.num_boost_round,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "objective": self.objective,
                    "reg_lambda": self.reg_lambda,
                    "subsample": self.subsample,
                    "colsample_by_tree": self.colsample_by_tree,
                    "sketch_eps": self.sketch_eps,
                    "base_score": self.base_score,
                    "seed": self.seed,
                },
                dtrain=x,
                label=y,
                checkpoint_data=checkpoint_data,
                dump_function=save_checkpoint_cb if ctx.enable_checkpoint else None,
            )

        checkpoint = ss_xgb_model_to_checkpoint_data(model, x, [self.label])

        model_db = Model(
            "xgb",
            DistDataType.SS_XGB_MODEL,
            SS_XGB_MODEL_MAX,
            objs=checkpoint.model_objs,
            public_info=json.dumps(checkpoint.model_metas),
            system_info=self.input_ds.system_info,
        )
        ctx.dump_to(model_db, self.output_model)

    def export(
        self, ctx: Context, builder: ServingBuilder, he_mode: bool = False
    ) -> None:
        return self.do_export(
            ctx, builder, self.input_ds, self.output_model.data, he_mode
        )
