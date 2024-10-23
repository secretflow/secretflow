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
    CompVDataFrame,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Model,
    Output,
    Reporter,
    ServingBuilder,
    VTable,
    VTableFieldKind,
    register,
)
from secretflow.device import SPUObject, reveal
from secretflow.ml.linear import SSRegression

from .ss_sgd import SSSGDExportMixin


@register(domain="ml.train", version="1.0.0", name="ss_sgd_train")
class SSSGDTrain(SSSGDExportMixin, Component):
    '''
    Train both linear and logistic regression
    linear models for vertical partitioning dataset with mini batch SGD training solver by using secret sharing.

    - SS-SGD is short for secret sharing SGD training.
    '''

    epochs: int = Field.attr(
        desc="The number of complete pass through the training data.",
        default=10,
        bound_limit=Interval.closed(1, None),
        is_checkpoint=True,
    )
    learning_rate: float = Field.attr(
        desc="The step size at each iteration in one iteration.",
        default=0.1,
        bound_limit=Interval.open(0, None),
        is_checkpoint=True,
    )
    batch_size: int = Field.attr(
        desc="The number of training examples utilized in one iteration.",
        default=1024,
        bound_limit=Interval.open(0, None),
        is_checkpoint=True,
    )
    sig_type: str = Field.attr(
        desc="Sigmoid approximation type.",
        default="t1",
        choices=["real", "t1", "t3", "t5", "df", "sr", "mix"],
        is_checkpoint=True,
    )
    reg_type: str = Field.attr(
        desc="Regression type",
        default="logistic",
        choices=["linear", "logistic"],
        is_checkpoint=True,
    )
    penalty: str = Field.attr(
        desc="The penalty(aka regularization term) to be used.",
        default="None",
        choices=["None", "l2"],
        is_checkpoint=True,
    )
    l2_norm: float = Field.attr(
        desc="L2 regularization term.",
        default=0.5,
        bound_limit=Interval.closed(0, None),
        is_checkpoint=True,
    )
    eps: float = Field.attr(
        desc="""If the change rate of weights is less than this threshold,
            the model is considered to be converged,
            and the training stops early. 0 to disable.""",
        default=0.001,
        bound_limit=Interval.closed(0, None),
        is_checkpoint=True,
    )
    report_weights: bool = Field.attr(
        desc="If this option is set to true, model will be revealed and model details are visible to all parties",
        default=False,
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
        types=[DistDataType.SS_SGD_MODEL],
    )
    report: Output = Field.output(
        desc="If report_weights is true, report model details",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        assert (
            self.label not in self.feature_selects
        ), f"col {self.label} used in both label and features"

        spu = ctx.make_spu()

        checkpoint: tuple[dict, list[SPUObject]] = None
        cp_model = ctx.load_checkpoint(
            DistDataType.SS_SGD_CHECKPOINT, SS_SGD_MODEL_MAX, spu=spu
        )
        if cp_model is not None:
            train_state = json.loads(cp_model.public_info)
            checkpoint = (train_state, cp_model.objs)

        def save_checkpoint_cb(epoch, check_point: tuple[dict, list[SPUObject]]):
            cp_uri = f"{self.output_model.uri}_checkpoint_{epoch}"
            train_state, spu_objs = check_point
            model = Model(
                "ss_sgd",
                DistDataType.SS_SGD_CHECKPOINT,
                SS_SGD_MODEL_MAX,
                objs=spu_objs,
                public_info=json.dumps(train_state),
                system_info=self.input_ds.system_info,
            )
            ctx.dump_checkpoint(epoch, model, cp_uri)

        tbl = VTable.from_distdata(self.input_ds)

        tbl_y = tbl.select([self.label])
        tbl_y.check_kinds(VTableFieldKind.FEATURE_LABEL)
        y = ctx.load_table(tbl_y).to_pandas()

        tbl_x = tbl.select(self.feature_selects)
        tbl_x.check_kinds(VTableFieldKind.FEATURE)
        x = ctx.load_table(tbl_x).to_pandas()

        with ctx.trace_running():
            reg = SSRegression(spu)
            reg.fit(
                x=x,
                y=y,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                sig_type=self.sig_type,
                reg_type=self.reg_type,
                penalty=self.penalty,
                l2_norm=self.l2_norm,
                eps=self.eps,
                epoch_callback=save_checkpoint_cb if ctx.enable_checkpoint else None,
                recovery_checkpoint=checkpoint,
            )

        model = reg.save_model()
        party_features_length = {
            device.party: len(columns)
            for device, columns in x.partition_columns.items()
        }
        model_meta = {
            "reg_type": model.reg_type.value,
            "sig_type": model.sig_type.value,
            "feature_names": x.columns,
            "label_col": [self.label],
            "party_features_length": party_features_length,
        }

        model_dd = Model(
            "ss_sgd",
            DistDataType.SS_SGD_MODEL,
            SS_SGD_MODEL_MAX,
            objs=[model.weights],
            public_info=json.dumps(model_meta),
            system_info=self.input_ds.system_info,
        )
        ctx.dump_to(model_dd, self.output_model)
        self.dump_report(reg, x)

    def dump_report(self, reg: SSRegression, x: CompVDataFrame):
        r = Reporter(name="weights", desc="model weights report")
        if self.report_weights:
            weights = list(map(float, list(reveal(reg.spu_w))))
            named_weight = {}
            for p in x.partitions.values():
                features = p.columns
                party_weight = weights[: len(features)]
                named_weight.update({f: w for f, w in zip(features, party_weight)})
                weights = weights[len(features) :]
            assert len(weights) == 1
            w_desc = {"_intercept_": weights[-1]}
            for f, w in named_weight.items():
                w_desc[f] = w
            r.add_tab(w_desc, name="weights", desc="model weights")
        r.dump_to(self.report, self.input_ds.system_info)

    def export(self, ctx: Context, builder: ServingBuilder, he_mode: bool) -> None:
        return self.do_export(
            ctx, builder, self.input_ds, self.output_model.data, he_mode
        )
