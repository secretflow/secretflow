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

from secretflow_spec.v1.component_pb2 import Attribute
from secretflow_spec.v1.data_pb2 import DistData
from secretflow_spec.v1.report_pb2 import Div, Report, Tab, Table

from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Model,
    Output,
    UnionGroup,
    Version,
    VTable,
    VTableFieldKind,
    register,
)
from secretflow.data.split import train_test_split

from ..core.utils import check_enabled_or_fail
from .base import (
    BUILTIN_COMPRESSORS,
    BUILTIN_LOSSES,
    BUILTIN_METRICS,
    BUILTIN_OPTIMIZERS,
    BUILTIN_STRATEGIES,
    DEFAULT_CUSTOM_LOSS_CODE,
    DEFAULT_MODELS_CODE,
    MODEL_MAX_MAJOR_VERSION,
    MODEL_MAX_MINOR_VERSION,
    ModelInputScheme,
    ModelMeta,
    mkdtemp,
)


@dataclass
class Loss(UnionGroup):
    builtin: str = Field.attr(
        desc="Builtin loss function.",
        default="mean_squared_error",
        choices=BUILTIN_LOSSES,
    )
    custom: str = Field.attr(
        desc="Custom loss function.",
        default=DEFAULT_CUSTOM_LOSS_CODE,
    )


@dataclass
class Optimizer:
    name: str = Field.attr(
        desc="Optimizer name.",
        is_optional=False,
        default="Adam",
        choices=BUILTIN_OPTIMIZERS,
    )
    params: str = Field.attr(
        desc="Additional optimizer parameters in JSON format.",
        default="",
    )


@dataclass
class Strategy:
    name: str = Field.attr(
        desc="Split learning strategy name.",
        default="pipeline",
        choices=BUILTIN_STRATEGIES,
    )
    params: str = Field.attr(
        desc="Additional strategy parameters in JSON format.",
        default='{"pipeline_size":2}',
    )


@dataclass
class Compressor:
    name: str = Field.attr(
        desc="Compressor name.",
        default="",
        choices=BUILTIN_COMPRESSORS,
    )
    params: str = Field.attr(
        desc="Additional compressor parameters in JSON format.",
        default="",
    )


@register(domain="ml.train", version="0.0.1", name="slnn_train")
class SLNNTrain(Component):
    '''
    Train nn models for vertical partitioning dataset by split learning.
    This component is not enabled by default, it requires the use of the full version
    of secretflow image and setting the ENABLE_NN environment variable to true.
    Since it is necessary to define the model structure using python code,
    although the range of syntax and APIs that can be used has been restricted,
    there are still potential security risks. It is recommended to use it in
    conjunction with process sandboxes such as nsjail.
    '''

    models: str = Field.attr(
        desc="Define the models for training.",
        is_optional=False,
        default=DEFAULT_MODELS_CODE,
    )
    epochs: int = Field.attr(
        desc="The number of complete pass through the training data.",
        default=10,
        bound_limit=Interval.closed(1, None),
    )
    learning_rate: float = Field.attr(
        desc="The step size at each iteration in one iteration.",
        default=0.001,
        bound_limit=Interval.open(0, None),
    )
    batch_size: int = Field.attr(
        desc="The number of training examples utilized in one iteration.",
        default=512,
        bound_limit=Interval.open(0, None),
    )
    validattion_prop: float = Field.attr(
        desc="The proportion of validation set to total data set.",
        default=0.1,
        bound_limit=Interval.closed_open(0, 1),
    )
    loss: Loss = Field.union_attr(desc="Loss function.")
    optimizer: Optimizer = Field.struct_attr(desc="Optimizer.")
    metrics: list[str] = Field.attr(
        desc="Metrics.",
        default=["AUC"],
        choices=BUILTIN_METRICS,
        list_limit=Interval.closed(None, 10),
    )
    model_input_scheme: str = Field.attr(
        desc="Input scheme of base model, tensor: merge all features into one tensor; tensor_dict: each feature as a tensor.",
        is_optional=False,
        default=str(ModelInputScheme.TENSOR),
        choices=ModelInputScheme.values(),
    )
    strategy: Strategy = Field.struct_attr(desc="Split learning strategy.")
    compressor: Compressor = Field.struct_attr(
        desc="Compressor for hiddens and gradients."
    )
    feature_selects: list[str] = Field.table_column_attr(
        "train_dataset",
        desc="which features should be used for training.",
        limit=Interval.closed(1, None),
    )
    label: str = Field.table_column_attr(
        "train_dataset",
        desc="Label of train dataset.",
    )
    train_dataset: Input = Field.input(
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_model: Output = Field.output(
        desc="Output model.",
        types=[DistDataType.SL_NN_MODEL],
    )
    reports: Output = Field.output(
        desc="Output report.",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        check_enabled_or_fail()

        # import after enabling check to avoid missing dependencies
        from .training import saver, trainer

        assert (
            self.label not in self.feature_selects
        ), f"col {self.label} used in both label and features"

        tbl = VTable.from_distdata(self.train_dataset)

        tbl_y = tbl.select([self.label])
        tbl_y.check_kinds(VTableFieldKind.FEATURE_LABEL)
        y = ctx.load_table(tbl_y).to_pandas(check_null=False)

        tbl_x = tbl.select(self.feature_selects)
        tbl_x.check_kinds(VTableFieldKind.FEATURE)
        x = ctx.load_table(tbl_x).to_pandas(check_null=False)

        val_x, val_y = None, None
        if self.validattion_prop > 0:
            x, val_x = train_test_split(
                x, test_size=self.validattion_prop, train_size=1 - self.validattion_prop
            )
            y, val_y = train_test_split(
                y, test_size=self.validattion_prop, train_size=1 - self.validattion_prop
            )

        pyus, label_pyu = trainer.prepare_pyus(x, y)

        if self.loss.is_selected("builtin"):
            loss_builtin = self.loss.builtin
            loss_custom = None
        else:
            loss_builtin = None
            loss_custom = self.loss.custom

        with ctx.trace_running():
            slmodel, history, model_configs = trainer.fit(
                x=x,
                y=y,
                val_x=val_x,
                val_y=val_y,
                models=self.models,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                loss=loss_builtin,
                custom_loss=loss_custom,
                optimizer=self.optimizer.name,
                optimizer_params=self.optimizer.params,
                metrics=self.metrics,
                model_input_scheme=self.model_input_scheme,
                strategy=self.strategy.name,
                strategy_params=self.strategy.params,
                compressor=self.compressor.name,
                compressor_params=self.compressor.params,
                initiator_party=ctx.initiator_party,
            )

        tmpdirs = mkdtemp(pyus)

        model, parts_meta = saver.save(slmodel, label_pyu, model_configs, tmpdirs)

        feature_names = x.columns
        model_meta = ModelMeta(
            parts=parts_meta,
            model_input_scheme=self.model_input_scheme,
            label_col=[self.label],
            feature_names=feature_names,
        )

        model_db = Model(
            "sl_nn",
            DistDataType.SL_NN_MODEL,
            Version(MODEL_MAX_MAJOR_VERSION, MODEL_MAX_MINOR_VERSION),
            objs=model,
            public_info=json.dumps(model_meta.to_dict()),
            system_info=self.train_dataset.system_info,
        )
        ctx.dump_to(model_db, self.output_model)

        self.reports.data = self.dump_training_reports(
            self.reports.uri,
            history,
            self.train_dataset.system_info,
        )

    @staticmethod
    def dump_training_reports(name, history, system_info):
        ret = DistData(
            name=name,
            system_info=system_info,
            type=str(DistDataType.REPORT),
        )

        headers = []
        rows = []
        for name, vals in history.items():
            headers.append(
                Table.HeaderItem(
                    name=name,
                    type="float",
                )
            )
            if not rows:
                for idx, v in enumerate(vals):
                    rows.append(
                        Table.Row(
                            name=f"epoch_{idx+1}",
                            items=[
                                Attribute(f=v),
                            ],
                        )
                    )
            else:
                for idx, v in enumerate(vals):
                    rows[idx].items.append(
                        Attribute(f=v),
                    )

        meta = Report(
            name="reports",
            desc="",
            tabs=[
                Tab(
                    name="metrics",
                    desc="train and eval metrics",
                    divs=[
                        Div(
                            name="",
                            desc="",
                            children=[
                                Div.Child(
                                    type="table",
                                    table=Table(
                                        name="",
                                        desc="",
                                        headers=headers,
                                        rows=rows,
                                    ),
                                ),
                            ],
                        )
                    ],
                )
            ],
        )
        ret.meta.Pack(meta)
        return ret
