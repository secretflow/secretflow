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

from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import DistDataType, load_table, model_dumps
from secretflow.data.split import train_test_split
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.report_pb2 import Div, Report, Tab, Table

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
from .training import saver, trainer

slnn_train_comp = Component(
    "slnn_train",
    domain="ml.train",
    version="0.0.1",
    desc="""Train nn models for vertical partitioning dataset by split learning.
    This component is not enabled by default, it requires the use of the full version
    of secretflow image and setting the ENABLE_NN environment variable to true.
    Since it is necessary to define the model structure using python code,
    although the range of syntax and APIs that can be used has been restricted,
    there are still potential security risks. It is recommended to use it in
    conjunction with process sandboxes such as nsjail.""",
)

slnn_train_comp.str_attr(
    name="models",
    desc="Define the models for training.",
    is_list=False,
    is_optional=False,
    default_value=DEFAULT_MODELS_CODE,
)
slnn_train_comp.int_attr(
    name="epochs",
    desc="The number of complete pass through the training data.",
    is_list=False,
    is_optional=True,
    default_value=10,
    allowed_values=None,
    lower_bound=1,
    upper_bound=None,
    lower_bound_inclusive=True,
)
slnn_train_comp.float_attr(
    name="learning_rate",
    desc="The step size at each iteration in one iteration.",
    is_list=False,
    is_optional=True,
    default_value=0.001,
    lower_bound=0,
    lower_bound_inclusive=False,
)
slnn_train_comp.int_attr(
    name="batch_size",
    desc="The number of training examples utilized in one iteration.",
    is_list=False,
    is_optional=True,
    default_value=512,
    lower_bound=0,
    lower_bound_inclusive=False,
)
slnn_train_comp.float_attr(
    name="validattion_prop",
    desc="The proportion of validation set to total data set.",
    is_list=False,
    is_optional=True,
    default_value=0.1,
    lower_bound=0,
    lower_bound_inclusive=True,
    upper_bound=1,
    upper_bound_inclusive=False,
)
slnn_train_comp.union_attr_group(
    name="loss",
    desc="Loss function.",
    group=[
        slnn_train_comp.str_attr(
            name="builtin",
            desc="Builtin loss function.",
            is_list=False,
            is_optional=True,
            default_value="mean_squared_error",
            allowed_values=BUILTIN_LOSSES,
        ),
        slnn_train_comp.str_attr(
            name="custom",
            desc="Custom loss function.",
            is_list=False,
            is_optional=True,
            default_value=DEFAULT_CUSTOM_LOSS_CODE,
        ),
    ],
)
slnn_train_comp.struct_attr_group(
    name="optimizer",
    desc="Optimizer.",
    group=[
        slnn_train_comp.str_attr(
            name="name",
            desc="Optimizer name.",
            is_list=False,
            is_optional=False,
            default_value="Adam",
            allowed_values=BUILTIN_OPTIMIZERS,
        ),
        slnn_train_comp.str_attr(
            name="params",
            desc="Additional optimizer parameters in JSON format.",
            is_list=False,
            is_optional=True,
            default_value="",
        ),
    ],
)
slnn_train_comp.str_attr(
    name="metrics",
    desc="Metrics.",
    is_list=True,
    is_optional=True,
    default_value=["AUC"],
    allowed_values=BUILTIN_METRICS,
    list_max_length_inclusive=10,
)
slnn_train_comp.str_attr(
    name="model_input_scheme",
    desc="Input scheme of base model, tensor: merge all features into one tensor; tensor_dict: each feature as a tensor.",
    is_list=False,
    is_optional=False,
    default_value=str(ModelInputScheme.TENSOR),
    allowed_values=ModelInputScheme.values(),
)
slnn_train_comp.struct_attr_group(
    name="strategy",
    desc="Split learning strategy.",
    group=[
        slnn_train_comp.str_attr(
            name="name",
            desc="Split learning strategy name.",
            is_list=False,
            is_optional=True,
            default_value="pipeline",
            allowed_values=BUILTIN_STRATEGIES,
        ),
        slnn_train_comp.str_attr(
            name="params",
            desc="Additional strategy parameters in JSON format.",
            is_list=False,
            is_optional=True,
            default_value='{"pipeline_size":2}',
        ),
    ],
)
slnn_train_comp.struct_attr_group(
    name="compressor",
    desc="Compressor for hiddens and gradients.",
    group=[
        slnn_train_comp.str_attr(
            name="name",
            desc="Compressor name.",
            is_list=False,
            is_optional=True,
            default_value="",
            allowed_values=BUILTIN_COMPRESSORS,
        ),
        slnn_train_comp.str_attr(
            name="params",
            desc="Additional compressor parameters in JSON format.",
            is_list=False,
            is_optional=True,
            default_value="",
        ),
    ],
)
slnn_train_comp.io(
    io_type=IoType.INPUT,
    name="train_dataset",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="feature_selects",
            desc="which features should be used for training.",
            col_min_cnt_inclusive=1,
        ),
        TableColParam(
            name="label",
            desc="Label of train dataset.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1,
        ),
    ],
)
slnn_train_comp.io(
    io_type=IoType.OUTPUT,
    name="output_model",
    desc="Output model.",
    types=[DistDataType.SL_NN_MODEL],
)
slnn_train_comp.io(
    io_type=IoType.OUTPUT,
    name="reports",
    desc="Output report.",
    types=[DistDataType.REPORT],
    col_params=None,
)


@slnn_train_comp.eval_fn
def slnn_train_eval_fn(
    *,
    ctx,
    models,
    epochs,
    learning_rate,
    batch_size,
    validattion_prop,
    loss_builtin,
    loss_custom,
    optimizer_name,
    optimizer_params,
    metrics,
    model_input_scheme,
    strategy_name,
    strategy_params,
    compressor_name,
    compressor_params,
    train_dataset,
    train_dataset_label,
    train_dataset_feature_selects,
    output_model,
    reports,
):
    check_enabled_or_fail()

    assert (
        train_dataset_label[0] not in train_dataset_feature_selects
    ), f"col {train_dataset_label[0]} used in both label and features"

    y = load_table(
        ctx,
        train_dataset,
        load_labels=True,
        load_features=True,
        col_selects=train_dataset_label,
    )
    x = load_table(
        ctx,
        train_dataset,
        load_features=True,
        col_selects=train_dataset_feature_selects,
    )

    val_x, val_y = None, None
    if validattion_prop > 0:
        x, val_x = train_test_split(
            x, test_size=validattion_prop, train_size=1 - validattion_prop
        )
        y, val_y = train_test_split(
            y, test_size=validattion_prop, train_size=1 - validattion_prop
        )

    pyus, label_pyu = trainer.prepare_pyus(x, y)

    with ctx.tracer.trace_running():
        slmodel, history, model_configs = trainer.fit(
            ctx=ctx,
            x=x,
            y=y,
            val_x=val_x,
            val_y=val_y,
            models=models,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            loss=loss_builtin,
            custom_loss=loss_custom,
            optimizer=optimizer_name,
            optimizer_params=optimizer_params,
            metrics=metrics,
            model_input_scheme=model_input_scheme,
            strategy=strategy_name,
            strategy_params=strategy_params,
            compressor=compressor_name,
            compressor_params=compressor_params,
        )

    tmpdirs = mkdtemp(pyus)

    model, parts_meta = saver.save(slmodel, label_pyu, model_configs, tmpdirs)

    feature_names = x.columns
    model_meta = ModelMeta(
        parts=parts_meta,
        model_input_scheme=model_input_scheme,
        label_col=train_dataset_label,
        feature_names=feature_names,
    )

    model_db = model_dumps(
        ctx,
        "sl_nn",
        DistDataType.SL_NN_MODEL,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        model,
        json.dumps(model_meta.to_dict()),
        output_model,
        train_dataset.system_info,
    )

    return {
        "output_model": model_db,
        "reports": dump_training_reports(
            reports,
            history,
            train_dataset.system_info,
        ),
    }


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
