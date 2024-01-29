# Copyright 2023 Ant Group Co., Ltd.
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
from typing import List, Tuple

from secretflow.component.component import (
    CompEvalError,
    Component,
    IoType,
    TableColParam,
)
from secretflow.component.data_utils import (
    DistDataType,
    load_table,
    model_dumps,
    model_loads,
    save_prediction_dd,
)
from secretflow.device.device.pyu import PYU
from secretflow.device.device.spu import SPU, SPUObject
from secretflow.ml.linear import LinearModel, RegType, SSRegression
from secretflow.utils.sigmoid import SigType

ss_sgd_train_comp = Component(
    "ss_sgd_train",
    domain="ml.train",
    version="0.0.1",
    desc="""Train both linear and logistic regression
    linear models for vertical partitioning dataset with mini batch SGD training solver by using secret sharing.

    - SS-SGD is short for secret sharing SGD training.
    """,
)
ss_sgd_train_comp.int_attr(
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
ss_sgd_train_comp.float_attr(
    name="learning_rate",
    desc="The step size at each iteration in one iteration.",
    is_list=False,
    is_optional=True,
    default_value=0.1,
    lower_bound=0,
    lower_bound_inclusive=False,
)
ss_sgd_train_comp.int_attr(
    name="batch_size",
    desc="The number of training examples utilized in one iteration.",
    is_list=False,
    is_optional=True,
    default_value=1024,
    lower_bound=0,
    lower_bound_inclusive=False,
)
ss_sgd_train_comp.str_attr(
    name="sig_type",
    desc="Sigmoid approximation type.",
    is_list=False,
    is_optional=True,
    default_value="t1",
    allowed_values=["real", "t1", "t3", "t5", "df", "sr", "mix"],
)
ss_sgd_train_comp.str_attr(
    name="reg_type",
    desc="Regression type",
    is_list=False,
    is_optional=True,
    default_value="logistic",
    allowed_values=["linear", "logistic"],
)
ss_sgd_train_comp.str_attr(
    name="penalty",
    desc="The penalty(aka regularization term) to be used.",
    is_list=False,
    is_optional=True,
    default_value="None",
    allowed_values=["None", "l1", "l2"],
)
ss_sgd_train_comp.float_attr(
    name="l2_norm",
    desc="L2 regularization term.",
    is_list=False,
    is_optional=True,
    default_value=0.5,
    lower_bound=0,
    lower_bound_inclusive=True,
)
ss_sgd_train_comp.float_attr(
    name="eps",
    desc="""If the change rate of weights is less than this threshold,
            the model is considered to be converged,
            and the training stops early. 0 to disable.""",
    is_list=False,
    is_optional=True,
    default_value=0.001,
    lower_bound=0,
    lower_bound_inclusive=True,
)
ss_sgd_train_comp.io(
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
ss_sgd_train_comp.io(
    io_type=IoType.OUTPUT,
    name="output_model",
    desc="Output model.",
    types=[DistDataType.SS_SGD_MODEL],
)

# current version 0.1
MODEL_MAX_MAJOR_VERSION = 0
MODEL_MAX_MINOR_VERSION = 1


@ss_sgd_train_comp.eval_fn
def ss_sgd_train_eval_fn(
    *,
    ctx,
    epochs,
    learning_rate,
    batch_size,
    sig_type,
    reg_type,
    penalty,
    l2_norm,
    eps,
    train_dataset,
    train_dataset_label,
    output_model,
    train_dataset_feature_selects,
):
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])

    reg = SSRegression(spu)

    assert len(train_dataset_label) == 1

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

    with ctx.tracer.trace_running():
        reg.fit(
            x=x,
            y=y,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            sig_type=sig_type,
            reg_type=reg_type,
            penalty=penalty,
            l2_norm=l2_norm,
            eps=eps,
        )

    model = reg.save_model()
    party_features_length = {
        device.party: len(columns) for device, columns in x.partition_columns.items()
    }
    model_meta = {
        "reg_type": model.reg_type.value,
        "sig_type": model.sig_type.value,
        "feature_selects": x.columns,
        "label_col": train_dataset_label,
        "party_features_length": party_features_length,
    }

    model_db = model_dumps(
        ctx,
        "ss_sgd",
        DistDataType.SS_SGD_MODEL,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        [model.weights],
        json.dumps(model_meta),
        output_model,
        train_dataset.system_info,
    )

    return {"output_model": model_db}


ss_sgd_predict_comp = Component(
    "ss_sgd_predict",
    domain="ml.predict",
    version="0.0.1",
    desc="Predict using the SS-SGD model.",
)
ss_sgd_predict_comp.int_attr(
    name="batch_size",
    desc="The number of training examples utilized in one iteration.",
    is_list=False,
    is_optional=True,
    default_value=1024,
    lower_bound=0,
    lower_bound_inclusive=False,
)
ss_sgd_predict_comp.str_attr(
    name="receiver",
    desc="Party of receiver.",
    is_list=False,
    is_optional=False,
)
ss_sgd_predict_comp.str_attr(
    name="pred_name",
    desc="Column name for predictions.",
    is_list=False,
    is_optional=True,
    default_value="pred",
    allowed_values=None,
)
ss_sgd_predict_comp.bool_attr(
    name="save_ids",
    desc=(
        "Whether to save ids columns into output prediction table. "
        "If true, input feature_dataset must contain id columns, and receiver party must be id owner."
    ),
    is_list=False,
    is_optional=True,
    default_value=True,
)
ss_sgd_predict_comp.bool_attr(
    name="save_label",
    desc=(
        "Whether or not to save real label columns into output pred file. "
        "If true, input feature_dataset must contain label columns and receiver party must be label owner."
    ),
    is_list=False,
    is_optional=True,
    default_value=False,
)
ss_sgd_predict_comp.io(
    io_type=IoType.INPUT,
    name="model",
    desc="Input model.",
    types=[DistDataType.SS_SGD_MODEL],
)
ss_sgd_predict_comp.io(
    io_type=IoType.INPUT,
    name="feature_dataset",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="saved_features",
            desc="which features should be saved with prediction result",
            col_min_cnt_inclusive=0,
        )
    ],
)
ss_sgd_predict_comp.io(
    io_type=IoType.OUTPUT,
    name="pred",
    desc="Output prediction.",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=None,
)


def load_ss_sgd_model(ctx, spu, model) -> Tuple[LinearModel, List[str]]:
    model_objs, model_meta_str = model_loads(
        ctx,
        model,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        DistDataType.SS_SGD_MODEL,
        spu=spu,
    )
    assert len(model_objs) == 1 and isinstance(
        model_objs[0], SPUObject
    ), f"model_objs {model_objs}, model_meta_str {model_meta_str}"

    model_meta = json.loads(model_meta_str)
    assert (
        isinstance(model_meta, dict)
        and "reg_type" in model_meta
        and "sig_type" in model_meta
    )
    model = LinearModel(
        weights=model_objs[0],
        reg_type=RegType(model_meta["reg_type"]),
        sig_type=SigType(model_meta["sig_type"]),
    )

    return model, model_meta


@ss_sgd_predict_comp.eval_fn
def ss_sgd_predict_eval_fn(
    *,
    ctx,
    batch_size,
    feature_dataset,
    feature_dataset_saved_features,
    model,
    receiver,
    pred_name,
    pred,
    save_ids,
    save_label,
):
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])

    model, model_meta = load_ss_sgd_model(ctx, spu, model)

    reg = SSRegression(spu)
    reg.load_model(model)

    x = load_table(
        ctx,
        feature_dataset,
        load_features=True,
        col_selects=model_meta["feature_selects"],
    )

    receiver_pyu = PYU(receiver)
    with ctx.tracer.trace_running():
        pyu_y = reg.predict(
            x=x,
            batch_size=batch_size,
            to_pyu=receiver_pyu,
        )

    with ctx.tracer.trace_io():
        y_db = save_prediction_dd(
            ctx,
            pred,
            receiver_pyu,
            pyu_y,
            pred_name,
            feature_dataset,
            feature_dataset_saved_features,
            model_meta["label_col"] if save_label else [],
            save_ids,
        )

    return {"pred": y_db}
