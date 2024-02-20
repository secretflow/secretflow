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
from typing import Dict, Tuple

from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import (
    DistDataType,
    load_table,
    model_loads,
    save_prediction_dd,
)
from secretflow.device.device.pyu import PYU, PYUObject

from ..core.utils import check_enabled_or_fail
from .base import MODEL_MAX_MAJOR_VERSION, MODEL_MAX_MINOR_VERSION, ModelMeta, mkdtemp
from .compile.compile import ModelConfig
from .training import predictor, saver

slnn_predict_comp = Component(
    "slnn_predict",
    domain="ml.predict",
    version="0.0.1",
    desc="""Predict using the SLNN model.
    This component is not enabled by default, it requires the use of the full version
    of secretflow image and setting the ENABLE_NN environment variable to true.""",
)
slnn_predict_comp.int_attr(
    name="batch_size",
    desc="The number of examples per batch.",
    is_list=False,
    is_optional=True,
    default_value=8192,
    lower_bound=0,
    lower_bound_inclusive=False,
)
slnn_predict_comp.str_attr(
    name="receiver",
    desc="Party of receiver.",
    is_list=False,
    is_optional=False,
)
slnn_predict_comp.str_attr(
    name="pred_name",
    desc="Column name for predictions.",
    is_list=False,
    is_optional=True,
    default_value="pred",
    allowed_values=None,
)
slnn_predict_comp.bool_attr(
    name="save_ids",
    desc=(
        "Whether to save ids columns into output prediction table. "
        "If true, input feature_dataset must contain id columns, and receiver party must be id owner."
    ),
    is_list=False,
    is_optional=True,
    default_value=False,
)
slnn_predict_comp.bool_attr(
    name="save_label",
    desc=(
        "Whether or not to save real label columns into output pred file. "
        "If true, input feature_dataset must contain label columns and receiver party must be label owner."
    ),
    is_list=False,
    is_optional=True,
    default_value=False,
)
slnn_predict_comp.io(
    io_type=IoType.INPUT,
    name="model",
    desc="Input model.",
    types=[DistDataType.SL_NN_MODEL],
)
slnn_predict_comp.io(
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
slnn_predict_comp.io(
    io_type=IoType.OUTPUT,
    name="pred",
    desc="Output prediction.",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=None,
)


def load_slnn_model(
    ctx, pyus, model, tmpdirs: Dict[PYU, Path]
) -> Tuple[Dict[PYU, ModelConfig], ModelMeta]:
    model_objs, model_meta_str = model_loads(
        ctx,
        model,
        MODEL_MAX_MAJOR_VERSION,
        MODEL_MAX_MINOR_VERSION,
        DistDataType.SL_NN_MODEL,
        pyus=pyus,
    )
    assert len(model_objs) == len(pyus) and isinstance(
        model_objs[0], PYUObject
    ), f"model_objs {model_objs}, model_meta_str {model_meta_str}"

    model_meta_dict = json.loads(model_meta_str)
    assert isinstance(model_meta_dict, Dict)

    model_meta = ModelMeta.from_dict(model_meta_dict)
    assert (
        len(model_meta.parts) == 2
        and model_meta.model_input_scheme
        and len(model_meta.feature_names) > 0
    )

    model = saver.load(model_objs, model_meta.parts, tmpdirs)

    return model, model_meta


@slnn_predict_comp.eval_fn
def ss_slnn_predict_eval_fn(
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
    check_enabled_or_fail()

    receiver_pyu = PYU(receiver)

    x = load_table(ctx, feature_dataset, load_features=True)
    pyus = set(x.partitions.keys())
    pyus.add(receiver_pyu)
    # ensure all parties have save order
    pyus = sorted(list(pyus))
    tmpdirs = mkdtemp(pyus)

    pyus = {str(pyu): pyu for pyu in pyus}
    model, model_meta = load_slnn_model(ctx, pyus, model, tmpdirs)
    feature_names = model_meta.feature_names
    x = x[feature_names]

    with ctx.tracer.trace_running():
        pyu_y = predictor.predict(
            ctx,
            batch_size=batch_size,
            feature_dataset=x,
            model=model,
            model_input_scheme=model_meta.model_input_scheme,
        )

        assert receiver_pyu in pyu_y.partitions, f"receiver must be the label provider"

    with ctx.tracer.trace_io():
        y_db = save_prediction_dd(
            ctx,
            pred,
            receiver_pyu,
            pyu_y,
            pred_name,
            feature_dataset,
            feature_dataset_saved_features,
            model_meta.label_col if save_label else [],
            save_ids,
        )

    return {"pred": y_db}
