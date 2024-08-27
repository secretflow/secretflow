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

import numpy as np

from secretflow.component.component import (
    CompEvalError,
    Component,
    IoType,
    TableColParam,
)
from secretflow.component.data_utils import DistDataType
from secretflow.component.dataframe import CompDataFrame
from secretflow.component.device_utils import make_spu
from secretflow.device.device.spu import SPU
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.report_pb2 import Descriptions, Div, Report, Tab
from secretflow.stats.ss_vif_v import VIF

ss_vif_comp = Component(
    "ss_vif",
    domain="stats",
    version="0.0.1",
    desc="""Calculate Variance Inflation Factor(VIF) for vertical partitioning dataset
    by using secret sharing.

    - For large dataset(large than 10w samples & 200 features), recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
    """,
)
ss_vif_comp.io(
    io_type=IoType.INPUT,
    name="input_data",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="feature_selects",
            desc="Specify which features to calculate VIF with. If empty, all features will be used.",
        )
    ],
)
ss_vif_comp.io(
    io_type=IoType.OUTPUT,
    name="report",
    desc="Output Variance Inflation Factor(VIF) report.",
    types=[DistDataType.REPORT],
)


@ss_vif_comp.eval_fn
def ss_vif_eval_fn(
    *,
    ctx,
    input_data,
    input_data_feature_selects,
    report,
):
    feature_selects = (
        input_data_feature_selects if len(input_data_feature_selects) else None
    )

    x = CompDataFrame.from_distdata(
        ctx,
        input_data,
        load_features=True,
        col_selects=feature_selects,
    ).to_pandas()

    if len(x.partitions) == 1:
        for pyu, _ in x.partitions.items():
            device = pyu
    else:
        device = make_spu(ctx)

    with ctx.tracer.trace_running():
        vif: np.ndarray = VIF(device).vif(x)

    feature_names = x.columns

    assert vif.shape[0] == len(feature_names)

    r_desc = Descriptions(
        items=[
            Descriptions.Item(
                name=feature_names[i], type="float", value=Attribute(f=vif[i])
            )
            for i in range(vif.shape[0])
        ],
    )

    report_mate = Report(
        name="vif",
        desc="vif list",
        tabs=[
            Tab(
                divs=[
                    Div(
                        children=[
                            Div.Child(
                                type="descriptions",
                                descriptions=r_desc,
                            )
                        ],
                    )
                ],
            )
        ],
    )

    report_dd = DistData(
        name=report,
        type=str(DistDataType.REPORT),
        system_info=input_data.system_info,
    )
    report_dd.meta.Pack(report_mate)

    return {"report": report_dd}
