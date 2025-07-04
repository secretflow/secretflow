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

from secretflow.component.core import (
    SPU_RUNTIME_CONFIG_FM128_FXP40,
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Output,
    Reporter,
    register,
)
from secretflow.stats.ss_vif_v import VIF


@register(domain="stats", version="1.0.0", name="ss_vif")
class SSVif(Component):
    '''
    Calculate Variance Inflation Factor(VIF) for vertical partitioning dataset
    by using secret sharing.

    - For large dataset(large than 10w samples & 200 features), recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
    '''

    feature_selects: list[str] = Field.table_column_attr(
        "input_ds",
        desc="Specify which features to calculate VIF with. If empty, all features will be used.",
    )
    input_ds: Input = Field.input(
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    )
    report: Output = Field.output(
        desc="Output Variance Inflation Factor(VIF) report.",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        x = ctx.load_table(self.input_ds, self.feature_selects).to_pandas()

        if len(x.partitions) == 1:
            device = next(iter((x.partitions.keys())))
        else:
            device = ctx.make_spu(config=SPU_RUNTIME_CONFIG_FM128_FXP40)

        with ctx.tracer.trace_running():
            vif: np.ndarray = VIF(device).vif(x)

        feature_names = x.columns

        assert vif.shape[0] == len(feature_names)

        descriptions = {feature_names[i]: float(vif[i]) for i in range(vif.shape[0])}

        r = Reporter(name="vif", desc="vif list", system_info=self.input_ds.system_info)
        r.add_tab(descriptions)
        self.report.data = r.to_distdata()
