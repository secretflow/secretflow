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
import pandas as pd

from secretflow.component.core import (
    SPU_RUNTIME_CONFIG_FM128_FXP40,
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Output,
    Reporter,
    VTable,
    VTableFieldKind,
    register,
)
from secretflow.stats.ss_pearsonr_v import PearsonR


@register(domain="stats", version="1.0.0", name="ss_pearsonr")
class SSPearsonr(Component):
    '''
    Calculate Pearson's product-moment correlation coefficient for vertical partitioning dataset
    by using secret sharing.

    - For large dataset(large than 10w samples & 200 features), recommend to use [Ring size: 128, Fxp: 40] options for SPU device.
    '''

    feature_selects: list[str] = Field.table_column_attr(
        "input_ds",
        desc="Specify which features to calculate correlation coefficient with. If empty, all features will be used",
    )
    input_ds: Input = Field.input(  # type: ignore
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE],
    )
    report: Output = Field.output(
        desc="Output Pearson's product-moment correlation coefficient report.",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        feature_selects = self.feature_selects if len(self.feature_selects) else None
        in_tbl = VTable.from_distdata(self.input_ds, columns=feature_selects)
        in_tbl.check_kinds(kinds=VTableFieldKind.FEATURE)

        x = ctx.load_table(in_tbl).to_pandas()
        if len(x.partitions) == 1:
            device = next(iter((x.partitions.keys())))
        else:
            device = ctx.make_spu(config=SPU_RUNTIME_CONFIG_FM128_FXP40)

        with ctx.tracer.trace_running():
            pr: np.ndarray = PearsonR(device).pearsonr(x)

        feature_names = x.columns

        assert pr.shape[0] == len(feature_names) and pr.shape[1] == len(feature_names)

        r_table = pd.DataFrame(pr, columns=feature_names)

        r = Reporter(name="corr", desc="corr table")
        r.add_tab(r_table)
        r.dump_to(self.report, self.input_ds.system_info)
