# Copyright 2022 Ant Group Co., Ltd.
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

from typing import Dict

import numpy as np
import pandas as pd

from secretflow.data import partition
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU, PYUObject, proxy


@proxy(PYUObject)
class VertBinSubstitutionPyuWorker:
    def sub(self, data: pd.DataFrame, r: Dict) -> pd.DataFrame:
        """
        PYU functions for binning substitution.

        Args:
            data: input dataset for this party.
            r: bin substitution rules.

        Returns:
            data: dataset after substituted.
        """
        assert isinstance(r, dict) and "variables" in r, f"not support rule format {r}"

        rules = {v['name']: v for v in r["variables"]}
        assert np.isin(
            list(rules.keys()), data.columns
        ).all(), "rule feature names [%s] mismatch with input dataset [%s]" % (
            str(rules.keys()),
            str(data.columns),
        )

        for v in rules:
            col_data = data[v]
            rule = rules[v]
            if rule["type"] == "string":
                condlist = [col_data == c for c in rule["categories"]]
                choicelist = rule["filling_values"]
                data[v] = np.select(condlist, choicelist, rule["else_filling_value"])
            else:
                condlist = list()
                split_points = rule["split_points"]
                # if no effective split points, we do no transformation
                if len(split_points) == 0:
                    continue
                for i in range(len(split_points)):
                    if i == 0:
                        condlist.append(col_data <= split_points[i])
                    else:
                        condlist.append(
                            (col_data > split_points[i - 1])
                            & (col_data <= split_points[i])
                        )
                if len(split_points) > 0:
                    condlist.append(col_data > split_points[-1])
                choicelist = rule["filling_values"]
                assert len(choicelist) == len(split_points) + 1, f"{choicelist}"
                assert len(condlist) == len(split_points) + 1, f"{condlist}"
                data[v] = np.select(condlist, choicelist, rule["else_filling_value"])

        return data


class VertBinSubstitution:
    def substitution(
        self, vdata: VDataFrame, rules: Dict[PYU, PYUObject]
    ) -> VDataFrame:
        """
        substitute dataset's value by binning substitution rules.

        Args:
            vdata: vertical slice dataset to be substituted.
            rules: binning substitution rules build by VertBinning.

        Returns:
            new_vdata: vertical slice dataset after substituted.
        """
        works: Dict[PYU, VertBinSubstitutionPyuWorker] = {}
        for device in rules:
            assert (
                device in vdata.partitions.keys()
            ), f"device {device} not exist in vdata"
            works[device] = VertBinSubstitutionPyuWorker(device=device)

        new_vdata = VDataFrame(
            {
                d: partition(
                    data=works[d].sub(vdata.partitions[d].data, rules[d]),
                    backend=vdata.partitions[d].backend,
                )
                for d in rules
            }
        )

        return new_vdata
