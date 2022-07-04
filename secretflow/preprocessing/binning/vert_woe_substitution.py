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
import pandas as pd
import numpy as np

from secretflow.device import proxy, PYUObject, PYU
from secretflow.data.vertical import VDataFrame
from secretflow.data.base import Partition


@proxy(PYUObject)
class VertWOESubstitutionPyuWorker:
    def sub(self, data: pd.DataFrame, r: Dict) -> pd.DataFrame:
        """
        PYU functions for woe substitution.

        Args:
            data: input dataset for this party.
            r: woe substitution rules.

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
                choicelist = rule["woes"]
                data[v] = np.select(condlist, choicelist, rule["else_woe"])
            else:
                condlist = list()
                split_points = rule["split_points"]
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
                choicelist = rule["woes"]
                data[v] = np.select(condlist, choicelist, rule["else_woe"])

        return data


class VertWOESubstitution:
    def substitution(
        self, vdata: VDataFrame, woe_rules: Dict[PYU, PYUObject]
    ) -> VDataFrame:
        """
        substitute dataset's value by woe substitution rules.

        Args:
            vdata: vertical slice dataset to be substituted.
            woe_rules: woe substitution rules build by VertWoeBinning.

        Returns:
            new_vdata: vertical slice dataset after substituted.
        """
        works: Dict[PYU, VertWOESubstitutionPyuWorker] = {}
        for device in woe_rules:
            assert (
                device in vdata.partitions.keys()
            ), f"device {device} not exist in vdata"
            works[device] = VertWOESubstitutionPyuWorker(device=device)

        new_vdata = VDataFrame(
            {
                d: Partition(data=works[d].sub(vdata.partitions[d].data, woe_rules[d]))
                for d in woe_rules
            }
        )

        return new_vdata
