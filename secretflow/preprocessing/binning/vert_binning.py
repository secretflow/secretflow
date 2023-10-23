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


from typing import Dict, List
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU, PYUObject

from secretflow.preprocessing.binning.vert_binning_pyu import VertBinningPyuWorker


class VertBinning:
    """
    equal range or frequency binning for vertical slice datasets.

    Split all features into bins by equal range or equal frequency.

    Finally, this method will output binning rules which are essentially split points.
    """

    def __init__(self):
        return

    def binning(
        self,
        vdata: VDataFrame,
        binning_method: str = "eq_range",
        bin_num: int = 10,
        bin_names: Dict[PYU, List[str]] = {},
    ):
        """
        Build bin substitution rules base on vdata.
        The split points for bins remains at each party.

        Attributes:
            vdata (VDataFrame): vertical slice datasets
                use {binning_method} to bin all number type features.
                for string type feature bin by it's categories.
                else bin is count for np.nan samples
            binning_method (str): how to bin number type features.
                Options: "quantile"(equal frequency)/"eq_range"(equal range)
                Default: "eq_range"
            bin_num (int): max bin counts for one features.
                Range: (0, ∞]
                Default: 10
            bin_names (Dict[PYU, List[str]]): which features should be binned.
                Default: {}

        Return:
            Dict[PYU, PYUObject], PYUObject contain a dict for all features' rule in this party.

            .. code:: python

                {
                    "variables":[
                        {
                            "name": str, # feature name
                            "type": str, # "string" or "numeric", if feature is discrete or continuous
                            "categories": list[str], # categories for discrete feature
                            "split_points": list[float], # left-open right-close split points
                            "total_counts": list[int], # total samples count in each bins.
                            "else_counts": int, # np.nan samples count
                            # for this binning method, we use [*range(f_num_bins)] as filling values
                            # that is 0 for bin with index 0, 1 for bin with index 1 etc.
                            "filling_values": list[float], # filling values for each bins.
                            # for this binning method, we use -1 as filling value for nan samples
                            "else_filling_value": float, # filling value for np.nan samples.
                        },
                        # ... others feature
                    ],
                }


        """
        assert binning_method in (
            "quantile",
            "eq_range",
        ), f"binning_method only support ('quantile', 'eq_range'), got {binning_method}"
        assert bin_num > 0, f"bin_num range (0, ∞], got {bin_num}"

        workers: Dict[PYU, VertBinningPyuWorker] = {}

        for device in bin_names:
            assert (
                device in vdata.partitions.keys()
            ), f"device {device} in bin_names not exist in vdata"
            workers[device] = VertBinningPyuWorker(
                vdata.partitions[device].data.data,
                binning_method,
                bin_num,
                bin_names[device],
                device=device,
            )

        rules: Dict[PYU, PYUObject] = {}
        for device in bin_names:
            rules[device] = workers[device].bin_feature_and_produce_rules()

        return rules
