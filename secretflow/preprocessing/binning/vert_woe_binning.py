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

from typing import List, Dict, Union
import jax.numpy as jnp
import numpy as np

from secretflow.device.device.base import MoveConfig
from secretflow.preprocessing.binning.vert_woe_binning_pyu import (
    VertWoeBinningPyuWorker,
)
from secretflow.device import SPU, HEU, PYU, PYUObject
from secretflow.data.vertical import VDataFrame
from secretflow.device import reveal


class VertWoeBinning:
    """
    woe binning for vertical slice datasets.

    Split all features into bins by equal frequency or ChiMerge.
    Then calculate woe value & iv value for each bin by SS or HE secure device to protect Y label.

    Finally, this method will output binning rules used to substitute features' value into woe by VertWOESubstitution.

    more details about woe/iv value:
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

    Attributes:
        secure_device: HEU or SPU for secure bucket summation.

    """

    def __init__(self, secure_device: Union[SPU, HEU]):
        self.secure_device = secure_device

    def _find_master_device(self, vdata: VDataFrame, label_name) -> PYU:
        """
        Find which holds the label column.

        Attributes:
            vdata: vertical slice datasets
            label_name: label column name.

        Return:
            PYU device
        """
        device_column_names = vdata.partition_columns
        label_count = 0
        for device in device_column_names:
            if np.isin(label_name, device_column_names[device]).all():
                master_device = device
                label_count += 1

        assert (
            label_count == 1
        ), f"One and only one party can have label, but found {label_count}"

        return master_device

    def binning(
        self,
        vdata: VDataFrame,
        binning_method: str = "quantile",
        bin_num: int = 10,
        bin_names: Dict[PYU, List[str]] = {},
        label_name: str = "",
        positive_label: str = "1",
        chimerge_init_bins: int = 100,
        chimerge_target_bins: int = 10,
        chimerge_target_pvalue: float = 0.1,
        audit_log_path: Dict[str, str] = {},
    ):
        """
        Build woe substitution rules base on vdata.
        Only support binary classification label dataset.

        Attributes:
            vdata: vertical slice datasets
                use {binning_method} to bin all number type features.
                for string type feature bin by it's categories.
                else bin is count for np.nan samples
            binning_method: how to bin number type features.
                Options: "quantile"(equal frequency)/"chimerge"(ChiMerge from AAAI92-019)
                Default: "quantile"
            bin_num: max bin counts for one features.
                Range: (0, ∞]
                Default: 10
            bin_names: which features should be binned.
            label_name: label column name.
            positive_label: which value represent positive value in label.
            chimerge_init_bins: max bin counts for initialization binning in ChiMerge.
                Range: (2, ∞]
                Default: 100
            chimerge_target_bins: stop merge if remain bin counts is less than or equal to this value.
                Range: [2, {chimerge_init_bins})
                Default: 10
            chimerge_target_pvalue: stop merge if biggest pvalue of remain bins is greater than this value.
                Range: (0, 1)
                Default: 0.1
            audit_log_path: output audit log for HEU encrypt to device's local path. empty means disable.
                example: {'alice': '/path/to/alice/audit/filename', 'bob': 'bob/audit/filename'}
                NOTICE: Please !!DO NOT!! touch this options, leave it empty and disabled.
                        Unless you really know this option's meaning and accept its risk.

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
                            "woes": list[float], # woe values for each bins.
                            "else_woe": float, # woe value for np.nan samples.
                            "ivs": list[float], # iv values for each bins.
                            "else_iv": float, # iv value for np.nan samples.
                        },
                        # ... others feature
                    ]
                }
        """
        assert binning_method in (
            "quantile",
            "chimerge",
        ), f"binning_method only support ('quantile', 'chimerge'), got {binning_method}"
        assert bin_num > 0, f"bin_num range (0, ∞], got {bin_num}"
        assert (
            chimerge_init_bins > 2
        ), f"chimerge_init_bins range (2, ∞], got {chimerge_init_bins}"
        assert (
            chimerge_target_bins >= 2 and chimerge_target_bins < chimerge_init_bins
        ), f"chimerge_target_bins range [2, chimerge_init_bins), got {chimerge_target_bins}"
        assert (
            chimerge_target_pvalue > 0 and chimerge_target_pvalue < 1
        ), f"chimerge_target_pvalue range (0, 1), got {chimerge_target_pvalue}"

        if audit_log_path:
            assert isinstance(self.secure_device, HEU), "only HEU support audit log"

        master_device = self._find_master_device(vdata, label_name)
        master_audit_log_path = None

        if isinstance(self.secure_device, HEU):
            assert len(bin_names) == 2, "only support two party binning in HEU mode"
            assert self.secure_device.sk_keeper_name() == master_device.party, (
                f"HEU sk keeper party {self.secure_device.sk_keeper_name()} "
                "mismatch with master device's party {master_device.party}"
            )
            if audit_log_path:
                assert (
                    master_device.party in audit_log_path
                ), "can not find sk keeper device's audit log path"
                master_audit_log_path = audit_log_path[master_device.party]

        workers: Dict[PYU, VertWoeBinningPyuWorker] = {}
        if master_device not in bin_names:
            bin_names[master_device] = list()

        for device in bin_names:
            assert (
                device in vdata.partitions.keys()
            ), f"device {device} in bin_names not exist in vdata"
            workers[device] = VertWoeBinningPyuWorker(
                vdata.partitions[device],
                binning_method,
                bin_num,
                bin_names[device],
                label_name if master_device == device else "",
                positive_label,
                chimerge_init_bins,
                chimerge_target_bins,
                chimerge_target_pvalue,
                device=device,
            )

        woe_rules: Dict[PYU, PYUObject] = {}

        # master build woe rules
        master_worker = workers[master_device]
        label, master_report = master_worker.master_work(
            vdata.partitions[master_device].data
        )
        woe_rules[master_device] = master_report

        secure_label = label.to(
            self.secure_device, MoveConfig(heu_audit_log=master_audit_log_path)
        )

        # all slaves
        for device in workers:
            if device == master_device:
                continue

            worker = workers[device]

            if isinstance(self.secure_device, HEU):
                if audit_log_path:
                    assert (
                        device.party in audit_log_path
                    ), f"can not find {device.party} device's audit log path"
                    worker_audit_path = audit_log_path[device.party]
                    secure_label.dump(worker_audit_path)
                    self.secure_device.get_participant(device.party).dump_pk.remote(
                        f'{worker_audit_path}.pk.pickle'
                    )
                idx_obj = worker.slave_build_sum_indices(vdata.partitions[device].data)
                # FIXME: avoid reveal, use remote function to calc sum in HEU when HEU support it
                bin_indices = reveal(idx_obj)
                bins_positive = [
                    secure_label[b].sum().to(master_device).to(device)
                    for b in bin_indices
                ]
                bin_stats = worker.slave_sum_bin(bins_positive)
            else:
                bin_select = worker.slave_build_sum_select(
                    vdata.partitions[device].data
                )

                def spu_work(label, select):
                    return jnp.matmul(label, select)

                bins_positive = self.secure_device(spu_work)(
                    secure_label, bin_select.to(self.secure_device)
                ).to(device)

                bin_stats = worker.slave_sum_bin(bins_positive)

            woe_ivs = master_worker.master_calc_woe_for_peer(
                bin_stats.to(master_device)
            )
            report = worker.slave_build_report(woe_ivs.to(device))
            woe_rules[device] = report

        return woe_rules
