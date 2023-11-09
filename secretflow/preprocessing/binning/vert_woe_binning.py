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

from typing import Dict, List, Union

import jax.numpy as jnp
import numpy as np
from heu import phe

from secretflow.data.vertical import VDataFrame
from secretflow.device import HEU, PYU, PYUObject, SPU
from secretflow.device.device.heu import HEUMoveConfig
from secretflow.preprocessing.binning.vert_woe_binning_pyu import (
    VertWoeBinningPyuWorker,
)


class VertWoeBinning:
    """
    woe binning for vertical slice datasets.

    Split all features into bins by equal frequency or ChiMerge.
    Then calculate woe value & iv value for each bin by SS or HE secure device to protect Y label.

    Finally, this method will output binning rules used to substitute features' value into woe by VertBinSubstitution.

    more details about woe/iv value:
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

    Attributes:
        secure_device: HEU or SPU for secure bucket summation.

    """

    def __init__(self, secure_device: Union[SPU, HEU]):
        self.secure_device = secure_device

    def _find_label_holder_device(self, vdata: VDataFrame, label_name) -> PYU:
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
                label_holder_device = device
                label_count += 1

        assert (
            label_count == 1
        ), f"One and only one party can have label, but found {label_count}, label name {label_name}, vdata {vdata.columns}."

        return label_holder_device

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
        The total sample numbers in each bin are not protected in current implementation.
        The split points for bins and the number of postive samples in each bin are protected.

        Attributes:
            vdata: vertical slice datasets
                use {binning_method} to bin all number type features.
                for string type feature bin by it's categories.
                else bin is count for np.nan samples
            binning_method: how to bin number type features.
                Options: "quantile"(equal frequency)/"chimerge"(ChiMerge from AAAI92-019)/"eq_range"(equal range)
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
                            "filling_values": list[float], # woe values for each bins.
                            "else_filling_value": float, # woe value for np.nan samples.
                        },
                        # ... others feature
                    ],
                    # label holder's PYUObject only
                    # warning: giving bin_ivs to other party will leak positive samples in each bin.
                    # it is up to label holder's will to give feature iv or bin ivs or all info to workers.
                    # for more information, look at: https://github.com/secretflow/secretflow/issues/565

                    # in the following comment, by safe we mean label distribution info is not leaked.
                    "feature_iv_info" :[
                        {
                            "name": str, #feature name
                            "ivs": list[float], #iv values for each bins, not safe to share with workers in any case.
                            "else_iv": float, #iv for nan values, may share to with workers
                            "feature_iv": float, #sum of bin_ivs, safe to share with workers when bin num > 2.
                        }
                    ]
                }


        """
        assert binning_method in (
            "quantile",
            "chimerge",
            "eq_range",
        ), f"binning_method only support ('quantile', 'chimerge', 'eq_range'), got {binning_method}"
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

        label_holder_device = self._find_label_holder_device(vdata, label_name)
        label_holder_audit_log_path = None

        if isinstance(self.secure_device, HEU):
            assert self.secure_device.sk_keeper_name() == label_holder_device.party, (
                f"HEU sk keeper party {self.secure_device.sk_keeper_name()} "
                "mismatch with label_holder device's party {label_holder_device.party}"
            )
            if audit_log_path:
                assert (
                    label_holder_device.party in audit_log_path
                ), "can not find sk keeper device's audit log path"
                label_holder_audit_log_path = audit_log_path[label_holder_device.party]

        workers: Dict[PYU, VertWoeBinningPyuWorker] = {}
        if label_holder_device not in bin_names:
            bin_names[label_holder_device] = list()

        for device in bin_names:
            assert (
                device in vdata.partitions.keys()
            ), f"device {device} in bin_names not exist in vdata"
            workers[device] = VertWoeBinningPyuWorker(
                vdata.partitions[device].data.data,
                binning_method,
                bin_num,
                bin_names[device],
                label_name if label_holder_device == device else "",
                positive_label,
                chimerge_init_bins,
                chimerge_target_bins,
                chimerge_target_pvalue,
                device=device,
            )

        bin_rules: Dict[PYU, PYUObject] = {}

        # label_holder build woe rules
        label_holder_worker = workers[label_holder_device]
        label, label_holder_report = label_holder_worker.label_holder_work(
            vdata.partitions[label_holder_device].data
        )

        if isinstance(self.secure_device, SPU):
            secure_label = label.to(self.secure_device)
        elif isinstance(self.secure_device, HEU):
            secure_label = label.to(
                self.secure_device,
                HEUMoveConfig(heu_audit_log=label_holder_audit_log_path),
            )
        else:
            raise NotImplementedError(
                f'Secure device should be SPU or HEU, but got {type(self.secure_device)}.'
            )

        # all participants
        for device in workers:
            if device == label_holder_device:
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
                move_config = HEUMoveConfig()
                move_config.heu_encoder = phe.BigintEncoderParams()
                bin_indices = worker.participant_build_sum_indices(
                    vdata.partitions[device].data
                )
                bins_positive = secure_label.batch_select_sum(bin_indices).to(
                    label_holder_device
                )
            else:
                bin_select = worker.participant_build_sum_select(
                    vdata.partitions[device].data
                )

                def spu_work(label, select):
                    return jnp.matmul(label, select)

                bins_positive = self.secure_device(spu_work)(
                    secure_label, bin_select.to(self.secure_device)
                ).to(label_holder_device)

            bim_sum_info = worker.get_bin_sum_info().to(label_holder_device)
            (
                bin_stats,
                total_counts,
                merged_split_point_indices,
            ) = label_holder_worker.label_holder_sum_bin(bins_positive, bim_sum_info)
            worker.participant_update_info(
                total_counts.to(device), merged_split_point_indices.to(device)
            )
            woes, ivs = label_holder_worker.label_holder_calc_woe_for_peer(bin_stats)
            # label_holder process and save the ivs, calculate the feature_ivs
            label_holder_worker.label_holder_collect_iv_for_participant(
                ivs, bim_sum_info
            )
            report = worker.participant_build_report(woes.to(device))
            bin_rules[device] = report

        # feature ivs are in label_holder report, which may be later shared to worker
        label_holder_report = label_holder_worker.generate_iv_report(
            label_holder_report
        )
        bin_rules[label_holder_device] = label_holder_report

        return bin_rules
