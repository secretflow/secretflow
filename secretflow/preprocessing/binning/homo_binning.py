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

"""
driver端程序
"""
import functools
import logging
import operator
from typing import Dict, List

import numpy as np

from secretflow.data.horizontal import HDataFrame
from secretflow.device import reveal
from secretflow.preprocessing.binning.homo_binning_base import HomoBinningBase


class HomoBinning(HomoBinningBase):
    """entrance of federate binning

    Attributes:
        bin_num: how many buckets need to be split
        compress_thres: compression threshold. If the value is greater than the threshold, do compression
        head_size: buffer size
        error: error tolerance
        bin_indexes: index of features to binning
        bin_names: name of features to binning
        abnormal_list: list of anomaly features
        allow_duplicate: whether to allow duplicate bucket values
        aggregator:  to aggregate values with aggregator
        max_iter: max iteration round
    """

    def __init__(
        self,
        bin_num: int = 10,
        compress_thres: int = 10000,
        head_size: int = 10000,
        error: float = 1e-4,
        bin_indexes: List[int] = [],
        bin_names: List[str] = [],
        abnormal_list: List[str] = None,
        allow_duplicate: bool = False,
        max_iter: int = 10,
        aggregator=None,
    ):
        self.bin_num = bin_num
        self.compress_thres = compress_thres
        self.head_size = head_size
        self.error = error
        self.bin_indexes = bin_indexes
        self.bin_names = bin_names
        self.abnormal_list = abnormal_list
        self.allow_duplicate = allow_duplicate
        self.max_iter = max_iter

        self._total_count = 0
        self._missing_counts = 0
        self._error_rank = None
        self._max_iter = max_iter
        self._workers = {}

    def _init_binning_worker(self, hdata: HDataFrame = None):
        self._workers = {}
        for device in hdata.partitions.keys():
            self._workers[device] = HomoBinningBase(
                bin_num=self.bin_num,
                bin_names=self.bin_names,
                bin_indexes=self.bin_indexes,
                compress_thres=self.compress_thres,
                error=self.error,
                head_size=self.head_size,
                allow_duplicate=self.allow_duplicate,
                abnormal_list=self.abnormal_list,
                device=device,
            )
        self.aggregator = hdata.aggregator

    def fit_split_points(self, hdata: HDataFrame):
        """entrance of federate binning

        Args:
            data: HDataFrame,input data to binning

        Returns:
            bin_result: a dict of binning result, PYUObject
        """
        if hdata is None:
            raise ValueError("Input data connot be none")
        logging.debug(f"abnormal_list: {self.abnormal_list}")

        bin_results = {}
        header = hdata.columns
        self._total_count = hdata.count()[0]
        self._error_rank = np.ceil(self.error * self._total_count)

        self.max_values = {}
        self.min_values = {}
        hdf_min_values = hdata.min()
        hdf_max_values = hdata.max()
        for col in header:
            self.max_values[col] = hdf_max_values[col]
            self.min_values[col] = hdf_min_values[col]
        self._init_binning_worker(hdata)

        new_header = [str(h) for h in header]
        (
            self.bin_names,
            self.bin_indexes,
            self.bin_idx_name,
            self.col_name_maps,
        ) = self.setup_header_param(
            header=new_header, bin_names=self.bin_names, bin_indexes=self.bin_indexes
        )
        missing_counts = []
        for device, worker in self._workers.items():
            worker.set_header_param(
                self.bin_names, self.bin_indexes, self.bin_idx_name, self.col_name_maps
            )
            worker.init_query_points(
                split_num=self.bin_num + 1,
                error_rank=self._error_rank,
                need_first=False,
                max_values=self.max_values,
                min_values=self.min_values,
                total_count=self._total_count,
            )
            worker.cal_summary_dict(hdata.partitions[device].data)
            missing_counts.append(worker.get_missing_count())

        g_missing_count = reveal(self.aggregator.sum(missing_counts, axis=0))

        for device, worker in self._workers.items():
            worker.set_missing_dict(g_missing_count)
            worker.set_aim_rank()
        local_ranks = [
            worker.query_values() for device, worker in self._workers.items()
        ]
        global_rank = reveal(self.aggregator.sum(local_ranks, axis=0))
        n_iter = 0
        logging.info("start recursive")
        while n_iter < self._max_iter:
            is_coverge = reveal(
                [
                    worker.renew_query_points(global_ranks=global_rank)
                    for device, worker in self._workers.items()
                ]
            )
            g_converge = functools.reduce(operator.and_, is_coverge)
            if g_converge:
                break
            local_ranks = [
                worker.query_values() for device, worker in self._workers.items()
            ]
            global_rank = reveal(self.aggregator.sum(local_ranks, axis=0))
            n_iter += 1
        bin_results = [
            worker.get_bin_result() for device, worker in self._workers.items()
        ]
        return bin_results[0]

    def setup_header_param(
        self, header: List[str], bin_names: List[str], bin_indexes: List[int]
    ) -> (List[str], List[int], Dict[int, str], Dict[int, str]):
        _bin_names = []
        _bin_indexes = []
        bin_idx_name = {}
        col_name_maps = {}

        if not bin_indexes and not bin_names:
            logging.warning("Both name and index are null, will use full columns")
            bin_names = header

        for index, col_name in enumerate(header):
            col_name_maps[index] = col_name

            if index in bin_indexes or col_name in bin_names:
                _bin_names.append(col_name)
                _bin_indexes.append(index)
                bin_idx_name[index] = col_name

        # Check that index and name are aligned
        for bin_index, bin_name in zip(_bin_indexes, _bin_names):
            header_name = header[bin_index]
            assert bin_name == header_name, (
                f"Bin name must be consistant with headers,but at index:{bin_index} "
                f"get bin_name={bin_name}, header_name={header_name}"
            )
        return _bin_names, _bin_indexes, bin_idx_name, col_name_maps
