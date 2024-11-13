#!/usr/bin/env python3
# *_* coding: utf-8 *_*

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


import abc
import logging
from abc import abstractmethod
from typing import Dict, List


class BaseBinning(abc.ABC):
    def __init__(
        self, bin_names: List, bin_indexes: List, bin_num: int, abnormal_list: List
    ):
        self.bin_indexes = bin_indexes
        self.bin_names = bin_names
        self.bin_num = bin_num

        self.col_name_maps = {}
        self.bin_idx_name = {}

        if abnormal_list is None:
            self.abnormal_list = []
        else:
            self.abnormal_list = abnormal_list
        self.bin_results = {}

    @property
    def split_points(self):
        return self.bin_results

    @abstractmethod
    def fit_split_points(self, data):
        pass

    @staticmethod
    def _setup_header_param(
        header: List[str], bin_names: List[str], bin_indexes: List[int]
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
