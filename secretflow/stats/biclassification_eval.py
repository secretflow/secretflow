# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

# This is a wrapper of evaluation functions
from typing import Union

from secretflow.data import FedNdarray
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYUObject

from .core import gen_biclassification_reports

# BiClassification Report is different from Regression Evaluation in that many binning related statistics
# are computed in a sequential batch processing manner, rather than independent evaluations
# on the whole of y_true and y_score

# TODO: HDataFrame, VDataFrame and SPU support in future


class BiClassificationEval:
    """Statistics Evaluation for a bi-classification model on a dataset.

    Attribute:
            y_true: Union[FedNdarray, VDataFrame]
                input of labels
            y_score: Union[FedNdarray, VDataFrame]
                input of prediction scores
            bucket_size: int
                input of number of bins in report
            min_item_cnt_per_bucket: int
                min item cnt per bucket. If any bucket doesn't meet the requirement, error raises.
    """

    # binning, n true positive and false positive sequence calculations all require sorting
    # which is not supported by spu yet
    # all implementations related to bi-classification uses single-party implementation only

    def __init__(
        self,
        y_true: Union[FedNdarray, VDataFrame],
        y_score: Union[FedNdarray, VDataFrame],
        bucket_size: int,
        min_item_cnt_per_bucket: int = None,
    ):
        assert isinstance(
            y_true, (FedNdarray, VDataFrame)
        ), "y_true should be FedNdarray or VDataFrame"
        assert isinstance(
            y_score, (FedNdarray, VDataFrame)
        ), "y_score should be FedNdarray or VDataFrame"

        # for now we only consider vertical splitting case
        # y_true and y_score belongs to the same and single party
        assert (
            y_true.shape == y_score.shape
        ), "y_true and y_score should have the same shapes"
        assert (
            y_true.shape[1] == 1
        ), "y_true must be a single column, reshape before proceed"
        assert len(y_true.partitions) == len(
            y_score.partitions
        ), "y_true and y_score should have the same partitions"
        assert len(y_score.partitions) == 1, "y_score should have one partition"

        device1 = [*y_score.partitions.keys()][0]
        device2 = [*y_true.partitions.keys()][0]
        assert (
            device1 == device2
        ), "Currently we requires both inputs belongs to the same party and computation happens locally."
        # Later may use spu

        self.device = device1
        if isinstance(y_true, FedNdarray):
            self.y_true = [*y_true.partitions.values()][0]
        else:
            self.y_true = ([*y_true.partitions.values()][0]).data

        if isinstance(y_score, FedNdarray):
            self.y_score = [*y_score.partitions.values()][0]
        else:
            self.y_score = ([*y_score.partitions.values()][0]).data

        self.bucket_size = bucket_size
        self.min_item_cnt_per_bucket = min_item_cnt_per_bucket

    def get_all_reports(self) -> PYUObject:
        """get all reports. The reports contains:

        summary_report: SummaryReport

        group_reports: List[GroupReport]

        eq_frequent_bin_report: List[EqBinReport]

        eq_range_bin_report: List[EqBinReport]

        head_report: List[PrReport]
            reports for fpr = 0.001, 0.005, 0.01, 0.05, 0.1, 0.2

        see more in core.biclassification_eval_core
        """
        # possible spu launch and reveal in the future
        return self.device(gen_biclassification_reports)(
            self.y_true, self.y_score, self.bucket_size, self.min_item_cnt_per_bucket
        )
