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


from secretflow.device import SPU
from secretflow.data import FedNdarray
from secretflow.data.ndarray import (
    mean,
    r2_score,
    rss,
    mean_abs_err,
    mean_abs_percent_err,
    mean_squared_error,
    root_mean_squared_error,
    residual_histogram,
)


class RegressionEval:
    """
    Statistics Evaluation for a regression model on a dataset.

    Attributes:
        y_true: FedNdarray
            If y_true is from a single party, then each statistics is a PYUObject.
            If y_true is from multiple parties, then a SPU device is required
            and each statistics is a SPUObject.
        y_pred: FedNdarray
            y_true and y_pred must have the same device and partition shapes
        r2_score: Union[PYUObject, SPUObject]

        mean_abs_err: Union[PYUObject, SPUObject]

        mean_abs_percent_err: Union[PYUObject, SPUObject]

        sum_squared_errors: Union[PYUObject, SPUObject]

        mean_squared_errors: Union[PYUObject, SPUObject]

        root_mean_squared_errors: Union[PYUObject, SPUObject]

        y_true_mean: Union[PYUObject, SPUObject]

        y_pred_mean: Union[PYUObject, SPUObject]

        residual_hist: Union[PYUObject, SPUObject]

    """

    def __init__(
        self, y_true: FedNdarray, y_pred: FedNdarray, spu_device: SPU = None, bins=10
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.spu = spu_device
        self.bins = bins

    def gen_all_reports(self):
        assert self.y_true.shape == self.y_pred.shape
        if self.y_true is None or self.y_true.shape[0] == 0:
            return
        self.r2_score = r2_score(self.y_true, self.y_pred, self.spu)
        self.mean_abs_err = mean_abs_err(self.y_true, self.y_pred, self.spu)
        self.mean_abs_percent_err = mean_abs_percent_err(
            self.y_true, self.y_pred, self.spu
        )
        self.sum_squared_errors = rss(self.y_true, self.y_pred, self.spu)
        self.mean_squared_errors = mean_squared_error(
            self.y_true, self.y_pred, self.spu
        )
        self.root_mean_squared_errors = root_mean_squared_error(
            self.y_true, self.y_pred, self.spu
        )
        self.y_true_mean = mean(self.y_true, self.spu)
        self.y_pred_mean = mean(self.y_pred, self.spu)
        self.residual_hist = residual_histogram(
            self.y_true, self.y_pred, self.spu, bins=self.bins
        )
