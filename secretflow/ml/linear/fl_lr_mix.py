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

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import spu

from secretflow.data.mix.dataframe import MixDataFrame, PartitionWay
from secretflow.device.device import PYU
from secretflow.device.device.heu import HEU
from secretflow.device.device.pyu import PYUObject
from secretflow.device.device.type_traits import spu_fxp_precision
from secretflow.device.driver import reveal
from secretflow.ml.linear.fl_lr_v import FlLogisticRegressionVertical
from secretflow.security.aggregation import Aggregator, SecureAggregator
from secretflow.utils.errors import InvalidArgumentError


class _CustomSecureAggregator(SecureAggregator):
    """This aggregator is based on secure aggregator
    while this aggregator supports multi inputs from one device.

    The multi inputs from same device will be aggregated first and secure
    aggregation will be executed then.
    """

    def __init__(self, device: PYU, participants: List[PYU], fxp_bits: int = 18):
        super().__init__(device, participants, fxp_bits)

    @staticmethod
    def _merge_data(data: List[PYUObject]):
        """Merge multi data in same pyu."""
        assert data, 'Data should be not be empty.'
        pyu_data = {}
        for datum in data:
            if datum.device in pyu_data:
                pyu_data[datum.device].append(datum)
            else:
                pyu_data[datum.device] = [datum]
        for pyu, ws in pyu_data.items():
            if len(ws) > 1:
                pyu_data[pyu] = [pyu(lambda ws: np.average(ws, axis=0))(ws)]

        return [ws[0] for ws in pyu_data.values()]

    def sum(self, data: List[PYUObject], axis=None):
        data = self._merge_data(data)
        if len(data) > 1:
            return super().sum(data, axis)
        else:
            return data[0]

    def average(self, data: List[PYUObject], axis=None, weights=None):
        data = self._merge_data(data)
        if len(data) > 1:
            return super().average(data, axis, weights)
        else:
            return data[0]


class FlLogisticRegressionMix:
    """SGD based logistic regression for mix partitioned data.

    The following is an example to illustrate the algorithm.

    Suppose alice has features and label,  while bob/carol/dave have features
    only.

    The perspective of MixDataFrame X is as follows:

    +--------------+------------+------------+----------+
    |                           X                       |
    +--------------+------------+------------+----------+
    | VDataFrame_0 | alice_x0   |   bob_x    | dave_x0  |
    +--------------+------------+------------+----------+
    | VDataFrame_1 | alice_x1   |   carol_x  | dave_x1  |
    +--------------+------------+------------+----------+

    The perspective of MixDataFrame Y is as follows:

    +---------------+-------------+
    |               Y             |
    +---------------+-------------+
    |  VDataFrame_0 | alice_y0    |
    +---------------+-------------+
    |  VDataFrame_1 | alice_y1    |
    +---------------+-------------+

    When fitted with the X and Y, two
    :py:class:`~secretflow.ml.linear.FlLogisticRegressionVertical` instances
    are constructed. The first one will be fitted with VDataFrame_0 of X and Y,
    while the second one will be fitted with  VDataFrame_1 of X and Y,.

    The main steps of one epoch are:

    1. The `FlLogisticRegressionVertical` are fitted with the `VDataFrame` of X
       and Y respectly.
    2. Aggregate :math:`{\\theta}` of the `FlLogisticRegressionVertical` with
       :py:class:`~secretflow.security.aggregation.SecureAggregator`.
    3. Send aggregated :math:`{\\theta}` to the `FlLogisticRegressionVertical`.

    """

    def _init_train_data(
        self,
        x: MixDataFrame,
        y: MixDataFrame,
        epochs: int,
        batch_size: int,
    ):
        for ver_lr, x_part, y_part in zip(self.ver_lr_list, x.partitions, y.partitions):
            ver_lr.init_train_data(
                x=x_part.values, y=y_part.values, epochs=epochs, batch_size=batch_size
            )

    def _agg_weights(self, aggr_hooks: List):
        weights_list = [
            list(ver_lr.get_weight().values()) for ver_lr in self.ver_lr_list
        ]
        agg_weight = [
            self.aggregators[i].average(weights, axis=0)
            for i, weights in enumerate(zip(*weights_list))
        ]

        for hook in aggr_hooks:
            agg_weight = hook.on_aggregate(agg_weight)
        for ver_lr in self.ver_lr_list:
            ver_lr.set_weight(dict(zip(ver_lr.workers.keys(), agg_weight)))

    def _compute_loss(
        self,
        x: MixDataFrame,
        y: MixDataFrame,
    ) -> float:
        """Compute the loss.

        Args:
            x: the samples.
            y: the label.

        Returns:
            the loss value.
        """
        loss_list = [
            ver_lr.compute_loss(x_part.values, y_part.values, False)
            for ver_lr, x_part, y_part in zip(
                self.ver_lr_list, x.partitions, y.partitions
            )
        ]
        loss_sum = reveal(self.aggregator_y.sum(loss_list, axis=0))
        return loss_sum[0][0] / x.shape[0]

    def fit(
        self,
        x: MixDataFrame,
        y: MixDataFrame,
        batch_size: int,
        epochs: int,
        aggregators: List[Aggregator],
        heus: List[HEU],
        fxp_bits: Optional[int] = spu_fxp_precision(spu.spu_pb2.FM64),
        tol: Optional[float] = 1e-4,
        learning_rate: Optional[float] = 0.1,
        agg_epochs: Optional[int] = 1,
        audit_log_dir: Dict[PYU, str] = None,
        aggr_hooks: List = None,
    ):
        """Fit the model.

        Args:
            x: training vector. X should be a horizontal partitioned
                :py:class:`~secretflow.data.mix.MixDataFrame`, which consists
                of :py:class:`~secretflow.data.vertical.VDataFrame`s.
            y: target vector relative to x. Y should be a horizontal partitioned
                :py:class:`~secretflow.data.mix.MixDataFrame` alos. X and y
                should have the same amount of `VDataFrame`s.
            batch_size: number of samples per gradient update.
            epochs: number of epochs to train the model.
            aggregators: aggregator used to compute vertical lr. Amount of
                aggregators should be same as the `VDataFrame` of X.
            heus: a list of heu used to compute vertical lr. Amount of
                heus should be same as the `VDataFrame` of X.
            fxp_bits: the fraction bit length for encoding before sending to
                heu device. Defaults to spu_fxp_precision(spu.spu_pb2.FM64).
            tol: optional, tolerance for stopping criteria. Defaults to 1e-4.
            learning_rate: optional, learning rate. Defaults to 0.1.
            agg_epochs: aggregate weights for every {agg_epochs} epochs.
                Defaults to 1.
            audit_log_dir: a dict specifying the audit log directory for each
                device. No audit log if is None. Default to None.
                Please leave it None unless you are very sure what the audit
                does and accept the risk.
            aggr_hooks: The hooks called on each aggregation
        """
        assert isinstance(
            x, MixDataFrame
        ), f'X should be a MixDataFrame but got {type(x)}.'
        assert (
            x.partition_way == PartitionWay.HORIZONTAL
        ), 'X should be horizontal partitioned.'
        assert isinstance(
            y, MixDataFrame
        ), f'Y should be a MixDataFrame but got {type(y)}.'
        assert (
            y.partition_way == PartitionWay.HORIZONTAL
        ), 'y should be horizontal partitioned.'
        assert len(x.partitions) == len(
            y.partitions
        ), f'X has {len(x.partitions)} partitions while y has {len(y.partitions)}.'
        for part in y.partitions:
            assert (
                len(part.partitions.keys()) == 1
            ), 'One and only one party should have y.'
        assert len(aggregators) == len(
            x.partitions
        ), 'Amount of aggregators should be same as `VDataFrame`s of X.'
        assert len(heus) == len(
            x.partitions
        ), 'Amount of heus should be same as `VDataFrame`s of X.'
        aggr_hooks = aggr_hooks or []  # change None -> []
        aggr_hooks = aggr_hooks if isinstance(aggr_hooks, List) else [aggr_hooks]

        devices_list = [list(part.partitions.keys()) for part in x.partitions]
        self.aggregators = []
        for ver_devices in zip(*devices_list):
            participants = list(set(ver_devices))
            participants.sort()
            self.aggregators.append(
                _CustomSecureAggregator(
                    ver_devices[0], participants=participants, fxp_bits=fxp_bits
                )
            )
        devices_y = [list(part.partitions.keys())[0] for part in y.partitions]
        devices_y = list(set(devices_y))
        devices_y.sort()
        self.aggregator_y = _CustomSecureAggregator(
            device=devices_y[0], participants=devices_y, fxp_bits=fxp_bits
        )

        self.ver_lr_list = [
            FlLogisticRegressionVertical(
                devices=devices_list[i],
                aggregator=aggregators[i],
                heu=heus[i],
                fxp_bits=fxp_bits,
                audit_log_dir=audit_log_dir,
            )
            for i in range(len(devices_list))
        ]
        self._init_train_data(x=x, y=y, epochs=epochs, batch_size=batch_size)

        for epoch in range(epochs):
            if epoch % agg_epochs == 0:
                self._agg_weights(aggr_hooks)
                loss = self._compute_loss(x, y)
                logging.info(f'MixLr epoch {epoch}: loss = {loss}')
                if loss <= tol:
                    return
            self._fit_in_steps(
                x, batch_size=batch_size, epoch=epoch, learning_rate=learning_rate
            )
        self._agg_weights(aggr_hooks)
        loss = self._compute_loss(x, y)
        logging.info(f'MixLr epoch {epoch + 1}: loss = {loss}')

    def _fit_in_steps(
        self,
        x: MixDataFrame,
        batch_size: int,
        epoch: int,
        learning_rate: Optional[float] = 0.1,
    ):
        """Fit in steps.

        Args:
            x: the training tensors.
            batch_size: number of samples per gradient update.
            learning_rate: learning rate.
        """
        for ver_lr, x_part in zip(self.ver_lr_list, x.partitions):
            n_step = math.ceil(x_part.shape[0] / batch_size)
            ver_lr.fit_in_steps(n_step, learning_rate, epoch)

    def predict(self, x: MixDataFrame) -> List[PYUObject]:
        """Predict the score.

        Args:
            x: the samples to predict.

        Returns:
            a list of PYUObjects holding prediction results.
        """
        assert isinstance(
            x, MixDataFrame
        ), f'X should be a MixDataFrame but got {type(x)}.'
        assert (
            x.partition_way == PartitionWay.HORIZONTAL
        ), 'X should be horizontal partitioned.'
        assert self.ver_lr_list, 'This estimator has not been fit yet.'

        devices_list = [list(ver_lr.workers.keys()) for ver_lr in self.ver_lr_list]
        preds = []
        for part in x.partitions:
            try:
                idx = devices_list.index(list(part.partitions.keys()))
                preds.append(self.ver_lr_list[idx].predict(part))
            except ValueError:
                raise InvalidArgumentError(
                    'Failed to predict as the devices of'
                    'VDataFrame in X do not appear during fit.'
                )

        return preds
