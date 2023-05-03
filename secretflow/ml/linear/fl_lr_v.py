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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import spu
from heu import phe
from numpy.random import RandomState

from secretflow.data.ndarray import FedNdarray
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device import proxy
from secretflow.device.device.heu import HEU, HEUMoveConfig
from secretflow.device.device.pyu import PYU, PYUObject
from secretflow.device.device.type_traits import spu_fxp_precision
from secretflow.device.driver import reveal
from secretflow.security.aggregation.aggregator import Aggregator


class FlLrVWorker(object):
    def _data_generator(
        self,
        x: np.ndarray,
        epochs: int,
        batch_size: int,
        y: np.ndarray = None,
        rdm: RandomState = None,
    ):
        sample_num = x.shape[0]
        splits = list(range(0, sample_num, batch_size)) + [sample_num]
        for _ in range(epochs):
            shuffle_idx = (
                rdm.permutation(sample_num)
                if rdm is not None
                else list(range(sample_num))
            )
            for s, e in zip(splits[:-1], splits[1:]):
                batch_idx = shuffle_idx[s:e]
                yield (x[batch_idx], y[batch_idx] if y is not None else None)

    def init_train_data(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        batch_size: int,
        epochs: int,
        y: Union[pd.DataFrame, np.ndarray] = None,
        shuffle_seed: int = None,
    ):
        """Initialize the training data.

        Args:
            x: the training vector.
            batch_size: number of samples per gradient update.
            epochs: number of epochs to train the model.
            y: optional; the target vector relative to x.
            shuffle_seed: optional; the data will be shuffled if not none.
        """
        assert isinstance(
            x, (pd.DataFrame, np.ndarray)
        ), f'X shall be a DataFrame or ndarray but got {type(x)}.'
        self.has_y = False
        if y is not None:
            assert isinstance(
                y, (pd.DataFrame, np.ndarray)
            ), f'Y shall be a DataFrame or ndarray but got {type(y)}.'
            y = y.values if isinstance(y, pd.DataFrame) else y
            y = y.reshape(-1, 1)
            self.has_y = True
        self.intercept = np.random.rand(1, 1)
        if isinstance(x, pd.DataFrame):
            x = x.values
        self.feat_size = x.shape[1]
        self.weight = np.zeros((x.shape[1], 1))
        rdm = RandomState(seed=shuffle_seed) if shuffle_seed is not None else None
        self.train_set = self._data_generator(
            x=x, epochs=epochs, batch_size=batch_size, y=y, rdm=rdm
        )

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get next batch of X and y.

        Returns:
            A tuple of (x batch, y batch), while y batch is None if no y.
        """
        xb, yb = next(self.train_set)
        self.batch_size = xb.shape[0]
        return xb, yb

    def compute_mul(self, x_batch: np.ndarray) -> np.ndarray:
        """Compute Xi*Wi."""
        return x_batch.dot(self.weight)

    def predict(self, mul: np.ndarray) -> np.ndarray:
        """Do prediction.

        Args:
            mul: the sum of Xi*Wi (i>0).

        Returns:
            The prediction results.
        """
        pred = mul + self.intercept
        ret = 1.0 / (1 + np.exp(-pred))
        return ret

    def compute_loss(self, y: np.ndarray, h: np.ndarray, avg_flag: bool) -> np.ndarray:
        h = np.clip(h, 1e-15, 1 - 1e-15)
        ret = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h), keepdims=True)
        if avg_flag:
            ret = (1.0 / y.shape[0]) * ret
        return ret

    def compute_residual(self, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.transpose(h - y)

    def encode(self, data: np.ndarray, frac_bits: int) -> np.ndarray:
        return data * (2**frac_bits)

    def decode(self, data: np.ndarray, frac_bits: int) -> np.ndarray:
        return data / (2**frac_bits)

    def generate_rand_mask(self, decode_frac: int) -> np.ndarray:
        # TODO: the random range is not secure.
        mask_int = np.random.randint(
            np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(1, self.feat_size)
        )
        # save for decoding
        self.rand_mask = self.decode(mask_int, decode_frac)
        # this is used to pass to heu later
        return mask_int

    def get_weight(self) -> np.ndarray:
        return (
            np.concatenate((self.intercept, self.weight)) if self.has_y else self.weight
        )

    def set_weight(self, w: np.ndarray):
        if self.has_y:
            self.intercept = w[0, :].reshape(1, 1)
            self.weight = w[1:, :]
        else:
            self.weight = w

    def update_weight(self, masked_gradient: np.ndarray, learning_rate: float):
        gradient = np.transpose(masked_gradient - self.rand_mask)
        self.weight = self.weight - learning_rate * gradient / self.batch_size

    def update_weight_agg(
        self, x_batch: np.ndarray, residual: np.ndarray, learning_rate: float
    ):
        gradient = np.transpose(np.dot(residual, x_batch))
        self.weight = self.weight - learning_rate * gradient / self.batch_size
        self.intercept = (
            self.intercept - learning_rate * np.sum(residual) / self.batch_size
        )


@proxy(PYUObject)
class PYUFlLrVWorker(FlLrVWorker):
    pass


def _gen_auth_file_path(
    audit_log_dir: str, device: PYU, epoch: int, step: int, hint: str
):
    if audit_log_dir is None:
        return None
    return f'{audit_log_dir[device]}/epoch_{epoch}_step_{step}_{hint}'


# TODO(zhouaihui): support linear regression.
class FlLogisticRegressionVertical:
    """Vertical logistic regression.

    Implement the basic SGD based logistic regression among multiple vertical
    participants.

    To explain this algorithm, suppose alice has features and label, while bob
    and charlie have features only.
    The main steps of SGD are:

    1. Alice does prediction using secure aggregation.
    2. Alice sends the residual to bob/charlie in HE(Homomorphic Encryption)
       ciphertext.
    3. Bob and charlie compute gradients in HE ciphertext and send masked
       gradients to alice.
    4. Alice decrypts the masked gradients and send them back to bob/charlie.
    5. Bob and charlie unmask gradients and update their weights independently.
    6. Alice updates its weights also.

    """

    def __init__(
        self,
        devices: List[PYU],
        aggregator: Aggregator,
        heu: HEU,
        fxp_bits: Optional[int] = spu_fxp_precision(spu.spu_pb2.FM64),
        audit_log_dir: Dict[PYU, str] = None,
    ):
        """Init VanillaVerLogisticRegression.

        Args:
            devices: a list of PYU devices taking part in the computation.
            aggregator: the aggregator instance.
            heu : the heu device instance.
            fxp_bits: the fraction bit length for encoding before send to
                heu device. Defaults to spu_fxp_precision(spu.spu_pb2.FM64).
            audit_log_dir: a dict specifying the audit log directory for each
                device. No audit log if is None. Default to None.
                Please leave it None unless you are very sure what the audit
                does and accept the risk.
        """
        assert isinstance(devices, list), 'device_list shall be a list!'
        assert len(devices) > 1, 'At least 2 devices are expected!'
        assert heu is not None, 'HEU must be provided.'
        assert set(list(heu.evaluator_names()) + [heu.sk_keeper_name()]) == set(
            [device.party for device in devices]
        ), 'The participants in HEU are inconsistent with device list.'

        self.workers = {device: PYUFlLrVWorker(device=device) for device in devices}
        self.heu = heu
        self.fxp_bits = fxp_bits
        self.aggregator = aggregator
        self.audit_log_dir = audit_log_dir
        if audit_log_dir:
            for device in devices:
                assert (
                    device in audit_log_dir
                ), f'The device {device} is not in audit_log_dir.'
            for party, evaluator in self.heu.evaluators.items():
                evaluator.dump_pk.remote(f'{audit_log_dir[party]}/public_key')
            self.heu.sk_keeper.dump_pk.remote(
                f'{audit_log_dir[self.heu.sk_keeper_name()]}/public_key'
            )

    def init_train_data(
        self,
        x: FedNdarray,
        y: FedNdarray,
        epochs: int,
        batch_size: int,
        shuffle_seed: Optional[int] = None,
    ):
        for device, worker in self.workers.items():
            worker.init_train_data(
                x=x.partitions[device],
                epochs=epochs,
                batch_size=batch_size,
                y=y.partitions.get(device, None),
                shuffle_seed=shuffle_seed,
            )

    def _next_batch(self) -> Tuple[List[PYUObject], PYUObject]:
        x_batchs = []
        y = None
        for device, worker in self.workers.items():
            x_batch, y_batch = worker.next_batch()
            x_batchs.append(x_batch)
            if device == self.y_device:
                y = y_batch
        return x_batchs, y

    def predict(self, x: Union[VDataFrame, FedNdarray, List[PYUObject]]) -> PYUObject:
        """Predict the score.

        Args:
            x: the samples to predict.

        Returns:
            PYUObject: a PYUObject holds prediction results.
        """
        # Compute Wi*Xi locally.
        assert isinstance(x, (VDataFrame, FedNdarray, List)), (
            'X should be a VDataFrame or FedNdarray or list of pyu objects but'
            f'got {type(x)}.'
        )
        if isinstance(x, VDataFrame):
            x_batchs = list(x.values.partitions.values())
        elif isinstance(x, FedNdarray):
            x_batchs = list(x.partitions.values())
        else:
            x_batchs = x
        assert set([x_batch.device for x_batch in x_batchs]) == set(
            self.workers.keys()
        ), 'Devices of x are different with this estimator.'

        muls = [
            self.workers[x_batch.device].compute_mul(x_batch) for x_batch in x_batchs
        ]
        mul = self.aggregator.sum(muls, axis=0).to(self.y_device)
        return self.workers[self.y_device].predict(mul)

    def compute_loss(
        self, x: FedNdarray, y: FedNdarray, avg_flag: Optional[bool] = True
    ) -> PYUObject:
        """Compute the loss.

        Args:
            x: the samples.
            y: the label.
            avg_flag: whether dividing the sample number. Defaults to True.

        Returns:
            PYUObject: a PYUObject holds loss value.
        """
        # compute at alice, metrics: loss, auc
        if not hasattr(self, 'y_device'):
            self.y_device = list(y.partitions.keys())[0]
        h = self.predict(x)
        return self.workers[self.y_device].compute_loss(
            y.partitions[self.y_device].data, h, avg_flag
        )

    def get_weight(self) -> Dict[PYU, PYUObject]:
        """Get weight from this estimator.

        Returns:
            A dict of pyu and its weight. Note that the intecept(w0)
            is the first column of the label deivce weight.
        """
        return {device: worker.get_weight() for device, worker in self.workers.items()}

    def set_weight(self, weight: Dict[PYU, Union[PYUObject, np.ndarray]]):
        """Set weight to this estimator.

        Args:
            weight: a dict of pyu and its weight.
        """
        assert isinstance(
            weight, dict
        ), f'Weight should be a dict but got {type(weight)}.'
        assert set(weight.keys()) == set(
            self.workers.keys()
        ), 'Devices of weight are different with this estimator.'
        for device, w in weight.items():
            if isinstance(w, PYUObject):
                w = w.to(device)
            self.workers[device].set_weight(w)

    def fit(
        self,
        x: Union[VDataFrame, FedNdarray],
        y: Union[VDataFrame, FedNdarray],
        batch_size: int,
        epochs: int,
        tol: Optional[float] = 1e-4,
        learning_rate: Optional[float] = 0.1,
    ):
        """Fit the model.

        Args:
            x: training vector.
            y: target vector relative to x.
            batch_size: number of samples per gradient update.
            epochs: number of epochs to train the model.
            tol: optional, tolerance for stopping criteria. Defaults to 1e-4.
            learning_rate: optional, learning rate. Defaults to 0.1.
        """
        assert isinstance(
            x, (VDataFrame, FedNdarray)
        ), f'X should be a VDataFrame or FedNdarray but got a {type(x)}.'
        assert len(x.partitions) > 1, f'Expect at least two partitipants.'
        assert set(x.partitions.keys()) == set(
            self.workers.keys()
        ), "X has different devices with this estimator."
        assert isinstance(
            y, (VDataFrame, FedNdarray)
        ), f'Y should be a VDataFrame or FedNdarray but got a {type(y)}.'
        assert len(y.partitions) == 1, f'One and only one participant should hold y.'
        y_device = list(y.partitions.keys())[0]
        assert y_device in self.workers, f'Device of y is not in this estimator.'
        assert y_device.party == self.heu.sk_keeper_name(), (
            'Y party shoule be same with heu sk keeper, '
            f'expect {self.heu.sk_keeper_name()} but got {y_device.party}.'
        )
        assert epochs > 0, 'Epochs should be greater then zero.'
        assert batch_size > 0, 'Batch size should be greater than zero.'

        self.y_device = y_device
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values

        x_shapes = list(x.partition_shape().values())
        assert (
            len(x_shapes[0]) == 2
        ), f'X should has two dimensions but got {len(x_shapes[0])}.'
        for shape in x_shapes[1:]:
            assert (
                shape[0] == x_shapes[0][0]
            ), f'Sample numbers of x partitionos are different.'
            # For security.
            assert (
                shape[1] < batch_size
            ), f'Feature number shall be smaller than batch size.'
        # Security Check.
        feature_num = sum(
            [
                shape[1]
                for device, shape in x.partition_shape().items()
                if device != self.y_device
            ]
        )
        if batch_size > feature_num:
            assert epochs < feature_num * batch_size / (batch_size - feature_num), (
                'Epochs shall be smaller than '
                'feature_num * batch_size / (batch_size - feature_num) '
                'when batch size is bigger than feature number for security. '
                'Note that feature number does not include features in y device.'
            )
        assert (
            list(y.partition_shape().values())[0][0] == x_shapes[0][0]
        ), f'Sample numbers of x and y are different.'
        sample_number = x_shapes[0][0]

        self.init_train_data(x=x, y=y, epochs=epochs, batch_size=batch_size)
        n_step = math.ceil(sample_number / batch_size)
        for epoch in range(epochs):
            loss = reveal(self.compute_loss(x, y))
            logging.info(f'Epoch {epoch}: loss = {loss}')
            if loss <= tol:
                logging.info(f'Stop training as loss is no greater than {tol} already.')
                return
            self.fit_in_steps(n_step, learning_rate, epoch)
        loss = reveal(self.compute_loss(x, y))
        logging.info(f'Epoch {epoch + 1}: loss = {loss}')

    def fit_in_steps(self, n_step: int, learning_rate: float, epoch: int):
        """Fit in steps.

        Args:
            n_step: the number of steps.
            learning_rate: learning rate.
            epoch: the current epoch.
        """
        for step in range(n_step):
            # step 1: y device compute residual, send to current device in
            # HE ciphertext.
            x_batchs, y_batch = self._next_batch()
            h = self.predict(x_batchs)
            r = self.workers[self.y_device].compute_residual(y_batch, h)
            self.workers[self.y_device].update_weight_agg(
                x_batchs[list(self.workers.keys()).index(self.y_device)],
                r,
                learning_rate,
            )
            for i, (device, worker) in enumerate(self.workers.items()):
                if device == self.y_device:
                    continue
                # step 2: current device compute the gradients locally,
                # and add mask and send to y device.
                x_heu = worker.encode(x_batchs[i], self.fxp_bits).to(
                    self.heu,
                    HEUMoveConfig(
                        heu_dest_party=device.party,
                        heu_encoder=phe.FloatEncoder(self.heu.schema, 1),
                        heu_audit_log=_gen_auth_file_path(
                            self.audit_log_dir, device, epoch, step, 'masked_g'
                        ),
                    ),
                )
                r_heu = r.to(
                    self.heu,
                    HEUMoveConfig(
                        heu_dest_party=device.party,
                        heu_audit_log=_gen_auth_file_path(
                            self.audit_log_dir, device, epoch, step, 'residual'
                        ),
                    ),
                )
                m_heu = worker.generate_rand_mask(self.fxp_bits).to(
                    self.heu,
                    HEUMoveConfig(
                        heu_dest_party=device.party,
                        heu_audit_log=_gen_auth_file_path(
                            self.audit_log_dir, device, epoch, step, 'rand_mask'
                        ),
                    ),
                )
                maskg_heu = r_heu @ x_heu + m_heu
                # step 3: y device decrypts the masked gradients and send
                # back to current device.
                maskg = (
                    self.workers[self.y_device]
                    .decode(maskg_heu.to(self.y_device), self.fxp_bits)
                    .to(device)
                )
                # step 4: update Wi.
                worker.update_weight(maskg, learning_rate)
