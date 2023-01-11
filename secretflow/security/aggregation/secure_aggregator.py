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

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

import secretflow.utils.ndarray_encoding as ndarray_encoding
from secretflow.device import PYU, DeviceObject, PYUObject, proxy, reveal
from secretflow.security.aggregation import Aggregator
from secretflow.security.aggregation._utils import is_nesting_list
from secretflow.security.diffie_hellman import DiffieHellman


@proxy(PYUObject)
class _Masker:
    def __init__(self, party, fxp_bits: int):
        self._party = party
        self._dh = DiffieHellman()
        self._pub_key, self._pri_key = self._dh.generate_key_pair()
        self._fxp_bits = fxp_bits

    def pub_key(self) -> int:
        return self._pub_key

    def gen_rng(self, pub_keys: Dict[str, int]) -> None:
        assert pub_keys, f'Public keys is None or empty.'
        self._rngs = {
            party: np.random.default_rng(
                int(self._dh.generate_secret(self._pri_key, peer_pub_key), base=16)
            )
            for party, peer_pub_key in pub_keys.items()
            if party != self._party
        }

    def mask(
        self,
        data: Union[
            List[Union[pd.DataFrame, pd.Series, np.ndarray]],
            Union[pd.DataFrame, pd.Series, np.ndarray],
        ],
        weight=None,
    ) -> Tuple[Union[List[np.ndarray], np.ndarray], np.dtype]:
        assert data is not None, 'Data shall not be None or empty.'
        is_list = isinstance(data, list)
        if not is_list:
            data = [data]
        if weight is None:
            weight = 1
        masked_data = []
        dtype = None
        for datum in data:
            if isinstance(datum, (pd.DataFrame, pd.Series)):
                datum = datum.values
            assert isinstance(
                datum, np.ndarray
            ), f'Accept ndarray or dataframe/series only but got {type(datum)}'
            if dtype is None:
                dtype = datum.dtype
            else:
                assert (
                    datum.dtype == dtype
                ), f'Data should have same dtypes but got {datum.dtype} {dtype}.'
            is_float = np.issubdtype(datum.dtype, np.floating)
            if not is_float:
                assert np.issubdtype(
                    datum.dtype, np.integer
                ), f'Data type are neither integer nor float.'
                if datum.dtype != np.int64:
                    datum = datum.astype(np.int64)
            # Do mulitple before encoding to finite field.
            masked_datum: np.ndarray = (
                ndarray_encoding.encode(datum * weight, self._fxp_bits)
                if is_float
                else datum * weight
            )
            for party, rng in self._rngs.items():
                if party == self._party:
                    continue
                mask = rng.integers(
                    low=np.iinfo(np.int64).min,
                    high=np.iinfo(np.int64).max,
                    size=masked_datum.shape,
                ).astype(masked_datum.dtype)
                if party > self._party:
                    masked_datum += mask
                else:
                    masked_datum -= mask

            masked_data.append(masked_datum)
        if is_list:
            return masked_data, dtype
        else:
            return masked_data[0], dtype


class SecureAggregator(Aggregator):
    """The secure aggregation implementation of `Masking with One-Time Pads`.

    `Masking with One-Time Pads` negotiates a secret for every two participants,
    then uses the secret to hide its input `x`, and each participant outputs.

    .. math::
        y_u = x_u + \sum _{u < v}S_{u,v} - \sum_{u>v}S_{u,v}\quad(mod\:R)

    the secrets are cancelled out after aggregation and then we can get the correct result.

    .. math::
        \sum y = \sum x


    For example, the participants Alice, Bob, and Carol each own :math:`x_1, x_2, x_3`,
    negotiate the secret :math:`s_{a,b}, s_{a,c}, s_{b,c}`, and then output:

    .. math::
        y_1 = x_1 + s_{a,b} + s_{a,c}

    .. math::
        y_2 = x_2 - s_{a,b} + s_{b,c}

    .. math::
        y_3 = x_3 - s_{a,c} - s_{b,c}

    then it is easy to get

    .. math::
        y_1 + y_2 + y_3 = x_1 + s_{a,b} + s_{a,c} + x_2 - s_{a,b} + s_{b,c} + x_3 - s_{a,c} - s_{b,c}
        = x_1 + x_2 + x_3

    Notes:
        `Masking with One-Time Pads` is based on semi-honest assumptions
        and does not support client dropping. For more information, please refer to
        `Practical Secure Aggregationfor Privacy-Preserving Machine Learning <https://eprint.iacr.org/2017/281.pdf>`_

    Warnings:
        The SecureAggregator uses :py:meth:`numpy.random.PCG64`. There are many
        discussions of whether PCG is a CSPRNG
        (e.g. https://crypto.stackexchange.com/questions/77101/is-the-pcg-prng-a-csprng-or-why-not),
        we perfer a conservative strategy unless a further security analysis came
        up. Therefore we recommend users to use a standardized CSPRNG in industrial
        scenarios.

    Examples:
        >>> # Alice and bob are both pyu instances.
        >>> aggregator = SecureAggregator(alice, [alice, bob])
        >>> a = alice(lambda : np.random.rand(2, 5))()
        >>> b = bob(lambda : np.random.rand(2, 5))()
        >>> sum_a_b = aggregator.sum([a, b], axis=0)
        >>> # Get the result.
        >>> sf.reveal(sum_a_b)
        array([[0.5954927 , 0.9381409 , 0.99397117, 1.551537  , 0.3269863 ],
        [1.288345  , 1.1820003 , 1.1769378 , 0.7396539 , 1.215364  ]],
        dtype=float32)
        >>> average_a_b = aggregator.average([a, b], axis=0)
        >>> sf.reveal(average_a_b)
        array([[0.29774636, 0.46907043, 0.49698558, 0.7757685 , 0.16349316],
        [0.6441725 , 0.5910001 , 0.5884689 , 0.3698269 , 0.607682  ]],
        dtype=float32)
    """

    def __init__(self, device: PYU, participants: List[PYU], fxp_bits: int = 18):
        assert len(set(participants)) == len(
            participants
        ), 'Should not have duplicated devices.'
        self._device = device
        self._participants = set(participants)
        self._fxp_bits = fxp_bits
        self._maskers = {
            pyu: _Masker(pyu.party, self._fxp_bits, device=pyu) for pyu in participants
        }
        pub_keys = reveal(
            {pyu.party: masker.pub_key() for pyu, masker in self._maskers.items()}
        )
        for masker in self._maskers.values():
            masker.gen_rng(pub_keys)

    def _check_data(self, data: List[PYUObject]):
        assert data, f'The data should not be None or empty.'
        assert len(data) == len(
            self._maskers
        ), f'Length of the data not equals devices: {len(data)} vs {len(self._maskers)}'
        devices_of_data = set(datum.device for datum in data)
        assert (
            devices_of_data == self._participants
        ), 'Devices of the data must be corresponding with this aggregator.'

    @classmethod
    def _is_list(cls, masked_data: Union[List, Any]) -> bool:
        is_list = isinstance(masked_data[0], list)
        for masked_datum in masked_data[1:]:
            assert (
                isinstance(masked_datum, list) == is_list
            ), f'Some data are list where some others are not.'
            assert not is_list or len(masked_datum) == len(
                masked_datum[0]
            ), f'Lengths of datum in data are different.'
        return is_list

    def sum(self, data: List[PYUObject], axis=None):
        def _sum(*masked_data: List[np.ndarray], dtypes: List[np.dtype], fxp_bits):
            for dtype in dtypes[1:]:
                assert (
                    dtype == dtypes[0]
                ), f'Data should have same dtypes but got {dtype} {dtypes[0]}.'
            is_float = np.issubdtype(dtypes[0], np.floating)

            if is_nesting_list(masked_data):
                results = [np.sum(element, axis=axis) for element in zip(*masked_data)]
                return (
                    [
                        ndarray_encoding.decode(result, fxp_bits)
                        for result in results
                    ]
                    if is_float
                    else results
                )
            else:
                result = np.sum(masked_data, axis=axis)
                return (
                    ndarray_encoding.decode(result, fxp_bits)
                    if is_float
                    else result
                )

        self._check_data(data)
        masked_data = [None] * len(data)
        dtypes = [None] * len(data)
        for i, datum in enumerate(data):
            masked_data[i], dtypes[i] = self._maskers[datum.device].mask(datum)
        masked_data = [d.to(self._device) for d in masked_data]
        dtypes = [dtype.to(self._device) for dtype in dtypes]
        return self._device(_sum)(*masked_data, dtypes=dtypes, fxp_bits=self._fxp_bits)

    def average(self, data: List[PYUObject], axis=None, weights=None):
        def _average(*masked_data: List[np.ndarray], dtypes: List[np.dtype], weights, fxp_bits):
            for dtype in dtypes[1:]:
                assert (
                    dtype == dtypes[0]
                ), f'Data should have same dtypes but got {dtype} {dtypes[0]}.'
            is_float = np.issubdtype(dtypes[0], np.floating)
            sum_weights = (
                np.sum(weights, axis=axis) if weights else len(masked_data)
            )
            if is_nesting_list(masked_data):
                sum_data = [np.sum(element, axis=axis) for element in zip(*masked_data)]
                if is_float:
                    sum_data = [
                        ndarray_encoding.decode(sum_datum, fxp_bits)
                        for sum_datum in sum_data
                    ]
                return [element / sum_weights for element in sum_data]
            else:
                if is_float:
                    return (
                        ndarray_encoding.decode(
                            np.sum(masked_data, axis=axis), fxp_bits
                        )
                        / sum_weights
                    )
                return np.sum(masked_data, axis=axis) / sum_weights

        self._check_data(data)
        masked_data = [None] * len(data)
        dtypes = [None] * len(data)
        _weights = []
        if weights is not None and isinstance(weights, (list, tuple, np.ndarray)):
            assert len(weights) == len(
                data
            ), f'Length of the weights not equals data: {len(weights)} vs {len(data)}.'
            for i, w in enumerate(weights):
                if isinstance(w, DeviceObject):
                    assert (
                        w.device == data[i].device
                    ), 'Device of weight is not same with the corresponding data.'
                    _weights.append(w.to(self._device))
                else:
                    _weights.append(w)
            for i, (datum, weight) in enumerate(zip(data, weights)):
                masked_data[i], dtypes[i] = self._maskers[datum.device].mask(
                    datum, weight
                )
        else:
            for i, datum in enumerate(data):
                masked_data[i], dtypes[i] = self._maskers[datum.device].mask(
                    datum, weights
                )
        masked_data = [d.to(self._device) for d in masked_data]
        dtypes = [dtype.to(self._device) for dtype in dtypes]
        return self._device(_average)(*masked_data, dtypes=dtypes, weights=_weights, fxp_bits=self._fxp_bits)
