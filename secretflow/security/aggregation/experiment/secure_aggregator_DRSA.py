import copy
import random
from typing import List, Tuple, Union

import galois
import numpy as np
import pandas as pd

import secretflow as sf
import secretflow.utils.ndarray_encoding as ndarray_encoding
from secretflow.device import PYU, PYUObject, proxy
from secretflow.security.aggregation import Aggregator
from secretflow.security.aggregation._utils import is_nesting_list
from secretflow.security.mp_paillier import (
    ThresholdPaillier,
    ThresholdPaillierPrivateKey,
    ThresholdPaillierPublicKey,
    combine_shares,
)


@proxy(PYUObject)
class _Masker:
    def __init__(
        self,
        party,
        fxp_bits: int,
        gamma: int,
        prime: int,
        pub_key: ThresholdPaillierPublicKey,
        pri_key: ThresholdPaillierPrivateKey,
    ):
        self._party = party
        self._gamma = gamma
        self._prime = prime
        self._fxp_bits = fxp_bits
        self._dimension = None
        self._field = galois.GF(prime)
        self._pub_key = pub_key
        self._pri_key = pri_key
        self._points = [None] * (self._gamma + 1)

        # sample local polynomial with degree 'degree'
        self._fi = galois.Poly.Random(degree=gamma, field=self._field)

    def mask(
        self,
        data: Union[
            List[Union[pd.DataFrame, pd.Series, np.ndarray]],
            Union[pd.DataFrame, pd.Series, np.ndarray],
        ],
        weight=None,
    ) -> Tuple[Union[List[np.ndarray], np.ndarray], np.dtype]:
        assert data is not None, "Data should not be None or empty."
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
            ), f"Only accept ndarray or dataframe/series but got {type(datum)}"
            if dtype is None:
                dtype = datum.dtype
            else:
                assert (
                    datum.dtype == dtype
                ), f"Data should have the same types but got {datum.dtype} {dtype}."
            is_float = np.issubdtype(datum.dtype, np.floating)
            if not is_float:
                assert np.issubdtype(
                    datum.dtype, np.integer
                ), f"Data type is neither integer nor float."
                if datum.dtype != np.int64:
                    datum = datum.astype(np.int64)
            # do multiplication before encoding to finite field
            masked_datum: np.ndarray = (
                ndarray_encoding.encode(datum * weight, self._fxp_bits)
                if is_float
                else datum * weight
            )

            # masking
            print("start masking...")
            # print('original datum:')
            # print(masked_datum)
            masked_datum_flatten = masked_datum.flatten()
            dimension = len(masked_datum_flatten)
            self._dimension = dimension
            for x_coordinate in range(1, dimension + 1):
                # type of 'r_i': <class 'galois.GF(p)'>
                # type of 'masked_datum': <class 'numpy.ndarray'>
                r_i = self._fi(x_coordinate)
                masked_datum_flatten[x_coordinate - 1] = (
                    self._field(int(masked_datum_flatten[x_coordinate - 1])) + r_i
                )
            masked_data.append(masked_datum_flatten)
            print("done.")
            # print('masking done. masked datum:')
            # print(masked_datum)

        if is_list:
            return masked_data, dtype
        else:
            return masked_data[0], dtype

    def sample(self):
        for i in range(self._dimension + 1, self._dimension + 1 + (self._gamma + 1)):
            self._points[i - (self._dimension + 1)] = int(self._fi(i))
        points = copy.deepcopy(self._points)
        points.pop(self._gamma)
        return points

    def encrypt(self):
        return self._pub_key.encrypt(self._points[self._gamma])

    def partial_decrypt(self, c):
        return self._pri_key.partial_decrypt(c)


class SecureAggregator_DRSA(Aggregator):
    """The secure aggregation implementation using locally generating masks based on randomly sampled polynomial, where
    DRSA means Dimension Reduction Secure Aggregation.

    Each participant :math: `p_i` locally chooses a random polynomial :math: `f_i` with specified degree :math: `d`,
    then generates the masks as:

    .. math::
        r_i = (f_i(1),f_i(2),\ldots,f_i(\ell))

    where :math: `\ell` is the dimension of original data.

    Then encrypt the data :math: `m_i` as

    .. math::
        \tilde{m}_i = m_i + r_i\quad(mod\: R)

    After aggregation, the sum will be

    .. math::
        \sum_i \tilde{m}_i = \sum_i m_i + \sum_i r_i\quad(mod\: R)

    To enable correct aggregation with privacy preservation at the same time, the aggregator
    needs to get the sum of all :math: `r_i`, but cannot get any single :math: `r_i`.
    To that end, each participant samples other :math: `d+1` points, sends :math: `d` points
    of them to the aggregator in the clear. Finally, all participants and the aggregator engage
    in a secure aggregation subprotocol to securely aggregate the remaining single point. With
    these information, aggregator can reconstruct the composite polynomial:

    .. math::
        f(\cdot) = f_1(\cdot) + f_2(\cdot) + \cdots

    and recover the sum of masks as :math: `\sum_i r_i=(f(1),f(2),\ldots,f(\ell))`, after that
    it can recover the correct aggregation result.import secretflow.utils.ndarray_encoding as ndarray_encoding


    Notes:
        This 'local mask generation based on polynomial' method indeed largely decrease interaction
        thus communication costs among participants, and all of above steps can be pre-executed offline
        or parallelly executed along with data generation. However, in the first place, it seems that
        the masks are not uniformly random, thus the security should be enhanced by combining with
        other methods, e.g., differential privacy.

    Examples:
        >>> # alice and bob are both pyu objects
        >>> participants = [alice, bob]
        >>> # to tolerate half of the participants drop out
        >>> threshold = int(len(participants) / 2)
        >>> # enable precomputation to improve efficiency
        >>> pre_computed = True
        >>> param_file = "params_2.ini" # if 'pre_computed' is True, set the parameter file
        >>> aggregator_DRSA = SecureAggregator_DRSA(alice, participants, threshold,
        >>>                  pre_computed = pre_computed, param_file = param_file)
        >>> a = alice(lambda: np.random.rand(2,5))()
        >>> b = bob(lambda: np.random.rand(2,5))()
        >>> sum = aggregator_DRSA.sum([a,b], axis=0)
        >>> avg = aggregator_DRSA.average([a,b], axis=0)
        >>> # get the result
        >>> sf.reveal(sum), sf.reveal(avg)
    """

    def __init__(
        self,
        device: PYU,
        participants: List[PYU],
        threshold: int,
        fxp_bits: int = 18,
        pre_computed=True,
        param_file="params_2.ini",
    ):
        assert len(set(participants)) == len(
            participants
        ), "Should not have duplicated devices"
        self._device = device
        self._participants = set(participants)
        self._fxp_bits = fxp_bits
        self._threshold = threshold

        # initialize parameters of MPPaillier scheme
        security_param = 1024
        num_clients = len(participants)
        if pre_computed:
            tp = ThresholdPaillier(
                security_param,
                num_clients,
                threshold,
                load=True,
                store=False,
                param_file=param_file,
            )
        else:
            tp = ThresholdPaillier(
                security_param,
                num_clients,
                threshold,
                load=False,
                store=True,
                param_file=param_file,
            )

        tp_pub_key = tp.pub_key
        self._pub_key = tp_pub_key
        tp_pri_keys = tp.priv_keys
        term_upper_bound = pow(2, 32)
        prime = galois.next_prime(num_clients * term_upper_bound)
        self._prime = prime
        self._field = galois.GF(prime)
        self._gamma = 2

        self._maskers = {
            pyu: _Masker(
                pyu.party,
                self._fxp_bits,
                gamma=self._gamma,
                prime=prime,
                device=pyu,
                pub_key=tp_pub_key,
                pri_key=tp_pri_keys[i],
            )
            for i, pyu in enumerate(participants)
        }

    def _check_data(self, data: List[PYUObject]):
        assert data, f"The data should not be None or empty."
        assert len(data) == len(
            self._maskers
        ), f"The length of data is not equal to number of devices: {len(data)} vs {len(self._maskers)}"
        devices_of_data = set(datum.device for datum in data)
        assert (
            devices_of_data == self._participants
        ), "Devices of the data must be corresponding with this aggrgator."

    def sum(self, data: List[PYUObject], axis=None):
        def _sum(*masked_data: List[np.ndarray], dtypes: List[np.dtype], fxp_bits):
            # check data types
            for dtype in dtypes[1:]:
                assert (
                    dtype == dtypes[0]
                ), f"Data should have the same data type but got {dtype} {dtypes[0]}."
            is_float = np.issubdtype(dtypes[0], np.floating)

            if is_nesting_list(masked_data):
                results = [np.sum(element, axis=axis) for element in zip(*masked_data)]
                return results
                # return (
                #     [ndarray_encoding.decode(result, fxp_bits) for result in results]
                #     if is_float
                #     else results
                # )
            else:
                result = np.sum(masked_data, axis=axis)
                return result
                # decoding and return
                # return ndarray_encoding.decode(result, fxp_bits) if is_float else result

        def _points_sum(*points_list, prime):
            gamma = len(points_list[0])
            result = [0] * gamma
            for i in range(len(points_list)):
                for j in range(gamma):
                    result[j] = (result[j] + points_list[i][j]) % prime
            return result

        def _aggregate(ciphertexts_list):
            c = ciphertexts_list[0]
            for i in range(1, len(ciphertexts_list)):
                c = c + ciphertexts_list[i]
            return c

        def _decrypt(decryption_shares: list, pub_key, prime):
            assert len(decryption_shares) >= pub_key.t
            beta = combine_shares(
                decryption_shares,
                pub_key.t,
                pub_key.delta,
                pub_key.combineSharesConstant,
                pub_key.nSPlusOne,
                pub_key.n,
                pub_key.ns,
            )
            beta = beta % prime
            return beta

        self._check_data(data)
        masked_data = [None] * len(data)
        dtypes = [None] * len(data)
        # save the shape
        shape = (sf.reveal(data))[0].shape

        # data encryption
        for i, datum in enumerate(data):
            masked_data[i], dtypes[i] = self._maskers[datum.device].mask(datum)
        masked_data = [d.to(self._device) for d in masked_data]
        dtypes = [dtype.to(self._device) for dtype in dtypes]

        _sum_results = self._device(_sum)(
            *masked_data, dtypes=dtypes, fxp_bits=self._fxp_bits
        )

        points_list = [None] * len(data)
        # sampling
        print("start points sampling...")
        for i, datum in enumerate(data):
            points_list[i] = self._maskers[datum.device].sample()
        print("done.")
        # send to the aggregator
        points_list = [d.to(self._device) for d in points_list]

        print("aggregator computing aggregation of points lists...")
        _points_sum_results = self._device(_points_sum)(*points_list, prime=self._prime)
        print("done.")

        ciphertexts_list = [None] * len(data)
        # encryption
        print("start encryption...")
        for i, datum in enumerate(data):
            ciphertexts_list[i] = self._maskers[datum.device].encrypt()
        print("done.")
        # send to the aggregator
        ciphertexts_list = [d.to(self._device) for d in ciphertexts_list]

        print("ciphertext aggregation...")
        _aggregate_result = self._device(_aggregate)(ciphertexts_list)
        print("done.")

        # partial decryption
        decryption_shares = [None] * len(data)
        print("start distributed decryption...")
        c = sf.reveal(_aggregate_result)
        for i, datum in enumerate(data):
            _c = datum.device(lambda: c)()
            decryption_share_i = self._maskers[datum.device].partial_decrypt(_c)
            decryption_shares[i] = sf.reveal(decryption_share_i)
        print("done.")

        # combine decryption shares
        print("combine decryption shares and recovering...")
        index = [i for i in range(len(data))]
        recon_index = random.sample(index, self._threshold)
        decryption_shares_collected = []
        for i in range(len(recon_index)):
            decryption_shares_collected.append(decryption_shares[recon_index[i]])

        beta = self._device(_decrypt)(
            decryption_shares_collected, self._pub_key, prime=self._prime
        )
        print("done.")

        print("start to recover correct aggregation result...")
        alpha = sf.reveal(_points_sum_results)
        gamma = len(alpha)
        alpha.append(sf.reveal(beta))
        tilde_m = sf.reveal(_sum_results)
        d = len(tilde_m)
        x = [i for i in range(d + 1, d + gamma + 2)]
        assert len(x) == len(alpha)
        x = self._field(x)
        y = self._field(alpha)
        # interpolation
        f = galois.lagrange_poly(x, y)
        m = copy.deepcopy(tilde_m)
        for i in range(1, d + 1):
            # recover aggregation of masks
            ri = f(i)
            # subtract masks
            m[i - 1] = (tilde_m[i - 1] - int(ri)) % self._prime

        print("done.")
        # decoding and return
        result = ndarray_encoding.decode(m, self._fxp_bits)

        return result.reshape(shape)

    def average(self, data: List[PYUObject], axis=None, weights=None):
        pass
