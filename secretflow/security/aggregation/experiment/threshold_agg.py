import random
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from phe import paillier
from phe.util import invert, mulmod, powmod, getprimeover, isqrt

import secretflow as sf
import secretflow.utils.ndarray_encoding as ndarray_encoding
from secretflow.device import PYU, DeviceObject, PYUObject, proxy, reveal
from secretflow.security.aggregation import Aggregator
from secretflow.security.aggregation._utils import is_nesting_list
from secretflow.security.diffie_hellman import DiffieHellman


@proxy(PYUObject)
class _Enc:
    def __init__(self, party, TA, fxp_bits: int = 18):
        self.party = party
        self.pub_key, self.pri_key, self.security_parameter = TA.genUserParameters(
            party
        )
        self.fxp_bits = fxp_bits

    def enc(
        self,
        data: Union[
            List[Union[pd.DataFrame, pd.Series, np.ndarray]],
            Union[pd.DataFrame, pd.Series, np.ndarray],
        ],
        exp: int = 0,
    ) -> Tuple[Union[List[np.ndarray], np.ndarray], np.dtype]:
        assert data is not None, 'Data shall not be None or empty.'
        is_list = isinstance(data, list)
        if not is_list:
            data = [data]
        enc_data = []
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
            raw_shape = datum.shape
            datum = datum.flatten()
            datum = np.array(datum * 2**self.fxp_bits // 1, dtype='int64').tolist()
            for d in datum:
                r = random.getrandbits(self.security_parameter // 3)
                c1 = powmod(self.pub_key.g, d, self.pub_key.nsquare)
                c2 = powmod(self.pri_key.h, r, self.pub_key.nsquare)
                c3 = powmod(self.pri_key.sk, int(exp), self.pub_key.nsquare)
                c = mulmod(c1, c2, self.pub_key.nsquare)
                c = mulmod(c, c3, self.pub_key.nsquare)
                enc_data.append(c)
            enc_data = np.array(enc_data).reshape(raw_shape)

            return enc_data, dtype


class TA:
    def __init__(
        self,
        device: PYU,
        participants: List[PYU],
        security_parameter: int = 1024,
        threshold: int = 1,
    ):
        self.security_parameter = security_parameter
        self.participants = set(participants)
        self.threshold = threshold
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=self.security_parameter
        )
        self.ids = {
            str(pyu.party): random.SystemRandom().randrange(self.private_key.p)
            for pyu in self.participants
        }
        self.p1 = getprimeover(self.security_parameter // 2)
        s = getprimeover(self.security_parameter)
        self.h = powmod(self.public_key.g, self.p1, self.public_key.nsquare)
        coff = [
            random.SystemRandom().randrange(self.private_key.p)
            for i in range(self.threshold - 1)
        ]
        self.sk = {}
        for party in self.ids.keys():
            tmp = [self.ids[str(party)] ** i for i in range(1, self.threshold)]
            fx = np.dot(tmp, coff) % self.private_key.p
            self.sk[str(party)] = user_sk(
                powmod(s, fx * self.private_key.q, self.public_key.nsquare), self.h
            )

    def genASParameters(self):
        return self.public_key, self.private_key, self.p1, self.ids

    def genUserParameters(self, party):
        return self.public_key, self.sk[str(party)], self.security_parameter


class user_sk(object):
    def __init__(self, sk, h):
        self.sk = sk
        self.h = h


class ThresholdAggregator(Aggregator):
    def __init__(self, device: PYU, TA, participants: List[PYU], fxp_bits: int = 18):
        assert len(set(participants)) == len(
            participants
        ), 'Should not have duplicated devices.'
        self.device = device
        self.participants = set(participants)
        self.fxp_bits = fxp_bits
        self.public_key, self.private_key, self.p1, self.ids = TA.genASParameters()

        self.encs = {
            pyu: _Enc(pyu.party, TA, fxp_bits=self.fxp_bits, device=pyu)
            for pyu in participants
        }
        pairs = [x - y for x in self.ids.values() for y in self.ids.values()]
        self.exp_base = 1
        for i in pairs:
            if i != 0:
                self.exp_base *= i

    def check_data(self, data: List[PYUObject]):
        assert data, f'The data should not be None or empty.'

        devices_of_data = set(datum.device for datum in data)

    def sum(self, data: List[PYUObject], axis=None):
        def _sum(*enc_data: List[np.ndarray], dtypes: List[np.dtype], fxp_bits):
            for dtype in dtypes[1:]:
                assert (
                    dtype == dtypes[0]
                ), f'Data should have same dtypes but got {dtype} {dtypes[0]}.'

            row = len(enc_data[0])
            column = len(enc_data[0][0])
            results = [[1 for i in range(column)] for j in range(row)]

            for i in range(row):
                for j in range(column):
                    for z in range(len(enc_data)):
                        results[i][j] = mulmod(
                            results[i][j], enc_data[z][i][j], self.public_key.nsquare
                        )
                    results[i][j] = (
                        self.private_key.raw_decrypt(results[i][j]) % self.p1
                    )

            return (np.array(results) / 2**self.fxp_bits).tolist()

        self.check_data(data)
        enc_data = [None] * len(data)
        dtypes = [None] * len(data)

        active_ids = []

        for i, datum in enumerate(data):
            active_ids.append(self.ids[str(datum.device)])

        for i, datum in enumerate(data):
            exp = self.exp_base
            idd = self.ids[str(datum.device)]
            for j in active_ids:
                if j != idd:
                    exp = exp * j // (j - idd)
            enc_data[i], dtypes[i] = self.encs[datum.device].enc(data=datum, exp=exp)
        enc_data = [d.to(self.device) for d in enc_data]
        dtypes = [dtype.to(self.device) for dtype in dtypes]
        return self.device(_sum)(*enc_data, dtypes=dtypes, fxp_bits=self.fxp_bits)

    def average(self, data: List[PYUObject], axis=None):
        def _average(*enc_data: List[np.ndarray], dtypes: List[np.dtype], fxp_bits):
            for dtype in dtypes[1:]:
                assert (
                    dtype == dtypes[0]
                ), f'Data should have same dtypes but got {dtype} {dtypes[0]}.'

            row = len(enc_data[0])
            column = len(enc_data[0][0])
            results = [[1 for i in range(column)] for j in range(row)]

            for i in range(row):
                for j in range(column):
                    for z in range(len(enc_data)):
                        results[i][j] = mulmod(
                            results[i][j], enc_data[z][i][j], self.public_key.nsquare
                        )
                    results[i][j] = (
                        self.private_key.raw_decrypt(results[i][j]) % self.p1
                    )

            return (np.array(results) / 2**self.fxp_bits / len(enc_data)).tolist()

        self.check_data(data)
        enc_data = [None] * len(data)
        dtypes = [None] * len(data)

        active_ids = []

        for i, datum in enumerate(data):
            active_ids.append(self.ids[str(datum.device)])

        for i, datum in enumerate(data):
            exp = self.exp_base
            idd = self.ids[str(datum.device)]
            for j in active_ids:
                if j != idd:
                    exp = exp * j // (j - idd)
            enc_data[i], dtypes[i] = self.encs[datum.device].enc(data=datum, exp=exp)
        enc_data = [d.to(self.device) for d in enc_data]
        dtypes = [dtype.to(self.device) for dtype in dtypes]
        return self.device(_average)(*enc_data, dtypes=dtypes, fxp_bits=self.fxp_bits)
