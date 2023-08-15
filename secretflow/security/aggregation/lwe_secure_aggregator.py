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
import pickle
import random
import struct
from typing import Any, Dict, List, Tuple, Union
import math
import numpy as np
import pandas as pd
import binascii

import secretflow.utils.ndarray_encoding as ndarray_encoding
from secretflow.device import PYU, DeviceObject, PYUObject, proxy, reveal
from secretflow.security.aggregation import Aggregator
from secretflow.security.aggregation._utils import is_nesting_list
from secretflow.security.aggregation._utils import is_prime
from secretflow.security.muti_secret_share import MutiSecretSharer
from secretflow.security.discrete_gaussian import sample_discrete_gaussian
from secretflow.security.diffie_hellman import DiffieHellman
from secretflow.utils.calculate_group import calculate_group_size
from Cryptodome.Cipher import AES

@proxy(PYUObject)
class _Masker:
    def __init__(self,self_device: PYU, dimension_red_rate : float, noise_scale:float,fxp_bits: int, L2_bound:float, prime:int, seed:int):
        self._device = self_device
        self._dh = DiffieHellman()
        self._pub_key, self._pri_key = self._dh.generate_key_pair()
        self._dimension_red_rate = dimension_red_rate
        self._noise_scale = noise_scale
        self._fxp_bits = fxp_bits
        self._L2_bound = L2_bound
        self._prime = prime
        self._seed = seed
        self.communication_graph = None
        self.t = None
        self.shard = None
        self.matrix_A = None
        self.secret_vector = None

    def pub_key(self) -> int:
        return self._pub_key
    def receive_key(self, communication_graph: Union[Dict[PYU, Dict], Dict[PYU, List[Dict]]], shard: bool,t:int) -> None:
        """Receive the communication topology diagram sent by the server.

        Args:
            communication_graph: Masker's communication node.
            shard: Whether to perform group aggregation.
            t : Thresholds for Secret Sharing Schemes

        """

        assert communication_graph is not None, 'The communication topology diagram shall not be None or empty.'
        self.communication_graph = communication_graph
        assert t is not None, 'A secret sharing scheme threshold needs to be specified.'
        self.t = t
        self.shard = shard

    def lwe_encrypt(self, data: np.ndarray, matrix_A: np.ndarray, secret_vector: np.ndarray, noise_scale:float, prime: int) -> np.ndarray:
        assert data is not None, f'matrix_A shall not be None or empty.'
        assert matrix_A is not None, f'matrix_A shall not be None or empty.'
        assert secret_vector is not None, f'secret_vector shall not be None or empty.'
        noise_vector = sample_discrete_gaussian(
            noise_scale ** 2,
            self._prime,
            shape=matrix_A.shape[0]
        )
        masked_gradient = (data + matrix_A.dot(secret_vector) + noise_vector) % prime
        return masked_gradient

    def mask(
            self,
            data: Union[
                List[Union[pd.DataFrame, pd.Series, np.ndarray]],
                Union[pd.DataFrame, pd.Series, np.ndarray],
            ],
            weight=None,
    ) -> Tuple[Dict[PYU, Union[List[np.ndarray], np.ndarray]], np.dtype]:
        """Use LWE to generate a random mask to complete the encryption operation on the gradient.

        Args:
            data: User's raw gradient vector.
            weight: The weight value of the gradient.

        Returns:
            Dict: encryption gradient.
            dtype: The data type of the original gradient
        """
        assert data is not None, 'Data shall not be None or empty.'
        is_list = isinstance(data, list)
        if not is_list:
            data = [data]
        else:
            for datum in data[1:]:
                assert (
                        len(datum) == len(data[0])
                ), f'The data dimensions of each item in the data list should be the same。'
        rng = np.random.default_rng(self._seed)
        self.matrix_A = rng.integers(
            low=0,
            high=self._prime,
            size=(len(data[0]), int(self._dimension_red_rate * len(data[0]))),
            dtype=np.int64
        )
        self.secret_vector = np.random.randint(
            low=0,
            high=self._prime,
            size=int(self._dimension_red_rate * len(data[0])),
            dtype=np.int64
        )
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

            # l2norm clip gradient
            datum = datum * weight
            L2_norm = np.linalg.norm(datum, 2)
            if L2_norm > self._L2_bound:
                datum = datum * (self._L2_bound / L2_norm)

            # Check whether the type of each item in the data list is consistent
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
            encoded_datum = ndarray_encoding.LWE_encode(datum, is_float, self._fxp_bits, self._prime)
            # encryption gradient
            masked_datum = self.lwe_encrypt(encoded_datum, self.matrix_A, self.secret_vector, self._noise_scale, self._prime)
            masked_data.append(masked_datum)

        if is_list:
            return [self._device, masked_data], dtype
        else:
            return [self._device, masked_data[0]], dtype

    def shard_data(self, nums: int) -> None:
        """
        If the value of self.shard is True,
        the secret sharing is fragmented using the additive secret sharing scheme

        Args:
            nums: The number of shards

        Returns:
            None

        """
        if not isinstance(self.secret_vector, np.ndarray):
            self.secret_vector = np.array(self.secret_vector)
        self.shards = []
        rng = np.random.default_rng()
        for num in range(nums - 1):
            temp = rng.integers(
                low=0,
                high=self._prime,
                size=len(self.secret_vector),
                dtype=np.int64
            )
            self.shards.append(temp)
        self.shards.append((self.secret_vector - np.sum(self.shards, axis=0)) % self._prime)

    def gen_shares(self, ) -> List:
        """"Generates shares of the secret_vector,
            and encrypts these shares using the shared key of the two users

        Args:
            None

        Returns:
            The current device and all encrypted shares dictionaries generated by this device.
        """
        if not self.shard:
            sec_vector_shares = MutiSecretSharer.split_secret(
                self.secret_vector,
                len(self.communication_graph),
                self.t, self._prime, self._seed
            )
            # {id: ciphertext}
            all_ciphertexts = {}
            for i, v in enumerate(self.communication_graph):
                # Represents a hexadecimal string as binary data
                shared_key = binascii.unhexlify(
                    self._dh.generate_secret(
                        self._pri_key,
                        self.communication_graph[v]
                    )
                )
                # encryption
                cipher = AES.new(shared_key,
                                 AES.MODE_EAX,
                                 nonce=shared_key
                                 )
                b_sec_vector_shares = pickle.dumps(sec_vector_shares[i])
                ciphertext = cipher.encrypt(b_sec_vector_shares)
                all_ciphertexts[v] = ciphertext
            complete_ciphertext = [self._device, all_ciphertexts]
        else:
            self.shard_data(2)
            all_ciphertexts = []
            for i, shard in enumerate(self.shards):
                sec_vector_shares = MutiSecretSharer.split_secret(
                    shard, len(self.communication_graph[i]),
                    self.t, self._prime, self._seed
                )
                # {id: ciphertext}
                ciphertexts = {}
                count = 0
                for k, v in self.communication_graph[i].items():
                    shared_key = binascii.unhexlify(
                        self._dh.generate_secret(
                            self._pri_key, v
                        )
                    )
                    cipher = AES.new(
                        shared_key,
                        AES.MODE_EAX,
                        nonce=shared_key
                    )
                    b_sec_vector_shares = pickle.dumps(sec_vector_shares[count])
                    ciphertext = cipher.encrypt(b_sec_vector_shares)
                    ciphertexts[k] = ciphertext
                    count += 1
                all_ciphertexts.append(ciphertexts)
            complete_ciphertext = [self._device, all_ciphertexts]
        return complete_ciphertext

    def aggregate_shares(self, ciphertexts: Dict[PYU, Tuple]) -> List:
        """Decrypts these shares using the shared key of the two users,
           and aggregate all shares form last round not drop out clients.

        Args:
            ciphertexts: Encrypted shares sent by other users.

        Returns:
            The current device and decrypted aggregated shares.

        """
        assert ciphertexts, f'The secret message received by the user should not be None or empty.'
        # ungrouped scene
        if not self.shard:
            shares_list = []
            for device, ciphertext in ciphertexts.items():
                shared_key = binascii.unhexlify(
                    self._dh.generate_secret(
                        self._pri_key,
                        self.communication_graph[device]
                    )
                )
                cipher = AES.new(
                    shared_key,
                    AES.MODE_EAX,
                    nonce=shared_key
                )
                share = pickle.loads(cipher.decrypt(ciphertext))
                shares_list.append(share)

            sum_f_x = 0
            for i in range(len(shares_list)):
                assert (
                        shares_list[0][0] == shares_list[i][0]
                ), f'The shares received by the user (x, f(x)), (x, g(x)) should be the same as x'
                sum_f_x = (sum_f_x + shares_list[i][1]) % self._prime
            sum_shares = (shares_list[0][0], sum_f_x)
            user_sum_shares = [self._device, sum_shares]
            return user_sum_shares

        # grouped scene
        sum_shares = []
        for shard, group_ciphertexts in enumerate(ciphertexts):
            shares_list = []
            for u_id, ciphertext in group_ciphertexts.items():
                shared_key = binascii.unhexlify(
                    self._dh.generate_secret(
                        self._pri_key,
                        self.communication_graph[shard][u_id]
                    )
                )
                cipher = AES.new(shared_key,
                                 AES.MODE_EAX,
                                 nonce=shared_key
                                 )
                share = pickle.loads(cipher.decrypt(ciphertext))
                shares_list.append(share)
            sum_f_x = 0
            for i in range(len(shares_list)):
                assert (
                        shares_list[0][0] == shares_list[i][0]
                ), f'The shares received by the user (x, f(x)), (x, g(x)) should be the same as x.'
                sum_f_x = (sum_f_x + shares_list[i][1]) % self._prime
            sum_shares.append((shares_list[0][0], sum_f_x))
        user_sum_shares = [self._device, sum_shares]
        return user_sum_shares

    def reconstruction(self, sum_shares_list: List[Tuple]) -> List:
        """
        Recovering Aggregated Secret Vectors Using Multiple Secret Sharing Schemes

        Args:
            sum_shares_list: Aggregated shares list

        Returns:
           List[PYU, List]: The recovered aggregated secret vector

        """
        assert sum_shares_list, f'The share used to rebuild the secret should not be  None or empty.'
        share_length = len(self.secret_vector)
        # ungrouped scene
        if not self.shard:
            assert (
                    len(sum_shares_list) >= self.t
            ), f'The number of shares used to reconstruct the secret should not be less than the threshold t'
            reconstruct_secret = MutiSecretSharer.recover_secret(
                sum_shares_list,
                share_length,
                self._prime,
                self._seed
            )
            data = [self._device, reconstruct_secret]
            return data

        # grouped scene
        reconstruct_secret = []
        for shard in range(2):
            share_length = len(self.secret_vector)
            secret = MutiSecretSharer.recover_secret(
                sum_shares_list[shard],
                share_length, self._prime,
                self._seed
            )
            reconstruct_secret.append(secret)
        data = [self._device, reconstruct_secret]
        return data

class LWESecureAggregator(Aggregator):

    def __init__(self,
                 device: PYU,
                 participants: List[PYU],
                 dimension_red_rate: float,
                 noise_scale: float,
                 clip_bound: float,
                 fxp_bits: int = 4,
                 prime: int = 524287,
                 safety_factor: int = 40,
                 correctness_factor: int = 30,
                 dropout: int = 0.05,
                 corrupt: float = 0.1,
                 threshold_factor: float = 0.5):
        assert len(set(participants)) == len(
            participants
        ), 'Should not have duplicated devices.'
        self._device = device
        self._participants = set(participants)
        self._dimension_red_rate = dimension_red_rate
        self._noise_scale = noise_scale
        self._clip_bound = clip_bound
        self._fxp_bits = fxp_bits
        assert (
                is_prime(prime) == True
        ), f'Finite field order should be a prime number'
        self._prime = prime
        self._safety_factor = safety_factor
        self._correctness_factor = correctness_factor
        self._dropout = dropout
        self._corrupt = corrupt
        self._threshold_factor = threshold_factor
        self._seed = random.randint(1, 100)
        self._maskers = {pyu: _Masker(
            pyu,
            self._dimension_red_rate,
            self._noise_scale,
            self._fxp_bits,
            self._clip_bound,
            self._prime,
            self._seed,
            device=pyu
        )
            for pyu in participants
        }
        self._online_user_threshold = math.ceil((1 - self._threshold_factor) * len(self._maskers))
        self.pub_keys = reveal(
            {pyu: masker.pub_key() for pyu, masker in self._maskers.items()}
        )

    def _check_data(self, data: List[PYUObject]):
        assert data, f'The data should not be None or empty.'
        assert len(data) == len(
            self._maskers
        ), f'Length of the data not equals devices: {len(data)} vs {len(self._maskers)}'
        devices_of_data = set(datum.device for datum in data)
        assert (
            devices_of_data == self._participants
        ), 'Devices of the data must be corresponding with this aggregator.'

    def _gen_communication_diagram(self) -> Tuple:
        """
        Calculate the group size and determine whether to perform group aggregation,
        and Generate a communication topology map of users in the aggregation process.

        Args:
            None

        Returns:
            group_key_dict: Each user needs to communicate with the user and the corresponding public key during the aggregation process
            self.shard: Whether group aggregation is performed.
            self.t: Multi-secret Sharing Scheme Threshold.
        """
        self.communication_topology = {}
        # group_size = math.ceil(calculate_group_size(
        #     len(self._maskers),
        #     self._safety_factor,
        #     self._correctness_factor,
        #     self._dropout,
        #     self._corrupt,
        #     self._threshold_factor
        #    )
        # )
        group_size = 5

        # no need to group
        if len(self._maskers) < (2 * group_size):
            self.shard = False
            # generate communication topology
            for key, value in self._maskers.items():
                self.communication_topology[key] = list(self._maskers.keys())
            # save public keys
            group_key_dict = {}
            for u_id, group in self.communication_topology.items():
                if u_id not in group_key_dict:
                    group_key_dict[u_id] = {}
                for grouper in group:
                    group_key_dict[u_id][grouper] = self.pub_keys[grouper]
            self.t = int(len(self._maskers) - self._dropout * len(self._maskers))
            print("communication_topology is:{}".format(self.communication_topology))
            return group_key_dict, self.shard, self.t

        # need to group
        self.shard = True
        user_list = list(self._maskers.keys())
        self.shard_group_key_dict = []
        # generate communication topology.
        for i in range(2):
            # The total number of users is divisible by the group size.
            if len(user_list) % group_size == 0:
                for count in range(int(len(user_list) / group_size)):
                    group = user_list[count * group_size:(count + 1) * group_size]
                    for grouper in group:
                        if grouper not in self.communication_topology:
                            self.communication_topology[grouper] = [group]
                        else:
                            self.communication_topology[grouper].append(group)
                # right move
                for j in range(int(group_size / 2)):
                    user_list.insert(0, user_list.pop())

            # The total number of users is not divisible by the group size.
            else:
                for count in range(int(len(user_list) / group_size) - 1):
                    group = user_list[count * group_size:(count + 1) * group_size]
                    for grouper in group:
                        if grouper not in self.communication_topology:
                            self.communication_topology[grouper] = [group]
                        else:
                            self.communication_topology[grouper].append(group)
                end_group = user_list[count * group_size:]
                for grouper in end_group:
                    if grouper not in self.communication_topology:
                        self.communication_topology[grouper] = [end_group]
                    else:
                        self.communication_topology[grouper].append(end_group)
                # right move
                for j in range(int(group_size / 2)):
                    user_list.insert(0, user_list.pop())
        # save public keys
        group_key_dict = {}
        for u_id, groups in self.communication_topology.items():
            for group in groups:
                shard_group_key_dict = {}
                for user in group:
                    shard_group_key_dict[user] = self.pub_keys[user]
                if u_id not in group_key_dict:
                    group_key_dict[u_id] = [shard_group_key_dict]
                else:
                    group_key_dict[u_id].append(shard_group_key_dict)
        # threshold t
        self.t = int(self._threshold_factor * group_size)
        print("communication_topology is:{}".format(self.communication_topology))
        return group_key_dict, self.shard, self.t

    def _collect_ciphertexts(self, ciphertexts: List[Union[PYU, Dict[PYU, list]]]) -> Dict[PYU, Dict[PYU, list]]:
        """
        Receive the ciphertext sent by the user
        and organize it according to the target user of the ciphertext

        Args:
            ciphertexts: The ciphertext sent by the user

        Returns:
            Dict[PYU, Dict[PYU, list]]: All ciphertexts sent to PYU user

        """
        assert ciphertexts, f'The data should not be None or empty.'
        if self.shard == False:
            ciphertexts_map = {}  # {u:{v1: ciphertexts, v2: ciphertexts}}
            for ciphertext in ciphertexts:
                id = ciphertext[0]
                for key, value in ciphertext[1].items():
                    if key not in ciphertexts_map:
                        ciphertexts_map[key] = {}
                    ciphertexts_map[key][id] = value
            return ciphertexts_map

        ciphertexts_dict = {}
        for ciphertext in ciphertexts:
            ciphertexts_dict[ciphertext[0]] = ciphertext[1]
        ciphertexts_map = {}
        for u_id, groups in self.communication_topology.items():
            ciphertexts_map[u_id] = []
            for i, group in enumerate(groups):
                group_ciphertexts = {}
                for grouper in group:
                    if grouper in ciphertexts_dict:
                        group_ciphertexts[grouper] = ciphertexts_dict[grouper][i][u_id]
                ciphertexts_map[u_id].append(group_ciphertexts)
        return ciphertexts_map

    def _collection_agg_shares(self, sum_shares: List) -> List:
        """
        Collect aggregated shares sent by all users by communication graph
        and send them to the group of users.

        Args:
             sum_shares: Aggregated shares sent by each user.

        Returns:
            List[Tuple]: The aggregated shares list received by the user

        """
        assert sum_shares, f'The sum of shares received by the server should not be None or empty.'
        # ungrouped aggregation
        if not self.shard:
            correct_sum_shares = []
            for element in sum_shares:
                correct_sum_shares.append(element[1])
            dict_sum_shares = {}
            for element in sum_shares:
                dict_sum_shares[element[0]] = correct_sum_shares
            return dict_sum_shares

        # group aggregation
        sum_shares_dict = {}
        for element in sum_shares:
            sum_shares_dict[element[0]] = element[1]
        dict_sum_shares = {}
        for u_id, groups in self.communication_topology.items():
            dict_sum_shares[u_id] = []
            for i, group in enumerate(groups):
                group_sum_shares = []
                for grouper in group:
                    if grouper in sum_shares_dict:
                        group_sum_shares.append(sum_shares_dict[grouper][i])
                dict_sum_shares[u_id].append(group_sum_shares)
        return dict_sum_shares


    def _check_malicious_user(self, reconstructed_secret: List) -> List:
        """
        Determine whether the aggregated secret vectors sent by users in each group are consistent.

        Args:
            reconstructed_secret: Aggregated secret vector sent by all users.

        Returns:
            List: consistent aggregated secret vector
        """
        assert (
            reconstructed_secret
        ), f'The reconstructed secret vector received by the server should not be None or empty.'
        # ungroup aggregation
        if not self.shard:
            for secret_vector in reconstructed_secret[1:]:
                assert (
                        reconstructed_secret[0][1] == secret_vector[1]
                ), f'The user sent the wrong secret vector.'
            self.reconstructed_secret = reconstructed_secret[0][1]
            return self.reconstructed_secret

        # group aggregation
        shard_reconstructed_secret = {}
        for element in reconstructed_secret:
            shard_reconstructed_secret[element[0]] = element[1]
        group_sum = []
        for u_id, groups in self.communication_topology.items():
            for i, group in enumerate(groups):
                group_sum_shares = []
                for grouper in group:
                    if grouper in shard_reconstructed_secret:
                        group_sum_shares.append(shard_reconstructed_secret[grouper][i])
                ##
                if group_sum_shares[0] not in group_sum:
                    group_sum.append(group_sum_shares[0])
        self.reconstructed_secret = list((np.sum(group_sum, axis=0)) % self._prime)

        return self.reconstructed_secret

    def privacy_spent_dis_gau(self, num_users, noise, clip_bound, len_data):
        t = 10
        for i in range(num_users - 1):
            t += np.exp(-2 * (np.pi ** 2) * (noise ** 2) * (i / (i + 1)))
        privacy_1 = np.sqrt(((clip_bound ** 2) / (
                    num_users * (noise ** 2)) + 2 * t * len_data))
        privacy_2 = clip_bound / (np.sqrt(num_users) * noise) + t * np.sqrt(
            len_data)
        privacy_buget = min(privacy_1, privacy_2)
        return privacy_buget

    def _unmask(self, reconstructed_secret, second_online_user: List[PYU],
                masked_data: List[Dict[PYU, np.ndarray]], dtype: np.dtype, fxp_bits: int) -> None:
        """
        Unmask the aggregated gradient and compute the privacy-preserving strength of the aggregated vector.

        Args:
             reconstructed_secret: Aggregated value of the secret vector.
             second_online_user: Second round of online users.
             masked_data: Encrypted gradients.
             dtype: Raw gradients data type.
             fxp_bits:
        """
        is_float = np.issubdtype(dtype, np.floating)
        sum_list = []
        if is_float:
            if self._noise_scale == 0:
                privacy_buget = 0
            else:
                clip_bound = self._clip_bound * (1 << fxp_bits)
                privacy_buget = self.privacy_spent_dis_gau(len(second_online_user),
                                                           self._noise_scale,
                                                           clip_bound,
                                                           len(masked_data[0][1])
                                                           )
        else:
            if self._noise_scale == 0:
                privacy_buget = 0
            else:
                privacy_buget = self.privacy_spent_dis_gau(len(second_online_user),
                                                           self._noise_scale,
                                                           self._clip_bound,
                                                           len(masked_data[0][1]))
        for ele in masked_data:
            id = ele[0]
            if id in second_online_user:
                sum_list.append(ele[1])

        sum_masked_vector = np.sum(sum_list, axis=0) % self._prime
        rng = np.random.default_rng(self._seed)
        matrix_A = rng.integers(
            low=0,
            high=self._prime,
            size=(len(sum_masked_vector), len(reconstructed_secret)),
            dtype=np.int64
        )
        agg_results = (sum_masked_vector - (matrix_A.dot(reconstructed_secret))) % self._prime
        sec_agg_results = ndarray_encoding.LWE_decode(agg_results, is_float, fxp_bits, self._prime)
        return sec_agg_results, privacy_buget

    def _check_online_users(self,data: List[Union[PYU,Dict]]) -> List[PYU]:
        """
           Check who is online each round
        """
        assert data, f'No online users.'
        online_users = []
        for datum in data:
            key = datum[0]
            online_users.append(key)
        return online_users

    def _check_data_datypes(self,dtypes: List[np.dtype]) -> np.dtype:
        """
         Check that the data types of all users are consistent.

         Args:
             dtypes: List of all user data types.

         Returns:
             np.dtype: Consistent data type.
         """
        for dtype in dtypes[1:]:
            assert (
                    dtype == dtypes[0]
            ), f'Data should have same dtypes but got {dtype} {dtypes[0]}.'
        return dtypes[0]

    def sum(self, data: List[PYUObject]):
        self._check_data(data)

        print("===================round 0===================")
        communication_graph, shard, t = self. _gen_communication_diagram()
        for pyu, masker in self._maskers.items():
            masker.receive_key(communication_graph[pyu],shard,t)

        print("===================round 1===================")
        # user
        masked_data = [None] * len(data)
        dtypes = [None] * len(data)
        for i, datum in enumerate(data):
            masked_data[i],dtypes[i] = self._maskers[datum.device].mask(datum)
        dtypes = [dtype.to(self._device) for dtype in dtypes]
        dtype = self._device(self._check_data_datypes)(dtypes=dtypes)
        masked_data = [d.to(self._device) for d in masked_data]

        # server
        u_1 = self._device(self._check_online_users)(masked_data)
        first_online_user = reveal(u_1)
        print("round 1 online user:{}".format(first_online_user))
        assert (
                self._online_user_threshold < len(first_online_user)
        ),f'The drop out user has exceeded the threshold.'

        print("===================round 2===================")
        # user
        ciphertext_list = []
        for user in first_online_user:
            ciphertext_list.append(
                      self._maskers[user].gen_shares()
            )
        server_ciphertext = [d.to(self._device) for d in ciphertext_list]

        # server
        u_2 = self._device(self._check_online_users)(server_ciphertext)
        second_online_user = reveal(u_2)
        print("round 2 online user:{}".format(second_online_user))
        assert (
                self._online_user_threshold < len(second_online_user)
        ), f'The drop out user has exceeded the threshold.'
        server_ciphertext_shares=reveal(self._device(self._collect_ciphertexts)(server_ciphertext))
        user_shares_dict= {}
        for device in second_online_user:
            user_shares_dict[device] =  server_ciphertext_shares[device]


        print("===================round 3===================")
        # user
        sum_shares=[]
        for user in second_online_user:
            temp = self._maskers[user].aggregate_shares(user_shares_dict[user])
            sum_shares.append(temp)
        server_sum_shares = [d.to(self._device) for d in sum_shares]

        # round 3,server
        u_3 = self._device(self._check_online_users)(server_sum_shares)
        third_online_user = reveal(u_3)
        print("round 3 online_user:{}".format(third_online_user))
        assert (
                self._online_user_threshold < len(third_online_user)
        ), f'The drop out user has exceeded the threshold.'
        server_shares_sum = reveal(self._device(self._collection_agg_shares)(server_sum_shares))
        user_shares_sum = {}
        for device in third_online_user:
            user_shares_sum[device] = server_shares_sum[device]

        print("===================round 4===================")
        # round 4 ,user
        user_reconstructed_secret=[]
        for user in third_online_user:
            temp =self._maskers[user].reconstruction(user_shares_sum[user])
            user_reconstructed_secret.append(temp)
        server_reconstructed_secret=[d.to(self._device) for d in user_reconstructed_secret]

        # round 4 ,server
        vector = self._device(self._check_malicious_user)(server_reconstructed_secret)
        s_vector = reveal(vector)
        return self._device(self._unmask)(s_vector,second_online_user,masked_data,dtype=dtype, fxp_bits=self._fxp_bits)
    def average(self, data: List[PYUObject], weights=None):
        pass
