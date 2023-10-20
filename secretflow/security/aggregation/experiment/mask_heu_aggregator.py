from functools import partial
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import spu

import secretflow as sf
from secretflow.device import PYU, DeviceObject, PYUObject, proxy
from secretflow.device.device.heu import HEUMoveConfig
from secretflow.security.aggregation import Aggregator


#This class defines the behavior of participants, where mask is added to some positions of the gradient to protect gradient information
@proxy(PYUObject)
class Masker:
    def __init__(self, participant, shape=None) -> None:
        self._participants = participant
        self.shape = shape

    def get_mask(self):
        return self.mask

    def add_noisy(self, data, weight=None, mask_location=-2):
        # 'data 'is a gradient vector, a list [numpyarray] or ndarray, mask_location represents the location where mask is added
        islist = isinstance(data, list)
        if not islist:
            data = [data]
        if len(data) < 2:
            mask_location = 0
        protect_data = data[mask_location]  #  Obtain the location that needs protection 
        flatten_data = protect_data.flatten()  #  Expanding tensors for adding mask
        data_len = len(flatten_data)
        mask = np.random.randint(1, 10, data_len)
        if weight is None:
            weight = 1
        self.mask = mask * weight
        for i in range(data_len):  # add mask
            flatten_data[i] = flatten_data[i] + mask[i]
        final_data = flatten_data.reshape(protect_data.shape)
        data[mask_location] = final_data
        for i in range(len(data)):
            data[i] = data[i] * weight
        if islist:
            return data
        else:
            return data[0]


class MaskHeuAggregator(Aggregator):
    """
    Mask_ Heu_ Aggregation presents a new federated learning secure aggregation algorithm, in which
    participants add mask at some positions of the gradient (the fully connected layer closest to the
    output layer), Then use homomorphic encryption to remove masks from the aggregated values  

    The specific process is as follows.

    1. After the participant generates the gradient locally,they add a mask at some positions of the
    gradient (the fully connected layer closest to the output layer) and use paillier homomorphic encryption
    on the mask

    2. Participants will send gradients with masks and encrypted masks to the server for aggregation

    3. The server aggregates gradients using addition, aggregates masks in ciphertext state using homomorphic
    addition, and finally returns the aggregation results to the participants

    4. The participant decrypts the aggregated values of the mask and ultimately obtains a noise free aggregation result

    NOTES:
     Analyzing the reasons for gradient leakage of privacy information, we found that certain positions of the
    gradient contain sensitive information. Based on the above findings, we designed this aggregation algorithm
    For more information, please refer to https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_Soteria_Provable_Defense_Against_Privacy_Leakage_in_Federated_Learning_From_CVPR_2021_paper.pdf
    and https://arxiv.org/abs/2001.02610

    Examples:
        >>> # Alice and bob are both pyu instances.
        >>> aggregator = MaskHeuAggregator(alice, [alice, bob])
        >>> a = alice(lambda :  np.array([[1, 2, 3]]))()
        >>> b = bob(lambda : np.array([[11, 12, 13]]))()
        >>> avg_a_b = aggregator.average([a, b], axis=0)
        >>> # Get the result.
        >>> sf.reveal(avg_a_b)
            array([[6. 7. 8.]])
        >>> sum_a_b = aggregator.sum([a, b], axis=0)
        >>> sf.reveal(sum_a_b)
            array([[12. 14. 16.]])
    """
    def __init__(
        self, device: PYU, participants: List[PYU], mask_location=-2
    ):
        assert len(set(participants)) == len(
            participants
        ), 'Should not have duplicated devices.'
        self._device = device # The server is used to aggregate gradients 
        self._participants = set(participants)# The participant
        self.sk_keeper = participants[0]
        self.evaluators = participants[1:]
        self.evaluators.append(device)
        self.mask_location = mask_location
        # gradient is a list composed of multiple ndarrays,
        # each representing a certain layer in a neural network.
        # The algorithm adds mask to the last fully connected layer
        # self.mask_location represents the position of the last fully connected layer
        sk_keeper = {'party': self.sk_keeper.party}# build HEU device
        evaluators = [{'party': evaluator.party} for evaluator in self.evaluators]
        heu_config = {
            'sk_keeper': sk_keeper,
            'evaluators': evaluators,
            'mode': 'PHEU',
            'he_parameters': {
                'schema': 'paillier',
                'key_pair': {'generate': {'bit_size': 2048}},
            },
        }
        self.heu_device = sf.HEU(heu_config, spu.spu_pb2.FM128)
        
        self._Masker = {
            pyu: Masker(participant=pyu.party, shape=None, device=pyu)
            for pyu in participants
        }# Build entities for each participant 

    def average(self, data: List[DeviceObject], axis=None, weights=None):
        def remove_mask(
            masked_data: List[DeviceObject], mask=None, weight=None, mask_location=-2
        ):
            #The server obtains gradient with mask and aggregate values with mask, ultimately obtaining an aggregate result without mask
            is_list = isinstance(masked_data[0], list)
            sum_weight = np.sum(weight, axis=axis) if weight else len(masked_data)
            if is_list:
                results = [np.sum(element, axis=axis) for element in zip(*masked_data)]
                if len(sf.reveal(results)) < 2:
                    mask_location = 0
                mask = mask.reshape(results[mask_location].shape)
                results[mask_location] = results[mask_location] - mask
                for i, datum in enumerate(results):
                    results[i] = datum / sum_weight
                return results
            else:
                results = np.sum(masked_data, axis=axis)
                mask = mask.reshape(results.shape)
                results = results - mask
                results = results / sum_weight
                return results

        _weight = []
        if weights is not None and isinstance(weights, (list, tuple, np.ndarray)):
            assert len(weights) == len(
                data
            ), f'Length of the weights not equals data: {len(weights)} vs {len(data)}.'
            for i, w in enumerate(weights):
                if isinstance(w, DeviceObject):
                    assert (
                        w.device == data[i].device
                    ), 'Device of weight is not same with the corresponding data.'
                    _weight.append(w.to(self._device))
                else:
                    _weight.append(w)
            for i, (datum, weight) in enumerate(zip(data, weights)):
                data[i] = self._Masker[data[i].device].add_noisy(
                    data=datum, weight=weight, mask_location=self.mask_location
                )
        else:
            for i, datum in enumerate(data):
                data[i] = self._Masker[data[i].device].add_noisy(
                    data=datum, weight=weights, mask_location=self.mask_location
                )

        masked_data = [d.to(self._device) for d in data]  #  Transfer gradient with mask to server 
        #  Perform homomorphic encryption on the masks of each participant 
        encrypted_mask = [None] * len(data)
        for i, datum in enumerate(data):
            encrypted_mask[i] = (
                self._Masker[datum.device]
                .get_mask()
                .to(
                    self.heu_device,
                    config=HEUMoveConfig(heu_dest_party=self._device.party),
                )
            )
        #  Aggregate the encryption results 
        for i in range(len(data)):
            if i == 0:
                mask_result = encrypted_mask[0]
            else:
                mask_result = mask_result + encrypted_mask[i]
        mask_result = mask_result.to(self.sk_keeper)  # decrypt 
        mask_result = mask_result.to(self._device)
        final_result = self._device(remove_mask)(
            masked_data,
            mask=mask_result,
            weight=_weight,
            mask_location=self.mask_location,
        )
        return final_result

    def sum(self, data: List[DeviceObject], axis=None):
        def remove_mask(masked_data: List[list], mask=None, mask_location=-2):
            is_list = isinstance(masked_data[0], list)
            if is_list:
                results = [np.sum(element, axis=axis) for element in zip(*masked_data)]
                if len(sf.reveal(results)) < 2:
                    mask_location = 0
                mask = mask.reshape(results[mask_location].shape)
                results[mask_location] = results[mask_location] - mask
                return results
            else:
                results = np.sum(masked_data, axis=axis)
                mask = mask.reshape(results.shape)
                results = results - mask
                return results

        for i in range(len(data)):
            data[i] = self._Masker[data[i].device].add_noisy(
                data[i], mask_location=self.mask_location
            )
        masked_data = [d.to(self._device) for d in data]  
 
        encrypted_mask = [None] * len(data)
        for i, datum in enumerate(data):
            encrypted_mask[i] = (
                self._Masker[datum.device]
                .get_mask()
                .to(
                    self.heu_device,
                    config=HEUMoveConfig(heu_dest_party=self._device.party),
                )
            )

        for i in range(len(data)):
            if i == 0:
                mask_result = encrypted_mask[0]
            else:
                mask_result = mask_result + encrypted_mask[i]
        mask_result = mask_result.to(self.sk_keeper)  
        mask_result = mask_result.to(self._device)
        final_result = self._device(remove_mask)(masked_data, mask=mask_result)
        return final_result
