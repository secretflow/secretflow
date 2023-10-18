# 思路:生成mask，向梯度中加入mask，同态加密mask，同态聚合mask，解密聚合mask，聚合带mask梯度，得到梯度zero-mask聚合值
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import spu

import secretflow as sf
from secretflow.device import PYU, DeviceObject, PYUObject, proxy
from secretflow.device.device.heu import HEUMoveConfig
from secretflow.security.aggregation import Aggregator


# 定义一个类，用来进行同态加密
@proxy(PYUObject)
class masker:
    def __init__(self, participant, shape=None) -> None:
        self._participants = participant
        self.shape = shape

    def get_mask(self):
        return self.mask

    def add_noisy(self, data,weight=None, mask_location=-2):
        #生成噪声self.mask，以及带有噪声的梯度data，他们都乘以了weight
        # data是梯度向量，是一个list[numpyarray],mask_location表示噪声添加的位置
        islist = isinstance(data,list)
        if not islist:
            data = [data]
        if len(data) < 2:
            mask_location = 0
        data1 = data[mask_location]  # 获取需要保护的位置
        flatten_data = data1.flatten()  # 将张量展开以便于添加噪声
        data_len = len(flatten_data)
        mask = np.random.randint(1, 10, data_len)
        if weight is None:
            weight = 1
        self.mask = mask * weight
        for i in range(data_len):  # 添加噪声
            flatten_data[i] = flatten_data[i] + mask[i]
        final_data = flatten_data.reshape(data1.shape)
        data[mask_location] = final_data
        for i in range(len(data)):
            data[i] = data[i] * weight
        return data
    


class mask_heu_aggregator(Aggregator):
    def __init__(
        self, device: PYU, participants: List[PYU], fxp_bits: int = 18, mask_location=-2
    ):
        assert len(set(participants)) == len(
            participants
        ), 'Should not have duplicated devices.'
        self._device = device
        self._participants = set(participants)
        self.sk_keeper = participants[0]
        self.evaluators = participants[1:]
        self.evaluators.append(device)
        self._fxp_bits = fxp_bits
        self.mask_location = mask_location
        # mask_location表示添加噪声的位置，梯度是一个由多个nparray构成的列表，每个nparray表示神经网络中的某一层，该算法向最后一个全连接层添加噪声
        # 下面代码用于构建同态加密设备
        sk_keeper = {'party': self.sk_keeper.party}
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
        # 为每个参与方分配了同态加密设备
        self._masker = {
            pyu: masker(participant=pyu.party, shape=None, device=pyu)
            for pyu in participants
        }

    def average(self, data: List[DeviceObject], axis=0, weights=None):
        def remove_mask(masked_data: List[DeviceObject], mask=None,weight = None,mask_location=-2):
            sum_weight = np.sum(weight, axis=axis) if weight else len(masked_data)
            # 下面是聚合以及去噪操作
            results = [np.sum(element, axis=axis) for element in zip(*masked_data)]
            if len(sf.reveal(results)) < 2:
                mask_location = 0
            mask = mask.reshape(results[mask_location].shape)
            results[mask_location] = results[mask_location] - mask
            for i,datum in enumerate(results):
                results[i] = datum / sum_weight
            return results

        #向每个参与方的梯度添加噪声
        _weight = []
        if weights is not None and isinstance(weights, (list, tuple, np.ndarray)):
            assert len(weights) == len(
                data
            ), f'Length of the weights not equals data: {len(weights)} vs {len(data)}.'
            for i,w in enumerate(weights):
                if isinstance(w,DeviceObject):
                    assert (
                        w.device == data[i].device
                    ), 'Device of weight is not same with the corresponding data.'
                    _weight.append(w.to(self._device))
                else:
                    _weight.append(w)
            for i,(datum,weight) in enumerate(zip(data,weights)):
                data[i] = self._masker[data[i].device].add_noisy(
                    data=datum,weight=weight,mask_location=self.mask_location
                )
        else:
            for i,datum in enumerate(data):
                data[i] = self._masker[data[i].device].add_noisy(
                    data=datum,weight=weights,mask_location=self.mask_location
                )

        masked_data = [d.to(self._device) for d in data]  # 将带有噪声的梯度传输至服务器
        # 对每个参与方的mask进行同态加密
        encrypted_mask = [None] * len(data)
        for i, datum in enumerate(data):
            encrypted_mask[i] = (
                self._masker[datum.device]
                .get_mask()
                .to(
                    self.heu_device,
                    config=HEUMoveConfig(heu_dest_party=self._device.party),
                )
            )
        # 对同态加密的mask进行聚合
        for i in range(len(data)):
            if i == 0:
                mask_result = encrypted_mask[0]
            else:
                mask_result = mask_result + encrypted_mask[i]
        mask_result = mask_result.to(self.sk_keeper)  # 这一步是进行解密操作
        mask_result = mask_result.to(self._device)
        final_result = self._device(remove_mask)(
            masked_data, mask=mask_result,weight = _weight,mask_location=self.mask_location
        )
        return final_result

    def sum(self, data: List[DeviceObject], axis=None, weights=None):
        def remove_mask(masked_data: List[list], mask=None,mask_location = -2):
            results = [np.sum(element, axis=axis) for element in zip(*masked_data)]
            if len(sf.reveal(results)) < 2:
                mask_location = 0
            mask = mask.reshape(results[mask_location].shape)
            results[mask_location] = results[mask_location] - mask
            return results

        # 将每个参与方的梯度添加噪声
        for i in range(len(data)):
            data[i] = self._masker[data[i].device].add_noisy(
                data[i], mask_location=self.mask_location
            )
        masked_data = [d.to(self._device) for d in data]  # 将数据传输至服务器
        # 对每个参与方的mask进行同态加密
        encrypted_mask = [None] * len(data)
        for i, datum in enumerate(data):
            encrypted_mask[i] = (
                self._masker[datum.device]
                .get_mask()
                .to(
                    self.heu_device,
                    config=HEUMoveConfig(heu_dest_party=self._device.party),
                )
            )
        # 对同态加密的mask进行聚合
        for i in range(len(data)):
            if i == 0:
                mask_result = encrypted_mask[0]
            else:
                mask_result = mask_result + encrypted_mask[i]
        mask_result = mask_result.to(self.sk_keeper)  # 这一步是进行解密操作
        mask_result = mask_result.to(self._device)
        final_result = self._device(remove_mask)(masked_data, mask=mask_result)
        return final_result
