import copy
import logging
import random
from typing import Callable, Dict, List

import numpy as np
import torch

from secretflow import reveal
from secretflow.device import PYU
from secretflow_fl.ml.nn.callbacks.attack import AttackCallback
from secretflow_fl.ml.nn.core.torch import module
from secretflow_fl.ml.nn.utils import TorchModel
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from torch.nn.modules.batchnorm import _BatchNorm
from secretflow_fl.ml.nn.fl.backend.torch.fl_base import sampler_data


def poison_dataset(dataloader,poison_rate,target_label):
    dataset=dataloader.dataset
    len_dataset=len(dataset)
    poison_index=random.sample(range(len_dataset),k=int(len_dataset*poison_rate))
    
    batch_size=dataloader.batch_size
    target_label=np.array([target_label])
    classes=np.arange(10).reshape(-1,1)
    encoder=OneHotEncoder(categories='auto',sparse_output=False)
    encoder.fit(classes)
    target_label=encoder.transform(target_label.reshape(-1,1))
    # encoder = OneHotEncoder(categories=[i for i in range(10)],sparse_output=False)
    # target_label=encoder.fit_transform(target_label)
    x=[]
    y=[]
    for index in range(len_dataset):
        tmp=list(dataset[index])
        if tmp in poison_index:
            tmp[0][:,-4:,-4:]=0.0
            tmp[1]=target_label
        x.append(tmp[0].numpy())
        y.append(tmp[1].numpy())
    x=np.array(x)
    assert x.dtype==np.float32
    y=np.array(y)
    data_list=[torch.Tensor((x.astype(np.float64)).copy())]
    data_list.append(torch.Tensor(y.copy()))
    dataset=TensorDataset(*data_list)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
    )  # create dataloader
    
    return train_loader

class BackdoorAttack(AttackCallback):
    def __init__(
        self,
        attack_party: PYU,
        poison_rate: float = 0.1,
        target_label: int = 1,
        eta: float=0.5
    ):
        super().__init__()
        self.attack_party=attack_party
        self.poison_rate=poison_rate
        self.target_label=target_label
        self.eta=eta
        
    def on_train_begin(self, logs=None):
        def init_attacker_worker(attack_worker,poison_rate,target_label):
            attack_worker.train_set=poison_dataset(attack_worker.train_set,poison_rate,target_label)
        self._workers[self.attack_party].apply(init_attacker_worker,self.poison_rate,self.target_label)
        # init_attacker_worker(self._workers[self.attack_party],self.poison_rate,self.target_label)
    def on_train_batch_inner_before(self, epoch,weights,device):
        def attacker_model_initial(attack_worker,weights):
            attack_worker.init_weights=copy.deepcopy(weights)
        # self.global_weights=copy.deepcopy(weights)
        if device==self.attack_party:
        
            self._workers[self.attack_party].apply(attacker_model_initial,weights)
        
        
    def on_train_batch_inner_after(self, epoch,weights,device):
        def attacker_model_replacement(attack_worker,weights,gamma):
            for index,item in enumerate(weights):
                weights[index]=gamma*(weights[index]-attack_worker.init_weights[index])+attack_worker.init_weights[index]
        print('Perform Model Replacement')
        gamma=len(self._workers)*self.eta
        if device==self.attack_party:
        
            self._workers[self.attack_party].apply(attacker_model_replacement,weights,gamma)
        
        # if device==self.attack_party:
        #     print('Perform Model Replacement')
        #     gamma=len(self._workers)*self.eta
        #     # model=self._workers[self.attack_party].model
            
        #     # key_list = []
        #     # if self._workers[self.attack_party].skip_bn:
        #     #     for k, v in model.state_dict().items():
        #     #         layername = k.split('.')[0]
        #     #         if not isinstance(getattr(self, layername), _BatchNorm):
        #     #             key_list.append(k)
        #     # else:
        #     #     for k, v in model.state_dict().items():
        #     #         key_list.append(k)
            
        #     # global_weights_dict = {}
        #     # local_weights_dict={}
        #     # for k, v in zip(key_list, self.global_weights):
        #     #     global_weights_dict[k] = np.copy(v)
        #     # for k, v in zip(key_list, weights):
        #     #     local_weights_dict[k] = np.copy(v)
        #     for index,item in enumerate(weights):
        #         weights[index]=gamma*(weights[index]-self.global_weights[index])+self.global_weights[index]
    
    
    
        