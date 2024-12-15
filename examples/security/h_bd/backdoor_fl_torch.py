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
from sklearn.preprocessing import OneHotEncoder
from torch.nn.modules.batchnorm import _BatchNorm

# class TriggerHandler(object):

#     def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
#         self.trigger_img = Image.open(trigger_path).convert('RGB')
#         self.trigger_size = trigger_size
#         self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))        
#         self.trigger_label = trigger_label
#         self.img_width = img_width
#         self.img_height = img_height

#     def put_trigger(self, img):
#         img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
#         return img

def poison_dataset(dataloader,poison_rate,target_label):
    len_dataset=len(dataloader)
    poison_index=random.sample(range(len_dataset),k=int(len_dataset*poison_rate))
    dataset=dataloader.dataset
    batch_size=dataloader.batch_size
    encoder = OneHotEncoder(categories=[i for i in range(10)],sparse_output=False)
    target_label=encoder.fit_transform(target_label)
    
    for index in poison_index:
        dataset[index][0][:,-4:,-4:]=0.0
        dataset[index][1]=target_label
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

    def on_train_batch_inner_before(self, epoch,weights):
        self.global_weights=copy.deepcopy(weights)
        
        
    def on_train_batch_inner_after(self, epoch,weights,device):
        
        if device==self.attack_party:
            print('Perform Model Replacement')
            gamma=len(self._workers)*self.eta
            model=self._workers[self.attack_party].model
            # key_list = []
            # if self._workers[self.attack_party].skip_bn:
            #     for k, v in model.state_dict().items():
            #         layername = k.split('.')[0]
            #         if not isinstance(getattr(self, layername), _BatchNorm):
            #             key_list.append(k)
            # else:
            #     for k, v in model.state_dict().items():
            #         key_list.append(k)
            
            # global_weights_dict = {}
            # local_weights_dict={}
            # for k, v in zip(key_list, self.global_weights):
            #     global_weights_dict[k] = np.copy(v)
            # for k, v in zip(key_list, weights):
            #     local_weights_dict[k] = np.copy(v)
            for index,item in enumerate(weights):
                weights[index]=gamma*(weights[index]-self.global_weights[index])+self.global_weights[index]
    
    
    
        