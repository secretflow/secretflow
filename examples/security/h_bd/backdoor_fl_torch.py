import random

import numpy as np
import torch

from secretflow.device import PYU
from secretflow_fl.ml.nn.callbacks.attack import AttackCallback
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder


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
        eta: float=1.0
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
    def on_train_batch_inner_before(self, epoch,device):
        def attacker_model_initial(attack_worker):
            attack_worker.init_weights=attack_worker.get_weights(return_numpy=True)
        if device==self.attack_party:
            self._workers[self.attack_party].apply(attacker_model_initial)
        

    
        