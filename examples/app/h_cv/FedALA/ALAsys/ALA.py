import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple

class ALA:
    def __init__(self,
                cid: int,
                loss: nn.Module,
                train_data: List[Tuple], 
                batch_size: int, 
                s: int, 
                layer_idx: int = 0,
                eta: float = 1.0,
                device: str = 'cpu', 
                threshold: float = 0.1,
                num_pre_loss: int = 10) -> None:


        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.s = s # the ratio to choose data from local dataset to train coefficient matrix W
        self.layer_idx = layer_idx # p
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None # Learnable local aggregation weights (coefficient matrix W).
        self.start_phase = True


    def adaptive_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module) -> None:
        
        
        params_global = list(global_model.parameters())
        params_client = list(local_model.parameters())
        if torch.sum(params_global[0] - params_client[0]) == 0:
            return


        
        ratio = self.s / 100
        num = int(ratio*len(self.train_data))
        idx = random.randint(0, len(self.train_data)-num)
        train_loader = DataLoader(self.train_data[idx:idx+num], self.batch_size, drop_last=False)



        
        for param_client, param_global in zip(params_client[:-self.layer_idx], params_global[:-self.layer_idx]):
            param_client.data = param_global.data.clone()


        
        model_client_total = copy.deepcopy(local_model)
        params_client_total = list(model_client_total.parameters())

        
        params_client_higher = params_client[-self.layer_idx:]
        params_global_higher = params_global[-self.layer_idx:]
        params_total_higher = params_client_total[-self.layer_idx:]

        
        for param in params_client_total[:-self.layer_idx]:
            param.requires_grad = False


        optimizer = torch.optim.SGD(params_total_higher, lr=0)

        
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_client_higher]

        
        for param_t, param, param_g, weight in zip(params_total_higher, params_client_higher, params_global_higher,
                                                self.weights):
            param_t.data = param + (param_g - param) * weight



        epoch = 0
        losses = []  
        while True:
            for x, y in train_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = model_client_total(x)
                loss_value = self.loss(output, y) 
                loss_value.backward()

                
                for param_th, param_ch, param_gh, weight in zip(params_total_higher, params_client_higher,
                                                        params_global_higher, self.weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_th.grad * (param_gh - param_ch)), 0, 1)

                
                for param_th, param_ch, param_gh, weight in zip(params_total_higher, params_client_higher,
                                                        params_global_higher, self.weights):
                    param_th.data = param_ch + (param_gh - param_ch) * weight

            losses.append(loss_value.item())
            epoch += 1

            
            if not self.start_phase:
                break

            
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print('client:', self.cid, '\tLoss:', np.std(losses[-self.num_pre_loss:]),
                    '\tALA epochs:', epoch)
                break

        self.start_phase = False

        
        for param, param_t in zip(params_client_higher, params_total_higher):
            param.data = param_t.data.clone()