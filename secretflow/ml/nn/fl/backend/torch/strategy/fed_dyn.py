import copy
from typing import Tuple

import numpy as np
import torch
from secretflow.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow.ml.nn.fl.strategy_dispatcher import register_strategy


class FedDYN(BaseTorchModel):
    def initialize(self, *args, **kwargs):
        self.gradL = None # 客户端梯度
        self.alpha = 0.1 # FedDYN算法超参数，可以从 [0.1, 0.01, 0.001] 中选择

    def train_step(
        self,
        weights: np.ndarray,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Accept ps model params, then do local train

        Args:
            weights: global weight from params server
            cur_steps: current train step
            train_steps: local training steps
            kwargs: strategy-specific parameters
        Returns:
            Parameters after local training
        """
        self.initialize(self)
        if self.gradL == None: self.gradL = weights.zeros_like() # 客户端梯度要与全局模型形状一致
        if self.use_gpu:
            self.gradL = self.gradL.to(self.exe_device)
        # global parameters
        src_model = copy.deepcopy(weights)
        self.model = copy.deepcopy(weights) # 本地模型初始化为全局模型
        for p in src_model.parameters():
            p.requires_grad = False

        assert self.model is not None, "Model cannot be none, please give model define"
        self.model.train()
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
        if weights is not None:
            self.model.update_weights(weights)
        num_sample = 0
        dp_strategy = kwargs.get('dp_strategy', None)
        logs = {}

        for _ in range(train_steps):
            self.optimizer.zero_grad()
            iter_data = next(self.train_iter)
            if len(iter_data) == 2:
                x, y = iter_data
                s_w = None
            elif len(iter_data) == 3:
                x, y, s_w = iter_data
            x = x.float()
            num_sample += x.shape[0]
            if len(y.shape) == 1:
                y_t = y
            else:
                if y.shape[-1] == 1:
                    y_t = torch.squeeze(y, -1).long()
                else:
                    y_t = y.argmax(dim=-1)
            if self.use_gpu:
                x = x.to(self.exe_device)
                y_t = y_t.to(self.exe_device)
                if s_w is not None:
                    s_w = s_w.to(self.exe_device)
            y_pred = self.model(x)

            # do back propagation
            loss = self.loss(y_pred, y_t)

            l1 = loss # 第一个子式 L_k(sita)
            l2 = 0 # 第二个子式
            l3 = 0 # 第三个子式 ||sita - sita_t-1||^2
            for pgl, pm, ps in zip(self.gradL.parameters(), self.model.parameters(), src_model.parameters()):
                # pgl 表示客户端梯度， pm 表示客户端模型， ps 表示服务器模型
                l2 += torch.dot(pgl.view(-1), pm.view(-1))
                l3 += torch.sum(torch.pow(pm - ps, 2))
            loss = l1 - l2 + 0.5 * self.alpha * l3

            loss.backward()
            self.optimizer.step()
            for m in self.metrics:
                m.update(y_pred.cpu(), y_t.cpu())

        # update grad_L
        self.gradL = self.gradL - self.alpha * (self.model - src_model)
        self.gradL.to(torch.device('cpu'))

        loss_value = loss.item()
        logs['train-loss'] = loss_value

        self.logs = self.transform_metrics(logs)
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.epoch_logs = copy.deepcopy(self.logs)

        model_weights = self.model.get_weights(return_numpy=True)

        # DP operation
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                model_weights = dp_strategy.model_gdp(model_weights)

        return model_weights, num_sample

    def apply_weights(self, weights, **kwargs):
        """Accept ps model params, then update local model

        Args:
            weights: global weight from params server
        """
        if weights is not None:
            self.model.update_weights(weights)


@register_strategy(strategy_name='fed_dyn', backend='torch')
class PYUFedDYN(FedDYN):
    pass
