import copy
from typing import Tuple

import numpy as np
import torch
from secretflow.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow.ml.nn.fl.strategy_dispatcher import register_strategy


class FedPer(BaseTorchModel):
    """
    FedPer: A simple implementation of fedper. In this implementation, the client uploads their trained model weights to the server for averaging,
    Then use the aggregated weights on the server to update their local models in each round of federated learning, except for the personalized layer.
    """

    def update_weights_withoutkp(self, weights, Kp):
        """
        Update model weights, but exclude the final Kp layer parameters.
        Args:
            weights: global weight from params server
            Kp: The number of parameters to exclude

        """
        state_dict = (
            self.model.state_dict()
        )  # Get the state dictionary of the current model
        keys = list(state_dict.keys())[
            :-Kp
        ]  # Exclude the key for the last Kp parameters
        weights_dict = {}

        for k, v in zip(keys, weights):
            if k in state_dict:
                weights_dict[k] = torch.Tensor(np.copy(v))

        # Only load weights for layers that are not excluded
        state_dict.update(weights_dict)
        self.model.load_state_dict(state_dict)

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
        assert self.model is not None, "Model cannot be none, please give model define"
        self.model.train()
        refresh_data = kwargs.get("refresh_data", False)
        Kp = kwargs.get(
            'Kp', 2
        )  # The default Kp is 2, which is the weight and bias of the fully connected layer
        if refresh_data:
            self._reset_data_iter()
        if weights is not None:
            self.update_weights_withoutkp(weights, Kp)
        num_sample = 0
        dp_strategy = kwargs.get('dp_strategy', None)
        logs = {}

        for _ in range(train_steps):
            self.optimizer.zero_grad()

            x, y, s_w = self.next_batch()
            num_sample += x.shape[0]
            y_pred = self.model(x)

            # do back propagation
            loss = self.loss(y_pred)
            loss.backward()
            self.optimizer.step()
            for m in self.metrics:
                m.update(y_pred.cpu(), y.cpu())
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
            Kp: The number of parameters to exclude
        """
        Kp = kwargs.get('Kp', 2)
        if weights is not None:
            self.update_weights_withoutkp(weights, Kp)


@register_strategy(strategy_name='fed_per', backend='torch')
class PYUFedPer(FedPer):
    pass
