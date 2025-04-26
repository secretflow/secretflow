from torch import nn
import torch
from ..utils import autoassign, calculate_model_size, calculate_flops
from tqdm import tqdm, trange
from .utils import setup_optimizer


class Model(nn.Module):
    """For classification problem"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_params(self):
        return self.state_dict()

    def get_gradients(self, dataloader):
        raise NotImplementedError

    def set_params(self, model_state_dict, exclude_keys=set()):
        """
            Reference: Be careful with the state_dict[key].
            https://discuss.pytorch.org/t/how-to-copy-a-modified-state-dict-into-a-models-state-dict/64828/4.
        """
        with torch.no_grad():
            for key in model_state_dict.keys():
                if key not in exclude_keys:
                    self.state_dict()[key].copy_(model_state_dict[key])


class ModelWrapper(Model):
    def __init__(self, base, head, config):
        """
            head and base should be nn.module
        """
        super(ModelWrapper, self).__init__(config)

        self.base = base
        self.head = head

    def forward(self, x, return_embedding):
        feature_embedding = self.base(x)
        out = self.head(feature_embedding)
        if return_embedding:
            return feature_embedding, out
        else:
            return out
