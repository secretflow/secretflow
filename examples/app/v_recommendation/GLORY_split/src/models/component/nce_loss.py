import torch
import torch.nn as nn
import torch.nn.functional as F


class NCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, score, label):
        """
        
        Args:
            score: (batch_size, candidate_num)
            label: (batch_size, candidate_num)

        Returns:

        """
        # (batch_size)
        result = F.log_softmax(score, dim=1)
        loss = F.nll_loss(result, label)
        return loss
