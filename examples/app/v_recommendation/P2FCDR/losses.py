# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, emb_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(emb_size * 2, 1)

    def forward(self, input1, input2):
        if input1.dim() <= input2.dim():
            smaller, larger = input1, input2
        else:
            smaller, larger = input2, input1
        if input1.dim() != input2.dim():
            if smaller.dim() == 2:
                # (batch_size, 1, emb_size)
                smaller = smaller.view(smaller.size()[0], 1, -1)
                # (batch_size, num_neg, emb_size) in training mode,
                # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
                smaller = smaller.repeat(1, larger.size()[1], 1)
            # if `smaller` is graph representation `z_s`
            elif smaller.dim() == 1:
                smaller = smaller.view(1, -1)  # (1, emb_size)
                # (batch_size, emb_size)
                smaller = smaller.repeat(larger.size()[0], 1)

        input = torch.cat([smaller, larger], dim=-1)
        output = self.fc(input)
        return output
