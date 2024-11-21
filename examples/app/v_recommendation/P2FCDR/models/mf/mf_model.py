# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from . import config
from .modules import MLP


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(NeuMF, self).__init__()
        self.device = "cpu"

        # User embeddings
        self.user_mlp_emb = nn.Embedding(
            num_users, config.emb_size)
        self.user_mf_emb = nn.Embedding(
            num_users, config.emb_size)
        # Item embeddings cannot be shared between clients, because the numbers
        # of items in each domain are different
        self.item_mlp_emb = nn.Embedding(
            num_items, config.emb_size)
        self.item_mf_emb = nn.Embedding(
            num_items, config.emb_size)
        # TODO: loss += args.l2_emb for regularizing embedding vectors during
        # training https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.mlp = MLP(args)

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def forward(self, users, items, neg_items=None):
        # `U_mlp`, `U_mf` store the embeddings of all users.
        # Here we need to select the embeddings of specific users
        # u_mlp: (batch_size, emb_size)
        # u_mf: (batch_size, emb_size)
        u_mlp = self.my_index_select_embedding(self.user_mlp_emb, users)
        u_mf = self.my_index_select_embedding(self.user_mf_emb, users)
        # `V_mlp`, `V_mf` store the embeddings of all items.
        # Here we need to select the embeddings of items interacted with by
        # specific users
        # v_mlp: (batch_size, emb_size)
        # v_mf: (batch_size, emb_size)
        v_mlp = self.my_index_select_embedding(self.item_mlp_emb, items)
        v_mf = self.my_index_select_embedding(self.item_mf_emb, items)

        if not self.training:  # Evaluation mode
            # (batch_size, 1, emb_size)
            u_mlp = u_mlp.view(u_mlp.size()[0], 1, -1)
            # (batch_size, 1 + num_test_neg, emb_size)
            u_mlp = u_mlp.repeat(1, v_mlp.size()[1], 1)
            # (batch_size, 1, emb_size)
            u_mf = u_mf.view(u_mf.size()[0], 1, -1)
            # (batch_size, 1 + num_test_neg, emb_size)
            u_mf = u_mf.repeat(1, v_mf.size()[1], 1)

        # The concatenated latent vector
        # mlp_vector: (batch_size, emb_size * 2) in training mode,
        # (batch_size, num_test_neg + 1, emb_size * 2) in evaluation mode
        mlp_vector = torch.cat([u_mlp, v_mlp], dim=-1)
        # Element-wise multiplication
        # mf_vector: (batch_size, emb_size) in training mode,
        # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
        mf_vector = torch.mul(u_mf, v_mf)

        # mlp_vector: (batch_size, emb_size) in training mode,
        # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
        mlp_vector = self.mlp(mlp_vector)

        if self.training:  # Training mode
            neg_v_mlp = self.my_index_select_embedding(
                self.item_mlp_emb, neg_items)
            neg_v_mf = self.my_index_select_embedding(
                self.item_mf_emb, neg_items)

            # (batch_size, 1, emb_size)
            u_mlp = u_mlp.view(u_mlp.size()[0], 1, -1)
            # (batch_size, num_neg, emb_size)
            u_mlp = u_mlp.repeat(1, neg_v_mlp.size()[1], 1)
            # (batch_size, 1, emb_size)
            u_mf = u_mf.view(u_mf.size()[0], 1, -1)
            # (batch_size, num_neg, emb_size)
            u_mf = u_mf.repeat(1, neg_v_mf.size()[1], 1)

            # The concatenated latent vector
            # neg_mlp_vector: (batch_size, num_neg, emb_size * 2)
            neg_mlp_vector = torch.cat([u_mlp, neg_v_mlp], dim=-1)
            # Element-wise multiplication
            # neg_mf_vector: (batch_size, num_neg, emb_size)
            neg_mf_vector = torch.mul(u_mf, neg_v_mf)

            # neg_mlp_vector: (batch_size, num_neg, emb_size)
            neg_mlp_vector = self.mlp(neg_mlp_vector)
            return mlp_vector, mf_vector, neg_mlp_vector, neg_mf_vector
        else:
            return mlp_vector, mf_vector
