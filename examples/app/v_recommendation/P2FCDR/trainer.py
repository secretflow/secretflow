# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.p2fcdr.p2fcdr_model import P2FCDR
from utils import train_utils
from losses import Discriminator


class Trainer(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def train_batch(self, *args, **kwargs):
        raise NotImplementedError

    def test_batch(self, *args, **kwargs):
        raise NotImplementedError



class ModelTrainer(Trainer):
    def __init__(self, args, num_users, num_items):
        self.args = args
        self.method = args.method
        self.device = "cpu"

        self.model = P2FCDR(num_users, num_items, args).to(self.device)
        self.U_mlp, self.U_mf, self.U_mlp_g, self.U_mf_g \
            = [None], [None], [None], [None]

        from models.mf import config
        self.discri = Discriminator(config.emb_size).to(self.device)

        self.params = list(self.model.parameters()) + \
                list(self.discri.parameters())


        self.optimizer = train_utils.get_optimizer(
                args.optimizer, self.params, args.lr)

        self.step = 0

    def train_batch(self, users, interactions, round, args,
                    UU_adj=None, VV_adj=None, M=None, perturb_UU_adj=None,
                    all_adj=None,
                    zeta=None, tilde_u_mu=None, tilde_u_sigma=None,
                    global_params=None):


        self.optimizer.zero_grad()


        self.U_mlp[0], self.U_mf[0] = self.model.get_user_embeddings()


        users = torch.LongTensor(users).to(self.device)
        interactions = [torch.LongTensor(x).to(
            self.device) for x in interactions]


        items, neg_items = interactions
        # mlp_vector, mf_vector: (batch_size, emb_size)
        # neg_mlp_vector, neg_mf_vector: (batch_size, num_neg, emb_size)
        mlp_vector, mf_vector, neg_mlp_vector, neg_mf_vector = \
            self.model(users, items, neg_items,
                        U_mlp_g=self.U_mlp_g[0], U_mf_g=self.U_mf_g[0])
        loss = self.p2fcdr_loss_fn(
            mlp_vector, mf_vector, neg_mlp_vector, neg_mf_vector)
        if "Fed" in self.method and args.mu:
            loss += self.prox_reg(
                [dict(self.model.encoder.named_parameters())],
                global_params, args.mu)

        loss.backward()
        self.optimizer.step()
        self.step += 1
        return loss.item()


    def p2fcdr_loss_fn(self, mlp_vector, mf_vector,
                       neg_mlp_vector, neg_mf_vector):
        pos_score = self.discri(mlp_vector, mf_vector)  # (batch_size, )
        neg_score = self.discri(neg_mlp_vector, neg_mf_vector)

        loss = -F.logsigmoid(pos_score).mean() \
            - F.logsigmoid(-neg_score).mean(dim=1).mean() \

        return loss

    @ staticmethod
    def flatten(source):
        return torch.cat([value.flatten() for value in source])

    def prox_reg(self, params1, params2, mu):
        params1_values, params2_values = [], []
        # Record the model parameter aggregation results of each branch
        # separately
        for branch_params1, branch_params2 in zip(params1, params2):
            branch_params2 = [branch_params2[key]
                              for key in branch_params1.keys()]
            params1_values.extend(branch_params1.values())
            params2_values.extend(branch_params2)

        # Multidimensional parameters should be compressed into one dimension
        # using the flatten function
        s1 = self.flatten(params1_values)
        s2 = self.flatten(params2_values)
        return mu/2 * torch.norm(s1 - s2)

    def test_batch(self, users, interactions):
        """Tests the model for one batch.

        Args:
            users: Input user IDs.
            interactions: Input user interactions.
        """
        users = torch.LongTensor(users).to(self.device)
        interactions = [torch.LongTensor(x).to(
            self.device) for x in interactions]

        # items: (batch_size, )
        # neg_items: (batch_size, num_test_neg)
        items, neg_items = interactions
        # all_items: (batch_size, num_test_neg + 1)
        # Note that the elements in the first column are the positive samples.
        all_items = torch.hstack([items.reshape(-1, 1), neg_items])

        mlp_vector, mf_vector = self.model(users, all_items)

        result = self.discri(mlp_vector, mf_vector)


        # (batch_size, num_test_neg + 1)
        result = result.view(result.size()[0],
                             result.size()[1])

        pred = []
        for score in result:
            # score:  (num_test_neg + 1)
            # Note that the first one is the positive sample.
            # `(-score).argsort().argsort()` indicates where the elements at
            # each position are ranked in the list of logits in descending
            # order (since `argsort()` defaults to ascending order, we use
            # `-score` here). Since the first one is the positive sample,
            # then `...[0].item()` indicates the ranking of the positive
            # sample.
            rank = (-score).argsort().argsort()[0].item()
            pred.append(rank + 1)  # `+1` makes the ranking start from 1

        return pred