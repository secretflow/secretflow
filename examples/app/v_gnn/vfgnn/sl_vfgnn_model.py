# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

os.environ['DGLBACKEND'] = 'pytorch'
import math  # noqa
from typing import List  # noqa

import dgl  # noqa
import dgl.nn.pytorch as dglnn  # noqa
import torch  # noqa
import torch.nn as nn  # noqa
import torch.nn.functional as F  # noqa


class SAGE(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        activation,
        batch_size,
        num_workers,
        dropout,
        aggregate,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, aggregate))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggregate))

        self.layers.append(nn.Linear(n_hidden, n_classes))
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, blocks, x):
        h = x

        for i in range(0, self.n_layers - 1):
            h = self.layers[i](blocks[i], h)
            emb = self.activation(h)
            h = self.dropout(emb)

        h = self.layers[self.n_layers - 1](h)

        return h

    def inference(self, g, x, batch_size, device):
        for _, layer in enumerate(self.layers[: len(self.layers) - 1]):
            y = torch.zeros(g.number_of_nodes(), self.n_hidden)
            embs = torch.zeros(g.number_of_nodes(), self.n_hidden)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers,
            )

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)

                emb = self.activation(h)
                h = self.dropout(emb)

                embs[output_nodes] = emb.cpu()
                y[output_nodes] = h.cpu()

            x = y

        y = self.layers[self.n_layers - 1](x.to(device))
        return y.cpu(), embs


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (
        torch.argmax(pred, dim=1) == torch.argmax(labels, dim=1)
    ).float().sum() / len(pred)


def evaluate_sage_target(model, g, inputs, labels, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with torch.no_grad():
        pred, embs = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), pred, embs


"""
Activation dict
"""


def get_activation(activation):
    _ACTIVATIONS = {'tanh': F.tanh, 'relu': F.relu, 'sigmoid': F.sigmoid}
    return _ACTIVATIONS[activation]


class VFGNN(nn.Module):
    def __init__(
        self,
        in_feats: List[int],
        n_classes,
        init_mode,
        init_model_num_hidden,
        base_model_num_hidden,
        base_model_layers,
        base_model_act,
        fuse_model_layers,
        fuse_model_num_hidden,
        fuse_model_act,
        batch_size,
        num_workers,
        dropout_rate,
        local_aggregate,
        global_aggregate,
    ):
        super().__init__()

        self.init_mode = init_mode
        self.init_model_num_hidden = init_model_num_hidden

        self.base_model_num_hidden = base_model_num_hidden
        self.base_model_layers = base_model_layers
        self.base_model_act = base_model_act
        self.local_aggregate = local_aggregate

        self.fuse_model_num_hidden = fuse_model_num_hidden
        self.fuse_model_layers = fuse_model_layers
        self.fuse_model_act = fuse_model_act
        self.global_aggregate = global_aggregate

        self.in_feats = in_feats
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(dropout_rate)

        # self.base_model_activaion = get_activation(self.base_model_act)
        # self.fuse_model_activation = get_activation(self.fuse_model_act)

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.init_model = self.get_init_model()
        self.base_models = self.get_base_models()

        self.fuse_model = self.get_fuse_model()

    def handle_data(self, dataloaders):
        self.dataloader_iters = []
        for dl in dataloaders:
            self.dataloader_iters.append(iter(dl))

        return self.dataloader_iters

    def get_init_model(self):
        init_model = InitModel(
            self.in_feats,
            self.n_classes,
            self.init_mode,
            self.init_model_num_hidden,
            self.batch_size,
            self.num_workers,
            self.dropout_rate,
        )
        return init_model

    def get_base_models(self):
        base_models = nn.ModuleList()
        for i in range(len(self.in_feats)):
            base_model = BaseModel(
                (
                    self.in_feats[i]
                    if self.init_mode == 'identity'
                    else self.init_model_num_hidden
                ),
                self.n_classes,
                self.base_model_num_hidden,
                self.base_model_layers,
                self.base_model_act,
                self.batch_size,
                self.num_workers,
                self.dropout_rate,
                self.local_aggregate,
            )
            base_models.append(base_model)
        return base_models

    def get_fuse_model(self):
        fuse_model = FuseModel(
            self.base_model_num_hidden,
            len(self.in_feats),
            self.n_classes,
            self.fuse_model_num_hidden,
            self.fuse_model_layers,
            self.fuse_model_act,
            self.batch_size,
            self.num_workers,
            self.dropout_rate,
            self.global_aggregate,
        )

        return fuse_model

    def forward(self, blocks, xs):
        # CG1: initial node embedding should be performed using MPC
        # print(f'input features num: {len(xs)}, shape: {xs[0].shape}')

        init_emb, _ = self.init_model(xs)
        # print(f'input init embedding num: {len(init_emb)}, shape: {init_emb[0].shape}')

        # CG2: local sage conv for each base model
        out_hiddens = []
        for i in range(len(self.base_models)):
            if self.init_mode == 'identity':
                h, local_emb = self.base_models[i]((blocks[i], init_emb[i]))
            elif self.init_mode == 'regression':  # each party owns the same embedding
                h, local_emb = self.base_models[i]((blocks[i], init_emb))
            else:
                raise NotImplementedError(
                    f'Init mode: {self.init_mode} not implemented!'
                )
            out_hiddens.append(h)

        # print(f'out hidden num: {len(out_hiddens)}, shape: {out_hiddens[0].shape}')

        # CG3: aggregate layer
        agg_hiddens, emb = self.fuse_model(out_hiddens)

        # print(agg_hiddens.shape)

        # CG4: output layer
        # h = self.layers[self.n_layers - 1](h)

        return agg_hiddens, emb

    def inference(self, g, x, batch_size, device):
        for _, layer in enumerate(self.layers[: len(self.layers) - 1]):
            y = torch.zeros(g.number_of_nodes(), self.n_hidden)
            embs = torch.zeros(g.number_of_nodes(), self.n_hidden)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.number_of_nodes()),
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers,
            )

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)

                emb = self.activation(h)
                h = self.dropout(emb)

                embs[output_nodes] = emb.cpu()
                y[output_nodes] = h.cpu()

            x = y

        y = self.layers[self.n_layers - 1](x.to(device))
        return y.cpu(), embs


class InitModel(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        init_mode,
        init_model_num_hidden,
        batch_size,
        num_workers,
        dropout_rate,
    ):
        super().__init__()

        self.in_feats = in_feats
        self.n_classes = n_classes

        self.init_mode = init_mode
        self.init_model_num_hidden = init_model_num_hidden

        self.dropout = nn.Dropout(dropout_rate)

        # combine outputs from init models
        self.in_feat_size = self.compute_in_feat_size()

        if self.init_mode == 'identity':
            self.layer = nn.Identity()
        elif self.init_mode == 'regression':
            print(self.in_feat_size, self.init_model_num_hidden)
            self.layer = nn.Linear(self.in_feat_size, self.init_model_num_hidden)
        else:
            raise NotImplementedError(f'Init method {self.init_mode} not implemented!')

        self.batch_size = batch_size
        self.num_workers = num_workers

    def compute_in_feat_size(self):
        if self.init_mode == 'identity':
            in_feat_size = self.in_feats[0]
        elif self.init_mode == 'regression':
            in_feat_size = 0
            for i in range(len(self.in_feats)):
                in_feat_size += self.in_feats[i]
        else:
            raise NotImplementedError(f'Init method {self.init_mode} not implemented!')

        return in_feat_size

    def forward(self, xs):
        if self.init_mode == 'identity':
            h = self.layer(xs)
        elif self.init_mode == 'regression':
            # FIXME: currently this has error since the sampled neighbours are different on each party.
            # Therefore, the joint initialization is not feasible.
            # E.g.,
            #   a in A has neighbors b, c
            #   a in B has neighbors d, e, f

            # Hence, the input shape is:
            #   A: [2, dim_a]
            #   B: [3, dim_b]

            # Ideally, input neighbors are the same, so that we can concat.
            # we want to have the input: [2, dim_a + dim_b]

            # The problme is neighbor set mismatch:
            #   1: 2 != 3
            #   2: [b, c] != [d, e, f]
            raise Exception(
                f'Init method: {self.init_mode} not supported in this case!'
            )
            # joint_input_feat = torch.concat(xs, dim=1)
            # h = self.layer(joint_input_feat)
        else:
            raise NotImplementedError(f'Init method {self.init_mode} not implemented!')

        return h, xs


class BaseModel(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        base_model_num_hidden,
        base_model_layers,
        base_model_act,
        batch_size,
        num_workers,
        dropout,
        local_aggregate,
    ):
        super().__init__()

        self.base_model_num_hidden = base_model_num_hidden
        self.base_model_layers = base_model_layers
        self.base_model_act = base_model_act
        self.local_aggregate = local_aggregate

        self.in_feats = in_feats
        self.n_classes = n_classes

        self.dropout = nn.Dropout(dropout)

        self.base_model_activaion = get_activation(self.base_model_act)

        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.SAGEConv(
                self.in_feats, self.base_model_num_hidden, self.local_aggregate
            )
        )
        for _ in range(1, self.base_model_layers):
            self.layers.append(
                dglnn.SAGEConv(
                    self.base_model_num_hidden,
                    self.base_model_num_hidden,
                    self.local_aggregate,
                )
            )

        # self.layers.append(nn.Linear(self.base_model_num_hidden, n_classes))

        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, data_x):
        blocks, x = data_x
        h = x

        for i in range(0, self.base_model_layers):
            h = self.layers[i](blocks[i], h)
            emb = self.base_model_activaion(h)
            h = self.dropout(emb)

        # h = self.layers[self.base_model_layers - 1](h)

        return h

    def output_num(self):
        return 1


class FuseModel(nn.Module):
    def __init__(
        self,
        in_feats,
        in_channel,
        n_classes,
        fuse_model_num_hidden,
        fuse_model_layers,
        fuse_model_act,
        batch_size,
        num_workers,
        dropout,
        global_aggregate,
    ):
        super().__init__()

        self.fuse_model_num_hidden = fuse_model_num_hidden
        self.fuse_model_layers = fuse_model_layers
        self.fuse_model_act = fuse_model_act
        self.global_aggregate = global_aggregate

        self.in_feats = in_feats
        self.in_channel = in_channel
        self.n_classes = n_classes

        self.dropout = nn.Dropout(dropout)

        self.fuse_model_activaion = get_activation(self.fuse_model_act)

        # combine outputs from base models
        self.in_feat_size = self.compute_in_feat_size()

        # add one layer for AGGREGATE the embeddings from all the base models
        self.fuse_model_layers = (
            self.fuse_model_layers + 1
            if self.global_aggregate == 'regression'
            else self.fuse_model_layers
        )

        self.layers = nn.ModuleList()

        self.lin_before_classifier = self.in_feat_size != self.fuse_model_num_hidden

        # sanity check
        if self.lin_before_classifier:
            assert self.fuse_model_layers > 1

        if self.fuse_model_layers > 1:
            self.layers.append(nn.Linear(self.in_feat_size, self.fuse_model_num_hidden))
            for _ in range(1, self.fuse_model_layers - 1):
                self.layers.append(
                    nn.Linear(self.fuse_model_num_hidden, self.fuse_model_num_hidden)
                )

        # only one classifier for the fuse model
        self.layers.append(nn.Linear(self.fuse_model_num_hidden, self.n_classes))

        self.batch_size = batch_size
        self.num_workers = num_workers

    def compute_in_feat_size(self):
        if self.global_aggregate == 'concat':
            # in_feat_size = 0
            # for i in range(len(self.in_feats)):
            #     in_feat_size += self.in_feats[i]
            in_feat_size = self.in_feats * self.in_channel
        elif self.global_aggregate == 'mean':
            # sanity check
            # for i in range(1, len(self.in_feats)):
            #     assert self.in_feats[i] == self.in_feats[0]
            in_feat_size = self.in_feats
        elif self.global_aggregate == 'regression':
            in_feat_size = self.in_feats * self.in_channel
        else:
            print(f'Global aggregate method {self.global_aggregate} not implemented')
            exit()

        return in_feat_size

    def forward(self, xs):
        if self.global_aggregate == 'concat':
            h = torch.concat(xs, dim=1)
        elif self.global_aggregate == 'mean':
            h = torch.mean(torch.stack(xs), dim=0)
        elif self.global_aggregate == 'regression':
            h = torch.concat(xs, dim=1)
        else:
            print(f'Global aggregate method {self.global_aggregate} not implemented')
            exit()

        for i in range(0, self.fuse_model_layers - 1):
            h = self.layers[i](h)
            h = self.fuse_model_activaion(h)
            h = self.dropout(h)

        h = self.layers[self.fuse_model_layers - 1](h)

        return h

    def output_num(self):
        return 1


def evaluate_vfgnn_target(model, in_graphs, num_layers, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    model : The input model
    in_graphs : The sub-graphs from all the parties.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """

    dataloaders = []

    for _, graph in enumerate(in_graphs):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)

        dataloader = dgl.dataloading.NodeDataLoader(
            graph,
            val_nid,
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            # num_workers=args.num_workers,
        )

        dataloaders.append(dataloader)

    iters = model.handle_data(dataloaders)

    model.eval()

    preds = []
    labels = []
    with torch.no_grad():
        steps_per_epoch = math.ceil(len(val_nid) / batch_size)
        for _ in range(steps_per_epoch):
            blocks_list = []
            inputs_list = []
            for _iter in iters:
                (input_nodes, seeds, blocks) = next(_iter)
                blocks = [block.int().to(device) for block in blocks]

                blocks_list.append(blocks)

                batch_inputs = blocks[0].srcdata['features']
                inputs_list.append(batch_inputs)

            # Load the input features as well as output labels
            # batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
            batch_labels = blocks_list[-1][-1].dstdata['labels']

            # Compute the forward propagation
            batch_pred, embs = model(blocks_list, inputs_list)

            # CG4: shall be performed on the label holder
            batch_pred = F.softmax(batch_pred, dim=1)

            preds.extend(batch_pred)
            labels.extend(batch_labels)
    model.train()
    return compute_acc(torch.stack(preds), torch.stack(labels)), preds, labels, embs
