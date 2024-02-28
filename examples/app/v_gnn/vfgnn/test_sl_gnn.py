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

# Install Python requirements
# 1. pip install dgl

# set the env
# 2. export PYTHONPATH=$PYTHONPATH:bazel-bin

# run
# 3. python tests/ml/nn/test_sl_gnn.py --mode vfgnn_train --num-epochs 5 --n-party 2 --fuse_model_layers 1 --init_mode identity

import argparse
import logging
import math
import time

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

import secretflow as sf
from examples.app.v_gnn.vfgnn.gnn_utils import (
    GraphEvalLoader,
    GraphTrainLoader,
    construct_dgl_graph_from_numpy,
    load_data_sf,
    load_data_sl,
)
from examples.app.v_gnn.vfgnn.sl_vfgnn_model import (
    SAGE,
    VFGNN,
    BaseModel,
    FuseModel,
    evaluate_sage_target,
    evaluate_vfgnn_target,
)
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.utils import (
    TorchModel,
    metric_wrapper,
    optim_wrapper,
    plot_with_tsne,
)


def local_train(dataset, args):
    """Centralized training on different partions of datasets.

    The graph is vertically splited, i.e.,
        1) Edges are evenly split;
        2) Node features are evenly split;
        3) Node set is the same among all the data holders
    """
    (
        _,
        split_adjs,
        split_feats,
        edge,
        nodes,
        labels,
        y_train,
        y_val,
        y_test,
        idx_train,
        idx_val,
        idx_test,
    ) = load_data_sl(
        dataset,
        n_party=args.n_party,
    )

    metrics = dict()

    for i in range(args.n_party + 1):
        name = "full" if i == args.n_party else "Party-" + str(i)
        print(f'Training on {name} dataset')

        # save metrics
        if name not in metrics:
            metrics[name] = dict()
            if 'train_acc' not in metrics[name]:
                metrics[name]['train_acc'] = np.array([])
            if 'eval_acc' not in metrics[name]:
                metrics[name]['eval_acc'] = np.array([])
            if 'test_acc' not in metrics[name]:
                metrics[name]['test_acc'] = np.array([])

        if i == args.n_party:  # full dataset
            edge_adj = edge
            features = nodes
        else:
            edge_adj = split_adjs[i]
            features = split_feats[i]

        # Note: only the features and edge are split, the train/val/test split are the same.
        data = (
            edge_adj,
            features,
            labels,
            y_train,
            y_val,
            y_test,
            idx_train,
            idx_val,
            idx_test,
        )

        # construct dgl graph from numpy adj_mat, raw features, raw labels, etc.
        graph, local_n_classes = construct_dgl_graph_from_numpy(data)

        # convert from bool array to node_idx array, serving as the input to dgl dataloader
        train_nid = np.array(np.where(idx_train)).squeeze(axis=0)
        val_nid = np.array(np.where(idx_val)).squeeze(axis=0)
        test_nid = np.array(np.where(idx_test)).squeeze(axis=0)

        in_feats = features.shape[1]

        # random neighbour sampler
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(',')]
        )
        dataloader = dgl.dataloading.NodeDataLoader(
            graph,
            train_nid,
            sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
        )

        # Define model and optimizer
        model = SAGE(
            in_feats,
            args.num_hidden,
            args.n_classes,
            args.num_layers,
            args.base_model_act,
            args.batch_size,
            args.num_workers,
            args.dropout,
            args.local_aggregate,
        )

        loss_fcn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Training loop
        avg = 0
        iter_tput = []
        for epoch in range(args.num_epochs):
            tic = time.time()

            # Loop over the dataloader to sample the computation dependency graph as a list of
            # blocks.
            tic_step = time.time()
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                # Load the input features as well as output labels
                # batch_inputs, batch_labels = load_subtensor(train_g, seeds, input_nodes, device)
                blocks = [block.int().to(args.device) for block in blocks]

                batch_inputs = blocks[0].srcdata['features']
                batch_labels = blocks[-1].dstdata['labels']

                # Compute loss and prediction
                batch_pred, embs = model(blocks, batch_inputs)
                batch_pred = F.softmax(batch_pred, dim=1)
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_tput.append(len(seeds) / (time.time() - tic_step))
                if epoch % args.log_every == 0:
                    # labels = labels.long()
                    acc = (
                        torch.argmax(batch_pred, dim=1)
                        == torch.argmax(batch_labels, dim=1)
                    ).float().sum() / len(batch_pred)
                    # gpu_mem_alloc = th.cuda.max_memory_allocated(
                    # ) / 1000000 if th.cuda.is_available() else 0
                    print(
                        'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f}'.format(
                            epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:])
                        )
                    )
                    metrics[name]['train_acc'] = np.append(
                        metrics[name]['train_acc'], acc
                    )

                tic_step = time.time()

            toc = time.time()

            if epoch >= 5:
                avg += toc - tic
            if epoch % args.eval_every == 0 and epoch != 0:
                eval_acc, pred, embs = evaluate_sage_target(
                    model,
                    graph,
                    graph.ndata['features'],
                    graph.ndata['labels'],
                    val_nid,
                    args.batch_size,
                    args.device,
                )
                test_acc, pred, embs = evaluate_sage_target(
                    model,
                    graph,
                    graph.ndata['features'],
                    graph.ndata['labels'],
                    test_nid,
                    args.batch_size,
                    args.device,
                )
                print('Eval Acc {:.4f}'.format(eval_acc))
                print('Test Acc: {:.4f}'.format(test_acc))

                metrics[name]['eval_acc'] = np.append(
                    metrics[name]['eval_acc'], eval_acc
                )
                metrics[name]['test_acc'] = np.append(
                    metrics[name]['test_acc'], test_acc
                )

                if epoch == args.num_epochs - args.eval_every:
                    """
                    t-SNE
                    """
                    plot_with_tsne(
                        torch.tensor(pred[test_nid]),
                        torch.tensor(graph.ndata['labels'][test_nid]),
                        "tsne-GraphSAGE",
                    )

        print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    for i in range(args.n_party + 1):
        name = "full" if i == args.n_party else "Party-" + str(i)
        print(f' === {dataset} : {name} ===')
        print(f'\033[1;31m Eval acc: {metrics[name]["eval_acc"]} \033[0m')
        print(f'\033[1;31m Test acc: {metrics[name]["test_acc"]} \033[0m')


def local_vfgnn_train(dataset, args):
    """VFGNN training on different partions of datasets.

    The graph is vertically splited, i.e.,
        1) Edges are evenly split;
        2) Node features are evenly split;
        3) Node set is the same among all the data holders
    """
    (
        _,
        split_adjs,
        split_feats,
        edge,
        nodes,
        labels,
        y_train,
        y_val,
        y_test,
        idx_train,
        idx_val,
        idx_test,
    ) = load_data_sl(
        dataset,
        n_party=args.n_party,
    )

    # save metrics
    metrics = dict()
    metrics_names = ['train_acc', 'eval_acc', 'test_acc']
    for metrics_name in metrics_names:
        if metrics_name not in metrics:
            metrics[metrics_name] = np.array([])

    # convert from bool array to node_idx array, serving as the input to dgl dataloader
    train_nid = np.array(np.where(idx_train)).squeeze(axis=0)
    val_nid = np.array(np.where(idx_val)).squeeze(axis=0)
    test_nid = np.array(np.where(idx_test)).squeeze(axis=0)

    in_feats = []
    in_graphs = []
    # construct dataloader for each singe party
    dataloaders = []

    for i in range(args.n_party):
        edge_adj = split_adjs[i]
        features = split_feats[i]
        # TODO: hack this part for debugging
        # features = nodes

        # Note: only the features and edge are split, the train/val/test split are the same.
        data = (
            edge_adj,
            features,
            labels,
            y_train,
            y_val,
            y_test,
            idx_train,
            idx_val,
            idx_test,
        )

        in_feats.append(features.shape[1])

        # construct dgl graph from numpy adj_mat, raw features, raw labels, etc.
        graph, local_n_classes = construct_dgl_graph_from_numpy(data)
        in_graphs.append(graph)

        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(',')]
        )

        dataloader = dgl.dataloading.NodeDataLoader(
            graph,
            train_nid,
            sampler,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        )

        dataloaders.append(dataloader)

    # Define model and optimizer
    model = VFGNN(
        in_feats,
        args.n_classes,
        args.init_mode,
        args.init_model_num_hidden,
        args.base_model_num_hidden,
        args.base_model_layers,
        args.base_model_act,
        args.fuse_model_layers,
        args.fuse_model_num_hidden,
        args.fuse_model_act,
        args.batch_size,
        args.num_workers,
        args.dropout,
        args.local_aggregate,
        args.global_aggregate,
    )

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()

        # handle input data
        iters = model.handle_data(dataloaders)

        steps_per_epoch = math.ceil(len(train_nid) / args.batch_size)
        for step in range(steps_per_epoch):
            blocks_list = []
            inputs_list = []
            for _iter in iters:
                (input_nodes, seeds, blocks) = next(_iter)
                blocks = [block.int().to(args.device) for block in blocks]

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
            loss = loss_fcn(batch_pred, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if epoch % args.log_every == 0:
                acc = (
                    torch.argmax(batch_pred, dim=1) == torch.argmax(batch_labels, dim=1)
                ).float().sum() / len(batch_pred)
                print(
                    'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f}'.format(
                        epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:])
                    )
                )
                metrics['train_acc'] = np.append(metrics['train_acc'], acc)

            tic_step = time.time()

        toc = time.time()
        # print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc, pred, _, embs = evaluate_vfgnn_target(
                model,
                in_graphs,
                args.base_model_layers,
                val_nid,
                args.batch_size,
                args.device,
            )
            test_acc, pred, label, embs = evaluate_vfgnn_target(
                model,
                in_graphs,
                args.base_model_layers,
                test_nid,
                args.batch_size,
                args.device,
            )
            print('Eval Acc {:.4f}'.format(eval_acc))
            print('Test Acc: {:.4f}'.format(test_acc))

            metrics['eval_acc'] = np.append(metrics['eval_acc'], eval_acc)
            metrics['test_acc'] = np.append(metrics['test_acc'], test_acc)

            if epoch == args.num_epochs - args.eval_every:
                """
                t-SNE
                """
                plot_with_tsne(torch.stack(pred), torch.stack(label), "tsne-VFGNN")

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    print(f'\033[1;31m Eval acc: {metrics["eval_acc"]} \033[0m')
    print(f'\033[1;31m Test acc: {metrics["test_acc"]} \033[0m')


def vfgnn_train(dataset, args):
    # In case you got a running secetflow runtime already.
    sf.shutdown()

    parties = [f"party-{i}" for i in range(args.n_party)]
    sf.init(parties=parties, address='local', logging_level='error')

    pyu_devices = [sf.PYU(party) for party in parties]

    """
    Step-1:
        assign each party with:
            1) separate graph and node features
            2) unified train_idx, val_idx, test_idx
            3) separate labels held by only one party
    """
    (
        split_adjs,
        split_feats,
        y_train,
        y_val,
        y_test,
        idx_train,
        idx_val,
        idx_test,
    ) = load_data_sf(
        dataset,
        parties=pyu_devices,
    )

    """
    Additioanl Step: Test the effect of introducing DP
    """
    from secretflow.security.privacy import DPStrategy
    from secretflow.security.privacy.mechanism.torch import GaussianEmbeddingDP

    # Define DP operations
    gaussian_embedding_dp = GaussianEmbeddingDP(
        noise_multiplier=0.5,
        l2_norm_clip=1.0,
        batch_size=args.batch_size,
        num_samples=y_train.partition_shape()[pyu_devices[0]][0],
        is_secure_generator=False,
    )

    dp_strategy_alice = DPStrategy(embedding_dp=gaussian_embedding_dp)
    dp_strategy_bob = DPStrategy(embedding_dp=gaussian_embedding_dp)

    dp_strategy_dict = {
        pyu_devices[0]: dp_strategy_alice,
        pyu_devices[1]: dp_strategy_bob,
    }

    """
    Additioanl Step: Compressor effect
    """
    from secretflow.utils.compressor import TopkSparse

    top_k_compressor = TopkSparse(0.3)

    """
    Step-2:
        Construct the model
    """
    partition_shapes = split_feats.partition_shape()
    in_feats = []
    for party in pyu_devices:
        in_feats.append(partition_shapes[party][1])

    def create_base_model(args, in_feat):
        # Compile model
        loss_fn = nn.CrossEntropyLoss

        model_def = TorchModel(
            model_fn=BaseModel,
            loss_fn=loss_fn,
            optim_fn=optim_wrapper(optim.Adam, lr=args.lr),
            metrics=[
                metric_wrapper(
                    torchmetrics.Accuracy,
                    task="multiclass",
                    num_classes=args.n_classes,
                    average='micro',
                ),
                metric_wrapper(
                    torchmetrics.Precision,
                    task="multiclass",
                    num_classes=args.n_classes,
                    average='micro',
                ),
            ],
            in_feats=(
                in_feat if args.init_mode == 'identity' else args.init_model_num_hidden
            ),
            n_classes=args.n_classes,
            base_model_num_hidden=args.base_model_num_hidden,
            base_model_layers=args.base_model_layers,
            base_model_act=args.base_model_act,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dropout=args.dropout,
            local_aggregate=args.local_aggregate,
        )
        return model_def

    def create_fuse_model(args):
        # Compile model
        loss_fn = nn.CrossEntropyLoss

        model_def = TorchModel(
            model_fn=FuseModel,
            loss_fn=loss_fn,
            optim_fn=optim_wrapper(optim.Adam, lr=args.lr),
            metrics=[
                metric_wrapper(
                    torchmetrics.Accuracy,
                    task='multiclass',
                    num_classes=args.n_classes,
                    average='micro',
                ),
                metric_wrapper(
                    torchmetrics.Precision,
                    task="multiclass",
                    num_classes=args.n_classes,
                    average='micro',
                ),
                metric_wrapper(
                    torchmetrics.AUROC,
                    task="multiclass",
                    num_classes=args.n_classes,
                ),
            ],
            in_feats=args.base_model_num_hidden,
            in_channel=args.n_party,
            n_classes=args.n_classes,
            fuse_model_num_hidden=args.fuse_model_num_hidden,
            fuse_model_layers=args.fuse_model_layers,
            fuse_model_act=args.fuse_model_act,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dropout=args.dropout,
            global_aggregate=args.global_aggregate,
        )

        return model_def

    base_model_def0 = create_base_model(args, in_feats[0])
    base_model_def1 = create_base_model(args, in_feats[1])
    fuse_model_def = create_fuse_model(args)

    print(base_model_def0)
    print(base_model_def1)
    print(fuse_model_def)

    base_model_def0.model_fn(**base_model_def0.kwargs)
    fuse_model_def.model_fn(**fuse_model_def.kwargs)

    sl_model = SLModel(
        base_model_dict={
            pyu_devices[0]: base_model_def0,
            pyu_devices[1]: base_model_def1,
        },
        device_y=pyu_devices[0],
        model_fuse=fuse_model_def,
        backend='torch',
        dp_strategy_dict=dp_strategy_dict,
        compressor=top_k_compressor,
    )

    """
    Step-3:
        Fit the model by `epochs`
    """
    print(f'===== Fit the model =====')

    def create_dataset_builder(
        fanout=[10, 15],
        batch_size=256,
        shuffle=False,
        drop_last=False,
        seed=321,
    ):
        def dataset_builder(x):
            return GraphTrainLoader(x, fanout, batch_size, shuffle, drop_last)

        return dataset_builder

    dataloader_train = create_dataset_builder(batch_size=args.batch_size)

    sl_model.fit(
        x=[split_feats, split_adjs, idx_train],
        y=y_train,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        sample_weight=idx_train,
        dataset_builder={
            pyu_devices[0]: dataloader_train,
            pyu_devices[1]: dataloader_train,
        },
        validation_data=([split_feats, split_adjs, idx_val], y_val, idx_val),
    )

    """
    Step-4:
        Evaluate the trained model using test dataset
    """
    print(f'===== Evaluate the model =====')

    def create_dataset_builder_infer(
        fanout=[10, 15],
        batch_size=256,
        shuffle=False,
        drop_last=False,
        seed=321,
    ):
        def dataset_builder(x):
            return GraphEvalLoader(x, fanout, batch_size, shuffle, drop_last)

        return dataset_builder

    dataloader_eval = create_dataset_builder_infer(batch_size=args.batch_size)

    sl_model.evaluate(
        x=[split_feats, split_adjs, idx_test],
        y=y_test,
        batch_size=args.batch_size,
        # sample_weight=idx_test,
        dataset_builder={
            pyu_devices[0]: dataloader_eval,
            pyu_devices[1]: dataloader_eval,
        },
    )

    """
    Step-5:
        Predict the test dataset
    """
    print(f'===== Prediction on the model =====')
    from secretflow.data.ndarray import FedNdarray, PartitionWay

    idx_infer_arr = FedNdarray(
        partitions={party: party(lambda: None)() for party in pyu_devices},
        partition_way=PartitionWay.HORIZONTAL,
    )
    dataloader_pred = create_dataset_builder_infer(batch_size=args.batch_size)

    pred_y = sl_model.predict(
        x=[split_feats, split_adjs, idx_infer_arr],
        batch_size=args.batch_size,
        dataset_builder={
            pyu_devices[0]: dataloader_pred,
            pyu_devices[1]: dataloader_pred,
        },
    )
    print(pred_y)

    """
    Step-6:
        Save/load the model
    """
    print(f'===== Save/Load the model =====')
    # sl_model.save_model("./base_mode.pth", "./fuse_model.pth")

    # sl_model.load_model("./base_mode.pth", "./fuse_model.pth")

    """
    Step-7:
        Export the model
    """
    print(f'===== Export the model =====')
    print(f'Currently not support export model to Torch, ONNX')
    # sl_model.export_model("~/base_mode", "~/fuse_model", save_format='onnx')


def parse_args():
    argparser = argparse.ArgumentParser("VFGNN training")
    # argparser.add_argument('--gpu', type=int, default=1,
    #                     help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--target-model', type=str, default='gat')
    argparser.add_argument(
        '--dataset',
        type=str,
        nargs='+',
        default=['cora'],
        help="['cora', 'pubmed', 'citeseer']",
    )
    argparser.add_argument('--num-epochs', type=int, default=30)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=256)
    argparser.add_argument('--val-batch-size', type=int, default=256)
    argparser.add_argument('--log-every', type=int, default=10)
    argparser.add_argument('--eval-every', type=int, default=50)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument(
        '--inductive', action='store_true', help="Inductive learning setting"
    )
    argparser.add_argument('--graphgallery', action='store_true', default=False)
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--head', type=int, default=4)
    argparser.add_argument('--wd', type=float, default=0)

    argparser.add_argument(
        '--mode',
        type=str,
        default='vfgnn_train',
        help="['local_train', 'local_vfgnn_train', 'vfgnn_train']",
    )
    argparser.add_argument('--n-party', type=int, default=2)
    argparser.add_argument('--device', type=str, default='cpu')
    argparser.add_argument(
        '--local_aggregate',
        type=str,
        default='mean',
        help="['mean', 'gcn', 'pool', 'lstm']",
    )
    argparser.add_argument(
        '--global_aggregate',
        type=str,
        default='mean',
        help="['mean', 'concat', 'regression']",
    )
    argparser.add_argument(
        '--init_mode', type=str, help='["identity, regression"]', default='identity'
    )
    argparser.add_argument(
        '--init_model_num_hidden', type=int, help='[128, 256]', default=256
    )

    argparser.add_argument('--base_model_layers', type=int, default=2)
    argparser.add_argument('--fuse_model_layers', type=int, default=1)
    argparser.add_argument(
        '--base_model_num_hidden', type=int, help="[32, 64, 128, 256]", default=128
    )
    argparser.add_argument(
        '--fuse_model_num_hidden', type=int, help="[32, 64, 128, 256]", default=128
    )
    argparser.add_argument('--base_model_act', type=str, default='tanh')
    argparser.add_argument('--fuse_model_act', type=str, default='sigmoid')

    argparser.add_argument(
        '--depth', type=int, help='The depth of the model structure', default='4'
    )

    # attack parameters
    # [LSA, LinkTeller, Our-attack targeted on SL]
    argparser.add_argument('--attack', action='store_true', default=False)
    argparser.add_argument(
        '--attack-mode',
        type=str,
        help='["LSA", "LinkTeller"]',
        default='LSA',
    )

    # defense parameters
    # [DP, sparsity, async split learning]
    args, _ = argparser.parse_known_args()
    return args


def main():
    args = parse_args()
    n_class_dcit = {'cora': 7, 'pubmed': 3, 'citeseer': 6}
    if args.mode == 'local_train':
        logging.info("local sage train")
        for ds in args.dataset:
            # manually set the class number
            args.n_classes = n_class_dcit[ds]
            local_train(ds, args)
    elif args.mode == 'local_vfgnn_train':
        logging.info("local vfgnn train")
        for ds in args.dataset:
            # manually set the class number
            args.n_classes = n_class_dcit[ds]
            local_vfgnn_train(ds, args)
    elif args.mode == 'vfgnn_train':
        logging.info("vfgnn train")
        for ds in args.dataset:
            # manually set the class number
            args.n_classes = n_class_dcit[ds]
            vfgnn_train(ds, args)


if __name__ == '__main__':
    main()
