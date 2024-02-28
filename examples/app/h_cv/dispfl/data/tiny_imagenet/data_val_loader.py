# MIT License
#
# Copyright (c) 2022 Rong Dai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import math
import random

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import tiny, tiny_truncated


def record_net_data_stats(y_train, net_dataidx_map, logger):
    net_cls_counts = []

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = []
        for i in range(200):
            if i in unq:
                tmp.append(unq_cnt[np.argwhere(unq == i)][0, 0])
            else:
                tmp.append(0)
        net_cls_counts.append(tmp)
    logger.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def record_part(y_test, train_cls_counts, test_dataidxs, logger):
    test_cls_counts = []

    for net_i, dataidx in enumerate(test_dataidxs):
        unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
        tmp = []
        for i in range(200):
            if i in unq:
                tmp.append(unq_cnt[np.argwhere(unq == i)][0, 0])
            else:
                tmp.append(0)
        test_cls_counts.append(tmp)
        logger.debug(
            'DATA Partition: Train %s; Test %s'
            % (str(train_cls_counts[net_i]), str(tmp))
        )
    # logger.debug('Data statistics: %s' % str(net_cls_counts))
    return


def _data_transforms_tiny():
    CIFAR_MEAN = [0.5, 0.5, 0.5]
    CIFAR_STD = [0.5, 0.5, 0.5]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    # train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    return train_transform, valid_transform


def load_tiny_data(datadir):
    train_transform, test_transform = _data_transforms_tiny()

    tiny_train_ds = tiny_truncated(
        datadir, train=True, download=True, transform=train_transform
    )
    tiny_test_ds = tiny_truncated(
        datadir, train=False, download=True, transform=test_transform
    )

    X_train, y_train = tiny_train_ds.data, tiny_train_ds.target
    X_test, y_test = tiny_test_ds.data, tiny_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data(datadir, partition, n_nets, alpha, logger):
    logger.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_tiny_data(datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 200
        N = y_train.shape[0]
        logger.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 200:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / n_nets)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == 'dir':
        n_client = n_nets
        n_cls = 200

        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(
            mean=np.log(n_data_per_clnt), sigma=0, size=n_client
        )
        clnt_data_list = (
            clnt_data_list / np.sum(clnt_data_list) * len(y_train)
        ).astype(int)
        cls_priors = np.random.dirichlet(alpha=[alpha] * n_cls, size=n_client)
        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            # If current node is full resample a client
            # print('Remaining Data: %d' %np.sum(clnt_data_list))
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if trn_y is out of that class
                if cls_amount[cls_label] <= 0:
                    # cls_amount[cls_label] = np.random.randint(0, len(idx_list[cls_label]))
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(
                    idx_list[cls_label][cls_amount[cls_label]]
                )

                break

    elif partition == 'n_cls':
        n_client = n_nets
        n_cls = 200

        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(
            mean=np.log(n_data_per_clnt), sigma=0, size=n_client
        )
        clnt_data_list = (
            clnt_data_list / np.sum(clnt_data_list) * len(y_train)
        ).astype(int)
        cls_priors = np.zeros(shape=(n_client, n_cls))
        for i in range(n_client):
            cls_priors[i][random.sample(range(n_cls), int(alpha))] = 1.0 / alpha

        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            # If current node is full resample a client
            # print('Remaining Data: %d' %np.sum(clnt_data_list))
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if trn_y is out of that class
                if cls_amount[cls_label] <= 0:
                    cls_amount[cls_label] = np.random.randint(
                        0, len(idx_list[cls_label])
                    )
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(
                    idx_list[cls_label][cls_amount[cls_label]]
                )

                break

    elif partition == 'my_part':
        n_shards = alpha
        n_client = n_nets
        n_cls = 200

        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(
            mean=np.log(n_data_per_clnt), sigma=0, size=n_client
        )
        clnt_data_list = (
            clnt_data_list / np.sum(clnt_data_list) * len(y_train)
        ).astype(int)
        cls_priors = np.zeros(shape=(n_client, n_cls))
        # default my_part 内部使用dirichlet=0.3的分布
        cls_priors_tmp = np.random.dirichlet(alpha=[0.3] * n_cls, size=int(n_shards))

        for i in range(n_client):
            cls_priors[i] = cls_priors_tmp[int(i / n_shards)]

        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            # If current node is full resample a client
            # print('Remaining Data: %d' %np.sum(clnt_data_list))
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if trn_y is out of that class
                if cls_amount[cls_label] <= 0:
                    cls_amount[cls_label] = np.random.randint(
                        0, len(idx_list[cls_label])
                    )
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(
                    idx_list[cls_label][cls_amount[cls_label]]
                )

                break

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logger)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


def get_dataloader_tiny(
    datadir,
    train_bs,
    test_bs,
    dataidxs=None,
    test_idxs=None,
    cache_train_data_set=None,
    cache_test_data_set=None,
):
    transform_train, transform_test = _data_transforms_tiny()

    dataidxs = np.array(dataidxs)
    # rand_perm = np.random.permutation(len(dataidxs))
    # train_dataidxs=dataidxs[rand_perm[:int(len(dataidxs) * 0.8)]]

    # logger.info("train_num{}  test_num{}".format(len(dataidxs),len(test_idxs)))
    train_ds = tiny_truncated(
        datadir,
        dataidxs=dataidxs,
        train=True,
        transform=transform_train,
        download=True,
        cache_data_set=cache_train_data_set,
    )
    # test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True,cache_data_set=cache_test_data_set)
    test_ds = tiny_truncated(
        datadir,
        dataidxs=test_idxs,
        train=False,
        transform=transform_test,
        download=True,
        cache_data_set=cache_test_data_set,
    )
    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False
    )
    test_dl = data.DataLoader(
        dataset=test_ds, batch_size=test_bs, shuffle=True, drop_last=False
    )
    # logger.info("train_loader{}  test_loader{}".format(len(train_dl), len(test_dl)))
    return train_dl, test_dl


def load_partition_data_tiny(
    data_dir, partition_method, partition_alpha, client_number, batch_size, logger
):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        traindata_cls_counts,
    ) = partition_data(
        data_dir, partition_method, client_number, partition_alpha, logger
    )
    # class_num = len(np.unique(y_train))
    # logger.info("traindata_cls_counts = " + str(traindata_cls_counts))
    # train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    #
    # train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    # logger.info("train_dl_global number = " + str(len(train_data_global)))
    # logger.info("test_dl_global number = " + str(len(test_data_global)))
    # test_data_num = len(test_data_global)

    val_dataidx_map = []
    for i in range(len(net_dataidx_map)):
        val_dataidx_map.append(
            random.sample(net_dataidx_map[i], int(0.1 * len(net_dataidx_map[0])))
        )
        net_dataidx_map[i] = list(set(net_dataidx_map[i]) - set(val_dataidx_map[i]))
    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    val_data_local_dict = dict()
    transform_train, transform_test = _data_transforms_tiny()

    cache_train_data_set = tiny(
        data_dir, train=True, transform=transform_train, download=True
    )
    cache_test_data_set = tiny(
        data_dir, train=False, transform=transform_test, download=True
    )
    idx_test = [[] for i in range(200)]
    # checking
    for label in range(200):
        idx_test[label] = np.where(y_test == label)[0]
    test_dataidxs = [[] for i in range(client_number)]
    tmp_tst_num = math.ceil(len(cache_test_data_set) / client_number)
    for client_idx in range(client_number):
        for label in range(200):
            # each has 2000 pieces of testing data
            label_num = math.ceil(
                traindata_cls_counts[client_idx][label]
                / sum(traindata_cls_counts[client_idx])
                * tmp_tst_num
            )
            rand_perm = np.random.permutation(len(idx_test[label]))
            if len(test_dataidxs[client_idx]) == 0:
                test_dataidxs[client_idx] = idx_test[label][rand_perm[:label_num]]
            else:
                test_dataidxs[client_idx] = np.concatenate(
                    (test_dataidxs[client_idx], idx_test[label][rand_perm[:label_num]])
                )
        dataidxs = net_dataidx_map[client_idx]
        val_dataidxs = val_dataidx_map[client_idx]
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader_tiny(
            data_dir,
            batch_size,
            batch_size,
            dataidxs,
            test_dataidxs[client_idx],
            cache_train_data_set=cache_train_data_set,
            cache_test_data_set=cache_test_data_set,
        )
        val_data_local, _ = get_dataloader_tiny(
            data_dir,
            batch_size,
            batch_size,
            val_dataidxs,
            test_dataidxs[client_idx],
            cache_train_data_set=cache_train_data_set,
            cache_test_data_set=cache_test_data_set,
        )
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        logger.info(
            "client_idx = %d, local_train_number = %d, local_val_number = %d, local_test_number = %d"
            % (
                client_idx,
                local_data_num,
                len(val_data_local.dataset),
                len(test_data_local.dataset),
            )
        )
        # logger.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        val_data_local_dict[client_idx] = val_data_local
    record_part(y_test, traindata_cls_counts, test_dataidxs, logger)
    # test= [0 for i in range(200000)]
    # for idx in test_dataidxs:
    #     for value in idx:
    #         test[value]+=1
    # print(np.count_nonzero(test))
    return (
        None,
        None,
        None,
        None,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
        traindata_cls_counts,
    )
