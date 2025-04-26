import torch
# from torchinfo import summary
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
import warnings
import matplotlib
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from itertools import compress
from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
from PIL import Image
from pathlib import Path
import glob


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


"""
helper functions
"""


def autoassign(lcls):
    """
        Map all inputs to class attributes.
        Reference: https://stackoverflow.com/questions/3652851/what-is-the-best-way-to-do-automatic-attribute-assignment-in-python-and-is-it-a
    """
    for key in lcls.keys():
        if key != "self":
            # flattern kwargs
            if key == 'kwargs':
                if key in lcls["self"].__dict__:
                    for k in lcls["self"].__dict__[key]:
                        lcls["self"].__dict__[
                            k] = lcls["self"].__dict__[key][k]
            else:
                lcls["self"].__dict__[key] = lcls[key]


def calculate_model_size(model_state_dict):
    """Show model size in MB"""
    # mem_params = sum([param.nelement() * param.element_size()
    #                   for param in model.parameters()])
    # mem_bufs = sum([buf.nelement() * buf.element_size()
    #                 for buf in model.buffers()])
    # mem = mem_params + mem_bufs  # in bytes
    mdict = model_state_dict
    mem = sum([mdict[key].nelement() * mdict[key].element_size()
               for key in mdict.keys()])
    return mem * 1e-6


def calculate_flops(model, inputs_size, device):
    """inputs_size: bacth size 1 input"""
    stat = summary(model, inputs_size, verbose=0, device=device)
    return stat.total_mult_adds


def save_to_pkl(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_from_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def mkdirs(dirpath):
    if not os.path.exists(dirpath):
        # multi-threading
        os.makedirs(dirpath, exist_ok=True)


def access_last_added_element(ordered_dict):
    """
        next(reversed(ordered_dict)) returns the last added key
    """
    try:
        key = next(reversed(ordered_dict))
        return ordered_dict[key]
    except StopIteration:
        # print("The OrderedDict is empty.")
        return None


from torch import nn


class Initializer:
    """
        ref: 
        1. https://github.com/3ammor/Weights-Initializer-pytorch/blob/master/weight_initializer.py
        2. https://github.com/kevinzakka/pytorch-goodies
    """

    def __init__(self):
        pass

    @staticmethod
    def initialize(model, initialization, **kwargs):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.Linear):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass
        model.apply(weights_init)


"""
Split Datasets

References:
1. https://github.com/Xtra-Computing/NIID-Bench
"""


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.targets = dataset.targets[self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return image, label


def remove_by_class(trainset, list_of_classes_to_remove):
    for cls in list_of_classes_to_remove:
        selected = trainset.targets != cls
        trainset.idxs = list(compress(trainset.idxs, selected))
        trainset.targets = trainset.dataset.targets[trainset.idxs]
    return trainset


def split_trainset_by_class(client_trainset):
    """
    Input: client_trainset, which is an object of DatasetSplit class
    Return: a dictionary of trainset, where the key is the label while the value is an object of the DatasetSplit class
    """
    all_classes = torch.unique(client_trainset.targets).tolist()
    class_dataset_dict = {}
    for c in all_classes:
        selected = client_trainset.targets == c
        idx_c = list(compress(client_trainset.idxs, selected))
        class_dataset_dict[c] = DatasetSplit(client_trainset.dataset, idx_c)
    return class_dataset_dict


def sampler(dataset, num_clients, partition, seed=None, minsize=10, **kwargs):
    """
        dataset: torch.utils.data.Dataset object
        partition:
            iid-equal-size: 
                uniformly randomly sample from the whole datasets and each party approximately has the same number of samples.

            iid-diff-size: 
                uniformly randomly sample from the whole datasets but each party approximately has different number of samples; 
                should also set the beta parameter.

            noniid-label-quantity:
                Each client will only contain `num_classes_per_client` classes of samples; for any two clients that have the same class, the samples will not overlap;
                should also set the num_class, num_classes_per_client, and ylabels parameters.
                samples in each classes are uniformly diveded. But this could still lead to class imbalance in each client.
                See https://arxiv.org/pdf/2102.02079.pdf a) Quantity-based label imbalance

            noniid-label-distribution:
                The number of classes per client own follow a dirichlet distribution with concentration parameter beta.
                should also set the num_class, beta, and ylabels parameters.

            shards:
                Suppose there are (n clients, c classes, N datapoints) and each clients own s shards of data. Then the 
                number of data per shard is  size_s = N / (n * s). And each class is split into c/size_s shards.
                Shards are randomly assigned to clients.

        kwargs:
            beta: concentration parameter for the **symmetric** Dirichlet distribution; float, larger than 0
            ylabels: 1d tensor of size as the len(dataset)
            num_class: int; larger than 0
            num_classes_per_client: int;  larger than 0 smaller than total number of classes in ylabels           

        Return: a dict; {cid: torch.utils.data.Dataset object}

        --- Notes ---
        Effect of the beta parameter:
            When beta = 1, the symmetric Dirichlet distribution is equivalent to a uniform distribution over the open standard (K − 1)-simplex, (the distribution over distributions is uniform)
            When beta > 1, it prefers variates that are dense, evenly distributed distributions, i.e. all the values within a single sample are similar to each other. 
            When beta < 1, it prefers sparse distributions, i.e. most of the values within a single sample will be close to 0, and the vast majority of the mass will be concentrated in a few of the values.

        --- References ---
        1. https://en.wikipedia.org/wiki/Dirichlet_distribution#The_concentration_parameter
    """
    # process arguments
    if partition in ['iid-diff-size', 'noniid-label-distribution']:
        if 'beta' not in kwargs:
            beta = 0.5
            warnings.warn(
                f"partition:{partition} | beta is not provided. Set to 0.5.")
        else:
            beta = kwargs['beta']
            temp = beta.split('b')
            if len(temp) == 1:
                beta = float(temp[0])
                is_balanced = False
            elif len(temp) == 2:
                beta = float(temp[0])
                is_balanced = True
            assert beta > 0, "beta needs to be non-negative"
    if partition == 'shards':
        if 'num_shards_per_client' not in kwargs:
            raise ValueError(
                f"The num_shards_per_client parameter needs to be set for the partition {partition}.")
        else:
            num_classes = kwargs['num_classes']

    if partition in ['noniid-label-quantity', 'noniid-label-distribution']:
        if 'num_classes' not in kwargs:
            raise ValueError(
                f"The num_classes parameter needs to be set for the partition {partition}.")
        else:
            num_classes = kwargs['num_classes']
        try:
            num_unique_class = len(torch.unique(dataset.targets))
        except TypeError:
            print('dataset.targets is not of tensor type! Proper actions are required.')
            exit()
        assert num_classes == num_unique_class, f"num_classes is set to {num_classes}, but number of unique class detected in ylables are {num_unique_class}."
        if 'ylabels' not in kwargs:
            raise ValueError(
                f"The ylabels parameter needs to be set for the partition {partition}.")
        else:
            ylabels = kwargs['ylabels']
    if seed is not None:
        np.random.seed(seed)
    num_samples = len(dataset)
    idxs = np.random.permutation(num_samples)
    cur_minsize = 0
    attemp = 0
    max_attemp = 3
    stats_dict = {}
    if partition == 'iid-equal-size':
        batch_idxs = np.array_split(idxs, num_clients)
        if len(batch_idxs[-1]) < minsize:
            warnings.warn(
                f"partition:{partition} | Some clients have less than {minsize} samples. Check it before continue.")
        cid_idxlst_dict = {
            cid: batch_idxs[cid].tolist() for cid in range(num_clients)}
    elif partition == 'iid-diff-size':
        """
        The number of samples per client follow a dirichlet distribution with concentration parameter beta.
        But the number of samples per classes in each client are approxumately the same
        """
        while cur_minsize < minsize:
            attemp += 1
            if attemp == max_attemp:
                raise RuntimeError(
                    f"partition:{partition} | Exceeds max allowed attempts. Consider change the random seed.")
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            proportions = proportions / proportions.sum()
            cur_minsize = np.min(proportions * len(idxs))

        proportions_to_num = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs, proportions_to_num)
        cid_idxlst_dict = {i: batch_idxs[i].tolist() for i in range(num_clients)}
        stats_dict['proportions'] = proportions
    elif partition == 'noniid-label-quantity':
        """
        Each client will only contain `num_classes_per_client` classes of samples.
        For any two clients that have the same class, the samples will not overlap.
        """
        # use user supplied partition
        if 'assigned_clients_per_class' in kwargs and 'assigned_classes_per_client' in kwargs:
            assigned_clients_per_class = kwargs['assigned_clients_per_class']
            assigned_classes_per_client = kwargs['assigned_classes_per_client']
            assert type(assigned_clients_per_class) == list, "assigned_clients_per_class has to a list"
            assert type(assigned_classes_per_client) == list, "assigned_classes_per_client has to a list"
            assert type(assigned_classes_per_client[0]) == set, "the elements of assigned_classes_per_client has to a set"
            cid_idxlst_dict = {cid: [] for cid in range(num_clients)}
            num_classes_per_client = [len(s) for s in assigned_classes_per_client]
        else:
            if 'num_classes_per_client' not in kwargs:
                raise ValueError(f"The num_classes_per_client parameter needs to be set for the partition {partition}.")
            else:
                num_classes_per_client = kwargs['num_classes_per_client']
            assert num_classes_per_client <= num_classes, "`num_classes_per_client` should be no bigger than `num_classes`"
            cid_idxlst_dict = {cid: [] for cid in range(num_clients)}
            assigned_clients_per_class = [0 for i in range(num_classes)]
            assigned_classes_per_client = []
            for cid in range(num_clients):
                # assign class `class_idx` to client `cid`
                class_idx = cid % num_classes
                current = set()
                current.add(class_idx)
                assigned_clients_per_class[class_idx] += 1
                assigned_class_count = 1
                while (assigned_class_count < num_classes_per_client):
                    ind = np.random.randint(0, num_classes)
                    if (ind not in current):
                        assigned_class_count += 1
                        current.add(ind)
                        assigned_clients_per_class[ind] += 1
                assigned_classes_per_client.append(current)

        missing_classes = []
        for k in range(num_classes):
            if assigned_clients_per_class[k] == 0:
                missing_classes.append(str(k))
        if len(missing_classes) > 0:
            warnings.warn("Classes " + ",".join(missing_classes) +
                          "are not used. Consider increase either num_clients or num_classes_per_client.")
        for k in range(num_classes):
            idx_k = np.where(ylabels == k)[0]
            np.random.shuffle(idx_k)
            try:
                split = np.array_split(idx_k, assigned_clients_per_class[k])
            except ValueError:
                pass
            ids = 0
            for cid in range(num_clients):
                if k in assigned_classes_per_client[cid]:
                    cid_idxlst_dict[cid] += split[ids].tolist()
                    ids += 1
        stats_dict['num_classes'] = num_classes
        stats_dict['num_classes_per_client'] = num_classes_per_client
        stats_dict['assigned_classes_per_client'] = assigned_classes_per_client
        stats_dict['assigned_clients_per_class'] = assigned_clients_per_class
    elif partition == 'noniid-label-distribution':
        """
        The number of classes per client own follow a dirichlet distribution with concentration parameter beta.
        feddf: https://github.com/epfml/federated-learning-public-code/blob/7e002ef5ff0d683dba3db48e2d088165499eb0b9/codes/FedDF-code/pcode/datasets/partition_data.py#L197
        """
        if is_balanced:
            np.random.seed(2022)
            server_config = kwargs['server_config']
            save_dir = f"../experiments/datapartition/{server_config['dataset']}_{beta}b_{server_config['num_clients']}.pkl"
            print("Doing balanced dir sampling")
            if os.path.exists(save_dir):
                print('Partition is found!')
                cid_idxlst_dict = load_from_pkl(save_dir)
            else:
                n_data_per_clnt = int(num_samples / num_clients)
                clnt_data_list = (np.ones(num_clients) * n_data_per_clnt).astype(int)
                cls_priors = np.random.dirichlet(alpha=[beta] * num_classes, size=num_clients)
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                idx_list = [np.where(ylabels == i)[0] for i in range(num_classes)]
                cls_amount = [len(idx_list[i]) for i in range(num_classes)]
                cid_idxlst_dict = {cid: [] for cid in range(num_clients)}
                while(np.sum(clnt_data_list) != 0):
                    curr_clnt = np.random.randint(num_clients)
                    # If current node is full resample a client
                    print('Remaining Data: %d' % np.sum(clnt_data_list))
                    if clnt_data_list[curr_clnt] <= 0:
                        continue
                    clnt_data_list[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if trn_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            continue
                        cls_amount[cls_label] -= 1
                        cid_idxlst_dict[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                        break
                mkdirs('../experiments/datapartition/')
                save_to_pkl(cid_idxlst_dict, save_dir)
                print('cid_idxlst_dict is saved to', save_dir)
        else:
            resample = False
            while cur_minsize < minsize or resample:
                attemp += 1
                if attemp > max_attemp:
                    count = 0
                    for cid in range(num_clients):
                        if allocated_classes[cid] <= 1:
                            count += 1
                    print(f" Warning: {count} clients have less than 2 classes")
                    break
                batch_idxs = [[] for _ in range(num_clients)]
                allocated_classes = [0] * num_clients
                for k in range(num_classes):
                    idx_k = np.where(ylabels == k)[0]
                    np.random.shuffle(idx_k)
                    # determine the fraction of samples in class k for each client;
                    proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                    # if number of samples in client j is already larger than the threshold num_samples / num_clients
                    # then the client won't contain any new class including the current class k
                    proportions = np.array(
                        [p * (len(allocated_idxs) < num_samples / num_clients) for p, allocated_idxs in zip(proportions, batch_idxs)])
                    proportions = proportions / proportions.sum()
                    stats_dict[f'proportions_{k}'] = proportions
                    proportions_to_num = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    # reference: https://numpy.org/doc/stable/reference/generated/numpy.split.html
                    # batch_idxs = [allocated_idxs + idx.tolist() for allocated_idxs,
                    #               idx in zip(batch_idxs, np.split(idx_k, proportions_to_num))]
                    chunks = np.split(idx_k, proportions_to_num)
                    # hack to fix class deficiency in some clients
                    if k >= 2:
                        if min(allocated_classes) <= 1:
                            cid_has_only_one_or_less_class = []
                            for cid in range(num_clients):
                                if allocated_classes[cid] <= 1:
                                    cid_has_only_one_or_less_class.append(cid)
                            replace_index = -1
                            for cid in cid_has_only_one_or_less_class:
                                temp_chunk = chunks[cid]
                                temp_ratio = proportions[cid]
                                chunks[cid] = chunks[replace_index]
                                chunks[replace_index] = temp_chunk
                                proportions[cid] = proportions[replace_index]
                                proportions[replace_index] = temp_ratio
                                replace_index -= 1
                    cid = 0
                    for allocated_idxs, idx in zip(batch_idxs, chunks):
                        added_samples = idx.tolist()
                        if len(added_samples) > 0:
                            allocated_idxs += added_samples
                            allocated_classes[cid] += 1
                        cid += 1
                    cur_minsize = min([len(allocated_idxs)
                                       for allocated_idxs in batch_idxs])
                if min(allocated_classes) <= 1:
                    resample = True
                    print(" [Info - Dirichlet Sampling]: At leaset one client only has one class label. Perform Resampling...")
                else:
                    resample = False
            cid_idxlst_dict = {cid: [] for cid in range(num_clients)}
            for cid in range(num_clients):
                np.random.shuffle(batch_idxs[cid])
                cid_idxlst_dict[cid] = batch_idxs[cid]
        stats_dict['num_classes'] = num_classes
    elif partition == 'shards':
        num_shards_per_client = kwargs['num_shards_per_client']
        dict_users, stats_dict['rand_set_all'] = sshards(dataset, num_clients, num_shards_per_client, server_data_ratio=0.0, rand_set_all=[])
        cid_idxlst_dict = {i: dict_users[i].tolist() for i in range(num_clients)}
    else:
        raise ValueError(f"partition:{partition} is not recognized.")
    # generate a set of sub-datasets
    dataset_per_client_dict = {
        cid: DatasetSplit(dataset, cid_idxlst_dict[cid]) for cid in range(num_clients)}
    stats_dict['num_clients'] = num_clients
    stats_dict['partition'] = partition
    stats_dict['seed'] = seed
    stats_dict['minsize'] = minsize
    return dataset_per_client_dict, stats_dict


def sampler_reuse(dataset, stats_dict, **kwargs):
    partition = stats_dict['partition']
    num_clients = stats_dict['num_clients']
    if stats_dict['seed'] is not None:
        np.random.seed(stats_dict['seed'])
    if partition in ['noniid-label-quantity', 'noniid-label-distribution']:
        num_classes = stats_dict['num_classes']
        num_unique_class = len(torch.unique(dataset.targets))
        assert num_classes == num_unique_class, f"num_class is set to, but number of unique class detected in ylables are {num_unique_class}. The dataset may have a different distribution!"
        if 'ylabels' not in kwargs:
            raise ValueError(
                f"The ylabels parameter needs to be set for the partition {partition}.")
        else:
            ylabels = kwargs['ylabels']
    num_samples = len(dataset)
    idxs = np.random.permutation(num_samples)
    cur_minsize = 0
    attemp = 0
    max_attemp = 100
    if partition == 'iid-equal-size':
        batch_idxs = np.array_split(idxs, num_clients)
        if len(batch_idxs[-1]) < stats_dict['minsize']:
            warnings.warn(
                f"partition:{partition} | Some clients have less than {stats_dict['minsize']} samples. Check it before continue.")
        cid_idxlst_dict = {
            cid: batch_idxs[cid].tolist() for cid in range(num_clients)}
    elif partition == 'iid-diff-size':
        proportions_to_num = (
            np.cumsum(stats_dict['proportions']) * len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs, proportions_to_num)
        cid_idxlst_dict = {i: batch_idxs[i].tolist()
                           for i in range(num_clients)}
    elif partition == 'noniid-label-quantity':
        num_classes_per_client = stats_dict['num_classes_per_client']
        assert num_classes_per_client <= num_classes, "`num_classes_per_client` should be no bigger than `num_classes`"
        cid_idxlst_dict = {cid: [] for cid in range(num_clients)}
        for k in range(num_classes):
            idx_k = np.where(ylabels == k)[0]
            np.random.shuffle(idx_k)
            try:
                split = np.array_split(
                    idx_k, stats_dict['assigned_clients_per_class'][k])
            except ValueError:
                pass
            ids = 0
            for cid in range(num_clients):
                if k in stats_dict['assigned_classes_per_client'][cid]:
                    cid_idxlst_dict[cid] += split[ids].tolist()
                    ids += 1
    elif partition == 'noniid-label-distribution':
        batch_idxs = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(ylabels == k)[0]
            np.random.shuffle(idx_k)
            proportions = stats_dict[f'proportions_{k}']
            proportions_to_num = (np.cumsum(proportions) *
                                  len(idx_k)).astype(int)[:-1]
            batch_idxs = [allocated_idxs + idx.tolist() for allocated_idxs,
                          idx in zip(batch_idxs, np.split(idx_k, proportions_to_num))]

        cid_idxlst_dict = {cid: [] for cid in range(num_clients)}
        for cid in range(num_clients):
            np.random.shuffle(batch_idxs[cid])
            cid_idxlst_dict[cid] = batch_idxs[cid]
    else:
        raise ValueError(f"partition:{partition} is not recognized.")
    # generate a set of sub-datasets
    dataset_per_client_dict = {
        cid: DatasetSplit(dataset, cid_idxlst_dict[cid]) for cid in range(num_clients)}
    return dataset_per_client_dict


def sshards(dataset, num_users, shard_per_user, server_data_ratio, rand_set_all=[]):
    setup_seed(2022)
    dict_users, all_idxs = {i: np.array([], dtype='int64') for i in range(num_users)}, [i for i in range(len(dataset))]

    idxs_dict = {}
    for i in range(len(dataset)):
        label = dataset.targets[i].item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        # collect all data in class ``label``
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(dataset.targets[value])
        assert(len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert(len(test) == len(dataset))
    assert(len(set(list(test))) == len(dataset))

    if server_data_ratio > 0.0:
        dict_users['server'] = set(np.random.choice(all_idxs, int(len(dataset) * server_data_ratio), replace=False))
    # print(dict_users)
    # exit()
    return dict_users, rand_set_all


"""
visualization tools
"""


def visualize_sampling(dataset_per_client_dict, num_classes, figsize=(10, 8), **kwargs):
    num_clients = len(dataset_per_client_dict)
    mat = np.zeros((num_clients, num_classes))
    targets = dataset_per_client_dict[0].dataset.targets
    for key in dataset_per_client_dict.keys():
        subset = dataset_per_client_dict[key]
        for k in range(num_classes):
            num_samples = torch.sum(torch.eq(targets[subset.idxs], k)).item()
            mat[key, k] = num_samples
    fig, ax = plt.subplots(figsize=figsize)

    im, _ = heatmap(mat, np.arange(num_clients), np.arange(num_classes), ax=ax,
                    cmap="YlGn", cbarlabel="#Samples")
    _ = annotate_heatmap(im, valfmt="{x:.0f}")

    fig.tight_layout()
    if 'fig_path_name' in kwargs:
        fig_path_name = kwargs['fig_path_name']
        dirpath = "/".join(fig_path_name.split("/")[:-1])
        mkdirs(dirpath)
        plt.savefig(fig_path_name)
    else:
        plt.show()
    return mat


def heatmap(data, x_labels, y_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data.T, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

#     # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Client ID")
    ax.set_yticks(np.arange(data.shape[1]))
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Class label")

    ax.set_xticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


class MulGaussian(Dataset):
    def __init__(self, mean_lst, n_lst):
        k = len(mean_lst)
        self.data = None
        self.targets = None
        for i in range(k):
            m = MultivariateNormal(torch.tensor(mean_lst[i]), torch.eye(len(mean_lst[i])))
            samples = m.sample(sample_shape=(n_lst[i],))
            labels = torch.ones((n_lst[i],), dtype=torch.int32) * i
            if i == 0:
                self.data = samples
                self.targets = labels
            else:
                self.data = torch.cat((self.data, samples))
                self.targets = torch.cat((self.targets, labels))
        self.targets = self.targets.type(torch.LongTensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.targets[idx]


class Spiral(Dataset):
    def __init__(self, n_lst, sigma=0.5):
        k = len(n_lst)
        self.data = None
        self.targets = None
        for i in range(k):
            r = torch.linspace(1, 10, n_lst[i])  # radius
            t = torch.linspace(i / k * 2 * torch.pi, (i + 1) / k * 2 * torch.pi, n_lst[i]) + torch.rand(n_lst[i]) * sigma
            x = r * torch.sin(t)
            y = r * torch.cos(t)
            samples = torch.stack((x, y), 1)
            labels = torch.ones((n_lst[i],), dtype=torch.int32) * i
            if i == 0:
                self.data = samples
                self.targets = labels
            else:
                self.data = torch.cat((self.data, samples))
                self.targets = torch.cat((self.targets, labels))
        self.targets = self.targets.type(torch.LongTensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.targets[idx]





"""
TinyImageNet Dataset
"""


EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = 'wnids.txt'
VAL_ANNOTATION_FILE = 'val_annotations.txt'


# lines = []
# with open('/export/home/data/tiny-imagenet-200/words.txt') as f:
#     lines = f.readlines()
# lookup = {}
# for i in range(len(lines)):
#     code, cls = lines[i].split('\t')
#     cls = cls.rstrip('\n')
#     lookup[code]=cls
# cls200 = []
# for code in trainset.label_texts:
#     cls200.append(lookup[code])

class TinyImageNet(Dataset):
    """
    Ref: https://github.com/leemengtaiwan/tiny-imagenet/blob/master/TinyImageNet.py
    Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """

    def __init__(self, root, split='train', transform=None, target_transform=None, in_memory=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(self.root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing
        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # get targets
        self.targets = []
        for index in range(len(self.image_paths)):
            file_path = self.image_paths[index]
            label_numeral = self.labels[os.path.basename(file_path)]
            self.targets.append(label_numeral)

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        if self.split == 'test':
            return img
        else:
            # file_name = file_path.split('/')[-1]
            return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img) if self.transform else img


"""
get datasets
"""


def get_datasets(datasetname, **kwargs):
    invTrans = None
    if datasetname == "FashionMnist":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                     download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                    download=True, transform=transform)
    elif datasetname == "Cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010]),
                                       transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465],
                                                            std=[1., 1., 1.]),
                                       ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        trainset.targets = torch.tensor(trainset.targets)
        testset.targets = torch.tensor(testset.targets)
    elif datasetname == 'Cifar100':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                   std=[0.267, 0.256, 0.276])])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                  std=[0.267, 0.256, 0.276])])

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                 download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform_test)
        trainset.targets = torch.tensor(trainset.targets)
        testset.targets = torch.tensor(testset.targets)

    elif datasetname == "TinyImageNet":
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        trainset = TinyImageNet("./data/tiny-imagenet-200", 'train', transform=transform_train, in_memory=False)
        testset = TinyImageNet("./data/tiny-imagenet-200", 'val', transform=transform_test, in_memory=False)
        trainset.targets = torch.tensor(trainset.targets)
        testset.targets = torch.tensor(testset.targets)

    elif datasetname == "Cifar10Aug":
        """
            On Bridging Generic and Personalized Federated Learning for Image Classification impl 
        """
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
        ])
        trainset = torchvision.datasets.CIFAR10(root='~/data', train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='~/data', train=False,
                                               download=True, transform=transform_test)
        trainset.targets = torch.tensor(trainset.targets)
        testset.targets = torch.tensor(testset.targets)
    elif datasetname == "GanEnhancedCifar10":
        trainset = GanEnhancedCifar10(
            kwargs['generator_path'],
            kwargs['dataset'],
            kwargs['upsample']
        )
        transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='~/data', train=False,
                                               download=True, transform=transform)
        testset.targets = torch.tensor(testset.targets)
    else:
        raise ValueError(f"Unrecognized dataset:{datasetname}")

    return trainset, testset, invTrans

    # if invTrans is not None:
    #     return trainset, testset, invTrans
    # else:
    #     return trainset, testset
