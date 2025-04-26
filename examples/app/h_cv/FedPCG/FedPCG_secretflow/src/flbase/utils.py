import torch
from collections import OrderedDict
import sys
sys.path.append("../../")
from copy import deepcopy
from src.utils import sampler, sampler_reuse, DatasetSplit, setup_seed
import numpy as np
"""
Configure Optimizer
"""


def setup_optimizer(model, config, round):
    if config['client_lr_scheduler'] == 'stepwise':
        if round < config['num_rounds'] // 2:
            lr = config['client_lr']
        else:
            lr = config['client_lr'] * 0.1

    elif config['client_lr_scheduler'] == 'diminishing':
        lr = config['client_lr'] * (config['lr_decay_per_round'] ** (round - 1))
    else:
        raise ValueError('unknown client_lr_scheduler')
    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=lr,
                                    momentum=config['sgd_momentum'], weight_decay=config['sgd_weight_decay'])
        # print('line 34: weight_decay=1e-3')
    elif config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr,
                                     weight_decay=1e-5)
    elif config['optimizer'] == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=lr,
                                        alpha=config['rmsprop_alpha'],
                                        eps=1e-08,
                                        weight_decay=config['rmsprop_weight_decay'],
                                        momentum=config['rmsprop_momentum'])
    else:
        raise ValueError(f"Unknown optimizer{config['optimizer']}")
    return optimizer


"""
client initialization
"""
def setup_clients(Client, trainset, testset, criterion, client_config_lst,client_pyus, **kwargs):
    """
        Client is a class constructor.
        A deepcopy is invoked such that each client has a unique model.
        **kwargs:
            weight_init: {'init.normal'}
            mean: params for init.normal
            std: 0.1
    """
    num_clients = kwargs['server_config']['num_clients']
    partition = kwargs['server_config']['partition']
    num_classes = kwargs['server_config']['num_classes']
    assert len(client_config_lst) == num_clients, "Inconsistent num_clients and len(client_config_lst)."
    if 'noniid' == partition[:6]:
        trainset_per_client_dict, stats_dict = sampler(trainset, num_clients, partition, ylabels=trainset.targets,
                                                       num_classes=num_classes, **kwargs)
        if testset is None:
            testset_per_client_dict = {cid: None for cid in range(num_clients)}
        else:
            if kwargs['same_testset']:
                testset_per_client_dict = {cid: testset for cid in range(num_clients)}
            else:
                testset_per_client_dict = sampler_reuse(testset, stats_dict, ylabels=testset.targets,
                                                        num_classes=num_classes, **kwargs)
    else:
        trainset_per_client_dict, stats_dict = sampler(trainset, num_clients, partition, num_classes=num_classes, **kwargs)
        if testset is None:
            testset_per_client_dict = {cid: None for cid in range(num_clients)}
        else:
            if kwargs['same_testset']:
                testset_per_client_dict = {cid: testset for cid in range(num_clients)}
            else:
                testset_per_client_dict = sampler_reuse(testset, stats_dict, num_classes=num_classes, **kwargs)
#     print('trainset_per_client_dict',trainset_per_client_dict)
    n_samples = {cid: len(dataset) for cid, dataset in trainset_per_client_dict.items()}
#     print('n_samples',n_samples)
    all_clients_dict = {}
    for cid in range(num_clients):
        # same initial weight
        setup_seed(2022)
        all_clients_dict[cid] = Client(device = client_pyus[cid],
            criterion=criterion,
            trainset=trainset_per_client_dict[cid],
            testset=testset_per_client_dict[cid],
            client_config=client_config_lst[cid],
            cid=cid,
            **kwargs)
    # print(all_clients_dict[9].trainset.targets)
    return all_clients_dict,n_samples


def create_clients_from_existing_ones(Client, clients_dict, newtrainset, increment, criterion, **kwargs):
    """
        Create new clients. All clients will maintain the same data distribution as specified in clients_dict.
    """
    num_clients = len(clients_dict)
    all_clients_dict = {}
    if 'same_pool' not in kwargs:
        same_pool = False
    else:
        same_pool = kwargs['same_pool']

    if 'scale' not in kwargs:
        scale = len(newtrainset) // increment - 1
    else:
        scale = kwargs['scale']

    for cid in range(num_clients):
        client = clients_dict[cid]
        data_idxs = client.trainset.idxs
        add_idxs = []
        if same_pool:
            for cls in client.count_by_class.keys():
                num_sample_cls = client.count_by_class[cls]
                target_num_sample_cls = min(num_sample_cls * scale, len(newtrainset.get_fake_imgs_idxs(cls)))
                add_idxs += np.random.choice(newtrainset.get_fake_imgs_idxs(cls), target_num_sample_cls, replace=False).tolist()
        else:
            for i in data_idxs:
                for j in range(scale):
                    add_idxs.append(i + increment * (j + 1))

        full_idxs = data_idxs + add_idxs
        client_newtrainset = DatasetSplit(newtrainset, full_idxs)
        all_clients_dict[cid] = Client(
            criterion,
            client_newtrainset,
            client.testset,
            client.client_config,
            client.cid,
            client.group,
            client.device, **kwargs)
    return all_clients_dict


"""
resume training
"""
from copy import deepcopy
from ..utils import load_from_pkl


def resume_training(server_config, checkpoint, model):
    server = load_from_pkl(checkpoint)
    server.server_config = server_config
    for c in server.clients_dict.values():
        c.model = deepcopy(model)
        c.set_params(server.server_model_state_dict)
        c.model.to(c.device)
        c.model.init()
    print("Resume Training")
    print(f"Rounds performed:{server.rounds}")
    return server


"""
state_dict operation
"""


def scale_state_dict(this, scale, inplace=True, exclude=set()):
    with torch.no_grad():
        if not inplace:
            ans = deepcopy(this)
        else:
            ans = this
        for state_key in this.keys():
            if state_key not in exclude:
                ans[state_key] = this[state_key] * scale
        return ans


def linear_combination_state_dict(this, other, this_weight=1.0, other_weight=1.0, exclude=set()):
    """
        this, other: state_dict
        this_weight * this + other_weight * other 
    """
    with torch.no_grad():
        ans = deepcopy(this)
        for state_key in this.keys():
            if state_key not in exclude:
                # print('agg', state_key)
                ans[state_key] = this[state_key] * this_weight + other[state_key] * other_weight
        return ans


def average_list_of_state_dict(state_dict_lst, exclude=set()):
    assert type(state_dict_lst) == list
    num_participants = len(state_dict_lst)
    keys = state_dict_lst[0].keys()
    with torch.no_grad():
        ans = OrderedDict()
        for key in keys:
            if state_key not in exclude:
                for idx, client_state_dict in enumerate(state_dict_lst):
                    if idx == 0:
                        # must do deepcopy; otherwise subsequent operation overwrittes the first client_state_dict
                        ans[key] = deepcopy(client_state_dict[key])
                    else:
                        ans[key] += client_state_dict[key]
                ans[key] = ans[key] / num_participants
    return ans


def weight_sum_of_dict_of_state_dict(dict_state_dict, weight_dict):
    layer_keys = next(iter(dict_state_dict.values())).keys()
    with torch.no_grad():
        ans = OrderedDict()
        for layer in layer_keys:
            count = 0
            for cid in dict_state_dict.keys():
                if count == 0:
                    # must do deepcopy; otherwise subsequent operation overwrittes the first client_state_dict
                    ans[layer] = deepcopy(dict_state_dict[cid][layer]) * weight_dict[cid]
                else:
                    ans[layer] += dict_state_dict[cid][layer] * weight_dict[cid]
                count += 1
    return ans
