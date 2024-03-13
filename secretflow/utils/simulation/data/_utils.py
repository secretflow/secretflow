# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum, unique

import numpy as np

from secretflow.device.device.pyu import PYU
from secretflow.utils.errors import InvalidArgumentError


@unique
class SPLIT_METHOD(Enum):
    IID = 1
    DIRICHLET = 2  # Please check `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`
    LABEL_SCREW = 3  # Please check `Federated Learning on Non-IID Data Silos: An Experimental Study <https://arxiv.org/abs/2102.02079>`


# Calculate samples for each client-> Dict[PYU,int]
def infer_client_num(parts, total_num, num_clients):
    device_list = (
        list(parts) if isinstance(parts, (list, tuple)) else list(parts.keys())
    )
    client_sample_dict = {}
    if isinstance(parts, (list, tuple)):
        for part in parts:
            assert isinstance(
                part, PYU
            ), f'Parts shall be list like of PYUs but got {type(part)}.'
        num_clients = len(parts)
        num_samples_per_client = int(round(total_num / num_clients))
        for i, device in enumerate(device_list):
            if i == len(parts) - 1:
                client_sample_dict[device] = total_num - i * num_samples_per_client
            else:
                client_sample_dict[device] = num_samples_per_client
        return client_sample_dict
    elif isinstance(parts, dict):
        devices = parts.keys()
        for device in devices:
            assert isinstance(device, PYU), f'Keys of parts shall be PYU'
        is_percent = isinstance(list(parts.values())[0], float)
        if is_percent:
            for percent in parts.values():
                assert isinstance(
                    percent, float
                ), f'Not all dict values are percentages.'
            assert sum(parts.values()) == 1.0, f'Sum of percentages shall be 1.0.'

            for i, (device, percent) in enumerate(parts.items()):
                client_sample_dict[device] = round(percent * total_num)
        else:
            raise InvalidArgumentError(
                "user assign index should be tackled directly in partition"
            )
        return client_sample_dict

    else:
        raise InvalidArgumentError(
            f'Parts should be a list/tuple or dict but got {type(parts)}.'
        )


def iid_partition(
    parts,
    total_num,
    random_seed=1234,
    shuffle=False,
):
    """Assign same sample sample for each client.

    Args:
        parts: The data partitions describe. The dataset will be distributed
            as evenly as possible to each PYU if parts is a array of PYUs.
            If parts is a dict {PYU: value}, the value shall be one of the
            followings.
            1) a float
            2) an interval in tuple closed on the left-side and open on the
               right-side.
        total_num: Total number of samples.
        random_seed: Random seed used when shuffling.
        shuffle: Whether to shuffle the data before partitioning. Default: False

    Returns:
        dict: ``{ PYU: indices}``.

    """
    np.random.seed(random_seed)
    clnt_idx_dict = {}
    device_list = (
        list(parts) if isinstance(parts, (list, tuple)) else list(parts.keys())
    )
    if isinstance(parts, dict) and isinstance(list(parts.values())[0], tuple):
        # user assign index
        for d, p in parts.items():
            assert len(p) == 2, "parts should be {PYU: (start,end)}"
            clnt_idx_dict[d] = np.arange(p[0], p[1])
        return clnt_idx_dict
    else:
        # split assign index
        if shuffle:
            rand_perm = np.random.permutation(total_num)
        else:
            rand_perm = np.arange(total_num)
        client_sample_dict = infer_client_num(parts, total_num, len(parts))
        client_sample_nums = np.array(list(client_sample_dict.values()))
        num_cumsum = np.cumsum(client_sample_nums).astype(int)
        split_point = np.split(rand_perm, num_cumsum)[:-1]
        for device in device_list:
            assert isinstance(device, PYU), f'Keys of parts shall be PYU'
        for p, idxs in zip(device_list, split_point):
            clnt_idx_dict[p] = idxs
        return clnt_idx_dict


def dirichlet_partition(
    parts,
    targets,
    num_classes,
    alpha,
    random_seed=1234,
):
    """Non-iid Dirichlet partition.
    This function can be used by given specific sample number for all clients ``client_sample_nums``.

    Args:
        parts: The data partitions describe. The dataset will be distributed
            as evenly as possible to each PYU if parts is a array of PYUs.
            If parts is a dict {PYU: value}, the value shall be one of the
            followings.
            1) a float

        targets (list or numpy.ndarray): Sample targets.
        num_classes (int): Number of classes in samples.
        alpha (float): Parameter alpha for Dirichlet distribution.
        random_seed: Random seed for generating random numbers.

    Returns:
        dict: ``{ PYU: indices}``.

    """
    np.random.seed(random_seed)
    num_clients = len(parts)
    device_list = (
        list(parts) if isinstance(parts, (list, tuple)) else list(parts.keys())
    )
    client_sample_dict = infer_client_num(parts, len(targets), len(parts))
    client_sample_nums = np.array(list(client_sample_dict.values()))
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    class_priors = np.random.dirichlet(
        alpha=[alpha] * num_classes,
        size=num_clients,
    )
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    client_indices = [
        np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in range(num_clients)
    ]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        curr_client = curr_cid
        # If current node is full resample a client
        if client_sample_nums[curr_client] <= 0:
            continue
        client_sample_nums[curr_client] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                continue
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_client]] = idx_list[
                curr_class
            ][class_amount[curr_class]]

            break

    client_dict = {
        device: client_indices[cid] for cid, device in enumerate(device_list)
    }
    return client_dict


def label_skew_partition(
    parts,
    targets,
    num_classes,
    max_class_nums,
    random_seed=1234,
):
    """Label-skew:quantity-based partition.

    Args:
        parts: The data partitions describe. The dataset will be distributed
            as evenly as possible to each PYU if parts is a array of PYUs.
            If parts is a dict {PYU: value}, the value shall be one of the
            followings.
            1) a float
            2) an interval in tuple closed on the left-side and open on the
               right-side.
        targets: Labels of dataset.
        num_classes: Number of unique classes.
        max_class_nums: Number of classes for each client, should be less then ``num_classes``.
        random_seed: Random seed for generating random numbers.

    Returns:
        dict: ``{ PYU: indices}``.

    """
    np.random.seed(random_seed)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    num_clients = len(parts)
    device_list = (
        list(parts) if isinstance(parts, (list, tuple)) else list(parts.keys())
    )

    idx_batch = [np.ndarray(0, dtype=np.int64) for _ in range(num_clients)]
    # only for max_class_nums < num_classes.
    # if max_class_nums = num_classes, it equals to IID partition
    times = [0 for _ in range(num_classes)]
    contain = []
    for cid in range(num_clients):
        current = [cid % num_classes]
        times[cid % num_classes] += 1
        j = 1
        while j < max_class_nums:
            ind = np.random.randint(num_classes)
            if ind not in current:
                j += 1
                current.append(ind)
                times[ind] += 1
        contain.append(current)
    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)

        if times[k] == 0:
            split = idx_k
        else:
            split = np.array_split(idx_k, times[k])
        ids = 0
        for cid in range(num_clients):
            if k in contain[cid]:
                idx_batch[cid] = np.append(idx_batch[cid], split[ids])
                ids += 1

    client_dict = {device_list[cid]: idx_batch[cid] for cid in range(num_clients)}
    return client_dict
