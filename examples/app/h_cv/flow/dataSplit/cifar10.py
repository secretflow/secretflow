"""
按照狄利克雷分布为每个客户端划分数据集
"""

import numpy as np
from torchvision import datasets

NUM_CLASSES = 10
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
TRAIN_EXAMPLES_PER_LABEL = int(TRAIN_EXAMPLES / 10)
TEST_EXAMPLES_PER_LABEL = int(TEST_EXAMPLES / 10)


def split_cifar10(
    dirichlet_parameter: float = 0.1,
    total_clients: int = 500,
):
    # 加载 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True)

    # 创建包含 10 个类别的二维数组，每个类别一个子列表
    train_indices_by_class = [[] for _ in range(10)]
    test_indices_by_class = [[] for _ in range(10)]

    # 将训练集中每个类别的样本索引添加到对应的子列表
    for idx, (_, label) in enumerate(train_dataset):
        train_indices_by_class[label].append(idx)
    # 将测试集中每个类别的样本索引添加到对应的子列表
    for idx, (_, label) in enumerate(test_dataset):
        test_indices_by_class[label].append(idx)
    train_indices_by_class = np.array(train_indices_by_class)
    test_indices_by_class = np.array(test_indices_by_class)

    # 每个客户端进行采样数据的狄利克雷分布
    train_multinomial_vals = []
    test_multinomial_vals = []
    for i in range(total_clients):
        proportion = np.random.dirichlet(
            dirichlet_parameter
            * np.ones(
                NUM_CLASSES,
            )
        )
        train_multinomial_vals.append(proportion)
        test_multinomial_vals.append(proportion)
    train_multinomial_vals = np.array(train_multinomial_vals)
    test_multinomial_vals = np.array(test_multinomial_vals)

    train_client_samples = [[] for _ in range(total_clients)]
    test_client_samples = [[] for _ in range(total_clients)]
    train_count = np.zeros(NUM_CLASSES).astype(int)
    test_count = np.zeros(NUM_CLASSES).astype(int)

    # 每个客户端的训练集和测试集中样本的数量
    train_examples_per_client = (int(TRAIN_EXAMPLES / total_clients))
    test_examples_per_client = (int(TEST_EXAMPLES / total_clients))

    for k in range(total_clients):
        for i in range(train_examples_per_client):
            sampled_label = np.argwhere(
                np.random.multinomial(1, train_multinomial_vals[k, :]) == 1
            )[0][0]
            train_client_samples[k].append(
                train_indices_by_class[sampled_label, train_count[sampled_label]]
            )
            train_count[sampled_label] += 1
            if train_count[sampled_label] == TRAIN_EXAMPLES_PER_LABEL:
                # print("该类别的训练集已取完")
                train_multinomial_vals[:, sampled_label] = 0
                train_multinomial_vals = (
                        train_multinomial_vals / train_multinomial_vals.sum(axis=1)[:, None]
                )
        for i in range(test_examples_per_client):
            sampled_label = np.argwhere(
                np.random.multinomial(1, test_multinomial_vals[k, :]) == 1
            )[0][0]
            test_client_samples[k].append(
                test_indices_by_class[sampled_label, test_count[sampled_label]]
            )
            test_count[sampled_label] += 1
            if test_count[sampled_label] == TEST_EXAMPLES_PER_LABEL:
                # print("该类别的测试集已取完")
                test_multinomial_vals[:, sampled_label] = 0
                test_multinomial_vals = (
                        test_multinomial_vals / test_multinomial_vals.sum(axis=1)[:, None]
                )
    return train_client_samples, test_client_samples
