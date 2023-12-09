import secretflow
print(secretflow.__file__)

import secretflow as sf

# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))
# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], address='local')
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')


from secretflow.ml.nn.utils import BaseModule, TorchModel
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
from secretflow.ml.nn import FLModel
from torchmetrics import Accuracy, Precision
from secretflow.security.aggregation import SecureAggregator
from secretflow.utils.simulation.datasets import load_mnist
import torch
from torch import nn, optim
from torch.nn import functional as F

class Model(BaseModule):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(784, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU()
        )
        self.head = nn.Linear(200, 10)
        self.cg = []
        self.c =[]
        for param in self.parameters():
            self.cg.append(torch.zeros_like(param))
            self.c.append(torch.zeros_like(param))
        self.delta_c = self.cg
        self.eta_l = 0.01
    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x


from typing import Dict, List, Tuple, Union
import numpy as np

from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.device.device.pyu import PYU

from torchvision.datasets import EMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def load_emnist(
        parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]],
        batch_size: int = 1,
        is_torch: bool = True,
        num_classes_per_client=4,
) -> Tuple[Tuple[FedNdarray, FedNdarray], Tuple[FedNdarray, FedNdarray]]:
    """Load EMNIST dataset to federated ndarrays.

    Args:
        parts: the data partitions.
        batch_size: Batch size for the DataLoader.
        is_torch: optional, return torch tensors if True. Default to True.

    Returns:
        A tuple consists of two tuples, (x_train, y_train) and (x_test, y_test).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载 CIFAR-10 训练集和测试集
    trainset = EMNIST(root='./data', train=True, split='mnist', download=True, transform=transform)
    testset = EMNIST(root='./data', train=False, split='mnist', download=True, transform=transform)

    # 使用 DataLoader 进行批处理
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # 将 DataLoader 转换为 SecretFlow 可处理的格式
    # 注意: 这里需要根据 SecretFlow 的要求进行适当的转换
    train_data, train_labels = _convert_to_fedndarray(trainloader, parts, is_torch)
    test_data, test_labels = _convert_to_fedndarray(testloader, parts, is_torch)

    return ((train_data, train_labels), (test_data, test_labels))

def _convert_to_fedndarray(dataloader, parts, is_torch):
    data_list = []
    label_list = []
    for data, label in dataloader:
        if is_torch:
            data = data.numpy()
            label = label.numpy()
        data_list.append(data)
        label_list.append(label)
    data_list, label_list = np.concatenate(data_list), np.concatenate(label_list)
    return create_emnist_ndarray(data_list, label_list, parts=parts, is_torch=is_torch)

def create_emnist_ndarray(data, labels, parts, is_torch=False, num_classes_per_client=4):
    assert len(data) == len(labels), "Data and labels must have the same length"
    class_indices = {i: np.where(labels == i)[0] for i in range(10)}

    # 分配给每个 PYU 的类别
    pyu_classes = {}
    total_samples = len(data)
    for idx, pyu in enumerate(parts.keys()):
        np.random.seed(idx)
        pyu_classes[pyu] = np.random.choice(10, num_classes_per_client, replace=False)  # 随机选择 4 个类别

    # 分配数据和标签给每个 PYU
    pyu_data = {}
    pyu_labels = {}
    for pyu, proportion in parts.items():
        pyu_sample_size = int(total_samples * proportion)  # 计算每个 PYU 的样本量
        # 为每个 PYU 从选定的类别中采样数据
        indices = np.concatenate(
            [np.random.choice(class_indices[cls], size=pyu_sample_size // num_classes_per_client, replace=True) for cls
             in pyu_classes[pyu]])
        np.random.shuffle(indices)  # 打乱索引
        pyu_data[pyu] = data[indices]
        pyu_labels[pyu] = labels[indices]
        print(len(pyu_data[pyu]), pyu_classes[pyu],pyu_labels[pyu][0])  # 打印每个PYU的数据量和随机类别

    # 将数据和标签转换为 FedNdarray
    data_fedndarray = FedNdarray(
        partitions={
            pyu: pyu(lambda arr: arr)(pyu_data[pyu]) for pyu in parts.keys()
        },
        partition_way=PartitionWay.HORIZONTAL,
    )
    labels_fedndarray = FedNdarray(
        partitions={
            pyu: pyu(lambda arr: arr)(pyu_labels[pyu]) for pyu in parts.keys()
        },
        partition_way=PartitionWay.HORIZONTAL,
    )

    return data_fedndarray, labels_fedndarray

(train_data, train_label), (test_data, test_label) = load_emnist(
    parts={alice: 0.1, bob: 0.1},
    # parts={client1: 0.1, client2: 0.1,client3: 0.1,client4: 0.1,client5: 0.1,client6: 0.1,client7: 0.1,client8: 0.1,client9: 0.1,client10: 0.1},
    # normalized_x=True,
    # categorical_y=True,
    is_torch=True,
    num_classes_per_client = 4,
)
loss_fn = nn.CrossEntropyLoss
optim_fn = optim_wrapper(optim.SGD, lr=0.01)
model_def = TorchModel(
    model_fn=Model,
    loss_fn=loss_fn,
    optim_fn=optim_fn,
    metrics=[
        metric_wrapper(Accuracy, task="multiclass", num_classes=10, average='micro'),
        metric_wrapper(Precision, task="multiclass", num_classes=10, average='micro'),
    ],
)

device_list = [alice, bob]
server = charlie
aggregator = SecureAggregator(server, [alice, bob])

# spcify params
fl_model = FLModel(
    server=server,
    device_list=device_list,
    model=model_def,
    aggregator=aggregator,
    # strategy='fed_avg_w',  # fl strategy
    strategy='scaffold',  # fl strategy
    backend="torch",  # backend support ['tensorflow', 'torch']
    # server_c=model_def.model_fn.parameters(),
)

history = fl_model.fit(
    train_data,
    train_label,
    validation_data=(test_data, test_label),
    epochs=20,
    batch_size=128,
    aggregate_freq=10,
)

print(history['local_history'])

from matplotlib import pyplot as plt

# Draw accuracy values for training & validation
plt.plot(history['global_history']['multiclassaccuracy'])
plt.plot(history['global_history']['val_multiclassaccuracy'])
plt.plot(history['local_history']['alice_train_multiclassaccuracy'])
plt.plot(history['local_history']['bob_train_multiclassaccuracy'])
plt.plot(history['local_history']['alice_val_eval_multiclassaccuracy'])
plt.plot(history['local_history']['bob_val_eval_multiclassaccuracy'])
plt.title('FLModel accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid','alice_T','bob_T','alice_V','bob_V'], loc='upper left')
plt.show()