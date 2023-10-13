from secretflow.ml.nn.utils import BaseModule, TorchModel
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
# from secretflow.ml.nn import FLModel
from fl_model import FLModel
from torchmetrics import Accuracy, Precision
# from secretflow.security.aggregation import SecureAggregator,PlainAggregator
from secretflow.utils.simulation.datasets import load_mnist
from torch import nn, optim
from torch.nn import functional as F
from matplotlib import pyplot as plt
import secretflow as sf
from typing import List
import numpy as np
from secretflow.device import PYU, DeviceObject, PYUObject
from secretflow.security.aggregation.aggregator import Aggregator
from scaffold_optimizer import ScaffoldOptimizer

sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], address='local')
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')

class ConvNet(BaseModule):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc_in_dim = 192
        self.fc = nn.Linear(self.fc_in_dim, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, self.fc_in_dim)
        x = self.fc(x)
        return F.softmax(x, dim=1)
    
(train_data, train_label), (test_data, test_label) = load_mnist(
    parts={alice: 0.4, bob: 0.6},
    normalized_x=True,
    categorical_y=True,
    is_torch=True,
)

# 自建聚合器
class PlainAggregator(Aggregator):
    def __init__(self, device: PYU):
        assert isinstance(device, PYU), f'Accepts PYU only but got {type(device)}.'
        self.device = device
        self.control = []
        self.delta_control = []
        self.x_weight=None
    
    @staticmethod
    def _get_dtype(arr):
        if isinstance(arr, np.ndarray):
            return arr.dtype
        else:
            try:
                import tensorflow as tf

                if isinstance(arr, tf.Tensor):
                    return arr.numpy().dtype
            except ImportError:
                return None

    def sum(self, data: List[DeviceObject], axis=None) -> PYUObject:
        assert data, 'Data to aggregate should not be None or empty!'
        data = [d.to(self.device) for d in data]

        def _sum(*data, axis):
            if isinstance(data[0], (list, tuple)):
                return [np.sum(element, axis=axis) for element in zip(*data)]
            else:
                return np.sum(data, axis=axis)

        return self.device(_sum)(*data, axis=axis)
    
    # update c   
    def update_scaffold(self, client_delta_control_list: List[DeviceObject], axis=None, weights=None):
        client_delta_control_list = [d.to(self.device) for d in client_delta_control_list]
        if isinstance(weights, (list, tuple)):
            weights = [
                w.to(self.device) if isinstance(w, DeviceObject) else w for w in weights
            ]
        def _update(*client_delta_control_list, control, axis, weights):
            results=[]
            for elements in zip(*client_delta_control_list):
                avg = np.average(elements, axis=axis, weights=weights)
                if control is not None and len(control)!=0:
                    avg+=control[len(results)]
                res_dtype = self._get_dtype(elements[0])
                if res_dtype:
                    avg = avg.astype(res_dtype)
                results.append(avg)
                self._control=results
        self.device(_update)(*client_delta_control_list, control=self._control, axis=axis, weights=weights)
            
    #初始化变量     
    def initial_scaffold(self, initial_weight: PYUObject):
        #增加控制变量         
        def _initial_weight(weight):
            x_weight=[]
            for layer_weight in weight:
                x_weight.append(layer_weight)
            return x_weight
        def _initial_control(weight):
            init_control=[]
            for layer_weight in weight:
                init_control.append(np.zeros_like(layer_weight))
            return init_control
        def _initial_delta_control(weight):
            init_delta_control=[]
            for layer_weight in weight:
                init_delta_control.append(np.zeros_like(layer_weight))
            return init_delta_control
        self.x_weight=self.device(_initial_weight)(initial_weight)
        self._control=self.device(_initial_control)(initial_weight)
        self._delta_control=self.device(_initial_delta_control)(initial_weight)
        

    def average(self, data: List[DeviceObject],axis=None, weights=None) -> PYUObject:
        assert data, 'Data to aggregate should not be None or empty!'
        data = [d.to(self.device) for d in data]
        if isinstance(weights, (list, tuple)):
            weights = [
                w.to(self.device) if isinstance(w, DeviceObject) else w for w in weights
            ]
        def _average(*data, axis, weights, x_weight: np.ndarray):
            if isinstance(data[0], (list, tuple)):
                results = []
                for elements in zip(*data):
                    avg = np.average(elements, axis=axis, weights=weights)
                    if x_weight is not None:
                        avg+=x_weight[len(results)]
                    res_dtype = self._get_dtype(elements[0])
                    if res_dtype:
                        avg = avg.astype(res_dtype)
                    results.append(avg)
                    
                #更新一下服务器参数
                return results
            else:
                res = np.average(data, axis=axis, weights=weights)
                res_dtype = self._get_dtype(data[0])
                return res.astype(res_dtype) if res_dtype else res
        
        ####改动的地方，之前写在函数里面，更新无效     
        self.x_weight=self.device(_average)(*data, axis=axis, weights=weights, x_weight=self.x_weight)
        return  self.x_weight

loss_fn = nn.CrossEntropyLoss
optim_fn = optim_wrapper(ScaffoldOptimizer, lr=1e-2)
model_def = TorchModel(
    model_fn=ConvNet,
    loss_fn=loss_fn,
    optim_fn=optim_fn,
    metrics=[
        metric_wrapper(Accuracy, task="multiclass", num_classes=10, average='micro'),
        metric_wrapper(Precision, task="multiclass", num_classes=10, average='micro'),
    ],
)

device_list = [alice, bob]
server = charlie
aggregator = PlainAggregator(charlie)

# spcify params
fl_model = FLModel(
    server=server,
    device_list=device_list,
    model=model_def,
    aggregator=aggregator,
    strategy='fed_scaffold',  # fl strategy
    backend="torch",  # backend support ['tensorflow', 'torch']
)

history = fl_model.fit(
    train_data,
    train_label,
    validation_data=(test_data, test_label),
    epochs=20,
    batch_size=32,
    aggregate_freq=1,
)

# Draw accuracy values for training & validation
plt.plot(history.global_history['multiclassaccuracy'])
plt.plot(history.global_history['val_multiclassaccuracy'])
plt.title('FLModel accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig("test.png")