#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""

import os
import tempfile
from typing import List, Union
import numpy as np
import tensorflow as tf
from torch import nn, optim
from torchmetrics import Accuracy, Precision

from secretflow.device import reveal
from secretflow.device import PYU, DeviceObject, PYUObject

# from secretflow.ml.nn import FLModel
from fl_model import FLModel #可更改文件位置
from scaffold_optimizer import ScaffoldOptimizer #可更改文件位置
# from secretflow.security.aggregation import PlainAggregator, SparsePlainAggregator
from secretflow.security.aggregation.aggregator import Aggregator

from secretflow.ml.nn.utils import TorchModel
from secretflow.ml.nn.fl.compress import COMPRESS_STRATEGY
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
from secretflow.preprocessing.encoder import OneHotEncoder
from secretflow.security.privacy import DPStrategyFL, GaussianModelDP
from secretflow.utils.simulation.datasets import load_iris, load_mnist
from tests.ml.nn.model_def import ConvNet, ConvRGBNet, MlpNet 


_temp_dir = tempfile.mkdtemp()

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

path_to_flower_dataset = tf.keras.utils.get_file(
    "flower_photos",
    "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/tf_flowers/flower_photos.tgz",
    untar=True,
    cache_dir=_temp_dir,
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
        if weights is not None:
            weights = [
                w.to(self.device) if isinstance(w, DeviceObject) else w for w in weights
            ]
        def _average(*data, axis, weights, x_weight: np.ndarray):
            if isinstance(data[0], (list, tuple)):
                results = []
                for elements in zip(*data):
                    avg = np.average(elements, axis=axis, weights=weights)
                    if x_weight is not None and len(results) < len(x_weight):
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
          
        self.x_weight=self.device(_average)(*data, axis=axis, weights=weights, x_weight=self.x_weight)
        return  self.x_weight

def _torch_model_with_mnist(
    devices, model_def, data, label, strategy, backend, **kwargs
):
    device_list = [devices.alice, devices.bob]
    server = devices.carol

    # if strategy in COMPRESS_STRATEGY:
    #     aggregator = SparsePlainAggregator(server)
    aggregator = PlainAggregator(server)

    # specify params
    dp_spent_step_freq = kwargs.get('dp_spent_step_freq', None)
    num_gpus = kwargs.get("num_gpus", 0)
    fl_model = FLModel(
        server=server,
        device_list=device_list,
        model=model_def,
        aggregator=aggregator,
        strategy=strategy,
        backend=backend,
        random_seed=1234,
        num_gpus=num_gpus,
    )
    history = fl_model.fit(
        data,
        label,
        validation_data=(data, label),
        epochs=20,
        batch_size=32,
        aggregate_freq=1,
        dp_spent_step_freq=dp_spent_step_freq,
    )
    result = fl_model.predict(data, batch_size=32)
    assert len(reveal(result[device_list[0]])) == 4000
    global_metric, _ = fl_model.evaluate(data, label, batch_size=32, random_seed=1234)

    assert (
        global_metric[0].result().numpy()
        == history.global_history['val_multiclassaccuracy'][-1]
    )

    assert global_metric[0].result().numpy() > 0.5

    model_path_test = os.path.join(_temp_dir, "base_model")
    fl_model.save_model(model_path=model_path_test, is_test=True)
    model_path_dict = {
        devices.alice: os.path.join(_temp_dir, "alice_model"),
        devices.bob: os.path.join(_temp_dir, "bob_model"),
    }
    fl_model.save_model(model_path=model_path_dict, is_test=False)

    new_fed_model = FLModel(
        server=server,
        device_list=device_list,
        model=model_def,
        aggregator=None,
        backend=backend,
        random_seed=1234,
        num_gpus=num_gpus,
    )
    new_fed_model.load_model(model_path=model_path_dict, is_test=False)
    new_fed_model.load_model(model_path=model_path_test, is_test=True)
    reload_metric, _ = new_fed_model.evaluate(
        data, label, batch_size=128, random_seed=1234
    )

    np.testing.assert_equal(
        [m.result().numpy() for m in global_metric],
        [m.result().numpy() for m in reload_metric],
    )

class TestFLModelTorchMnist:
    def test_torch_model(self, sf_simulation_setup_devices):
        (_, _), (mnist_data, mnist_label) = load_mnist(
            parts={
                sf_simulation_setup_devices.alice: 0.4,
                sf_simulation_setup_devices.bob: 0.6,
            },
            normalized_x=True,
            categorical_y=True,
            is_torch=True,
        )

        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(ScaffoldOptimizer, lr=1e-2)
        model_def = TorchModel(
            model_fn=ConvNet,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average='micro'
                ),
            ],
        )

        # Test fed_scaffold with mnist
        _torch_model_with_mnist(
            devices=sf_simulation_setup_devices,
            model_def=model_def,
            data=mnist_data,
            label=mnist_label,
            strategy='fed_scaffold',
            backend="torch",
        )

class TestFLModelTorchMlp:
    def test_torch_model_mlp(self, sf_simulation_setup_devices):
        aggregator = PlainAggregator(sf_simulation_setup_devices.carol)
        hdf = load_iris(
            parts=[
                sf_simulation_setup_devices.alice,
                sf_simulation_setup_devices.bob,
            ],
            aggregator=aggregator,
        )

        label = hdf['class']
        # do preprocess
        encoder = OneHotEncoder()
        label = encoder.fit_transform(label)

        data = hdf.drop(columns='class', inplace=False)
        data = data.fillna(data.mean(numeric_only=True).to_dict())

        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(ScaffoldOptimizer, lr=5e-3)
        model_def = TorchModel(
            model_fn=MlpNet,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=3, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=3, average='micro'
                ),
            ],
        )
        device_list = [
            sf_simulation_setup_devices.alice,
            sf_simulation_setup_devices.bob,
        ]

        fl_model = FLModel(
            server=sf_simulation_setup_devices.carol,
            device_list=device_list,
            model=model_def,
            aggregator=aggregator,
            sparsity=0.0,
            strategy="fed_scaffold",
            backend="torch",
            random_seed=1234,
            # num_gpus=0.5,  # here is no GPU in the CI environment, so it is temporarily commented out.
        )

        history = fl_model.fit(
            data,
            label,
            validation_data=(data, label),
            epochs=20,
            batch_size=32,
            aggregate_freq=1,
        )
        global_metric, _ = fl_model.evaluate(
            data, label, batch_size=32, random_seed=1234
        )
        assert (
            global_metric[0].result().numpy()
            == history.global_history['val_multiclassaccuracy'][-1]
        )
        model_path = os.path.join(_temp_dir, "base_model")
        fl_model.save_model(model_path=model_path, is_test=True)
        # FIXME(fengjun.feng)
        # assert os.path.exists(model_path) != None

        # test load model
        new_model_def = TorchModel(
            model_fn=MlpNet,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=3, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=3, average='micro'
                ),
            ],
        )
        new_fed_model = FLModel(
            server=sf_simulation_setup_devices.carol,
            device_list=device_list,
            model=new_model_def,
            aggregator=None,
            backend="torch",
            random_seed=1234,
        )
        new_fed_model.load_model(model_path=model_path, is_test=True)
        reload_metric, _ = new_fed_model.evaluate(
            data, label, batch_size=128, random_seed=1234
        )

        np.testing.assert_equal(
            [m.result().numpy() for m in global_metric],
            [m.result().numpy() for m in reload_metric],
        )

class TestFLModelTorchDataBuilder:
    def test_torch_model_databuilder(self, sf_simulation_setup_devices):
        aggregator = PlainAggregator(sf_simulation_setup_devices.carol)
        hdf = load_iris(
            parts=[sf_simulation_setup_devices.alice, sf_simulation_setup_devices.bob],
            aggregator=aggregator,
        )

        label = hdf['class']

        def create_dataset_builder(
            batch_size=32,
            train_split=0.8,
            shuffle=True,
            random_seed=1234,
        ):
            def dataset_builder(x, stage="train"):
                import math

                import numpy as np
                from torch.utils.data import DataLoader
                from torch.utils.data.sampler import SubsetRandomSampler
                from torchvision import datasets, transforms

                # Define dataset
                flower_transform = transforms.Compose(
                    [
                        transforms.Resize((180, 180)),
                        transforms.ToTensor(),
                    ]
                )
                flower_dataset = datasets.ImageFolder(x, transform=flower_transform)
                dataset_size = len(flower_dataset)
                # Define sampler

                indices = list(range(dataset_size))
                if shuffle:
                    np.random.seed(random_seed)
                    np.random.shuffle(indices)
                split = int(np.floor(train_split * dataset_size))
                train_indices, val_indices = indices[:split], indices[split:]
                train_sampler = SubsetRandomSampler(train_indices)
                valid_sampler = SubsetRandomSampler(val_indices)

                # Define databuilder
                train_loader = DataLoader(
                    flower_dataset, batch_size=batch_size, sampler=train_sampler
                )
                valid_loader = DataLoader(
                    flower_dataset, batch_size=batch_size, sampler=valid_sampler
                )

                # Return
                if stage == "train":
                    train_step_per_epoch = math.ceil(split / batch_size)
                    return train_loader, train_step_per_epoch
                elif stage == "eval":
                    eval_step_per_epoch = math.ceil((dataset_size - split) / batch_size)
                    return valid_loader, eval_step_per_epoch

            return dataset_builder

        data_builder_dict = {
            sf_simulation_setup_devices.alice: create_dataset_builder(
                batch_size=32,
                train_split=0.8,
                shuffle=False,
                random_seed=1234,
            ),
            sf_simulation_setup_devices.bob: create_dataset_builder(
                batch_size=32,
                train_split=0.8,
                shuffle=False,
                random_seed=1234,
            ),
        }

        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(ScaffoldOptimizer, lr=1e-3)
        model_def = TorchModel(
            model_fn=ConvRGBNet,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=5, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=5, average='micro'
                ),
            ],
        )
        device_list = [
            sf_simulation_setup_devices.alice,
            sf_simulation_setup_devices.bob,
        ]

        fl_model = FLModel(
            server=sf_simulation_setup_devices.carol,
            device_list=device_list,
            model=model_def,
            aggregator=aggregator,
            sparsity=0.0,
            strategy="fed_scaffold",
            backend="torch",
            random_seed=1234,
        )
        data = {
            sf_simulation_setup_devices.alice: path_to_flower_dataset,
            sf_simulation_setup_devices.bob: path_to_flower_dataset,
        }

        history = fl_model.fit(
            data,
            None,
            validation_data=data,
            epochs=2,
            aggregate_freq=1,
            dataset_builder=data_builder_dict,
        )
        global_metric, _ = fl_model.evaluate(
            data,
            label,
            batch_size=32,
            dataset_builder=data_builder_dict,
        )

        assert (
            global_metric[0].result().numpy()
            == history.global_history['val_multiclassaccuracy'][-1]
        )
        model_path = os.path.join(_temp_dir, "base_model")
        fl_model.save_model(model_path=model_path, is_test=True)
        # FIXME(fengjun.feng)
        # assert os.path.exists(model_path) != None
