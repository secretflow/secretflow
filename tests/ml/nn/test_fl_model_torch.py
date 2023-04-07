#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""

import os
import tempfile

import numpy as np
import tensorflow as tf
from torch import nn, optim
from torchmetrics import Accuracy, Precision

from secretflow.device import reveal
from secretflow.ml.nn import FLModel
from secretflow.ml.nn.fl.backend.torch.utils import TorchModel
from secretflow.ml.nn.fl.compress import COMPRESS_STRATEGY
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
from secretflow.preprocessing.encoder import OneHotEncoder
from secretflow.security.aggregation import PlainAggregator, SparsePlainAggregator
from secretflow.security.privacy import DPStrategyFL, GaussianModelDP
from secretflow.utils.simulation.datasets import load_iris, load_mnist
from tests.basecase import MultiDriverDeviceTestCase, SingleDriverDeviceTestCase
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


class TestFLModelTorchMnist(MultiDriverDeviceTestCase):
    def torch_model_with_mnist(
        self, model_def, data, label, strategy, backend, **kwargs
    ):

        device_list = [self.alice, self.bob]
        server = self.carol

        if strategy in COMPRESS_STRATEGY:
            aggregator = SparsePlainAggregator(server)
        else:
            aggregator = PlainAggregator(server)

        # spcify params
        dp_spent_step_freq = kwargs.get('dp_spent_step_freq', None)

        fl_model = FLModel(
            server=server,
            device_list=device_list,
            model=model_def,
            aggregator=aggregator,
            strategy=strategy,
            backend=backend,
            random_seed=1234,
            **kwargs,
        )
        history = fl_model.fit(
            data,
            label,
            validation_data=(data, label),
            epochs=2,
            batch_size=128,
            aggregate_freq=2,
            dp_spent_step_freq=dp_spent_step_freq,
        )
        result = fl_model.predict(data, batch_size=128)
        self.assertEquals(len(reveal(result[device_list[0]])), 4000)
        global_metric, _ = fl_model.evaluate(
            data, label, batch_size=128, random_seed=1234
        )

        self.assertEquals(
            global_metric[0].result().numpy(),
            history.global_history['val_multiclassaccuracy'][-1],
        )

        self.assertGreater(global_metric[0].result().numpy(), 0.5)

        model_path_test = os.path.join(_temp_dir, "base_model")
        fl_model.save_model(model_path=model_path_test, is_test=True)
        model_path_dict = {
            self.alice: os.path.join(_temp_dir, "alice_model"),
            self.bob: os.path.join(_temp_dir, "bob_model"),
        }
        fl_model.save_model(model_path=model_path_dict, is_test=False)

        new_fed_model = FLModel(
            server=server,
            device_list=device_list,
            model=model_def,
            aggregator=None,
            backend=backend,
            random_seed=1234,
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

    def test_torch_model(self):
        (_, _), (mnist_data, mnist_label) = load_mnist(
            parts={self.alice: 0.4, self.bob: 0.6},
            normalized_x=True,
            categorical_y=True,
            is_torch=True,
        )

        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(optim.Adam, lr=1e-2)
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

        # Test fed_avg_w with mnist
        self.torch_model_with_mnist(
            model_def=model_def,
            data=mnist_data,
            label=mnist_label,
            strategy='fed_avg_w',
            backend="torch",
        )

        # Test fed_avg_g with mnist
        self.torch_model_with_mnist(
            model_def=model_def,
            data=mnist_data,
            label=mnist_label,
            strategy='fed_avg_g',
            backend="torch",
        )

        # Test fed_avg_u with mnist
        self.torch_model_with_mnist(
            model_def=model_def,
            data=mnist_data,
            label=mnist_label,
            strategy='fed_avg_u',
            backend="torch",
        )

        # Test fed_prox with mnist
        self.torch_model_with_mnist(
            model_def=model_def,
            data=mnist_data,
            label=mnist_label,
            strategy='fed_prox',
            backend="torch",
            mu=0.1,
        )

        # Test fed_stc with mnist
        self.torch_model_with_mnist(
            model_def=model_def,
            data=mnist_data,
            label=mnist_label,
            strategy='fed_stc',
            backend="torch",
            sparsity=0.9,
        )

        # Test fed_scr with mnist
        self.torch_model_with_mnist(
            model_def=model_def,
            data=mnist_data,
            label=mnist_label,
            strategy='fed_scr',
            backend="torch",
            threshold=0.8,
        )

        # Define DP operations
        gaussian_model_gdp = GaussianModelDP(
            noise_multiplier=0.001,
            l2_norm_clip=0.1,
            num_clients=2,
            is_secure_generator=False,
        )
        dp_strategy_fl = DPStrategyFL(model_gdp=gaussian_model_gdp)
        dp_spent_step_freq = 10

        self.torch_model_with_mnist(
            model_def=model_def,
            data=mnist_data,
            label=mnist_label,
            strategy='fed_stc',
            backend="torch",
            threshold=0.9,
            dp_strategy=dp_strategy_fl,
            dp_spent_step_freq=dp_spent_step_freq,
        )


class TestFLModelTorchMlp(MultiDriverDeviceTestCase):
    def test_torch_model_mlp(self):
        aggregator = PlainAggregator(self.carol)
        hdf = load_iris(parts=[self.alice, self.bob], aggregator=aggregator)

        label = hdf['class']
        # do preprocess
        encoder = OneHotEncoder()
        label = encoder.fit_transform(label)

        data = hdf.drop(columns='class', inplace=False)
        data = data.fillna(data.mean(numeric_only=True).to_dict())

        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(optim.Adam, lr=5e-3)
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
        device_list = [self.alice, self.bob]

        fl_model = FLModel(
            server=self.carol,
            device_list=device_list,
            model=model_def,
            aggregator=aggregator,
            sparsity=0.0,
            strategy="fed_avg_w",
            backend="torch",
            random_seed=1234,
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
        self.assertEquals(
            global_metric[0].result().numpy(),
            history.global_history['val_multiclassaccuracy'][-1],
        )
        model_path = os.path.join(_temp_dir, "base_model")
        fl_model.save_model(model_path=model_path, is_test=True)
        self.assertIsNotNone(os.path.exists(model_path))

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
            server=self.carol,
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


class TestFLModelTorchDataBuilder(SingleDriverDeviceTestCase):
    def test_torch_model_databuilder(self):
        aggregator = PlainAggregator(self.carol)
        hdf = load_iris(parts=[self.alice, self.bob], aggregator=aggregator)

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
            self.alice: create_dataset_builder(
                batch_size=32,
                train_split=0.8,
                shuffle=False,
                random_seed=1234,
            ),
            self.bob: create_dataset_builder(
                batch_size=32,
                train_split=0.8,
                shuffle=False,
                random_seed=1234,
            ),
        }

        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(optim.Adam, lr=1e-3)
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
        device_list = [self.alice, self.bob]

        fl_model = FLModel(
            server=self.carol,
            device_list=device_list,
            model=model_def,
            aggregator=aggregator,
            sparsity=0.0,
            strategy="fed_avg_w",
            backend="torch",
            random_seed=1234,
        )
        data = {
            self.alice: path_to_flower_dataset,
            self.bob: path_to_flower_dataset,
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

        self.assertEquals(
            global_metric[0].result().numpy(),
            history.global_history['val_multiclassaccuracy'][-1],
        )
        model_path = os.path.join(_temp_dir, "base_model")
        fl_model.save_model(model_path=model_path, is_test=True)
        self.assertIsNotNone(os.path.exists(model_path))
