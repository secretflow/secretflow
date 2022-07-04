import numpy as np
from jax import numpy as jnp
from torch import nn, optim

from secretflow import reveal
from secretflow.ml.nn.fl_base import PYUModel
from tests.basecase import DeviceTestCase
from tests.ml.nn.model_def import ConvNet, calculate_accu, get_mnist_dataloader


class TestRawFedAVG(DeviceTestCase):
    def setUp(self) -> None:
        super(TestRawFedAVG, self).setUp()
        self.epochs = 100

        # Setup for FedAVG
        self.model_fn = ConvNet
        self.loss_fn = nn.NLLLoss
        self.optim_fn = lambda param: optim.Adam(param, lr=1e-3)
        self.dataloader_fn = {
            'train': lambda: get_mnist_dataloader(is_train=True),
            'test': lambda: get_mnist_dataloader(is_train=False),
        }

    def test_simple_fedavg(self):
        model1 = PYUModel(
            device=self.alice,
            model_fn=self.model_fn,
            loss_fn=self.loss_fn,
            optim_fn=self.optim_fn,
            dataloader_fn=self.dataloader_fn,
        )
        model2 = PYUModel(
            device=self.bob,
            model_fn=self.model_fn,
            loss_fn=self.loss_fn,
            optim_fn=self.optim_fn,
            dataloader_fn=self.dataloader_fn,
        )

        accu = 0.0
        for i in range(self.epochs):
            model1.zero_grad()
            model2.zero_grad()

            old_w1 = reveal(model1.get_weights())['conv1.weight']
            old_w2 = reveal(model2.get_weights())['conv1.weight']
            grad1 = model1.backward()
            grad2 = model2.backward()
            # grad1, grad2 = reveal(grad1), reveal(grad2)
            # TODO: extract this aggregator and formed FedAVG/FedProx/etc.
            aggregate_fn = self.spu(
                lambda grad1, grad2: [
                    jnp.mean(jnp.array([g1, g2]), axis=0)
                    for g1, g2 in zip(grad1, grad2)
                ]
            )
            grad = aggregate_fn(grad1.to(self.spu), grad2.to(self.spu))
            model1.apply_gradients(grad.to(model1.device))
            model2.apply_gradients(grad.to(model2.device))
            new_w1 = reveal(model1.get_weights())['conv1.weight']
            new_w2 = reveal(model2.get_weights())['conv1.weight']

            # Check weights change
            with self.assertRaises(AssertionError):
                np.testing.assert_array_almost_equal(new_w1, old_w1)
                np.testing.assert_array_almost_equal(new_w2, old_w2)

            if (i + 1) % 10 == 0:
                accu = self.eval(model1, model2)

        self.assertLess(0.5, accu)

    @staticmethod
    def eval(model1, model2):
        y_pred1, y1 = model1.forward(used_name='test')
        y_pred2, y2 = model2.forward(used_name='test')
        y_pred1, y1 = reveal([y_pred1, y1])
        y_pred2, y2 = reveal([y_pred2, y2])
        accu1 = calculate_accu(y_pred1, y1)
        accu2 = calculate_accu(y_pred2, y2)
        accu = (accu1 + accu2) / 2
        print(f"Accuracy: {accu}")
        return accu
