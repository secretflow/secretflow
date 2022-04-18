import numpy as np
from torch import nn
from torch import optim

from secretflow import reveal
from secretflow.model.sl_base import PYUModel
from tests.basecase import DeviceTestCase
from tests.model.model_def import SLConvNet, get_mnist_dataloader, calculate_accu


class TestRawSL(DeviceTestCase):
    def setUp(self) -> None:
        super(TestRawSL, self).setUp()
        self.epochs = 100
        # SetUp for Split Learning
        cols_1 = [0, 2, 3, 4, 5, 14, 15, 19, 21, 24, 26]
        cols_2 = [i for i in range(28) if i not in cols_1]
        self.dl_fn_without_y = {'train': lambda: get_mnist_dataloader(is_train=True, use_cols=cols_1, with_y=False),
                                'test': lambda: get_mnist_dataloader(is_train=False, use_cols=cols_1, with_y=False)}
        self.dl_fn_with_y = {'train': lambda: get_mnist_dataloader(is_train=True, use_cols=cols_2),
                             'test': lambda: get_mnist_dataloader(is_train=False, use_cols=cols_2)}
        self.model_fn_without_y = lambda: SLConvNet(
            output_shape=32, fc_in_dim=72)
        self.model_fn_with_y = lambda: SLConvNet(output_shape=10, fc_in_dim=120,
                                                 embed_shape_from_other=32, with_softmax=True)
        self.loss_fn = nn.NLLLoss
        self.optim_fn = lambda param: optim.Adam(param, lr=1e-3)

    def test_simple_split_learning(self):
        # Model without y
        model1 = PYUModel(device=self.alice, model_fn=self.model_fn_without_y, loss_fn=self.loss_fn,
                          optim_fn=self.optim_fn, dataloader_fn=self.dl_fn_without_y)
        # Model with y
        model2 = PYUModel(device=self.bob, model_fn=self.model_fn_with_y, loss_fn=self.loss_fn,
                          optim_fn=self.optim_fn, dataloader_fn=self.dl_fn_with_y)
        accu = 0
        for i in range(self.epochs):
            model1.zero_grad()
            model2.zero_grad()

            embed, _ = model1.forward()
            embed = reveal(embed)

            # Update only happened on side with label
            grad2 = model2.backward(external_input={'embed_from_other': embed})
            # TODO: need to reveal even sending back?
            model2.apply_gradients(reveal(grad2))

            old_w1 = reveal(model1.get_weights())['conv1.weight']
            embed_grad1 = model2.call_model_fn('return_embed_grad')
            embed_grad1 = reveal(embed_grad1)
            model1.call_model_fn('backward_embed_grad', embed_grad1)
            model1.optim_step()
            new_w1 = reveal(model1.get_weights())['conv1.weight']

            # Check side without label successfully update model
            with self.assertRaises(AssertionError):
                np.testing.assert_array_almost_equal(old_w1, new_w1)

            if (i + 1) % 10 == 0:
                accu = self.eval(model1, model2)

        self.assertLess(0.5, accu)

    @staticmethod
    def eval(model1, model2):
        embed = model1.forward(used_name='test')
        embed, _ = reveal(embed)
        y_pred_y = model2.forward(used_name='test', external_input={
                                  'embed_from_other': embed})
        y_pred, y = reveal(y_pred_y)

        accu = calculate_accu(y_pred, y)
        print(f"Accuracy: {accu}")
        return accu
