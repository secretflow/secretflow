from unittest import TestCase
from secretflow.ml.nn.fl.metrics import Mean, Precision
from secretflow.ml.nn.fl.utils import History


class TestHistory(TestCase):
    def test_record_should_ok(self):
        # GIVEN
        mean = Mean(name='mean', total=10.0, count=2.0)
        acc = Precision('acc', [0.5], [10.0], [20.0])
        his = History()

        # WHEN
        his.record_global_history(metrics=[mean, acc])
        his.record_local_history('alice', metrics=[mean])

        # THEN
        self.assertEqual(his.global_history['mean'], [mean.result()])
        self.assertEqual(his.global_history['acc'], [acc.result().numpy()])
        self.assertEqual(
            his.global_detailed_history['mean'][0].result().numpy(),
            mean.result().numpy(),
        )
        self.assertEqual(
            his.global_detailed_history['acc'][0].result().numpy(), acc.result().numpy()
        )
        self.assertEqual(his.local_history['alice']['mean'], [mean.result().numpy()])
        self.assertEqual(
            his.local_detailed_history['alice']['mean'][0].result().numpy(),
            mean.result().numpy(),
        )
