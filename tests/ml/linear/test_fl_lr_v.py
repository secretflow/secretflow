import numpy as np
import spu
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

import secretflow as sf
from secretflow.data.base import Partition
from secretflow.data.vertical import VDataFrame
from secretflow.ml.linear.fl_lr_v import FlLogisticRegressionVertical
from secretflow.preprocessing import StandardScaler
from secretflow.security.aggregation.plain_aggregator import PlainAggregator

from tests.basecase import MultiDriverDeviceTestCase


class TestFlLogisticRegressionVertical(MultiDriverDeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.num_cpus = 32
        super().setUpClass()

        heu_config = {
            'sk_keeper': {'party': 'alice'},
            'evaluators': [{'party': 'bob'}, {'party': 'carol'}],
            'mode': 'PHEU',
            'he_parameters': {
                'schema': 'paillier',
                'key_pair': {'generate': {'bit_size': 2048}},
            },
        }

        cls.heu = sf.HEU(heu_config, spu.spu_pb2.FM128)

        features, label = load_breast_cancer(return_X_y=True, as_frame=True)
        label = label.to_frame()
        feat_list = [
            features.iloc[:, :10],
            features.iloc[:, 10:20],
            features.iloc[:, 20:],
        ]
        cls.x = VDataFrame(
            partitions={
                cls.alice: Partition(cls.alice(lambda: feat_list[0])()),
                cls.bob: Partition(cls.bob(lambda: feat_list[1])()),
                cls.carol: Partition(cls.carol(lambda: feat_list[2])()),
            }
        )
        cls.x = StandardScaler().fit_transform(cls.x)
        cls.y = VDataFrame(
            partitions={cls.alice: Partition(cls.alice(lambda: label)())}
        )

    def test_model_should_ok_when_fit_dataframe(self):
        # GIVEN
        aggregator = PlainAggregator(self.alice)

        model = FlLogisticRegressionVertical(
            [self.alice, self.bob, self.carol], aggregator, self.heu
        )

        # WHEN
        model.fit(self.x, self.y, epochs=3, batch_size=64)

        y_pred = model.predict(self.x)

        y = self.y.values.partitions[self.alice]
        auc = self.alice(roc_auc_score)(y, y_pred)
        acc = self.alice(lambda y_true, y_pred: np.mean((y_pred > 0.5) == y_true))(
            y, y_pred
        )

        # THEN
        auc = sf.reveal(auc)
        acc = sf.reveal(acc)
        print(f'auc={auc}, acc={acc}')

        self.assertGreater(auc, 0.99)
        self.assertGreater(acc, 0.94)

    def test_model_should_ok_when_fit_ndarray(self):
        # GIVEN
        aggregator = PlainAggregator(self.alice)

        model = FlLogisticRegressionVertical(
            [self.alice, self.bob, self.carol], aggregator, self.heu
        )
        x = self.x.values
        y = self.y.values

        # WHEN
        model.fit(x, y, epochs=3, batch_size=64)

        y_pred = model.predict(x)

        y = y.partitions[self.alice]
        auc = self.alice(roc_auc_score)(y, y_pred)
        acc = self.alice(lambda y_true, y_pred: np.mean((y_pred > 0.5) == y_true))(
            y, y_pred
        )

        # THEN
        auc = sf.reveal(auc)
        acc = sf.reveal(acc)
        print(f'auc={auc}, acc={acc}')

        self.assertGreater(auc, 0.99)
        self.assertGreater(acc, 0.94)

    def test_fit_should_error_when_mismatch_heu_sk_keeper(self):
        # GIVEN
        aggregator = PlainAggregator(self.alice)

        model = FlLogisticRegressionVertical(
            [self.alice, self.bob, self.carol], aggregator, self.heu
        )
        x = self.x.values
        y = VDataFrame(partitions={self.bob: Partition(self.bob(lambda: [1, 2, 3])())})

        # WHEN
        with self.assertRaisesRegex(
            AssertionError, 'Y party shoule be same with heu sk keeper'
        ):
            model.fit(x, y, epochs=3, batch_size=64)
