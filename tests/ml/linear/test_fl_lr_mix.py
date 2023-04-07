from typing import List

import numpy as np
import spu
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

import secretflow as sf
from secretflow.data.base import Partition
from secretflow.data.mix import MixDataFrame
from secretflow.data.split import train_test_split
from secretflow.data.vertical import VDataFrame
from secretflow.ml.linear.fl_lr_mix import FlLogisticRegressionMix
from secretflow.preprocessing.scaler import StandardScaler
from secretflow.security.aggregation import SecureAggregator
from tests.basecase import MultiDriverDeviceTestCase


class TestFlLrMix(MultiDriverDeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.num_cpus = 64
        super().setUpClass()

        def heu_config(sk_keeper: str, evaluators: List[str]):
            return {
                'sk_keeper': {'party': sk_keeper},
                'evaluators': [{'party': evaluator} for evaluator in evaluators],
                'mode': 'PHEU',
                'he_parameters': {
                    'schema': 'paillier',
                    'key_pair': {'generate': {'bit_size': 2048}},
                },
            }

        cls.heu0 = sf.HEU(heu_config('alice', ['bob', 'carol']), spu.spu_pb2.FM128)
        cls.heu1 = sf.HEU(heu_config('alice', ['bob', 'davy']), spu.spu_pb2.FM128)

        features, label = load_breast_cancer(return_X_y=True, as_frame=True)
        label = label.to_frame()
        feat_list = [
            features.iloc[:, :10],
            features.iloc[:, 10:20],
            features.iloc[:, 20:],
        ]
        x = VDataFrame(
            partitions={
                cls.alice: Partition(cls.alice(lambda: feat_list[0])()),
                cls.bob: Partition(cls.bob(lambda: feat_list[1])()),
                cls.carol: Partition(cls.carol(lambda: feat_list[2])()),
            }
        )
        x = StandardScaler().fit_transform(x)
        y = VDataFrame(partitions={cls.alice: Partition(cls.alice(lambda: label)())})
        x1, x2 = train_test_split(x, train_size=0.5, shuffle=False)
        y1, y2 = train_test_split(y, train_size=0.5, shuffle=False)

        # davy holds same x
        x2_davy = x2.partitions[cls.carol].data.to(cls.davy)
        del x2.partitions[cls.carol]
        x2.partitions[cls.davy] = Partition(x2_davy)

        cls.x = MixDataFrame(partitions=[x1, x2])
        cls.y = MixDataFrame(partitions=[y1, y2])
        cls.y_arr = label.values

    def test_model_should_ok_when_fit_dataframe(self):
        # GIVEN
        aggregator0 = SecureAggregator(self.alice, [self.alice, self.bob, self.carol])
        aggregator1 = SecureAggregator(self.alice, [self.alice, self.bob, self.davy])
        # aggregator2 = SecureAggregator(self.alice, [self.alice, self.bob, self.eric])

        model = FlLogisticRegressionMix()

        # WHEN
        model.fit(
            self.x,
            self.y,
            epochs=3,
            batch_size=64,
            learning_rate=0.1,
            aggregators=[aggregator0, aggregator1],
            heus=[self.heu0, self.heu1],
        )

        y_pred = np.concatenate(sf.reveal(model.predict(self.x)))

        auc = roc_auc_score(self.y_arr, y_pred)
        acc = np.mean((y_pred > 0.5) == self.y_arr)

        # THEN
        auc = sf.reveal(auc)
        acc = sf.reveal(acc)
        print(f'auc={auc}, acc={acc}')

        self.assertGreater(auc, 0.98)
        self.assertGreater(acc, 0.93)
