import copy
from typing import Tuple

import numpy as np
import secretflow as sf
from secretflow.data import FedNdarray
from secretflow.ml.linear import HESSLogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from tests.basecase import DeviceTestCase
from sklearn.metrics import roc_auc_score


class TestHESSLogisticRegression(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        config_x = copy.deepcopy(cls.heu.config)
        config_x['encoding'] = {'cleartext_type': 'DT_I32'}
        cls.heu_x = sf.HEU(config_x, cls.spu.cluster_def['runtime_config']['field'])

        config_y = copy.deepcopy(config_x)
        sk_keeper = config_y["sk_keeper"]
        evaluator = config_y["evaluators"][0]
        config_y["sk_keeper"] = evaluator
        config_y["evaluators"][0] = sk_keeper

        cls.heu_y = sf.HEU(config_y, cls.spu.cluster_def['runtime_config']['field'])

    def load_dataset(self):
        def _load_dataset(return_label=False) -> Tuple[np.ndarray, np.ndarray]:
            features, label = load_breast_cancer(return_X_y=True)

            if return_label:
                return features[:, 15:], label
            else:
                return features[:, :15], None

        def _transform(data):
            scaler = StandardScaler()
            return scaler.fit_transform(data)

        x1, _ = self.alice(_load_dataset)(return_label=False)
        x2, y = self.bob(_load_dataset)(return_label=True)

        x1 = self.alice(_transform)(x1)
        x2 = self.bob(_transform)(x2)

        self.x = FedNdarray(partitions={x1.device: x1, x2.device: x2})
        self.y = FedNdarray(partitions={y.device: y})

    def test_model(self):
        self.load_dataset()

        model = HESSLogisticRegression(self.spu, self.heu_x, self.heu_y)
        model.fit(self.x, self.y, epochs=3, batch_size=64)

        y = self.y.partitions[self.bob]
        y_pred = model.predict(self.x)

        auc = self.bob(roc_auc_score)(y, y_pred)
        acc = self.bob(lambda y_true, y_pred: np.mean((y_pred > 0.5) == y_true))(
            y, y_pred
        )

        auc = sf.reveal(auc)
        acc = sf.reveal(acc)
        print(f'auc={auc}, acc={acc}')

        self.assertGreater(auc, 0.99)
        self.assertGreater(acc, 0.94)
