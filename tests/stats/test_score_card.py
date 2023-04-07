import numpy as np

import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.stats import ScoreCard
from tests.basecase import MultiDriverDeviceTestCase


class TestScoreCard(MultiDriverDeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.sc = ScoreCard(20, 600, 20)
        cls.pred1 = np.random.random((10, 1))
        cls.ds1 = FedNdarray(
            partitions={cls.alice: cls.alice(lambda x: x)(cls.pred1)},
            partition_way=PartitionWay.VERTICAL,
        )

        cls.alice_pred2 = np.random.random((10, 1))
        cls.bob_pred2 = np.random.random((10, 1))
        cls.ds2 = FedNdarray(
            partitions={
                cls.alice: cls.alice(lambda x: x)(cls.alice_pred2),
                cls.bob: cls.bob(lambda x: x)(cls.bob_pred2),
            },
            partition_way=PartitionWay.HORIZONTAL,
        )

    def test_sc(self):
        scord = self.sc.transform(self.ds1)
        assert scord.shape[0] == 10
        assert len(scord.partitions) == 1
        scord1 = sf.reveal(list(scord.partitions.items())[0])
        print(f"pred \n{self.pred1}\n -> \n{scord1}")

        scord = self.sc.transform(self.ds2)
        assert scord.shape[0] == 20
        assert len(scord.partitions) == 2
        scord2_alice = sf.reveal(list(scord.partitions.items())[0])
        scord2_bob = sf.reveal(list(scord.partitions.items())[1])
        print(f"pred \n{self.alice_pred2}\n -> \n{scord2_alice}")
        print(f"pred \n{self.bob_pred2}\n -> \n{scord2_bob}")
