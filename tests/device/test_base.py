import numpy as np

import secretflow as sf
from tests.basecase import MultiDriverDeviceTestCase, SingleDriverDeviceTestCase


class TestDeviceBase(MultiDriverDeviceTestCase, SingleDriverDeviceTestCase):
    def test_reveal(self):
        x = self.alice(lambda : np.random.rand(3, 4))()
        vals = sf.reveal(
            {
                self.spu: x.to(
                    self.spu,
                ),
                self.carol: x.to(self.carol),
            }
        )
        x = sf.reveal(x)
        np.testing.assert_almost_equal(x, vals[self.spu], decimal=6)
        np.testing.assert_almost_equal(x, vals[self.carol], decimal=6)
