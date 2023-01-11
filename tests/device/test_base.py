import numpy as np

import secretflow as sf

from tests.basecase import (MultiDriverDeviceTestCase,
                            SingleDriverDeviceTestCase)


class TestDeviceBase(MultiDriverDeviceTestCase, SingleDriverDeviceTestCase):
    def reveal(self):
        x = np.random.rand(3, 4)
        vals = sf.reveal(
            {self.spu: sf.to(self.spu, x), self.carol: sf.to(self.carol, x)}
        )
        np.testing.assert_almost_equal(x, vals[self.spu], decimal=6)
        np.testing.assert_almost_equal(x, vals[self.carol], decimal=6)
