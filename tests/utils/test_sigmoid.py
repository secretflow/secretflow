from tests.basecase import DeviceTestCase
import numpy as np
from secretflow.device.driver import reveal

from secretflow.utils import sigmoid


class SigmoidAllCase(DeviceTestCase):
    def get_spu_x(self, x):
        pyu = self.alice(lambda: x)()
        return pyu.to(self.spu)

    def do_test(self, fn):
        x = np.random.normal(0, 5, size=(5, 5))
        spu = reveal(self.spu(fn)(self.get_spu_x(x)))
        jnp = fn(x)
        np.testing.assert_almost_equal(spu, jnp, decimal=2)
        np.testing.assert_almost_equal(fn(0), 0.5, decimal=2)
        np.testing.assert_almost_equal(fn(100), 1, decimal=2)
        np.testing.assert_almost_equal(fn(-100), 0, decimal=2)

    def test_t1(self):
        self.do_test(sigmoid.t1_sig)

    def test_t3(self):
        self.do_test(sigmoid.t3_sig)

    def test_t5(self):
        self.do_test(sigmoid.t5_sig)

    def test_seg3(self):
        self.do_test(sigmoid.seg3_sig)

    def test_df(self):
        self.do_test(sigmoid.df_sig)

    def test_sr(self):
        self.do_test(sigmoid.sr_sig)
