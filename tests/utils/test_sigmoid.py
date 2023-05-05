import numpy as np

from secretflow.device.driver import reveal
from secretflow.utils import sigmoid


def get_spu_x(env, x):
    pyu = env.alice(lambda: x)()
    return pyu.to(env.spu)


def do_test(env, fn):
    x_ = env.alice(lambda: np.random.normal(0, 5, size=(5, 5)))()
    x = reveal(x_)
    spu = reveal(env.spu(fn)(x_.to(env.spu)))
    jnp = fn(x)
    np.testing.assert_almost_equal(spu, jnp, decimal=2)
    np.testing.assert_almost_equal(fn(0), 0.5, decimal=2)
    np.testing.assert_almost_equal(fn(100), 1, decimal=2)
    np.testing.assert_almost_equal(fn(-100), 0, decimal=2)


def test_t1(sf_production_setup_devices):
    do_test(sf_production_setup_devices, sigmoid.t1_sig)


def test_t3(sf_production_setup_devices):
    do_test(sf_production_setup_devices, sigmoid.t3_sig)


def test_t5(sf_production_setup_devices):
    do_test(sf_production_setup_devices, sigmoid.t5_sig)


def test_seg3(sf_production_setup_devices):
    do_test(sf_production_setup_devices, sigmoid.seg3_sig)


def test_df(sf_production_setup_devices):
    do_test(sf_production_setup_devices, sigmoid.df_sig)


def test_sr(sf_production_setup_devices):
    do_test(sf_production_setup_devices, sigmoid.sr_sig)
