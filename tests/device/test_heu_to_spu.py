import numpy as np
import pytest

import secretflow.device as ft
from secretflow import reveal


def _test_device(devices):
    x = ft.with_device(devices.alice)(np.random.rand)(3, 4)
    x_ = x.to(devices.heu)
    assert x_.device == devices.heu

    # Can't place a user-defined function to HEU Device
    with pytest.raises(NotImplementedError):

        @ft.with_device(devices.heu)
        def add(a, b):
            return a + b

        y = add(x_, x_)

    # Can't convert an HEUTensor to CPUTensor without secret key
    with pytest.raises(AssertionError):
        y = x_.to(devices.bob)

    y = x_.to(devices.alice)
    assert y.device == devices.alice
    np.testing.assert_almost_equal(reveal(x), reveal(y), decimal=4)

    y_spu = x_.to(devices.spu)
    assert y_spu.device == devices.spu
    np.testing.assert_almost_equal(reveal(x), reveal(y_spu), decimal=4)


def test_device_prod(sf_production_setup_devices_cheetah):
    _test_device(sf_production_setup_devices_cheetah)
