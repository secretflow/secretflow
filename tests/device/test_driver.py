import tempfile
import time

import numpy as np

from secretflow.device.device.spu import SPUObject
from secretflow.device.driver import reveal, to, wait
from tests.basecase import MultiDriverDeviceTestCase, SingleDriverDeviceTestCase


class TestWait(MultiDriverDeviceTestCase, SingleDriverDeviceTestCase):
    def test_wait_should_ok(self):
        # GIVEN
        def simple_write():
            _, temp_file = tempfile.mkstemp()
            time.sleep(5)
            with open(temp_file, 'w') as f:
                f.write('This is alice.')
            return temp_file

        o = self.alice(simple_write)()

        # WHEN
        wait([o])

        # THEN
        def check(temp_file):
            with open(temp_file, 'r') as f:
                assert f.read() == 'This is alice.'
            return True

        file_path = reveal(o)
        self.assertTrue(reveal(self.alice(check)(file_path)))

    def wait_should_error_when_task_failure(self):
        # GIVEN
        def task():
            raise AssertionError('This exception is expected by design.')

        o = self.alice(task)()

        # WHEN
        with self.assertRaisesRegex(
            AssertionError, 'This exception is expected by design.'
        ):
            wait([o])

    def test_spu_reveal(self):
        with self.assertRaises(ValueError):
            x = to(self.spu, 32)

        x = to(self.alice, 32).to(self.spu)
        self.assertTrue(isinstance(x, SPUObject))

        x_ = reveal(x)
        self.assertTrue(isinstance(x_, np.ndarray))
        self.assertEqual(x_, 32)
