import os
import tempfile
import time

from secretflow.device.driver import wait

from tests.basecase import DeviceTestCase


class TestWait(DeviceTestCase):
    def test_wait_should_ok(self):
        # GIVEN
        _, tmp = tempfile.mkstemp()

        def simple_write(filepath):
            time.sleep(5)
            with open(filepath, 'w') as f:
                f.write('This is alice.')

        o = self.alice(simple_write)(tmp)

        # WHEN
        wait([o])

        # THEN
        with open(tmp, 'r') as f:
            self.assertEqual(f.read(), 'This is alice.')

        os.remove(tmp)

    def test_wait_should_error_when_task_failure(self):
        # GIVEN
        def task():
            raise AssertionError('This exception is expected by design.')

        o = self.alice(task)()

        # WHEN
        with self.assertRaisesRegex(
            AssertionError, 'This exception is expected by design.'
        ):
            wait([o])
