import logging
import random
import time

import numpy as np

from secretflow.device import proxy, PYUObject, reveal
from tests.basecase import DeviceTestCase
from secretflow.device.link import Link


@proxy(PYUObject, max_concurrency=2)
class Worker(Link):
    def __init__(self, device=None, ps_device=None):
        self._ps_device = ps_device
        super().__init__(device)

    def run(self, epochs, steps_per_epoch):
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                step_id = epoch * steps_per_epoch + step

                # simulate latency
                time.sleep(random.uniform(0.1, 0.5))
                weights = [np.random.rand(3, 4)]

                self.send('weights', weights, self._ps_device, step_id)
                weights = self.recv('weights', self._ps_device, step_id)
                logging.info(f'worker {self._device} finish step {step_id}')


@proxy(PYUObject, max_concurrency=2)
class ParameterServer(Link):
    def __init__(self, device=None, worker_device=None):
        self._worker_device = worker_device
        super().__init__(device)

    def run(self, epochs, steps_per_epoch):
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                step_id = epoch * steps_per_epoch + step
                weights = self.recv('weights', self._worker_device, step_id)
                weights = [np.average(weight, axis=0) for weight in zip(weights)]
                self.send('weights', weights, self._worker_device, step_id)
                logging.info(f'parameter server {self._device} finish step {step_id}')


class TestLink(DeviceTestCase):
    def test_parameter_server(self):
        ps = ParameterServer(
            device=self.davy, worker_device=[self.alice, self.bob, self.carol]
        )

        workers = [
            Worker(device=self.alice, ps_device=self.davy),
            Worker(device=self.bob, ps_device=self.davy),
            Worker(device=self.carol, ps_device=self.davy),
        ]

        # 集群组网
        for worker in workers:
            worker.initialize({self.davy: ps.data})
        ps.initialize({worker.device: worker.data for worker in workers})

        epochs, steps_per_epoch = 1, 10
        res = [worker.run(epochs, steps_per_epoch) for worker in workers]
        res.append(ps.run(epochs, steps_per_epoch))

        reveal(res)  # wait all tasks done
