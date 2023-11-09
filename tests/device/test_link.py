import logging
import random
import time

import numpy as np

from secretflow.device import proxy, PYUObject, reveal
from secretflow.device.link import init_link, Link


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


@proxy(PYUObject, _simulation_max_concurrency=2)
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


def _test_parameter_server(devices):
    ps = ParameterServer(
        device=devices.davy, worker_device=[devices.alice, devices.bob, devices.carol]
    )

    workers = [
        Worker(device=devices.alice, ps_device=devices.davy),
        Worker(device=devices.bob, ps_device=devices.davy),
        Worker(device=devices.carol, ps_device=devices.davy),
    ]

    # 集群组网
    for worker in workers:
        init_link(worker, ps)

    init_link(ps, workers)

    epochs, steps_per_epoch = 1, 10
    res = [worker.run(epochs, steps_per_epoch) for worker in workers]
    res.append(ps.run(epochs, steps_per_epoch))

    reveal(res)  # wait all tasks done


def test_parameter_server_prod(sf_production_setup_devices_grpc):
    _test_parameter_server(sf_production_setup_devices_grpc)


def test_parameter_server_sim(sf_simulation_setup_devices):
    _test_parameter_server(sf_simulation_setup_devices)
