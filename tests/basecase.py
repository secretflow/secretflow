#!/usr/bin/env python3
# *_* coding: utf-8 *_*
"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""
import unittest

import numpy as np
import ray
import spu

import secretflow as sf
from secretflow.utils.testing import unused_tcp_port

aby3_cluster_def = {
    'nodes': [
        {
            'party': 'alice',
            'id': 'local:0',
            'address': f'127.0.0.1:{unused_tcp_port()}',
        },
        {'party': 'bob', 'id': 'local:1', 'address': f'127.0.0.1:{unused_tcp_port()}'},
        {
            'party': 'carol',
            'id': 'local:2',
            'address': f'127.0.0.1:{unused_tcp_port()}',
        },
    ],
    'runtime_config': {
        'protocol': spu.spu_pb2.ABY3,
        'field': spu.spu_pb2.FM64,
        'enable_pphlo_profile': False,
        'enable_hal_profile': False,
        'enable_pphlo_trace': False,
        'enable_action_trace': False,
    },
}

semi2k_cluster_def = {
    'nodes': [
        {
            'party': 'alice',
            'id': 'local:0',
            'address': f'127.0.0.1:{unused_tcp_port()}',
        },
        {'party': 'bob', 'id': 'local:1', 'address': f'127.0.0.1:{unused_tcp_port()}'},
    ],
    'runtime_config': {
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
        'enable_pphlo_profile': False,
        'enable_hal_profile': False,
        'enable_pphlo_trace': False,
        'enable_action_trace': False,
    },
}

heu_config = {
    'sk_keeper': {'party': 'alice'},
    'evaluators': [{'party': 'bob'}],
    # The HEU working mode, choose from PHEU / LHEU / FHEU_ROUGH / FHEU
    'mode': 'PHEU',
    'he_parameters': {
        'schema': 'paillier',
        'key_pair': {'generate': {'bit_size': 2048}},
    },
}


class DeviceTestCaseBase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        print(f"##### Testing: {self.__class__.__name__}.{self._testMethodName} #####")

    @classmethod
    def setUpClass(cls) -> None:
        '''
        You can set `cls.num_cpus = xx` for larger resources.
        '''
        super().setUpClass()
        num_cpus = cls.num_cpus if hasattr(cls, 'num_cpus') else 16
        sf.init(
            ['alice', 'bob', 'carol', 'davy', 'eric'],
            num_cpus=num_cpus,
            log_to_driver=False,
        )

        cls.alice = sf.PYU('alice')
        cls.bob = sf.PYU('bob')
        cls.carol = sf.PYU('carol')
        cls.davy = sf.PYU('davy')
        cls.eric = sf.PYU('eric')

    @classmethod
    def tearDownClass(cls) -> None:
        ray.shutdown()


class DeviceTestCase(DeviceTestCaseBase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.spu = sf.SPU(semi2k_cluster_def)
        cls.heu = sf.HEU(heu_config, semi2k_cluster_def['runtime_config']['field'])


class ABY3DeviceTestCase(DeviceTestCaseBase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.spu = sf.SPU(aby3_cluster_def)
        cls.heu = sf.HEU(heu_config, aby3_cluster_def['runtime_config']['field'])


def array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    # Ignore nan.
    return ((a == b) | ((a != a) & (b != b))).all()
