#!/usr/bin/env python3
# *_* coding: utf-8 *_*
"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""
import unittest

import multiprocess
import numpy as np
import spu

import secretflow as sf
import secretflow.distributed as sfd
from secretflow.utils.testing import unused_tcp_port

from tests.cluster import cluster, get_self_party

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


def array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    # Ignore nan.
    return ((a == b) | ((a != a) & (b != b))).all()


def semi2k_cluster():
    return {
        'nodes': [
            {
                'party': 'alice',
                'id': 'local:0',
                'address': f'127.0.0.1:{unused_tcp_port()}',
            },
            {
                'party': 'bob',
                'id': 'local:1',
                'address': f'127.0.0.1:{unused_tcp_port()}',
            },
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


def aby3_cluster():
    return {
        'nodes': [
            {
                'party': 'alice',
                'id': 'local:0',
                'address': f'127.0.0.1:{unused_tcp_port()}',
            },
            {
                'party': 'bob',
                'id': 'local:1',
                'address': f'127.0.0.1:{unused_tcp_port()}',
            },
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


class DeviceTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        print(
            f"##### {get_self_party()} start Testing : {self.__class__.__name__}.{self._testMethodName} #####"
        )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not sfd.production_mode():
            num_cpus = getattr(cls, 'num_cpus', 16)
            sf.init(
                ['alice', 'bob', 'carol', 'davy'],
                address='local',
                num_cpus=num_cpus,
                log_to_driver=True,
                omp_num_threads=multiprocess.cpu_count(),
            )
        else:
            sf.init(
                address='local',
                num_cpus=8,
                log_to_driver=True,
                cluster_config=cluster(),
                exit_on_failure_cross_silo_sending=True,
            )

        cls.alice = sf.PYU('alice')
        cls.bob = sf.PYU('bob')
        cls.carol = sf.PYU('carol')
        cls.davy = sf.PYU('davy')

        if not getattr(cls, 'no_spu_semi2k', False):
            cluster_def = sf.reveal(cls.alice(semi2k_cluster)())
            print(cluster_def)
            cls.spu = sf.SPU(
                cluster_def,
                link_desc={
                    'connect_retry_interval_ms': 1000,
                },
            )
            cls.heu = sf.HEU(heu_config, cluster_def['runtime_config']['field'])

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        sf.shutdown()

    def tearDown(self) -> None:
        super().tearDown()
        print(
            f"##### {get_self_party()} finish Testing: {self.__class__.__name__}.{self._testMethodName} #####"
        )


class MultiDriverDeviceTestCase(DeviceTestCase):
    pass


class SingleDriverDeviceTestCase(DeviceTestCase):
    pass


class ABY3DeviceTestCase(DeviceTestCase):
    @classmethod
    def setUpClass(cls):
        cls.no_spu_semi2k = True
        super().setUpClass()
        cluster_def = sf.reveal(cls.alice(aby3_cluster)())
        print(cluster_def)
        cls.spu = sf.SPU(
            cluster_def,
            link_desc={'connect_retry_interval_ms': 1000},
        )
        cls.heu = sf.HEU(heu_config, cluster_def['runtime_config']['field'])


class ABY3MultiDriverDeviceTestCase(ABY3DeviceTestCase):
    pass


class ABY3SingleDriverDeviceTestCase(ABY3DeviceTestCase):
    pass
