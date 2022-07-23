#!/usr/bin/env python3
# *_* coding: utf-8 *_*
"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""
import unittest

import numpy as np
import spu
import ray

import secretflow as sf

cluster_def = {
    'nodes': [
        {'party': 'alice', 'id': 'local:0', 'address': '127.0.0.1:12345'},
        {'party': 'bob', 'id': 'local:1', 'address': '127.0.0.1:12346'},
    ],
    'runtime_config': {
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
        'sigmoid_mode': spu.spu_pb2.RuntimeConfig.SIGMOID_REAL,
    },
}

heu_config = {
    'sk_keeper': {'party': 'alice'},
    'evaluators': [{'party': 'bob'}],
    # The HEU working mode, choose from PHEU / LHEU / FHEU_ROUGH / FHEU
    'mode': 'PHEU',
    'he_parameters': {
        'schema': 'zpaillier',
        'key_pair': {
            'generate': {
                'bit_size': 2048,
            },
        },
    },
}


class DeviceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # TODO: check the resources setting since cluster requires more then we expected
        # reproduce the warning by using cluster_util.Cluster and add node with 'alice'...resources
        sf.init(['alice', 'bob', 'carol', 'davy'], num_cpus=16, log_to_driver=False)

        cls.alice = sf.PYU('alice')
        cls.bob = sf.PYU('bob')
        cls.carol = sf.PYU('carol')
        cls.davy = sf.PYU('davy')
        cls.spu = sf.SPU(cluster_def)
        cls.heu = sf.HEU(heu_config, cluster_def['runtime_config']['field'])

    @classmethod
    def tearDownClass(cls) -> None:
        ray.shutdown()


def array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    # Ignore nan.
    return ((a == b) | ((a != a) & (b != b))).all()
