#!/usr/bin/env python3
# *_* coding: utf-8 *_*
"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""
import unittest

import numpy as np
import ppu
import ray

import secretflow as sf

cluster_def = {
    'nodes': [
        {
            'party': 'alice',
            'id': 'local:0',
            'address': '127.0.0.1:12345'
        },
        {
            'party': 'bob',
            'id': 'local:1',
            'address': '127.0.0.1:12346'
        },
    ],
    'runtime_config': {
        'protocol': ppu.ppu_pb2.SEMI2K,
        'field': ppu.ppu_pb2.FM128,
        'sigmoid_mode': ppu.ppu_pb2.REAL,
    }
}

heu_config = {
    'generator': 'alice',
    'evaluator': 'bob',
    'key_size': 2048,
}


class DeviceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # TODO: check the resources setting since cluster requires more then we expected
        # reproduce the warning by using cluster_util.Cluster and add node with 'alice'...resources
        sf.init(['alice', 'bob', 'carol'], num_cpus=8, log_to_driver=True)

        cls.alice = sf.PYU('alice')
        cls.bob = sf.PYU('bob')
        cls.carol = sf.PYU('carol')
        cls.ppu = sf.PPU(cluster_def)
        cls.heu = sf.HEU(heu_config)

    @classmethod
    def tearDownClass(cls) -> None:
        ray.shutdown()


def array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    # Ignore nan.
    return ((a == b) | ((a != a) & (b != b))).all()
