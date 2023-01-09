#!/usr/bin/env python3
# *_* coding: utf-8 *_*

""" Main test function """
import multiprocessing
import signal
import unittest
import xml.etree.ElementTree as ET
import sys
from typing import List

import xmlrunner

from tests.cluster import cluster, get_self_party, set_self_party


class MultiDriverTestLoader(unittest.TestLoader):
    def getTestCaseNames(self, testCaseClass):
        from tests.basecase import (
            ABY3MultiDriverDeviceTestCase,
            MultiDriverDeviceTestCase,
        )

        if issubclass(
            testCaseClass, (MultiDriverDeviceTestCase, ABY3MultiDriverDeviceTestCase)
        ):
            return super().getTestCaseNames(testCaseClass)
        else:
            return []


class SingleDriverTestLoader(unittest.TestLoader):
    def getTestCaseNames(self, testCaseClass):
        from tests.basecase import (
            ABY3SingleDriverDeviceTestCase,
            SingleDriverDeviceTestCase,
        )

        if issubclass(
            testCaseClass, (SingleDriverDeviceTestCase, ABY3SingleDriverDeviceTestCase)
        ):
            return super().getTestCaseNames(testCaseClass)
        else:
            return []


def party_run(self_party):
    set_self_party(self_party)

    from secretflow.distributed.primitive import set_production

    set_production(True)

    suite = unittest.TestSuite()
    global test_file
    all_cases = MultiDriverTestLoader().discover('./tests', test_file)
    for case in all_cases:
        # add all tests into suite
        suite.addTests(case)
    result_file = f'./result_{get_self_party()}.xml'
    with open(result_file, 'wb') as output:
        runner = xmlrunner.XMLTestRunner(output=output, failfast=True)
        runner.run(suite)


PROCS = None


def run_multi_driver_test():
    parties = list(cluster()['parties'].keys())
    global PROCS
    PROCS = [
        multiprocessing.Process(target=party_run, args=(party,), daemon=True)
        for party in parties[1:]
    ]
    for p in PROCS:
        p.start()
    # Main process acts as alice.
    party_run('alice')

    for i, p in enumerate(PROCS):
        p.join()
        print(
            f'Party {parties[i+1]} {p.pid} finished execution with exit code {p.exitcode}.'
        )
        assert (
            p.exitcode == 0
        ), f'{parties[i+1]}: exception occurred with exit code {p.exitcode} while running ut.'


def run_single_driver_test():
    from secretflow.distributed.primitive import set_production

    set_production(False)
    suite = unittest.TestSuite()
    global test_file
    all_cases = SingleDriverTestLoader().discover('./tests', test_file)
    for case in all_cases:
        # add all tests into suite
        suite.addTests(case)
    result_file = f'./result_single.xml'
    with open(result_file, 'wb') as output:
        runner = xmlrunner.XMLTestRunner(output=output, failfast=True)
        runner.run(suite)


def signal_handler(sig, frame):
    if sig == signal.SIGINT.value:
        global PROCS
        if PROCS:
            for proc in PROCS:
                try:
                    if proc is not None:
                        proc.kill()
                except Exception:
                    # ignore
                    pass
            PROCS = None

        exit(-1)


signal.signal(signal.SIGINT, signal_handler)


def combine_results(files: List[str], output: str = 'results.xml'):
    tree = None
    for f in files:
        cur_tree = ET.parse(f)
        if tree is None:
            tree = cur_tree
        else:
            root = cur_tree.getroot()
            tree.getroot().extend(root)

    tree.write(output, encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    test_file = 'test_*.py'
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    run_multi_driver_test()
    run_single_driver_test()
    combine_results(['result_alice.xml', 'result_single.xml'])
