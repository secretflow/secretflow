# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import multiprocessing
import string
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import pytest

from .sf_config import (
    ALL_PARTIES,
    generate_port_by_node_index,
    get_parties,
    get_storage_root,
)
from .sf_fixtures import (
    DeviceInventory,
    MPCFixture,
    build_cluster_config,
    build_devices,
    get_mpc_fixture,
    is_mpc_fixture,
)
from .sf_services import get_service_params, get_services


def pytest_addoption(parser):
    parser.addoption(
        "--env",
        action="store",
        default="sim",
        help="env option: simulation or production",
        choices=("sim", "prod", "ray_prod"),
    )
    parser.addoption(
        "--simple-report",
        action="store_true",
        default=False,
        help="Generate a JSON test report (default: False)",
    )
    parser.addoption(
        "--simple-report-file",
        action="store",
        default="./report.json",
        help="Directory to save the report (default: ./report.json)",
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    import json
    from pathlib import Path

    if not config.getoption("--simple-report"):
        return

    report_file = config.getoption("--simple-report-file")

    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict = terminalreporter.stats

    empty_nodes = []
    test_items = []
    for reports in stats.values():
        for report in reports:
            if hasattr(report, "nodeid") and report.nodeid:
                test_items.append(report.nodeid)
            else:
                empty_nodes.append(report)

    report = {
        "passed": len(stats.get("passed", [])),
        "failed": len(stats.get("failed", [])),
        "skipped": len(stats.get("skipped", [])),
        "total": sum(len(x) for x in stats.values() if x),
        "items": test_items,
    }

    def _add_extra(key):
        v = stats.get(key, None)
        if v:
            report[key] = len(v)

    _add_extra("xfailed")
    _add_extra("xpassed")
    _add_extra("error")
    if empty_nodes:
        report["empty"] = len(empty_nodes)

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False)
        terminalreporter.write_sep(
            "-", f"Simplified JSON report saved to: {report_path}"
        )
    except IOError as e:
        terminalreporter.write_sep("!", f"Failed to save report: {str(e)}")


def pytest_configure(config: pytest.Config):
    # avoid multiprocessing.fork error
    # os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded,
    # so this will likely lead to a deadlock.
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')

    config.addinivalue_line(
        "markers",
        "mpc:mark test to run as mpc test in Multiple child processes",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    env = config.getoption("--env")

    def should_skip(item: pytest.Item) -> bool:
        mark = item.get_closest_marker("mpc")
        if mark is not None:
            # mpc test
            is_ray = mark.kwargs.get("is_ray", False)
            if env == "sim":
                return True
            elif env == "prod":
                return is_ray
            else:
                return not is_ray
        else:
            # basic test
            return env != "sim"

    # Reverse traversal avoids index misalignment
    for i in reversed(range(len(items))):
        item = items[i]
        _check_mpc_marker(item)
        if should_skip(item):
            item.add_marker(pytest.mark.skip(reason=f"test requires env in {env}"))
            items.pop(i)

    # Specify the unique index serial number
    for index, item in enumerate(items):
        item.user_properties.append(("index", index))

    _check_services(items)

    logging.info(f"storage_root: {get_storage_root()}")


def _check_mpc_marker(item: pytest.Item):
    mark = item.get_closest_marker("mpc")
    if mark:
        return
    test_func: Callable = item.obj
    fixtures = _get_mpc_fixtures(test_func, [])
    if fixtures:
        pytest.fail(f"not found mpc marker but it has mpc fixtures, {item.nodeid}")


def _check_services(items: list[pytest.Item]):
    services = get_services()

    service_flags = {}
    for item in items:
        if len(service_flags) == services:
            # all services
            break

        mark = item.get_closest_marker("mpc")
        if mark is None:
            continue

        # prod mpc
        extra_fixtures: list = mark.kwargs.get("fixtures", [])
        test_func: Callable = item.obj
        fixtures = _get_mpc_fixtures(test_func, extra_fixtures)
        for fix in fixtures:
            if not fix.services:
                continue
            for serv_name in fix.services:
                service_flags[serv_name] = True

    for name, fn in services.items():
        if name in service_flags:
            logging.info(f"start service: {name}")
            fn()
        else:
            logging.info(f"ignore service: {name}")


def pytest_generate_tests(metafunc: pytest.Metafunc):
    for fix_name in metafunc.fixturenames:
        if is_mpc_fixture(fix_name):
            metafunc.parametrize(fix_name, [None], indirect=False)


def _prepare_kwargs(func: Callable, params: dict) -> dict:
    res = {}
    sign = inspect.signature(func)
    for key in sign.parameters.keys():
        if key in params:
            res[key] = params[key]
    return res


def _run_mpc_worker(
    test_func: Callable,
    nodeid: str,  # id with filename and unit test info
    self_party: str,
    node_params: dict,
    fixtures: list[Callable],
):
    node_params["self_party"] = self_party
    request = node_params

    fixture_res = {}
    teardown_generators = []
    for func in fixtures:
        # Usually, alias is equal to name
        func_name = func.__alias__
        kwargs = _prepare_kwargs(func, request)
        try:
            if inspect.isgeneratorfunction(func):
                gen = func(**kwargs)
                res = next(gen)  # Execute setup code (before yield)
                teardown_generators.append(gen)
            else:
                # Ordinary callbacks are executed directly (only setup logic)
                res = func(**kwargs)
        except Exception as e:
            logging.exception("process fixture fail.")
            raise ValueError(
                f"process fixture fail. func={func_name}, node={nodeid}, err={e}, kwargs={kwargs}"
            )

        if res is not None:
            fixture_res[func_name] = res
            request[func_name] = res

    try:
        test_kwargs = _prepare_kwargs(test_func, request)
        test_func(**test_kwargs)
    except Exception as e:
        logging.exception("run mpc test fail.")
        raise
    finally:
        for gen in reversed(teardown_generators):
            next(gen, None)


def _kill_workers(exec: ProcessPoolExecutor):
    if hasattr(exec, "kill_workers"):
        # Added in version 3.14.
        exec.kill_workers()
    else:
        # Simulate kill_workers
        for p in exec._processes.values():
            if p.is_alive():
                p.kill()


def _get_property(
    properties: list[tuple[str, object]], key: str, default: object = None
) -> object:
    for prop in properties:
        if prop[0] == key:
            return prop[1]

    return default


def _get_mpc_fixtures(fn: Callable, extras: list[str] = None) -> list[MPCFixture]:
    extra_fixtures = {}
    if extras:
        for name in extras:
            mf = get_mpc_fixture(name)
            if mf is None:
                raise ValueError(f"cannot find mpc fixture, {name}")
            alias = mf.func.__alias__
            extra_fixtures[alias] = mf

    def _walk_fixtures(fixtures: dict, marks: set, func: Callable):
        sign = inspect.signature(func)
        for key in sign.parameters:
            if not is_mpc_fixture(key) or key in fixtures:
                continue
            if key in marks:
                raise ValueError(f"{key} has circular dependencies. ")
            marks.add(key)
            mf = get_mpc_fixture(key, func.__module__)
            if mf is None:
                alias_mf = extra_fixtures.get(key)
                if alias_mf is None:
                    raise ValueError(f"cannot find mpc fixture by alias<{key}>.")
                mf = alias_mf
            _walk_fixtures(fixtures, marks, mf.func)
            fixtures[key] = mf
            marks.remove(key)

    fixtures = {}
    # Used for detecting circular dependencies
    marks = set()
    _walk_fixtures(fixtures, marks, fn)
    return list(fixtures.values())


def _rand_id(count: int = 4) -> str:
    import random

    characters = string.ascii_lowercase + string.digits
    res = ''.join(random.choice(characters) for _ in range(count))
    return res


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """
    e.g.
    ```
    @pytest.mark.mpc(parties=["alice", "bob"])
    def test_xxx(self_party: str):
        ...
    ```
    """
    mark = item.get_closest_marker("mpc")
    if not mark:
        yield
        return

    parties = get_parties(mark.kwargs.get("parties"))
    mpc_params: dict = mark.kwargs.get("params", {})
    extra_fixtures: list = mark.kwargs.get("fixtures", [])

    test_func: Callable = item.obj
    nodeid: str = item.nodeid
    node_index: int = _get_property(item.user_properties, "index", 0)
    call_params: dict = item.callspec.params if hasattr(item, "callspec") else {}
    # remove mock pytest.mark.parametrize if it is mpc_fixture
    call_params = {k: v for k, v in call_params.items() if not is_mpc_fixture(k)}

    # logging.warning(f'====> node_params={node_params},fixtures {item.fixturenames}')
    # for fix_name in item.fixturenames:
    #     if fix_name != "request":
    #         node_params[fix_name] = item._request.getfixturevalue(fix_name)

    buildin_params = {
        # "self_party":"", fill in later
        "parties": parties,
        "nodeid": nodeid,
        "testid": f"{_rand_id()}{node_index}",
        "cluster": build_cluster_config(parties, node_index),
    }
    serv_params = get_service_params()
    node_params = {**call_params, **mpc_params, **serv_params, **buildin_params}

    mpc_fixtures = _get_mpc_fixtures(test_func, extra_fixtures)
    fixtures = [mf.func for mf in mpc_fixtures]

    logging.info(f"start to run test. {nodeid}-{node_index}")

    # run mpc test in multiple child processes
    with ProcessPoolExecutor(max_workers=len(parties)) as exec:
        futures = [
            exec.submit(
                _run_mpc_worker, test_func, nodeid, party, node_params, fixtures
            )
            for party in parties
        ]
        errors = []
        for fu in as_completed(futures):
            try:
                fu.result()
            except Exception as e:
                errors.append(e)
                _kill_workers(exec)

        if errors:
            pytest.fail(
                f"Run test failed. node={nodeid}, errors={errors}, node_params={node_params}"
            )
        else:
            logging.info(f"Run test success. node={nodeid}-{node_index}")

    outcome = yield
    outcome.force_result(None)


def prepare_storage_path(party):
    import os
    import uuid

    storage_path = os.path.join(get_storage_root(), party, str(uuid.uuid4()))
    os.makedirs(storage_path, exist_ok=True)
    return storage_path


@pytest.fixture(scope="function", params=[{"parties": ["alice", "bob"]}])
def sf_memory_setup_devices(request):
    import secretflow as sf
    import secretflow.distributed as sfd
    from secretflow.distributed.const import DISTRIBUTION_MODE

    param = request.param if hasattr(request, "param") else {}
    parties = get_parties(param.get("parties"))
    assert parties, f"parties={parties}"

    sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.DEBUG)
    sf.shutdown()
    sf.init(parties, debug_mode=True)

    devices = DeviceInventory()
    pyus = {p: sf.PYU(p) for p in parties}
    devices.alice = pyus.get("alice")
    devices.bob = pyus.get("bob")
    devices.carol = pyus.get("carol")
    devices.davy = pyus.get("davy")
    devices.spu = pyus.get("spu")
    devices.heu = None

    yield devices
    del devices
    sf.shutdown()


@pytest.fixture(scope="function")
def sf_simulation_setup():
    import multiprocess

    import secretflow as sf
    import secretflow.distributed as sfd
    from secretflow.distributed.const import DISTRIBUTION_MODE

    address = "local"
    parties = ALL_PARTIES

    logging.info(f"try to init sf, {address}")
    sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.SIMULATION)
    sf.shutdown()
    sf.init(
        parties,
        address=address,
        num_cpus=32,
        log_to_driver=True,
        omp_num_threads=multiprocess.cpu_count(),
    )

    yield parties

    sf.shutdown()


@pytest.fixture(
    scope="function",
    params=[{"spu_protocol": "semi2k", "spu_parties": 2, "is_tune": False}],
)
def sf_simulation_setup_devices(request: pytest.FixtureRequest, sf_simulation_setup):
    """
    By default, parties is ["alice", "bob"], spu_protocol is semi2k.
    The parameters can be adjusted using mark.parametrize.
    ```
    @pytest.mark.parametrize("sf_simulation_setup_devices", [{"is_tune": True}], indirect=True)
    def test_foo(sf_simulation_setup_devices):...
    ```
    """

    param = request.param if hasattr(request, "param") else {}
    parties = sf_simulation_setup
    spu_protocol = param.get("spu_protocol", "semi2k")
    spu_parties = get_parties(param.get("spu_parties", 2))
    heu_sk_keeper = param.get("heu_sk_keeper", "alice")
    is_tune = param.get("is_tune", False)
    assert spu_protocol and heu_sk_keeper, f"spu={spu_protocol},heu={heu_sk_keeper}"

    # tests/tuner don't need spu and heu
    if is_tune:
        spu_protocol = ""
        heu_sk_keeper = ""
        spu_addrs = None
    else:
        index = _get_property(request.node.user_properties, "index", 1)
        port_gen = generate_port_by_node_index(index)
        spu_addrs = {party: f"127.0.0.1:{next(port_gen)}" for party in spu_parties}
    devices = build_devices(parties, spu_protocol, spu_addrs, heu_sk_keeper)
    return devices
