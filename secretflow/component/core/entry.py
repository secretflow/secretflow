# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os

from secretflow.component.core import Definition
from secretflow.device.driver import init, shutdown
from secretflow.error_system.exceptions import SFTrainingHyperparameterError
from secretflow.spec.extend.cluster_pb2 import SFClusterConfig
from secretflow.spec.v1.data_pb2 import StorageConfig
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam, NodeEvalResult

from .context import Context
from .registry import Registry


def setup_sf_cluster(config: SFClusterConfig):  # type: ignore
    import multiprocess

    cross_silo_comm_backend = (
        config.desc.ray_fed_config.cross_silo_comm_backend
        if len(config.desc.ray_fed_config.cross_silo_comm_backend)
        else 'brpc_link'
    )

    # From https://grpc.github.io/grpc/core/md_doc_statuscodes.html
    # We take an aggressive strategy that all error code are retriable except OK.
    # The code could even be inconsistent with the original meaning because the
    # complex real produciton environment.
    _GRPC_RETRY_CODES = [
        'CANCELLED',
        'UNKNOWN',
        'INVALID_ARGUMENT',
        'DEADLINE_EXCEEDED',
        'NOT_FOUND',
        'ALREADY_EXISTS',
        'PERMISSION_DENIED',
        'RESOURCE_EXHAUSTED',
        'ABORTED',
        'OUT_OF_RANGE',
        'UNIMPLEMENTED',
        'INTERNAL',
        'UNAVAILABLE',
        'DATA_LOSS',
        'UNAUTHENTICATED',
    ]

    if cross_silo_comm_backend == 'grpc':
        cross_silo_comm_options = {
            'proxy_max_restarts': 3,
            'grpc_retry_policy': {
                # The maximum is 5.
                # ref https://github.com/grpc/proposal/blob/master/A6-client-retries.md#validation-of-retrypolicy
                "maxAttempts": 5,
                "initialBackoff": "2s",
                "maxBackoff": "3600s",
                "backoffMultiplier": 2,
                "retryableStatusCodes": _GRPC_RETRY_CODES,
            },
        }
    elif cross_silo_comm_backend == 'brpc_link':
        cross_silo_comm_options = {
            'proxy_max_restarts': 3,
            'timeout_in_ms': 300 * 1000,
            # Give recv_timeout_ms a big value, e.g.
            # The server does nothing but waits for task finish.
            # To fix the psi timeout, got a week here.
            'recv_timeout_ms': 7 * 24 * 3600 * 1000,
            'connect_retry_times': 3600,
            'connect_retry_interval_ms': 1000,
            'brpc_channel_protocol': 'http',
            'brpc_channel_connection_type': 'pooled',
        }

    else:
        raise RuntimeError(
            f"unknown cross_silo_comm_backend: {cross_silo_comm_backend}"
        )

    cluster_config = {
        "parties": {},
        "self_party": config.private_config.self_party,
    }
    for party, addr in zip(
        list(config.public_config.ray_fed_config.parties),
        list(config.public_config.ray_fed_config.addresses),
    ):
        if cross_silo_comm_backend == 'brpc_link':
            # if port is not present, use default 80 port.
            if len(addr.split(":")) < 2:
                addr += ":80"

            splits = addr.split(":")
            if len(splits) != 2:
                raise SFTrainingHyperparameterError.sf_cluster_config_error(
                    f"wrong ray_fed_config address: {addr}"
                )

            cluster_config["parties"][party] = {
                # add "http://" to force brpc to set the correct host
                "address": f'http://{addr}',
                'listen_addr': f'0.0.0.0:{splits[1]}',
            }
        else:
            cluster_config["parties"][party] = {"address": addr}

    init(
        address=config.private_config.ray_head_addr,
        num_cpus=32,
        log_to_driver=True,
        cluster_config=cluster_config,
        omp_num_threads=multiprocess.cpu_count(),
        logging_level='info',
        cross_silo_comm_backend=cross_silo_comm_backend,
        cross_silo_comm_options=cross_silo_comm_options,
        enable_waiting_for_other_parties_ready=True,
        ray_mode=False,
    )


def comp_eval(
    param: NodeEvalParam,  # type: ignore
    storage_config: StorageConfig,  # type: ignore
    cluster_config: SFClusterConfig,  # type: ignore
    tracer_report: bool = False,
) -> NodeEvalResult:  # type: ignore
    comp_def = Registry.get_definition(param.domain, param.name, param.version)
    if comp_def is None:
        raise RuntimeError(
            f"component is not found, {param.domain}, {param.name}, {param.version}."
        )

    on_error = False
    try:
        if cluster_config is not None:
            setup_sf_cluster(cluster_config)

        minor = Definition.parse_minor(param.version)
        kwargs = comp_def.parse_param(param)
        cp = comp_def.make_checkpoint(kwargs, param.checkpoint_uri)
        ctx = Context(storage_config, cluster_config, cp)
        comp = comp_def.make_component(kwargs)
        comp.evaluate(ctx)

        output_defs = comp_def.get_output_defs(minor)
        outputs = []
        for out_def in output_defs:
            out = getattr(comp, out_def.name)
            assert out.data is not None, f"output {out_def.name} must be not None"
            assert (
                out.data.type in out_def.types
            ), f"DistData type<{out.data.type}> must be in {out_def.types}"
            outputs.append(out.data)

        res = NodeEvalResult(outputs=outputs)
        if 'PYTEST_CURRENT_TEST' not in os.environ:
            logging.info(f'\n--\n*res* \n\n{res}\n--\n')
        if tracer_report:
            res = {"eval_result": res, "tracer_report": ctx.trace_report()}
        return res
    except Exception as e:
        on_error = True
        logging.exception(f"comp eval failed, error <{e}>")
        raise
    finally:
        if cluster_config is not None:
            shutdown(
                barrier_on_shutdown=cluster_config.public_config.barrier_on_shutdown,
                on_error=on_error,
            )
