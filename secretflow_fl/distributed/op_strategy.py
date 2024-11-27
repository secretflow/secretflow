# Copyright 2024 Ant Group Co., Ltd.
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
import os
import multiprocess
from functools import partial
from typing import List, Optional, Union

import fed as rayfed
import ray
from ray import Language
from ray.actor import ActorClass, _inject_tracing_into_class, ray_constants
from ray._private import ray_option_utils
from ray.remote_function import RemoteFunction

import secretflow_fl.ic.remote as ic
from secretflow.distributed.config import get_cluster_config, parse_tls_config
from secretflow.distributed.const import FED_OBJECT_TYPES
from secretflow.distributed.op_strategy import SFOpStrategy
from secretflow.distributed.ray_op import resolve_args
from secretflow_fl.utils.ray_compatibility import ray_version_less_than_2_0_0


class _RayStrategy(SFOpStrategy):
    def _init_ray(
        self,
        parties: Union[str, List[str]] = None,
        address: str = None,
        simulation_mode: bool = False,
        omp_num_threads: int = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        log_to_driver: bool = True,
        **kwargs,
    ):
        if ray_version_less_than_2_0_0():
            if address:
                local_mode = False
            else:
                local_mode = True
        else:
            local_mode = address == 'local'
        if not local_mode and num_cpus is not None:
            num_cpus = None
            logging.warning(
                'When connecting to an existing cluster, num_cpus must not be provided. Num_cpus is neglected at this moment.'
            )
        if local_mode and num_cpus is None:
            num_cpus = multiprocess.cpu_count()
            if simulation_mode:
                # Give num_cpus a min value for better simulation.
                num_cpus = max(num_cpus, 32)

        if 'include_dashboard' not in kwargs:
            kwargs['include_dashboard'] = False

        if simulation_mode and local_mode:
            # party resources is not for scheduler cpus, but set num_cpus for convenient.
            kwargs['resources'] = {party: num_cpus for party in parties}

        if not address and omp_num_threads:
            os.environ['OMP_NUM_THREADS'] = f'{omp_num_threads}'

        ray.init(
            address,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            log_to_driver=log_to_driver,
            **kwargs,
        )


class RayProdStrategy(_RayStrategy):
    def _init_rayfed(
        self,
        cluster_config,
        tls_config,
        enable_waiting_for_other_parties_ready,
        cross_silo_comm_options,
        cross_silo_comm_backend,
        logging_level,
        job_name,
    ):
        self_party, all_parties = get_cluster_config(cluster_config)
        if tls_config:
            parse_tls_config(tls_config, self_party)

        assert cross_silo_comm_backend.lower() in [
            'grpc',
            'brpc_link',
        ], 'Invalid cross_silo_comm_backend, [grpc, brpc_link] are available now.'

        cross_silo_comm_options = cross_silo_comm_options or {}
        if 'exit_on_sending_failure' not in cross_silo_comm_options:
            cross_silo_comm_options['exit_on_sending_failure'] = True
        sending_failure_handler = cross_silo_comm_options.pop(
            'sending_failure_handler', None
        )
        config = {
            'cross_silo_comm': cross_silo_comm_options,
            'barrier_on_initializing': enable_waiting_for_other_parties_ready,
        }
        receiver_sender_proxy_cls = None
        if cross_silo_comm_backend.lower() == 'brpc_link':
            from fed.proxy.brpc_link.link import BrpcLinkSenderReceiverProxy

            receiver_sender_proxy_cls = BrpcLinkSenderReceiverProxy
            if enable_waiting_for_other_parties_ready:
                if 'connect_retry_times' not in config['cross_silo_comm']:
                    config['cross_silo_comm']['connect_retry_times'] = 3600
                    config['cross_silo_comm']['connect_retry_interval_ms'] = 1000
        addresses = {}
        for party, addr in all_parties.items():
            if party == self_party:
                addresses[party] = addr.get('listen_addr', addr['address'])
            else:
                addresses[party] = addr['address']
        rayfed.init(
            addresses=addresses,
            party=self_party,
            config=config,
            logging_level=logging_level,
            tls_config=tls_config,
            receiver_sender_proxy_cls=receiver_sender_proxy_cls,
            sending_failure_handler=sending_failure_handler,
            job_name=job_name,
        )

    def init(self, **kwargs):
        parties = kwargs.pop("parties", None)
        address = kwargs.pop("address", None)
        simulation_mode = kwargs.pop("simulation_mode", False)
        omp_num_threads = kwargs.pop("omp_num_threads", False)
        num_cpus = kwargs.pop("num_cpus", None)
        num_gpus = kwargs.pop("num_gpus", None)
        log_to_driver = kwargs.pop("log_to_driver", True)
        cluster_config = kwargs.pop("cluster_config", None)
        tls_config = kwargs.pop("tls_config", None)
        enable_waiting_for_other_parties_ready = kwargs.pop(
            "enable_waiting_for_other_parties_ready", True
        )
        cross_silo_comm_options = kwargs.pop("cross_silo_comm_options", None)
        cross_silo_comm_backend = kwargs.pop("cross_silo_comm_backend", "grpc")
        logging_level = kwargs.pop("logging_level", "info")
        job_name = kwargs.pop("job_name", None)
        self._init_ray(
            parties,
            address,
            simulation_mode,
            omp_num_threads,
            num_cpus,
            num_gpus,
            log_to_driver,
            **kwargs,
        )
        self._init_rayfed(
            cluster_config,
            tls_config,
            enable_waiting_for_other_parties_ready,
            cross_silo_comm_options,
            cross_silo_comm_backend,
            logging_level,
            job_name,
        )

    def remote(self, *args, **kwargs):
        return rayfed.remote(*args, **kwargs)

    def get(
        self,
        object_refs: Union[
            FED_OBJECT_TYPES,
            List[FED_OBJECT_TYPES],
            Union[object, List[object]],
        ],
    ):
        return rayfed.get(object_refs)

    def kill(self, actor, *, no_restart=True):
        return rayfed.kill(actor, no_restart=no_restart)

    def shutdown(self, on_error=None):
        rayfed.shutdown(on_error=on_error)
        ray.shutdown(_exiting_interpreter=True)


class RemoteFunctionWrapper(RemoteFunction):
    def _remote(self, *args, **kwargs):
        args, kwargs = resolve_args(*args, **kwargs)
        return super()._remote(*args, **kwargs)

    def party(self, party: str):
        self.party = party
        if 'resources' in self._default_options:
            self._default_options['resources'].update({self.party: 1})
        else:
            self._default_options.update({'resources': {self.party: 1}})
        return self

    def options(self, **task_options):
        if hasattr(self, 'party') and self.party:
            if 'resources' in task_options:
                task_options['resources'].update({self.party: 1})
            else:
                task_options.update({'resources': {self.party: 1}})
        return super().options(**task_options)


class ActorClassWrapper(ActorClass):
    def party(self, party: str):
        self.party = party
        if 'resources' in self._default_options:
            self._default_options['resources'].update({self.party: 1})
        else:
            self._default_options.update({'resources': {self.party: 1}})
        return self

    def options(self, **actor_options):
        if hasattr(self, 'party') and self.party:
            if 'resources' in actor_options:
                actor_options['resources'].update({self.party: 1})
            else:
                actor_options.update({'resources': {self.party: 1}})
        return super().options(**actor_options)

    def remote(self, *args, **kwargs):
        args, kwargs = resolve_args(*args, **kwargs)
        return super().remote(*args, **kwargs)


class SimulationStrategy(_RayStrategy):
    def init(self, **kwargs):
        parties = kwargs.pop("parties", None)
        address = kwargs.pop("address", None)
        simulation_mode = kwargs.pop("simulation_mode", False)
        omp_num_threads = kwargs.pop("omp_num_threads", False)
        num_cpus = kwargs.pop("num_cpus", None)
        num_gpus = kwargs.pop("num_gpus", None)
        log_to_driver = kwargs.pop("log_to_driver", True)

        self._init_ray(
            parties,
            address,
            simulation_mode,
            omp_num_threads,
            num_cpus,
            num_gpus,
            log_to_driver,
            **kwargs,
        )

    def remote(self, *args, **kwargs):
        return self._ray_remote(*args, **kwargs)

    def get(
        self,
        object_refs: Union[
            FED_OBJECT_TYPES,
            List[FED_OBJECT_TYPES],
            Union[object, List[object]],
        ],
    ):
        return ray.get(object_refs)

    def kill(self, actor, *, no_restart=True):
        return ray.kill(actor, no_restart=no_restart)

    def shutdown(self, on_error=None):
        ray.shutdown()

    def get_cluster_available_resources(self):
        return ray.available_resources()

    def _make_actor(self, cls, actor_options):
        from secretflow_fl.utils.ray_compatibility import ray_version_less_than_2_0_0

        if ray_version_less_than_2_0_0():
            from ray import ActorClassID
            from ray.actor import modify_class as _modify_class
        else:
            from ray.actor import ActorClassID, _modify_class

        Class = _modify_class(cls)
        _inject_tracing_into_class(Class)

        if "max_restarts" in actor_options:
            if actor_options["max_restarts"] != -1:  # -1 represents infinite restart
                # Make sure we don't pass too big of an int to C++, causing
                # an overflow.
                actor_options["max_restarts"] = min(
                    actor_options["max_restarts"], ray_constants.MAX_INT64_VALUE
                )

        return ActorClassWrapper._ray_from_modified_class(
            Class,
            ActorClassID.from_random(),
            actor_options,
        )

    def _is_cython(self, obj):
        """Check if an object is a Cython function or method"""

        # TODO(suo): We could split these into two functions, one for Cython
        # functions and another for Cython methods.
        # TODO(suo): There doesn't appear to be a Cython function 'type' we can
        # check against via isinstance. Please correct me if I'm wrong.
        def check_cython(x):
            return type(x).__name__ == "cython_function_or_method"

        # Check if function or method, respectively
        return check_cython(obj) or (
            hasattr(obj, "__func__") and check_cython(obj.__func__)
        )

    def _make_remote(self, function_or_class, options):
        if inspect.isfunction(function_or_class) or self._is_cython(function_or_class):
            ray_option_utils.validate_task_options(options, in_options=False)
            return RemoteFunctionWrapper(
                Language.PYTHON,
                function_or_class,
                None,
                options,
            )
        if inspect.isclass(function_or_class):
            ray_option_utils.validate_actor_options(options, in_options=False)
            return self._make_actor(function_or_class, options)

        raise TypeError(
            "The @ray.remote decorator must be applied to either a function or a class."
        )

    def _ray_remote(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # This is the case where the decorator is just @ray.remote.
            # "args[0]" is the class or function under the decorator.
            return self._make_remote(args[0], {})
        assert (
            len(args) == 0 and len(kwargs) > 0
        ), ray_option_utils.remote_args_error_string
        return partial(self._make_remote, options=kwargs)


class InterConnStrategy(SFOpStrategy):
    def init(self, **kwargs):
        pass

    def remote(self, *args, **kwargs):
        return ic.remote(*args, **kwargs)

    def get(
        self,
        object_refs: Union[
            FED_OBJECT_TYPES,
            List[FED_OBJECT_TYPES],
            Union[object, List[object]],
        ],
    ):
        return ic.get(object_refs)

    def kill(self, actor, *, no_restart=True):
        logging.warning("Actor be killed")

    def shutdown(self, on_error=None):
        logging.warning("Shut Down!")
