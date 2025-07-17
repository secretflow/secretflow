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

import logging
import multiprocessing
from abc import ABC, abstractmethod
from typing import List, Union

from . import fed as sf_fed
from .config import get_cluster_config, parse_tls_config
from .const import FED_OBJECT_TYPES
from .memory.memory_api import mem_remote


class SFOpStrategy(ABC):
    @abstractmethod
    def init(self, **kwargs):
        pass

    @abstractmethod
    def remote(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(
        self,
        object_refs: Union[
            FED_OBJECT_TYPES,
            List[FED_OBJECT_TYPES],
            Union[object, List[object]],
        ],
    ):
        pass

    @abstractmethod
    def kill(self, actor, *, no_restart=True):
        pass

    @abstractmethod
    def shutdown(self, on_error=None):
        pass

    def get_cluster_available_resources(self):
        raise NotImplementedError(
            "Need but not implement. This will be used by tune when production mode."
        )


class DebugStrategy(SFOpStrategy):
    def init(self, **kwargs):
        pass

    def remote(self, *args, **kwargs):
        return mem_remote(*args, **kwargs)

    def get(
        self,
        object_refs: Union[
            FED_OBJECT_TYPES,
            List[FED_OBJECT_TYPES],
            Union[object, List[object]],
        ],
    ):
        return object_refs

    def kill(self, actor, *, no_restart=True):
        logging.warning("Actor be killed")

    def shutdown(self, on_error=None):
        logging.warning("Shut Down!")

    def get_cluster_available_resources(self):
        from secretflow.device import global_state

        cpu_counts = multiprocessing.cpu_count()
        available_resources = {party: cpu_counts for party in global_state.parties()}
        available_resources['CPU'] = cpu_counts
        return available_resources


class ProdStrategy(SFOpStrategy):
    def _init_sf_fed(
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

        cross_silo_comm_options = cross_silo_comm_options or {}
        config = {
            'cross_silo_comm': cross_silo_comm_options,
            'barrier_on_initializing': enable_waiting_for_other_parties_ready,
            'cross_silo_comm_backend': cross_silo_comm_backend,
        }

        addresses = {}
        for party, addr in all_parties.items():
            if party == self_party:
                addresses[party] = addr.get('listen_addr', addr['address'])
            else:
                addresses[party] = addr['address']
        sf_fed.init(
            addresses=addresses,
            party=self_party,
            config=config,
            logging_level=logging_level,
            tls_config=tls_config,
            job_name=job_name,
        )

    def init(self, **kwargs):
        cluster_config = kwargs.pop("cluster_config", None)
        tls_config = kwargs.pop("tls_config", None)
        enable_waiting_for_other_parties_ready = kwargs.pop(
            "enable_waiting_for_other_parties_ready", True
        )
        cross_silo_comm_options = kwargs.pop("cross_silo_comm_options", None)
        cross_silo_comm_backend = kwargs.pop("cross_silo_comm_backend", "grpc")
        logging_level = kwargs.pop("logging_level", "info")
        job_name = kwargs.pop("job_name", None)
        self._init_sf_fed(
            cluster_config,
            tls_config,
            enable_waiting_for_other_parties_ready,
            cross_silo_comm_options,
            cross_silo_comm_backend,
            logging_level,
            job_name,
        )

    def remote(self, *args, **kwargs):
        return sf_fed.remote(*args, **kwargs)

    def get(
        self,
        object_refs: Union[
            FED_OBJECT_TYPES,
            List[FED_OBJECT_TYPES],
            Union[object, List[object]],
        ],
    ):
        return sf_fed.get(object_refs)

    def kill(self, actor, *, no_restart=True):
        pass

    def shutdown(self, on_error=None):
        sf_fed.shutdown(on_error=on_error)
