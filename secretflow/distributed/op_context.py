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
from typing import List, Union

from secretflow.utils.errors import NotSupportedError

from .const import DISTRIBUTION_MODE, FED_OBJECT_TYPES
from .op_strategy import DebugStrategy, ProdStrategy


class SFOpContext:
    _lite_strategy_dict = {
        DISTRIBUTION_MODE.PRODUCTION: ProdStrategy,
        DISTRIBUTION_MODE.DEBUG: DebugStrategy,
    }

    _fl_strategy_dict = {
        DISTRIBUTION_MODE.SIMULATION: "SimulationStrategy",
        DISTRIBUTION_MODE.RAY_PRODUCTION: "RayProdStrategy",
        DISTRIBUTION_MODE.INTERCONNECTION: "InterConnStrategy",
    }

    def __init__(self, mode: DISTRIBUTION_MODE):
        self._mode = mode
        self._is_cluster_active = False
        self._current_cluster_id = -1
        self._init_strategy()

    def _init_strategy(self):
        if self._mode in self._lite_strategy_dict:
            strategy = self._lite_strategy_dict[self._mode]
        elif self._mode in self._fl_strategy_dict:
            try:
                # lazy import
                from secretflow_fl.distributed import op_strategy as fl_op_strategy

                assert hasattr(
                    fl_op_strategy, self._fl_strategy_dict[self._mode]
                ), f"fl strategy {self._fl_strategy_dict[self._mode]} not found"
                strategy = getattr(fl_op_strategy, self._fl_strategy_dict[self._mode])
            except ImportError:
                raise NotSupportedError(
                    f"{self._mode} mode is not supported in lite version, please try it in full version."
                )
        else:
            raise NotSupportedError(
                f"Illegal distribute mode, only support ({DISTRIBUTION_MODE})"
            )
        self._strategy = strategy()

    def set_distribution_mode(self, mode: DISTRIBUTION_MODE):
        if mode not in DISTRIBUTION_MODE:
            raise NotSupportedError(
                f"Illegal distribute mode, only support ({DISTRIBUTION_MODE})"
            )

        logging.info(f"set distribution mode to {mode}")

        self._mode = mode
        self._init_strategy()

    def get_distribution_mode(self):
        return self._mode

    def init(self, **kwargs):
        return self._strategy.init(**kwargs)

    def remote(self, *args, **kwargs):
        return self._strategy.remote(*args, **kwargs)

    def get(
        self,
        object_refs: Union[
            FED_OBJECT_TYPES,
            List[FED_OBJECT_TYPES],
            Union[object, List[object]],
        ],
    ):
        return self._strategy.get(object_refs)

    def kill(self, actor, *, no_restart=True):
        return self._strategy.kill(actor, no_restart=no_restart)

    def shutdown(self, on_error=None):
        return self._strategy.shutdown(on_error=on_error)

    def get_current_cluster_idx(self):
        """
        Get the current secretflow cluster index.
        In general situation, users will execute sf.init() at the first of their codes,
        and execute sf.shutdown() at the end.
        But in some cases, the following code may appear:
            - sf.init()
            - do_something_in_sf()
            - sf.shutdown()
            - do_something_else()
            - sf.init()
            - do_something_in_sf()
            - sf.shutdown()
        To ensure some variable can work as expect when it across different clusters,
        we use an index to indicate different active cluster.
        As shown above, between the first sf.init() and sf.shutdown(), the cluster index is 0,
        and the between the second sf.init() and sf.shutdown(), the cluster index is 1,
        In do_something_else(), the cluster index is -1 (no active cluster).

        Returns: The current cluster id, -1 for not active.

        """
        if self._is_cluster_active:
            return self._current_cluster_id
        else:
            return -1

    def active_sf_cluster(self):
        """
        Record the cluster's active status and generate the current cluster index.
        """
        self._current_cluster_id = self._current_cluster_id + 1
        self._is_cluster_active = True

    def deactivate_sf_cluster(self):
        self._is_cluster_active = False

    def get_cluster_available_resources(self):
        return self._strategy.get_cluster_available_resources()
