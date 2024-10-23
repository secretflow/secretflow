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
import uuid
from abc import abstractmethod
from typing import Any, Dict, List

from secretflow.component.storage.impl import BuildStorageImpl
from secretflow.device.driver import PYU, reveal, wait
from secretflow.spec.v1.data_pb2 import DistData, StorageConfig
from secretflow.utils import secure_pickle as pickle


class CompCheckpoint:
    def __init__(
        self,
        kwargs: Dict[str, Any],
        input_names: List[str],
        uri: str,
        storage: StorageConfig,
    ) -> None:
        arg_names = self.associated_arg_names()
        assert set(arg_names).issubset(set(kwargs.keys()))
        self.args = {n: kwargs[n] for n in arg_names}
        parties = set()
        assert set(input_names).issubset(set(kwargs.keys()))
        inputs = {i: kwargs[i] for i in input_names}
        self.args.update(inputs)
        for i in inputs.values():
            assert isinstance(i, DistData)
            for dr in i.data_refs:
                parties.add(dr.party)
        assert len(parties) > 0
        self.parties = sorted(list(parties))
        self.uri = uri
        self.storage = storage

    @abstractmethod
    def associated_arg_names(self) -> List[str]:
        """
        Specify which parameters will affect the validity of the checkpoint.
        If these parameters are changed, the checkpoint will become invalid.
        """
        pass

    def load(self) -> DistData:
        def _try_load(storage):
            impl = BuildStorageImpl(storage)
            check_points = []
            step = 0
            while True:
                try:
                    with impl.get_reader(self._step_uri(step)) as f:
                        cp = pickle.load(f)
                    if not isinstance(cp, dict):
                        break
                    if set(cp.keys()) != set(["step", "uuid", "args", "payload"]):
                        break
                    if cp["step"] != step:
                        break

                    check_points.append(cp)
                    step += 1
                except:
                    break

            return check_points

        parties_check_points = []
        for party in self.parties:
            parties_check_points.append(PYU(party)(_try_load)(self.storage))
        parties_check_points = reveal(parties_check_points)

        cp_len = [len(cps) for cps in parties_check_points]
        max_step = min(cp_len)
        logging.info(
            f"try load checkpoint from {self.uri}, cp len of each party: {cp_len}"
        )

        while max_step > 0:
            max_step -= 1
            check_points = [cps[max_step] for cps in parties_check_points]
            if len(set([cp["uuid"] for cp in check_points])) > 1:
                # uuid miss match
                logging.info(f"uuid miss match, checkpoint from step {max_step}")
                continue

            if not all([cp["args"] == self.args for cp in check_points]):
                # args miss match
                logging.info(f"args miss match, checkpoint from step {max_step}")
                continue

            logging.info(f"found usable checkpoint from step {max_step}")
            return check_points[0]["payload"]

        # no usable checkpoint
        logging.info(f"no usable checkpoint")
        return None

    def _step_uri(self, step: int):
        return f"{self.uri}_{step}"

    def save(self, step: int, payload: DistData):
        check_point = dict()
        check_point["step"] = step
        check_point["uuid"] = reveal(PYU(self.parties[0])(lambda: str(uuid.uuid4()))())
        check_point["args"] = self.args
        check_point["payload"] = payload

        check_point = pickle.dumps(check_point)

        def _save(cp, storage, uri):
            impl = BuildStorageImpl(storage)
            with impl.get_writer(uri) as f:
                f.write(cp)

        waits = []
        for party in self.parties:
            waits.append(
                PYU(party)(_save)(check_point, self.storage, self._step_uri(step))
            )

        wait(waits)
