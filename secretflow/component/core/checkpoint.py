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

from secretflow_spec import Storage
from secretflow_spec.v1.data_pb2 import DistData

from secretflow.device.driver import PYU, reveal, wait
from secretflow.utils import secure_pickle as pickle


class Checkpoint:
    def __init__(self, uri: str, args: dict, parties: list[str]) -> None:
        self.uri = uri
        self.args = args
        self.parties = parties

    @staticmethod
    def parse_parties(kwargs: dict) -> list[str]:
        parties = set()
        for v in kwargs.values():
            if not isinstance(v, DistData):
                continue
            for dr in v.data_refs:
                parties.add(dr.party)

        assert len(parties) > 0
        return sorted(list(parties))

    def load(self, storage: Storage) -> DistData:
        def _try_load():
            check_points = []
            step = 0
            while True:
                try:
                    uri = self._step_uri(step)
                    if not storage.exists(uri):
                        break
                    with storage.get_reader(uri) as f:
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
            parties_check_points.append(PYU(party)(_try_load)())
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

    def dump(self, storage: Storage, step: int, payload: DistData):
        parties = [dr.party for dr in payload.data_refs]
        check_point = dict()
        check_point["step"] = step
        check_point["uuid"] = reveal(PYU(parties[0])(lambda: str(uuid.uuid4()))())
        check_point["args"] = self.args
        check_point["payload"] = payload

        check_point = pickle.dumps(check_point)

        def _save(cp, storage, uri):
            with storage.get_writer(uri) as f:
                f.write(cp)

        waits = []
        for party in parties:
            waits.append(PYU(party)(_save)(check_point, storage, self._step_uri(step)))

        wait(waits)
