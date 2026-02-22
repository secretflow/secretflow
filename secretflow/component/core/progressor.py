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


import abc
import functools
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor

import requests


class IProgressor(abc.ABC):
    @abc.abstractmethod
    def update(self, percent: float, infos: dict): ...

    @abc.abstractmethod
    def done(self): ...


class MockProgressor(IProgressor):
    def update(self, percent, infos):
        logging.debug(f"update progress: {percent} {infos}")

    def done(self):
        logging.debug("mock progress done")


class HttpProgressor(IProgressor):
    def __init__(self, url):
        self.url = url
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._feature: Future = None
        self._percent = 0.0

    def update(self, percent: float, infos: dict):
        if percent > 1.0:
            logging.warning(f"invalid percent{percent}")
            percent = 1.0

        if percent <= self._percent:
            return
        self._percent = percent
        logging.info(f"update progress: {percent * 100}%")

        payload = {"progress": percent}
        if infos:
            custom_fields = {}
            for k, v in infos:
                custom_fields[str(k)] = str(v)
            payload["custom_fields"] = custom_fields

        if self._feature:
            self._feature.cancel()
        self._feature = self._executor.submit(
            functools.partial(HttpProgressor.send_post, self.url, payload, percent)
        )

    def done(self):
        self.update(1.0, None)
        self._executor.shutdown()
        logging.info("update progress done")

    @staticmethod
    def send_post(
        url: str,
        payload: dict,
        percent: float,
        max_retries: int = 3,
        max_delay_seconds: float = 1,
    ):
        for attempt in range(max_retries):
            try:
                rsp = requests.post(url, json=payload, timeout=3)
                if rsp.status_code != 200:
                    logging.warning(
                        f"update progress fail, {rsp.status_code} {rsp.reason} {percent}"
                    )
                return
            except Exception as e:
                logging.warning(
                    f"update progress raise exception, {attempt} {percent} {e}"
                )
            if attempt < max_retries - 1:
                time.sleep(max_delay_seconds)


def new_progressor(url: str | None) -> IProgressor | None:
    if not url:
        return None

    if url.startswith("mock://"):
        return MockProgressor()
    return HttpProgressor(url)
