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


def config_logger(mode="log", fname="output.log"):
    logger = logging.getLogger(mode)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    fh = logging.FileHandler(fname, "a", encoding="utf-8")

    # formatter = logging.Formatter('%(asctime)s-%(filename)s-%(message)s')
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def print_log(msg_list, oriention="print", mode="log", delimiter=""):
    msg = ""
    logger = logging.getLogger(mode)
    for m in msg_list:
        if msg != "":
            msg = msg + delimiter + str(m)
        else:
            msg = str(m)

    if oriention == "print":
        print(msg)
    elif oriention == "logger":
        logger.info(msg)


# config_logger(mode='log', fname='./logs/split.log')
