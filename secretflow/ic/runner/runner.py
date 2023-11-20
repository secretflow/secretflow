# Copyright 2023 Ant Group Co., Ltd.
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
import secretflow.distributed as sfd
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.utils.logging import set_logging_level, get_logging_level, LOG_FORMAT
from secretflow.ic.proxy import LinkProxy


def _init_log(self_party: str, logging_level: str):
    set_logging_level(logging_level)
    logging.basicConfig(
        level=get_logging_level(),
        format=LOG_FORMAT,
        filename=self_party + '.log',
        filemode='w',
        force=True,
    )


def _init_link(link_config: dict):
    self_party = link_config['self_party']
    parties = link_config['parties']

    for party in parties.values():
        assert (
            'address' in party
        ), f'There is no address for party {party} in link config.'

    addresses = {}
    for party, addr in parties.items():
        if party == self_party:
            addresses[party] = addr.get('listen_addr', addr['address'])
        else:
            addresses[party] = addr['address']

    LinkProxy.init(addresses=addresses, self_party=self_party)


def _stop_link():
    LinkProxy.stop()


def _get_ic_handler(config: dict, dataset: dict):
    if config['algo'] == 'sgb':
        from secretflow.ic.handler import SgbIcHandler

        return SgbIcHandler(config, dataset)
    else:
        raise NotImplementedError(f'Unsupported algorithm {config["algo"]}')


def run(config: dict, dataset: dict, logging_level: str = 'info'):
    """Initialize runtime environment and run the specified algorithm

    Args:
        config : The configuration for network communication and
            specified algorithm
        dataset : The input dataset for training the model
        logging_level : Optional; The logging level, could be `debug`,
            `info`, `warning`, `error`, `critical`, not case-sensitive.
    """

    sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.INTERCONNECTION)

    _init_log(config['link']['self_party'], logging_level)

    _init_link(config['link'])

    handler = _get_ic_handler(config, dataset)
    handler.run()

    _stop_link()
