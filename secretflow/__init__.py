# Copyright 2022 Ant Group Co., Ltd.
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

import sys

if sys.version_info.major == 3 and sys.version_info.minor == 8:
    # python 3.8
    import pkg_resources

    package_name = 'secretflow-ray'
    try:
        cool_package_dist_info = pkg_resources.get_distribution(package_name)
    except pkg_resources.DistributionNotFound:
        # Not found.
        pass
    else:
        print(
            "The secretflow-ray package is now deprecated.\n"
            "Please uninstall secretflow-ray before proceeding.\n"
            "Follow the commands below to complete the uninstallation process.\n"
            "> pip uninstall secretflow-ray\n"
            "> pip uninstall ray\n"
            "> pip install ray"
        )
        exit(1)


from . import component, data, device, ic, kuscia, ml, preprocessing, security, utils
from .device import (
    HEU,
    PYU,
    SPU,
    TEEU,
    Device,
    DeviceObject,
    HEUObject,
    PYUObject,
    SPUObject,
    init,
    proxy,
    reveal,
    shutdown,
    to,
    wait,
)
from .version import __version__  # type: ignore

__all__ = [
    'kuscia',
    'data',
    'device',
    'ml',
    'preprocessing',
    'security',
    'utils',
    'ic',
    'HEU',
    'PYU',
    'SPU',
    'TEEU',
    'Device',
    'DeviceObject',
    'HEUObject',
    'PYUObject',
    'SPUObject',
    'init',
    'proxy',
    'reveal',
    'shutdown',
    'to',
    'wait',
    'component',
]
