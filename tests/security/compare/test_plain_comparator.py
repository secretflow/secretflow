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

import pytest

from secretflow.security.compare.plain_comparator import PlainComparator
from tests.security.compare.test_comparator_base import ComparatorBase
from tests.sf_fixtures import mpc_fixture


@mpc_fixture(alias="env_and_comparator")
def plain_env_and_comparator(sf_production_setup_devices):
    return sf_production_setup_devices, PlainComparator(
        sf_production_setup_devices.alice
    )


@pytest.mark.mpc(fixtures=["plain_env_and_comparator"])
class TestPlainComparator(ComparatorBase): ...
