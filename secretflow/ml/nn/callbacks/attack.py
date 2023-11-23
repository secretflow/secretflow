# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .callback import Callback


class AttackCallback(Callback):
    """This class is the base class of AttackCallback.
    Specific attack callbacks should inherit from this class and embed the specific attack logic into the on_attack method.
    This ensures that each specific attack callback has the same attackoutput structure.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_attack_metrics(self):
        raise NotImplementedError()
