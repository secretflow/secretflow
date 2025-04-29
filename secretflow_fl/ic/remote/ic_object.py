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


class IcObjectSendingContext:
    """The class that's used for holding the all contexts about sending side."""

    def __init__(self) -> None:
        # This field holds the target(downstream) parties that this ic object
        # is sending or sent to.
        # The key is the party name and the value is a boolean indicating whether
        # this object is sending or sent to the party.
        self._is_sending_or_sent = {}

    def mark_is_sending_to_party(self, target_party: str):
        self._is_sending_or_sent[target_party] = True

    def was_sending_or_sent_to_party(self, target_party: str):
        return target_party in self._is_sending_or_sent


class IcObject:
    def __init__(self, node_party: str, data):
        self._node_party = node_party
        self._data = data
        self._sending_context = IcObjectSendingContext()
        self._received = False

    def get_party(self):
        return self._node_party

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def mark_is_sending_to_party(self, target_party: str):
        """Mark this ic object is sending to the target party."""
        self._sending_context.mark_is_sending_to_party(target_party)

    def was_sending_or_sent_to_party(self, target_party: str):
        """Query whether this ic object was sending or sent to the target party."""
        return self._sending_context.was_sending_or_sent_to_party(target_party)

    @property
    def received(self):
        return self._received

    def mark_received(self):
        self._received = True
