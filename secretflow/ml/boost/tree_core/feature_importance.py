#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


LEGAL_TYPE = ['split', 'gain']


class FeatureImportance(object):
    """Feature importance class

    Attributes:
        main_importance: main importance value, ref main_type
        other_importance: other importance value, ref opposite to main_type
        main_type: type of importance, eg:gain
    """

    def __init__(
        self,
        main_importance: float = 0,
        other_importance: float = 0,
        main_type: str = 'split',
    ):

        assert main_type in LEGAL_TYPE, f"illegal importance type {main_type}"
        self.main_importance = main_importance
        self.other_importance = other_importance
        self.main_type = main_type

    def add_gain(self, val: float):
        if self.main_type == 'gain':
            self.main_importance += val
        else:
            self.other_importance += val

    def add_split(self, val: float):
        if self.main_type == 'split':
            self.main_importance += val
        else:
            self.other_importance += val

    def __eq__(self, other):
        return self.main_importance == other.main_importance

    def __lt__(self, other):
        return self.main_importance < other.main_importance

    def __gt__(self, other):
        return self.main_importance > other.main_importance

    def __repr__(self):
        return 'importance type: {}, main_importance: {}, other_importance {}'.format(
            self.main_type, self.main_importance, self.other_importance
        )

    def __add__(self, other):
        assert (
            self.main_type == other.main_type
        ), "Self.main_type and other.main_type must be same! "
        new_importance = FeatureImportance(
            main_type=self.main_type,
            main_importance=self.main_importance + other.main_importance,
            other_importance=self.other_importance + other.other_importance,
        )
        return new_importance
