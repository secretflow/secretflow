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

import copy

from secretflow_fl.ml.boost.homo_boost.tree_core.feature_importance import (
    FeatureImportance,
)


class TestFeatureImportance:
    def test_feature_importance_split(self):
        feature_importance = FeatureImportance(
            main_importance=5, other_importance=2.57, main_type="split"
        )

        # test add_gain
        feature_importance.add_gain(7.43)
        assert feature_importance.other_importance == 10
        # test add_split
        feature_importance.add_split(5)
        assert feature_importance.main_importance == 10

        # test repr
        assert (
            repr(feature_importance)
            == "importance type: split, main_importance: 10, other_importance 10.0"
        )

        # test add
        new_feature_importance = copy.deepcopy(feature_importance)
        double_feature_importance = feature_importance + new_feature_importance

        expect_feature_importance = {
            "main_type": "split",
            "main_importance": 20,
            "other_importance": 20,
        }
        for key in expect_feature_importance:
            assert expect_feature_importance[key] == getattr(
                double_feature_importance, key
            )
        assert (
            feature_importance.main_importance * 2
            == double_feature_importance.main_importance
        )

        # test bool operation
        assert feature_importance < double_feature_importance
        assert feature_importance == new_feature_importance
        assert double_feature_importance > feature_importance

    def test_feature_importance_gain(self):
        feature_importance = FeatureImportance(
            main_importance=2.57, other_importance=5, main_type="gain"
        )

        # test add_gain
        feature_importance.add_gain(7.43)
        assert feature_importance.other_importance == 5
        # test add_split
        feature_importance.add_split(5)
        assert feature_importance.main_importance == 10

        # test add
        new_feature_importance = copy.deepcopy(feature_importance)
        double_feature_importance = feature_importance + new_feature_importance

        expect_feature_importance = {
            "main_type": "gain",
            "main_importance": 20,
            "other_importance": 20,
        }
        for key in expect_feature_importance:
            assert expect_feature_importance[key] == getattr(
                double_feature_importance, key
            )

        assert (
            feature_importance.main_importance * 2
            == double_feature_importance.main_importance
        )
        # test bool operation
        assert feature_importance < double_feature_importance
        assert feature_importance == new_feature_importance
        assert double_feature_importance > feature_importance
