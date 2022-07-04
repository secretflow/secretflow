import copy
import unittest

from secretflow.ml.boost.tree_core.feature_importance import FeatureImportance


class TestFeatureImportance(unittest.TestCase):
    def setUp(self):
        pass

    def test_feature_importance_split(self):
        feature_importance = FeatureImportance(
            main_importance=5, other_importance=2.57, main_type='split'
        )

        # test add_gain
        feature_importance.add_gain(7.43)
        self.assertTrue(feature_importance.other_importance, 8)
        # test add_split
        feature_importance.add_split(5)
        self.assertTrue(feature_importance.main_importance, 10)

        # test repr
        self.assertTrue(
            repr(feature_importance),
            "importance type: split, importance: 10, importance2 10",
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
            self.assertTrue(
                expect_feature_importance[key]
                == getattr(double_feature_importance, key)
            )
        self.assertTrue(
            feature_importance.main_importance * 2,
            double_feature_importance.main_importance,
        )
        # test bool operation
        self.assertTrue(feature_importance < double_feature_importance)
        self.assertTrue(feature_importance == new_feature_importance)
        self.assertTrue(double_feature_importance > feature_importance)

    def test_feature_importance_gain(self):
        feature_importance = FeatureImportance(
            main_importance=2.57, other_importance=5, main_type='gain'
        )

        # test add_gain
        feature_importance.add_gain(7.43)
        self.assertTrue(feature_importance.other_importance, 8)
        # test add_split
        feature_importance.add_split(5)
        self.assertTrue(feature_importance.main_importance, 10)

        # test add
        new_feature_importance = copy.deepcopy(feature_importance)
        double_feature_importance = feature_importance + new_feature_importance

        expect_feature_importance = {
            "main_type": "gain",
            "main_importance": 20,
            "other_importance": 20,
        }
        for key in expect_feature_importance:
            self.assertTrue(
                expect_feature_importance[key]
                == getattr(double_feature_importance, key)
            )
        self.assertTrue(
            feature_importance.main_importance * 2,
            double_feature_importance.main_importance,
        )
        # test bool operation
        self.assertTrue(feature_importance < double_feature_importance)
        self.assertTrue(feature_importance == new_feature_importance)
        self.assertTrue(double_feature_importance > feature_importance)


if __name__ == '__main__':
    unittest.main()
