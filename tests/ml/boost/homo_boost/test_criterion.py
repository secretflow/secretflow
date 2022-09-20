import unittest

from secretflow.ml.boost.homo_boost.tree_core.criterion import XgboostCriterion


class TestXgboostCriterion(unittest.TestCase):
    def setUp(self):
        self.reg_lambda = 0.3
        self.criterion = XgboostCriterion(reg_lambda=self.reg_lambda)

    def test_init(self):
        self.assertTrue(self.criterion.reg_lambda, self.reg_lambda)

    def test_split_gain(self):
        node = [0.5, 0.6]
        left = [0.1, 0.2]
        right = [0.4, 0.4]
        gain_all = node[0] * node[0] / (node[1] + self.reg_lambda)
        gain_left = left[0] * left[0] / (left[1] + self.reg_lambda)
        gain_right = right[0] * right[0] / (right[1] + self.reg_lambda)
        split_gain = gain_left + gain_right - gain_all
        self.assertTrue(self.criterion.split_gain(node, left, right), split_gain)

    def test_node_gain(self):
        grad = 0.5
        hess = 6
        gain = grad * grad / (hess + self.reg_lambda)
        self.assertTrue(self.criterion.node_gain(grad, hess), gain)

    def test_node_weight(self):
        grad = 0.5
        hess = 6
        weight = -grad / (hess + self.reg_lambda)
        self.assertTrue(self.criterion.node_weight(grad, hess), weight)


if __name__ == '__main__':
    unittest.main()
