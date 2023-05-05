import math

import pytest

from secretflow.ml.boost.homo_boost.tree_core.criterion import XgboostCriterion

reg_lambda = 0.3


@pytest.fixture(scope='function')
def criterion():
    yield XgboostCriterion(reg_lambda=reg_lambda)


def test_init(criterion):
    assert criterion.reg_lambda == reg_lambda


def test_split_gain(criterion):
    node = [0.5, 0.6]
    left = [0.1, 0.2]
    right = [0.4, 0.4]
    gain_all = node[0] * node[0] / (node[1] + reg_lambda)
    gain_left = left[0] * left[0] / (left[1] + reg_lambda)
    gain_right = right[0] * right[0] / (right[1] + reg_lambda)
    split_gain = gain_left + gain_right - gain_all
    assert math.isclose(
        criterion.split_gain(node, left, right), split_gain, rel_tol=0.01
    )


def test_node_gain(criterion):
    grad = 0.5
    hess = 6
    gain = grad * grad / (hess + reg_lambda)
    assert math.isclose(criterion.node_gain(grad, hess), gain, rel_tol=0.01)


def test_node_weight(criterion):
    grad = 0.5
    hess = 6
    weight = -grad / (hess + reg_lambda)

    assert math.isclose(criterion.node_weight(grad, hess), weight, rel_tol=0.01)
