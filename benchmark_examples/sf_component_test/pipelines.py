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

import os

from config_reader import load_config_dict

from secretflow.component.test_framework.test_case import PipelineCase, TestComp


def sgb(current_dir, config_dict):
    sgb_test = PipelineCase("sgb")
    attrs = load_config_dict(os.path.join(current_dir, config_dict["sgb_train"]))

    sgb = TestComp("sgb", "ml.train", "sgb_train", "0.0.1", attrs)

    sgb_test.add_comp(sgb, ["DAGInput.train_data"])

    sgb_pred_attrs = load_config_dict(
        os.path.join(current_dir, config_dict["sgb_pred"])
    )
    sgb_pred = TestComp(
        "sgb_pred", "ml.predict", "sgb_predict", "0.0.1", sgb_pred_attrs
    )

    sgb_test.add_comp(sgb_pred, ["sgb.0", "DAGInput.test_data"])

    biclassification_attr = load_config_dict(
        os.path.join(current_dir, config_dict["biclassification_eval"])
    )
    biclassification_eval = TestComp(
        "biclassification_eval",
        "ml.eval",
        "biclassification_eval",
        "0.0.1",
        biclassification_attr,
    )

    sgb_test.add_comp(biclassification_eval, ["DAGInput.test_data", "sgb_pred.0"])
    return sgb_test


def woe(current_dir, config_dict):
    woe_test = PipelineCase("woe")
    attrs = load_config_dict(os.path.join(current_dir, config_dict["woe_he"]))

    woe = TestComp("woe", "feature", "vert_woe_binning", "0.0.1", attrs)

    woe_test.add_comp(woe, ["DAGInput.train_data"])

    return woe_test


def logistic_regression(current_dir, config_dict):
    logistic_regression_test = PipelineCase("logistic_regression")
    attrs = load_config_dict(
        os.path.join(current_dir, config_dict["logistic_regression_train"])
    )

    logistic_regression_train = TestComp(
        "ss_sgd_train", "ml.train", "ss_sgd_train", "1.0.0", attrs
    )

    logistic_regression_test.add_comp(
        logistic_regression_train, ["DAGInput.train_data"]
    )

    logistic_regression_pred_attrs = load_config_dict(
        os.path.join(current_dir, config_dict["logistic_regression_pred"])
    )
    logistic_regression_pred = TestComp(
        "ss_sgd_predict",
        "ml.predict",
        "ss_sgd_predict",
        "1.0.0",
        logistic_regression_pred_attrs,
    )

    logistic_regression_test.add_comp(
        logistic_regression_pred, ["ss_sgd_train.0", "DAGInput.test_data"]
    )

    biclassification_attr = load_config_dict(
        os.path.join(current_dir, config_dict["biclassification_eval"])
    )
    biclassification_eval = TestComp(
        "biclassification_eval",
        "ml.eval",
        "biclassification_eval",
        "0.0.1",
        biclassification_attr,
    )

    logistic_regression_test.add_comp(
        biclassification_eval, ["DAGInput.test_data", "ss_sgd_predict.0"]
    )
    return logistic_regression_test


pipeline_map = {"sgb": sgb, "woe": woe, "logistic_regression": logistic_regression}
