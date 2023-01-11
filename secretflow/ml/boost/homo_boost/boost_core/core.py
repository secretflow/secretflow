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

import logging
import os
import pathlib
import uuid
from typing import Callable, Dict, List, Union

import numpy
import xgboost.core as xgb_core

import secretflow.device.link as link
from secretflow.data.horizontal import HDataFrame
from secretflow.data.split import train_test_split
from secretflow.ml.boost.homo_boost.homo_decision_tree import HomoDecisionTree
from secretflow.ml.boost.homo_boost.tree_param import TreeParam
from secretflow.utils.errors import InvalidArgumentError


class FedBooster(xgb_core.Booster):
    """Federated Booster internal
    Internal implementation, it is not recommended for users to call directly! ! !

    Attributes:
        params : Parameters for boosters.
        cache : List of cache items.
        model_file : Path to the model file if it's string or PathLike.
    """

    def __init__(
        self,
        params: Dict = None,
        cache: List = (),
        model_file: Union[str, os.PathLike, xgb_core.Booster, bytearray] = None,
    ):
        checkpoint_dir = '.checkpoint'
        pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.model_path = f"{checkpoint_dir}/{link.get_device()}_{uuid.uuid1()}.json"
        if 'hess_key' in params:
            self.hess_key = params.pop("hess_key")
        else:
            raise InvalidArgumentError("hess_key must be assignd!")
        if 'grad_key' in params:
            self.grad_key = params.pop("grad_key")
        else:
            raise InvalidArgumentError("grad_key must be assignd")
        if 'label_key' in params:
            self.label_key = params.pop("label_key")
        else:
            raise InvalidArgumentError("label_key must be assignd")
        self.role = link.get_role()
        super(FedBooster, self).__init__(
            params=params, cache=cache, model_file=model_file
        )
        self.save_model(self.model_path)

    def federate_update(
        self,
        params: Dict,
        dtrain: xgb_core.DMatrix,
        hdata: HDataFrame,
        bin_split_points: List,
        iter_round: int = None,
        fobj: Callable = None,
    ):
        """
        federated update function, a variant in xgboost update
        Args:
            params: Training params dict
            dtrain: Training data in dmatrix format
            hdata: Training data in HdataFrame format
            bin_split_points: Global split point
            iter_round: Iteration rounds
            fobj: Custom evaluation function

        """
        if not isinstance(dtrain, xgb_core.DMatrix):
            raise TypeError('invalid training matrix: {}'.format(type(dtrain).__name__))
        self._validate_features(dtrain)
        # Create tree_params
        tree_param = TreeParam(
            max_depth=params['max_depth'] if 'max_depth' in params else 3,
            eta=params['eta'] if 'eta' in params else 0.3,
            objective=params['objective'],
            verbosity=params['verbosity'] if 'verbosity' in params else 0,
            tree_method=params['tree_method'] if 'tree_method' in params else 'hist',
            reg_lambda=params['lambda'] if 'lambda' in params else 0.1,
            reg_alpha=params['alpha'] if 'alpha' in params else 0.0,
            gamma=params['gamma'] if 'gamma' in params else 1e-4,
            colsample_bytree=params['colsample_bytree']
            if 'colsample_bytree' in params
            else 1.0,
            colsample_byleval=params['colsample_bylevel']
            if 'colsample_bylevel' in params
            else 1.0,
            base_score=params['base_score'] if 'base_score' in params else 0.5,
            random_state=params['random_state'] if 'random_state' in params else 1234,
            num_parallel=params['n_thread'] if 'n_thread' in params else None,
            subsample=params['subsample'] if 'subsample' in params else 1.0,
            decimal=params['decimal'] if 'decimal' in params else 10,
            num_class=params['num_class'] if 'num_class' in params else 0,
        )
        # sample by row
        if tree_param.subsample < 1.0:
            train_data, _ = train_test_split(
                hdata, ratio=tree_param.subsample, random_state=tree_param.random_state
            )
        else:
            train_data = hdata

        pred = self.predict(dtrain, output_margin=True, training=True)
        grad, hess = fobj(pred, dtrain)
        group_num = numpy.expand_dims(pred, axis=-1).shape[1]

        # single thread
        if group_num > 2:
            assert params['objective'] in [
                "multi:softmax",
                "multi:softprob",
            ], "Use only 'multi:softmax' for multi-category tasks"
            assert (
                group_num == params['num_class']
            ), "group_num and num_class not aligned"
        for group_id in range(group_num):
            if group_num > 2:
                hdata[self.grad_key], hdata[self.hess_key] = (
                    grad[:, group_id],
                    hess[:, group_id],
                )
            else:
                hdata[self.grad_key], hdata[self.hess_key] = grad, hess

            tree_id = iter_round * group_num + group_id
            decision_tree = HomoDecisionTree(
                tree_param=tree_param,
                data=train_data,
                bin_split_points=bin_split_points,
                group_id=group_id,
                tree_id=tree_id,
                iter_round=iter_round,
                hess_key=self.hess_key,
                grad_key=self.grad_key,
                label_key=self.label_key,
            )
            decision_tree.fit()

            if self.role == link.CLIENT:
                if tree_id == 0:
                    decision_tree.init_xgboost_model(self.model_path)
                decision_tree.save_xgboost_model(
                    self.model_path, decision_tree.tree_node
                )
        logging.info(f"fit for iter_round={iter_round} done")
        if self.role == link.CLIENT:
            self.load_model(self.model_path)

    def save_model(self, fname: Union[str, os.PathLike]):
        """Save the model to a file.

        Attributes:
            fname : string or os.PathLike, model path, if the suffix is json, store the model in json format

        """
        if isinstance(fname, (str, os.PathLike)):  # assume file name
            fname = os.fspath(os.path.expanduser(fname))
            xgb_core._check_call(
                xgb_core._LIB.XGBoosterSaveModel(self.handle, xgb_core.c_str(fname))
            )
        else:
            raise TypeError("fname must be a string or os PathLike")
