#!/usr/bin/env python3
# *_* coding: utf-8 *_*

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


"""Homo Booster"""
import os
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
import xgboost as xgb

import secretflow.device.link as link
from secretflow.device import PYUObject, proxy
from secretflow.device.device import PYU
from secretflow.ml.boost.homo_boost.boost_core.core import FedBooster
from secretflow.ml.boost.homo_boost.boost_core.training import train
from secretflow.ml.boost.homo_boost.tree_core.loss_function import LossFunction


@proxy(PYUObject, _simulation_max_concurrency=2)
class HomoBooster(link.Link):
    def __init__(
        self,
        device: PYU = None,
        clients: List[PYU] = None,
        server: PYU = None,
        msg_id_prefix='',
    ):
        super().__init__(device, msg_id_prefix)
        self.clients = clients
        self.server = server
        self.device = device
        self.role = link.SERVER if device == server else link.CLIENT
        self.bst = None

    def set_split_point(self, bin_split_points):
        # set global binning
        self.bin_split_points = bin_split_points

    def gen_mock_data(
        self,
        data_num: int = 100,
        columns: List[str] = None,
        label_name: str = None,
        num_class: int = None,
    ) -> pd.DataFrame:
        """mock data with the same schema for the SERVER to synchronize
        the training process

        Args:
            data_num: rows of data
            columns: feature names of data
            label_name: label name of data
        Returns:
            data: mock data,has same schema with HdataFrame
        """
        fea_num = len(columns)
        index_colname_map = {}
        for index, name in enumerate(columns):
            index_colname_map[index] = name
        features = np.random.random((data_num, fea_num))
        labels = np.random.randint(0, num_class, (data_num, 1))

        data = pd.DataFrame(features)
        data.rename(columns=index_colname_map, inplace=True)
        data[label_name] = labels
        return data

    def homo_train(
        self,
        train_hdf: pd.DataFrame,
        valid_hdf: pd.DataFrame,
        params: Dict = None,
        num_boost_round: int = 10,
        obj=None,
        feval=None,
        maximize: bool = None,
        early_stopping_rounds: int = None,
        evals_result: Dict = None,
        verbose_eval: Union[int, bool] = True,
        xgb_model=None,
        callbacks: List[Callable] = None,
    ) -> FedBooster:
        """Fed xgboost entrance

        Args:
            train_hdf: federate table for training
            valid_hdf: federate table for valid
            params: a dict of all params
            num_boost_round: num of boost round
            obj: user define obj, objective type will be squared_error
            feval: user define eval function
            maximize: is feval going to maximize
            early_stopping_rounds: same as xgboost early_stooping_round
            evals_result: a container store results of evaluation
            verbose_eval: same as xgboost verbose_eval
            xgb_model: xgb model path, be used for training continuation
            callbacks: callback function list
        """
        link.set_mesh(self)

        if "label_key" not in params:
            return
        if obj is not None:
            raise NotImplementedError(f"Custom object function is not supported")
        columns = [x for x in train_hdf.columns]
        if params['hess_key'] in columns:
            raise Exception(
                f"The value of hess_key must be different from other columns in the data"
            )
        if params['grad_key'] in columns:
            raise Exception(
                f"The value of grad_key must be different from other columns in the data"
            )

        columns.remove(params["label_key"])

        dtrain = xgb.DMatrix(
            train_hdf.drop(columns=[params["label_key"]]),
            train_hdf[params["label_key"]],
        )
        dvalid = xgb.DMatrix(
            valid_hdf.drop(columns=[params["label_key"]]),
            valid_hdf[params["label_key"]],
        )

        obj_func = LossFunction(params['objective']).obj_function()

        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        self.bst = train(
            params,
            dtrain,
            train_hdf,
            self.bin_split_points,
            num_boost_round=num_boost_round,
            evals=watchlist,
            obj=obj_func,
            feval=feval,
            maximize=maximize,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=verbose_eval,
            xgb_model=xgb_model,
            callbacks=callbacks,
        )
        return self.bst

    def homo_eval(
        self,
        eval_hdf: pd.DataFrame,
        params: Dict,
        model_path: str,
    ):
        link.set_mesh(self)

        if self.role == link.CLIENT:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"model file {model_path} can not found")
            try:
                bst = xgb.Booster(params)
                bst.load_model(model_path)
            except Exception as e:
                raise InterruptedError(f"Load model interrupted! detail:{e}")
            deval = xgb.DMatrix(
                eval_hdf.drop(columns=[params["label_key"]]),
                eval_hdf[params["label_key"]],
            )
            score = bst.eval(data=deval)
            score = score.split()[1:]
            score = [tuple(s.split(':')) for s in score]
            link.send_to_server(name=f"eval_local", value=score, version=0)

            return score
        if self.role == link.SERVER:
            all_score = link.recv_from_clients(
                name=f"eval_local",
                version=0,
            )

            num_party = len(all_score)
            all_score_dict = [dict(score) for score in all_score]
            sum_score = {
                k: sum(float(d[k]) for d in all_score_dict) / num_party
                for k in all_score_dict[0]
            }
            agg_score = [(k, v) for k, v in sum_score.items()]

            # prepare summary eval_metrics
            # prepare global infos
            metrics = sum_score.keys()
            metrics = [m.replace("_", "-") for m in metrics]
            global_metrics = []
            for s in agg_score:
                global_metrics.append(s)
            return global_metrics
        assert False, 'Should never get here.'

    def save_model(self, model_path: Union[str, os.PathLike]):
        if self.role == link.CLIENT:
            self.bst.save_model(model_path)

    def dump_model(self, model_path: Union[str, os.PathLike]):
        if self.role == link.CLIENT:
            self.bst.dump_model(model_path)
