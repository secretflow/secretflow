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

from typing import Callable, Dict, List, Union

from validator import GreaterThan, In, LessThan, Not, Range, Required, validate

from secretflow.data.horizontal.dataframe import HDataFrame
from secretflow.device import reveal
from secretflow.device.device import PYU
from secretflow.device.link import init_link
from secretflow.ml.boost.homo_boost.homo_booster_worker import HomoBooster
from secretflow.preprocessing.binning.homo_binning import HomoBinning
from secretflow.utils.random import global_random


class SFXgboost:
    def __init__(self, server, clients):
        self.server = server
        self.clients = clients
        self.fed_bst = {}
        self._workers = []
        msg_id_prefix = str(global_random(self.server, 100000000))
        for client in self.clients:
            self._workers.append(
                HomoBooster(
                    device=client,
                    clients=self.clients,
                    server=self.server,
                    msg_id_prefix=msg_id_prefix,
                )
            )
        self._workers.append(
            HomoBooster(
                device=self.server,
                clients=self.clients,
                server=self.server,
                msg_id_prefix=msg_id_prefix,
            )
        )
        for worker in self._workers:
            partners = [w for w in self._workers if w != worker]
            init_link(worker, partners)

    def check_params(self, params):
        rules = {
            'max_depth': [Required, Range(1, 64, inclusive=True)],
            'eta': [Required, Range(0, 1, inclusive=True)],
            'objective': [
                Required,
                In(
                    [
                        "binary:logistic",
                        "reg:logistic",
                        "multi:softmax",
                        "multi:softprob",
                        "reg:squarederror",
                    ]
                ),
            ],
            'min_child_weight': [GreaterThan(0)],
            'lambda': [Not(LessThan(0))],
            'alpha': [Not(LessThan(0))],
            'max_bin': [Required, GreaterThan(0)],
            'gamma': [Not(LessThan(0))],
            'subsample': [Range(0, 1, inclusive=True)],
            'hess_key': [Required],
            'grad_key': [Required],
            'label_key': [Required],
        }
        if 'colsample_bytree' in params:
            rules['colsample_bytree'] = [Range(0, 1, inclusive=True)]
        if 'colsample_bylevel' in params:
            rules['colsample_bylevel'] = [Range(0, 1, inclusive=True)]
        if 'objective' == "multi:softmax":
            rules['num_class'] = [GreaterThan(0)]
            rules['eval_metric'] = [In(["merror", "mlogloss"])]
        if 'objective' == "multi:softprob":
            rules['num_class'] = [GreaterThan(0)]
            rules['eval_metric'] = [In(["merror", "mlogloss", "auc"])]

        return validate(rules, params)

    def train(
        self,
        train_hdf: HDataFrame,
        valid_hdf: HDataFrame,
        params: Dict = None,
        num_boost_round: int = 10,
        obj=None,
        feval=None,
        maximize: bool = None,
        early_stopping_rounds: int = None,
        evals_result: Dict = None,
        verbose_eval: Union[int, bool] = True,
        xgb_model: Dict = None,
        callbacks: List[Callable] = None,
    ) -> "SFXgboost":
        """Federated xgboost interface for training

        Args:
            train_hdf: horizontal federation table used for training
            valid_hdf: horizontal federated table for validation
            params: dictionary of parameters
            num_boost_round: Number of spanning trees required
            obj: custom objective function, objective type is squared_error
            feval: custom eval evaluation function
            maximize: whether feval is maximized
            early_stopping_rounds: same as xgboost early_stooping_round option
            evals_result: container for storing evaluation results
            verbose_eval: same as xgboost verbose_eval
            xgb_model: xgb model file path, used for breakpoint retraining (training continuation)
            callbacks: list of callback functions
        """
        # set up thread pool
        valid, errors = self.check_params(params)
        if not valid:
            raise Exception(f"param not legal,err msg: {errors}")

        columns = [x for x in train_hdf.columns]
        columns.remove(params["label_key"])
        bin_obj = HomoBinning(
            bin_num=params['max_bin'],
            bin_names=columns,
            error=1e-10,
            max_iter=200,
            compress_thres=5 * params['max_bin'],
        )
        bin_dict = reveal(bin_obj.fit_split_points(train_hdf))
        bin_split_points = []
        for col in columns:
            bin_split_points.append(bin_dict[col])

        for worker in self._workers:
            worker.set_split_point(bin_split_points)
        for worker in self._workers:
            if xgb_model is not None and worker.device in xgb_model.keys():
                xgb_partition = xgb_model[worker.device]

            else:
                xgb_partition = None
            if worker.device in train_hdf.partitions.keys():
                self.fed_bst[worker.device] = worker.homo_train(
                    train_hdf=train_hdf.partitions[worker.device].data,
                    valid_hdf=valid_hdf.partitions[worker.device].data,
                    params=params,
                    num_boost_round=num_boost_round,
                    obj=obj,
                    feval=feval,
                    maximize=maximize,
                    early_stopping_rounds=early_stopping_rounds,
                    evals_result=evals_result,
                    verbose_eval=verbose_eval,
                    xgb_model=xgb_partition,
                    callbacks=callbacks,
                )
            else:
                num_class = params["num_class"] if "num_class" in params else 2
                mock_df = worker.gen_mock_data(
                    data_num=100,
                    columns=columns,
                    label_name=params["label_key"],
                    num_class=num_class,
                )
                self.fed_bst[worker.device] = worker.homo_train(
                    train_hdf=mock_df,
                    valid_hdf=mock_df,
                    params=params,
                    num_boost_round=num_boost_round,
                    obj=obj,
                    feval=feval,
                    maximize=maximize,
                    early_stopping_rounds=early_stopping_rounds,
                    evals_result=evals_result,
                    verbose_eval=verbose_eval,
                    xgb_model=xgb_partition,
                    callbacks=callbacks,
                )
        reveal(self.fed_bst)  # wait all tasks done

    def save_model(self, model_path: Dict):
        """Federated xgboost save model interface

        Args:
            model_path: Path of the model stored
        """
        assert self.fed_bst is not None, "FedBooster must be train before save model"
        res = {}
        for worker in self._workers:
            if worker.device in model_path:
                res[worker.device] = worker.save_model(model_path[worker.device])
        reveal(res)

    def dump_model(self, model_path: Dict):
        """Federated xgboost dump model interface

        Args:
            model_path: Path of the model stored
        """
        assert self.fed_bst is not None, "FedBooster must be train before dump model"
        res = {}
        for worker in self._workers:
            if worker.device in model_path.keys():
                res[worker.device] = worker.dump_model(model_path[worker.device])
        reveal(res)

    def eval(
        self,
        model_path: Union[str, Dict[PYU, str]],
        hdata: HDataFrame,
        params: Dict,
    ):
        """Federated xgboost eval interface

        Args:
            model_path: Path of the model stored
            hdata: Horizontal dataframe to be evaluated
            params: Xgboost params

        Returns:
            result: Dict evaluate result
        """
        assert isinstance(
            model_path, (str, Dict)
        ), f'Model path accepts string or dict but got {type(model_path)}.'
        if isinstance(model_path, str):
            model_path = {device: model_path for device in self._workers.keys()}

        res = {}
        for worker in self._workers:
            device = worker.device
            if device == self.server:
                hdata_partition = None
                model_path_device = "mock_path"
            else:
                hdata_partition = hdata.partitions[worker.device].data
                model_path_device = model_path[device]
            res[device.party] = worker.homo_eval(
                hdata_partition, params, model_path_device
            )
        return reveal(res)
