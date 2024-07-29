# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np

from secretflow.data import FedNdarray
from secretflow.data.vertical import VDataFrame
from secretflow.device import HEU, PYUObject, wait
from secretflow.ml.boost.core.callback import (
    CallBackCompatibleModel,
    CallbackContainer,
    TrainingCallback,
    VData,
)
from secretflow.ml.boost.core.data_preprocess import prepare_dataset
from secretflow.ml.boost.core.metric import Metric
from secretflow.ml.boost.sgb_v.checkpoint import (
    SGBCheckpointData,
    checkpoint_data_to_model_and_train_state,
    sgb_model_to_checkpoint_data,
)
from secretflow.ml.boost.sgb_v.core.params import default_params

from ...model import SgbModel
from ..components import DataPreprocessor, ModelBuilder, OrderMapManager, TreeTrainer
from ..components.component import Composite, Devices, label_have_feature, print_params
from ..sgb_actor import SGBActor


@dataclass
class GlobalOrdermapBoosterComponents:
    preprocessor: DataPreprocessor = DataPreprocessor()
    order_map_manager: OrderMapManager = OrderMapManager()
    model_builder: ModelBuilder = ModelBuilder()


@dataclass
class GlobalOrdermapBoosterParams:
    """params specifically belonged to global ordermap booster, not its components.

    num_boost_round : int, default=10
                Number of boosting iterations. Same as number of trees.
                range: [1, 1024]
    first_tree_with_label_holder_feature: bool, default=False
                Whether to train the first tree with label holder's own features.
                Can increase training speed and label security.
                The training loss may increase.
                If label holder has no feature, set this to False.
    eval_metric: str, default=None, must be one of 'roc_auc', 'rmse' and 'mse'
    """

    num_boost_round: int = default_params.num_boost_round
    first_tree_with_label_holder_feature: bool = (
        default_params.first_tree_with_label_holder_feature
    )


class GlobalOrdermapBooster(Composite, CallBackCompatibleModel):
    """
    This class provides both classification and regression tree boosting (also known as GBDT, GBM)
    for vertical split dataset setting by using level wise boost.
    """

    def __init__(self, heu: HEU, tree_trainer: TreeTrainer) -> None:
        self.components = GlobalOrdermapBoosterComponents()
        self.params = GlobalOrdermapBoosterParams()
        self.user_config = {}
        self.heu = heu
        self.tree_trainer = tree_trainer
        self._best_iteration = None
        self._best_score = None
        self.save_best = False
        self.eval_predict_cache = {}

    def _set_booster_params(self, params: dict):
        trees = params.get('num_boost_round', default_params.num_boost_round)
        self.params.num_boost_round = trees
        self.params.first_tree_with_label_holder_feature = params.get(
            'first_tree_with_label_holder_feature',
            default_params.first_tree_with_label_holder_feature,
        )

    def _get_booster_params(self, params: dict):
        params['num_boost_round'] = self.params.num_boost_round
        params['first_tree_with_label_holder_feature'] = (
            self.params.first_tree_with_label_holder_feature
        )

    def show_params(self):
        super().show_params()
        self.tree_trainer.show_params()
        print_params(self.params)

    def set_params(self, params: dict):
        super().set_params(params)
        self.tree_trainer.set_params(params)
        self._set_booster_params(params)
        self.user_config = params

    def get_params(self, params: dict = {}) -> dict:
        super().get_params(params)
        self.tree_trainer.get_params(params)
        self._get_booster_params(params)
        return params

    def set_devices(self, devices: Devices):
        super().set_devices(devices)
        self.tree_trainer.set_devices(devices)

    def set_actors(self, actors: List[SGBActor]):
        super().set_actors(actors)
        self.tree_trainer.set_actors(actors)

    def del_actors(self):
        super().del_actors()
        self.tree_trainer.del_actors()

    def fit(
        self,
        dataset: Union[FedNdarray, VDataFrame],
        label: Union[FedNdarray, VDataFrame],
        callbacks: List[TrainingCallback] = [],
        eval_sets: List[Tuple[VData, VData, str]] = [],
        metric: Metric = None,
        checkpoint_data: SGBCheckpointData = None,
        sample_weight: Union[FedNdarray, VDataFrame] = None,
    ) -> SgbModel:
        import secretflow.distributed as sfd

        checkpoint_model = None
        history = None
        if checkpoint_data is not None:
            checkpoint_model, history = checkpoint_data_to_model_and_train_state(
                checkpoint_data
            )
        call_backs = CallbackContainer(callbacks, metric, history)
        data_names = set([e[2] for e in eval_sets])
        assert len(data_names) == len(
            eval_sets
        ), f"Each data_name in evals must be unique, got {data_names}"
        self.eval_predict_cache = {data_name: None for data_name in data_names}
        if sfd.in_ic_mode():
            x = dataset
            y = list(label.partitions.values())[0]
            sample_weight_object = (
                list(sample_weight.partitions.values())[0]
                if sample_weight is not None
                else None
            )
            y = y.device(lambda y: y.reshape(-1, 1, order='F'))(y)
            sample_num = y.device(lambda y: y.shape[0])(y)
        else:
            x, x_shape, y, _, sample_weight_object = (
                self.components.preprocessor.validate(
                    dataset, label, sample_weight=sample_weight
                )
            )
            sample_num = x_shape[0]
        # set devices
        devices = Devices(y.device, [*x.partitions.keys()], self.heu)
        self.devices = devices
        actors = [SGBActor(device=device) for device in devices.workers]
        if not label_have_feature(devices):
            logging.warning(
                "label holder has no feature, setting first tree with label holder to be False."
            )
            # disable train using label holder's device
            self.set_params({"first_tree_with_label_holder_feature": False})
            # add label holder to actors
            actors.append(SGBActor(device=devices.label_holder))
        self.set_devices(devices)
        # set actors

        logging.debug("actors are created.")
        self.set_actors(actors)
        logging.debug("actors are set.")

        pred = self.components.model_builder.init_pred(sample_num, checkpoint_model, x)
        logging.debug("pred initialized.")
        self.components.order_map_manager.build_order_map(x)
        logging.debug("ordermap built.")
        self.components.model_builder.init_model(checkpoint_model)
        logging.debug("model initialized.")
        self.components.model_builder.set_parition_shapes(x)

        begin_tree_num = self.components.model_builder.get_tree_num()

        call_backs.before_training(self)
        for tree_index in range(begin_tree_num, self.params.num_boost_round):
            call_backs.before_iteration(self, tree_index)
            start = time.perf_counter()
            if self.params.first_tree_with_label_holder_feature and tree_index == 0:
                # we are sure the config is small, so ok to copy
                config = copy.deepcopy(self.user_config)
                config['label_holder_feature_only'] = True
                self.tree_trainer.set_params(config)
                logging.info("training the first tree with label holder only.")
            tree = self.tree_trainer.train_tree(
                tree_index,
                self.components.order_map_manager,
                y,
                pred,
                sample_num,
                sample_weight=sample_weight_object,
            )
            if self.params.first_tree_with_label_holder_feature and tree_index == 0:
                config['label_holder_feature_only'] = False
                self.tree_trainer.set_params(config)

            # check if the tree is meaningful
            # if more than root node exists, then insert
            # else stop the training process
            if tree.is_empty():
                logging.info(f"tree {tree_index} is empty, stop training.")
                break
            self.components.model_builder.insert_tree(tree)
            cur_tree_num = self.components.model_builder.get_tree_num()

            if cur_tree_num < self.params.num_boost_round:
                pred = y.device(lambda x, y: x + np.array(y, order='F'))(
                    pred, tree.predict(x.partitions)
                )
                wait([pred])
            else:
                wait(tree)

            logging.info(
                f"epoch {cur_tree_num - 1} time {time.perf_counter() - start}s"
            )
            should_stop = call_backs.after_iteration(self, tree_index, eval_sets)
            if should_stop:
                logging.info(
                    f"early_stopped, current tree num: {self.components.model_builder.get_tree_num()}"
                )
                break
        self.del_actors()
        call_backs.after_training(self)
        if self.save_best:
            return self.components.model_builder.finish()[: self.best_iteration() + 1]
        return self.components.model_builder.finish()

    def _eval_predict(self, data: VData, data_name: str) -> PYUObject:
        """Special eval prediction for call back evaluation
        assumed prediction is used exactly once per boosting round
        used cache to avoid repeated computation
        """
        import secretflow.distributed as sfd

        model = self.components.model_builder.finish()
        if sfd.in_ic_mode():
            x = data
        else:
            x, _ = prepare_dataset(data)
        if self.eval_predict_cache[data_name] is not None:
            # predict on the new tree is sufficient
            cache = self.eval_predict_cache[data_name]
            pred = model.get_trees()[-1].predict(x.partitions)
            new_pred = pred.device(lambda x, y: np.add(x, y).reshape(-1, 1))(
                pred, cache
            )

        else:
            new_pred = model.predict_with_trees(x)
        self.eval_predict_cache[data_name] = new_pred
        return model.apply_activation(new_pred)

    def eval_set(
        self,
        evals: Sequence[Tuple[VData, VData, str]],
        feval: Metric,
    ) -> List:
        """Evaluate a set of data.

        Parameters
        ----------
        evals :
            List of items to be evaluated.
        feval :
            Custom evaluation function.

        Returns
        -------
        result: List
            Evaluation result List.
        """
        res = []
        if feval is None:
            return res

        for data, label, data_name in evals:
            if not (isinstance(data, FedNdarray) or isinstance(data, VDataFrame)):
                raise TypeError(
                    f"expected FedNdarray or VDataFrame, got {type(data).__name__}"
                )

            if not (isinstance(label, FedNdarray) or isinstance(label, VDataFrame)):
                raise TypeError(
                    f"expected  FedNdarray or VDataFrame, got {type(label).__name__}"
                )

            if not isinstance(data_name, str):
                raise TypeError(f"expected string, got {type(data_name).__name__}")

            data, _ = prepare_dataset(data)
            label, _ = prepare_dataset(label)
            feval_ret = feval(
                label,
                self._eval_predict(data, data_name),
            )
            if isinstance(feval_ret, list):
                for metric_name, metric_val in feval_ret:
                    res.append((data_name + "-" + metric_name, metric_val))
            else:
                metric_name, metric_val = feval_ret
                res.append((data_name + "-" + metric_name, metric_val))
        return res

    def set_save_best(self, save_best: bool):
        self.save_best = save_best

    def best_iteration(self):
        return self._best_iteration

    def best_score(self):
        return self._best_score

    def set_best_iteration_score(self, iteration, score):
        self._best_iteration = iteration
        self._best_score = score

    def get_model(self):
        return self.components.model_builder.finish()


def build_checkpoint(
    booster: GlobalOrdermapBooster,
    evals_log: TrainingCallback.EvalsLog,
    x: VDataFrame,
    label_name: str,
) -> SGBCheckpointData:
    """Build checkpoint from booster and evals log."""
    sgb_model = booster.get_model()
    return sgb_model_to_checkpoint_data(sgb_model, evals_log, x, label_name)
