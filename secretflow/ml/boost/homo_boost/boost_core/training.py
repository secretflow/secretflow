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

# coding: utf-8

"""Training Library containing training routines."""
import copy
import logging
import os
from typing import Dict, List, Union, Optional, Sequence

import xgboost.core as xgb_core
from xgboost.core import Metric, Objective
from xgboost.callback import (
    TrainingCallback,
    CallbackContainer,
    EvaluationMonitor,
    EarlyStopping,
)
import secretflow.device.link as link
from secretflow.data.horizontal import HDataFrame
from secretflow.ml.boost.homo_boost.boost_core.core import FedBooster


def _configure_custom_metric(
    feval: Optional[Metric], custom_metric: Optional[Metric]
) -> Optional[Metric]:
    if feval is not None:
        link = (
            "https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html"
        )
        logging.warn(
            "`feval` is deprecated, use `custom_metric` instead.  They have "
            "different behavior when custom objective is also used."
            f"See {link} for details on the `custom_metric`."
        )
    if feval is not None and custom_metric is not None:
        raise ValueError(
            "Both `feval` and `custom_metric` are supplied.  Use `custom_metric` instead."
        )
    eval_metric = custom_metric if custom_metric is not None else feval
    return eval_metric


def _assert_new_callback(
    callbacks: Optional[Sequence[TrainingCallback]],
) -> None:
    is_new_callback: bool = not callbacks or all(
        isinstance(c, TrainingCallback) for c in callbacks
    )
    if not is_new_callback:
        link = "https://xgboost.readthedocs.io/en/latest/python/callbacks.html"
        raise ValueError(
            f"Old style callback was removed in version 1.6.  See: {link}."
        )


def _train_internal(
    params: Dict,
    dtrain: xgb_core.DMatrix,
    hdata: HDataFrame,
    global_bin: List,
    num_boost_round: int = 10,
    evals: List = (),
    obj: Optional[Objective] = None,
    feval: Optional[Metric] = None,
    xgb_model: Union[str, os.PathLike, xgb_core.Booster, bytearray] = None,
    callbacks: Optional[Sequence[TrainingCallback]] = None,
    evals_result: Dict = None,
    maximize: bool = None,
    verbose_eval: Union[bool, int] = None,
    early_stopping_rounds: int = None,
    custom_metric: Optional[Metric] = None,
):
    """internal training function"""
    role = link.get_role()
    callbacks = [] if callbacks is None else copy.copy(callbacks)
    metric_fn = _configure_custom_metric(feval, custom_metric)
    evals = list(evals)
    start_iteration = 0
    pre_round = 0
    if xgb_model is None:
        bst = FedBooster(params, [dtrain] + [d[0] for d in evals])

    else:
        bst = FedBooster(params, [dtrain] + [d[0] for d in evals], model_file=xgb_model)

    start_iteration = 0
    _assert_new_callback(callbacks)
    if verbose_eval:
        verbose_eval = 1 if verbose_eval is True else verbose_eval
        callbacks.append(EvaluationMonitor(period=verbose_eval))
    if early_stopping_rounds:
        callbacks.append(EarlyStopping(rounds=early_stopping_rounds, maximize=maximize))
    cb_container = CallbackContainer(
        callbacks,
        metric=metric_fn,
        # For old `feval` parameter, the behavior is unchanged.  For the new
        # `custom_metric`, it will receive proper prediction result when custom objective
        # is not used.
        output_margin=callable(obj) or metric_fn is feval,
    )

    bst = cb_container.before_training(bst)
    # finetune need align iteration round between server and client.
    if role == link.CLIENT:
        pre_round = bst.num_boosted_rounds()
        link.send_to_server(name="pre_round", value=pre_round, version=0)
    if role == link.SERVER:
        pre_round_list = link.recv_from_clients(
            name="pre_round",
            version=0,
        )
        if len(set(pre_round_list)) != 1:
            raise ValueError(
                f"num round before training for clients must aligned, but got {pre_round_list}"
            )
        pre_round = pre_round_list[0]
    start_iteration += pre_round
    for i in range(start_iteration, start_iteration + num_boost_round):
        if cb_container.before_iteration(bst, i, dtrain, evals):
            break
        # bst calls federate_update to build the tree of this iteration in federated mode, and merges it into the xgboost model after the construction is complete
        bst.federate_update(params, dtrain, hdata, global_bin, iter_round=i, fobj=obj)

        if cb_container.after_iteration(bst, i, dtrain, evals):
            break

    bst = cb_container.after_training(bst)

    if evals_result is not None:  # and is_new_callback:
        evals_result.update(callbacks.history)

    # These should be moved into callback functions `after_training`, but until old
    # callbacks are removed, the train function is the only place for setting the
    # attributes.
    num_parallel, _ = xgb_core._get_booster_layer_trees(bst)
    if bst.attr('best_score') is not None:
        bst.best_score = float(bst.attr('best_score'))
        bst.best_iteration = int(bst.attr('best_iteration'))
        # num_class is handled internally
        bst.set_attr(best_ntree_limit=str((bst.best_iteration + 1) * num_parallel))
        bst.best_ntree_limit = int(bst.attr("best_ntree_limit"))
    else:
        # Due to compatibility with version older than 1.4, these attributes are added
        # to Python object even if early stopping is not used.
        bst.best_iteration = bst.num_boosted_rounds() - 1
        bst.best_ntree_limit = (bst.best_iteration + 1) * num_parallel

    # Copy to serialise and unserialise booster to reset state and free
    # training memory
    return bst.copy()


def train(
    params: Dict,
    dtrain: xgb_core.DMatrix,
    hdata: HDataFrame,
    bin_split_points: List,
    num_boost_round: int = 10,
    evals: List = (),
    obj: Optional[Objective] = None,
    feval: Optional[Metric] = None,
    maximize: bool = None,
    early_stopping_rounds: int = None,
    evals_result: Dict = None,
    verbose_eval: Union[bool, int] = True,
    xgb_model: Union[str, os.PathLike, xgb_core.Booster, bytearray] = None,
    callbacks: Optional[Sequence[TrainingCallback]] = None,
    custom_metric: Optional[Metric] = None,
):
    """Specifies the parameter training level federated version of xgboost.

    Args:
        params: Booster parameters. Reference: https://xgboost.readthedocs.io/en/latest/parameter.html
        dtrain: The training data passed in in DMatrix format.
        hdata: The training data passed in in HDataFrame format, the content is consistent with dtrain
        bin_split_points: The global equal-frequency binning of each feature is recorded
        num_boost_round: Number of iteration rounds.
        evals: A list of validation sets to evaluate during training to help us monitor training effects during training.
        obj: custom objective function.
        feval: Custom eval function.
        maximize: Whether the function in feval is optimized in the maximize direction.
        early_stopping_rounds: used to activate the early_stop strategy, the eval metric of the validation set needs to be executed every early stop round
            Raised at least once. Correspondingly, at least one item must be added to evals. If there are multiple items in eval metric, then the last item
            The indicator will be used for early stop strategy
        evals_result: The evals_result dictionary, used to store the evaluation (eval) results of all items in the watch list.
        verbose_eval: If verbose_eval is True, the evaluation metrics on the validation set will be printed out at each iteration, and if verbose_eval is true
            If it is an int, the evaluation metric of the validation set will be printed after each verbose_eval* iterations
        xgb_model: The trained xgb model can transfer the path or the loaded model for relay training or breakpoint retraining
        callbacks: a list of callback functions that will be applied to each iteration of training
        custom_metric:  Custom metric function.  See :doc:`Custom Metric </tutorials/custom_metric_obj>` for details.
    Returns:
        Booster : a trained booster model
    """
    bst = _train_internal(
        params,
        dtrain,
        hdata,
        bin_split_points,
        num_boost_round=num_boost_round,
        evals=evals,
        obj=obj,
        feval=feval,
        xgb_model=xgb_model,
        callbacks=callbacks,
        verbose_eval=verbose_eval,
        evals_result=evals_result,
        maximize=maximize,
        early_stopping_rounds=early_stopping_rounds,
        custom_metric=custom_metric,
    )
    return bst
