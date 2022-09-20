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
from typing import Callable, Dict, List, Union

import xgboost.core as xgb_core
from secretflow.data.horizontal import HDataFrame
from secretflow.ml.boost.homo_boost.boost_core import callback
from secretflow.ml.boost.homo_boost.boost_core.core import FedBooster
from xgboost import callback as xgb_callback


def _configure_deprecated_callbacks(
    verbose_eval,
    early_stopping_rounds,
    maximize,
    start_iteration,
    num_boost_round,
    feval,
    evals_result,
    callbacks,
    show_stdv,
    cvfolds,
):
    link = 'https://xgboost.readthedocs.io/en/latest/python/callbacks.html'
    logging.warn(f'Old style callback is deprecated.  See: {link}', UserWarning)
    # Most of legacy advanced options becomes callbacks
    if early_stopping_rounds is not None:
        callbacks.append(
            callback.early_stop(
                early_stopping_rounds, maximize=maximize, verbose=bool(verbose_eval)
            )
        )
    if isinstance(verbose_eval, bool) and verbose_eval:
        callbacks.append(callback.print_evaluation(show_stdv=show_stdv))
    else:
        if isinstance(verbose_eval, int):
            callbacks.append(
                callback.print_evaluation(verbose_eval, show_stdv=show_stdv)
            )
    if evals_result is not None:
        callbacks.append(callback.record_evaluation(evals_result))
    callbacks = callback.LegacyCallbacks(
        callbacks, start_iteration, num_boost_round, feval, cvfolds=cvfolds
    )
    return callbacks


def _is_new_callback(callbacks):
    return (
        any(isinstance(c, callback.TrainingCallback) for c in callbacks)
        or not callbacks
    )


def _train_internal(
    params: Dict,
    dtrain: xgb_core.DMatrix,
    hdata: HDataFrame,
    global_bin: List,
    num_boost_round: int = 10,
    evals: List = (),
    obj: Callable = None,
    feval: Callable = None,
    xgb_model: Union[str, os.PathLike, xgb_core.Booster, bytearray] = None,
    callbacks: List = None,
    evals_result: Dict = None,
    maximize: bool = None,
    verbose_eval: Union[bool, int] = None,
    early_stopping_rounds: int = None,
):
    """internal training function"""
    callbacks = [] if callbacks is None else copy.copy(callbacks)
    evals = list(evals)

    bst = FedBooster(params, [dtrain] + [d[0] for d in evals])

    if xgb_model is not None:
        bst = FedBooster(params, [dtrain] + [d[0] for d in evals], model_file=xgb_model)

    start_iteration = 0
    is_new_callback = _is_new_callback(callbacks)
    if is_new_callback:
        assert all(
            isinstance(c, xgb_callback.TrainingCallback) for c in callbacks
        ), "You can't mix new and old callback styles."
        if verbose_eval:
            verbose_eval = 1 if verbose_eval is True else verbose_eval
            callbacks.append(xgb_callback.EvaluationMonitor(period=verbose_eval))
        if early_stopping_rounds:
            callbacks.append(
                xgb_callback.EarlyStopping(
                    rounds=early_stopping_rounds, maximize=maximize
                )
            )
        callbacks = callback.FedCallbackContainer(callbacks, metric=feval)
    else:
        callbacks = _configure_deprecated_callbacks(
            verbose_eval,
            early_stopping_rounds,
            maximize,
            start_iteration,
            num_boost_round,
            feval,
            evals_result,
            callbacks,
            show_stdv=False,
            cvfolds=None,
        )

    bst = callbacks.before_training(bst)

    for i in range(start_iteration, num_boost_round):
        if callbacks.before_iteration(bst, i, dtrain, evals):
            break
        # bst调用federate_update使用联邦方式构建本次迭代的树，构建完成后合并到xgboost模型中
        bst.federate_update(params, dtrain, hdata, global_bin, iter_round=i, fobj=obj)

        if callbacks.after_iteration(bst, i, dtrain, evals):
            break

    bst = callbacks.after_training(bst)

    if evals_result is not None and is_new_callback:
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
    obj: Callable = None,
    feval: Callable = None,
    maximize: bool = None,
    early_stopping_rounds: int = None,
    evals_result: Dict = None,
    verbose_eval: Union[bool, int] = True,
    xgb_model: Union[str, os.PathLike, xgb_core.Booster, bytearray] = None,
    callbacks: List = None,
):
    """指定参数训练水平联邦版本的xgboost.

    Args:
        params:Booster的参数. 参考:https://xgboost.readthedocs.io/en/latest/parameter.html
        dtrain : 以DMatrix格式传入的训练数据.
        hdata : 以HDataFrame格式传入的训练数据，内容和dtrain保持一致
        bin_split_points : 各个特征的global等频分箱记过
        num_boost_round: 迭代轮数.
        evals: 在训练过程中进行评估的验证集列表，用来帮助我们在训练过程中监听训练效果.
        obj : 自定义目标函数.
        feval : 自定义eval函数.
        maximize : feval中的函数是否是最大化方向优化.
        early_stopping_rounds: 用于激活early_stop 策略， 验证集的eval metric 需要在每 early stop round
            提升至少一次。对应必须在evals中添加至少一个item，如果在eval metric中有多个item，那么最后一个item
            的指标会用来做early stop策略
        evals_result: evals_result字典，用于存储watch list中所有项目的评价（eval）结果.
        verbose_eval: 如果verbose_eval为True，在验证集上的评估指标会在每一次迭代阶段打印出来，而如果verbose_eval
            是int的话，验证集的评估指标会在每个verbose_eval*迭代次数后打印
        xgb_model : 训练好的xgb模型，可以传输路径或者加载后的模型，用于接力训练，或断点重训
        callbacks : 回调函数列表，将会被apply到训练的每一个迭代中

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
    )
    return bst
