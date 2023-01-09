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
from typing import Callable, List

from xgboost.callback import CallbackContainer, TrainingCallback, _aggcv

import secretflow.device.link as link


class FedCallbackContainer(CallbackContainer):
    '''Federate version, encapsulates xgboostCallbackContainer.
    Attributes:
        callbacks: list of training callback functions
        metric: eval function callable
        is_cv: whether to do cross validation
    '''

    EvalsLog = TrainingCallback.EvalsLog

    def __init__(
        self,
        callbacks: List[TrainingCallback],
        metric: Callable = None,
        is_cv: bool = False,
    ):
        super(FedCallbackContainer, self).__init__(
            callbacks=callbacks, metric=metric, is_cv=is_cv
        )
        self._exist_key = {}
        self._sync_version = 0
        self.role = link.get_role()

    def key(self, name: str) -> str:
        if name in self._exist_key:
            key = self._exist_key[name]
        else:
            key = f"XGB_callback/{name}"
            self._exist_key[name] = key
        return key

    def after_iteration(self, model, epoch, dtrain, evals) -> bool:
        """A function to call after training iterations.
        Args:
            model: xgboost booster object, which stores training parameters and states
            epoch: number of iteration rounds
            dtrain: DMatrix xgboost format training data
            evals: List[(DMatrix, string)] List of data to evaluate
        Returns:
            ret: Whether the training should be terminated, if the callbacks are successfully executed, return true (eg: EarlyStop returns True, the training is terminated early)
        """
        if self.is_cv:
            scores = model.eval(epoch, self.metric)
            scores = _aggcv(scores)
            self.aggregated_cv = scores
            self._update_history(scores, epoch)
        else:
            if dtrain is not None:  # required and cannot delete
                pass
            evals = [] if evals is None else evals
            for _, name in evals:
                assert name.find('-') == -1, 'Dataset name should not contain `-`'
            if self.role == link.CLIENT:
                score = model.eval_set(evals, epoch, self.metric)
                score = score.split()[1:]  # into datasets
                score = [tuple(s.split(':')) for s in score]
                link.send_to_server(
                    name=self.key("score"), value=score, version=self._sync_version
                )
            if self.role == link.SERVER:
                all_score = link.recv_from_clients(
                    name=self.key("score"),
                    version=self._sync_version,
                )
                num_party = len(all_score)
                all_score_dict = [dict(score) for score in all_score]
                sum_score = {
                    k: sum(float(d[k]) for d in all_score_dict) / num_party
                    for k in all_score_dict[0]
                }
                agg_score = [(k, v) for k, v in sum_score.items()]
                link.send_to_clients(
                    name=self.key("agg_score"),
                    value=agg_score,
                    version=self._sync_version,
                )
                self._update_history(agg_score, epoch)
            if self.role == link.CLIENT:
                agg_score = link.recv_from_server(
                    name=self.key("agg_score"),
                    version=self._sync_version,
                )

                self._update_history(score, epoch)
        self._sync_version += 1
        # TODO: 聚合需要根据不同的算子来做不同的策略
        ret = any(c.after_iteration(model, epoch, self.history) for c in self.callbacks)
        return ret
