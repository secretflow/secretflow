# Copyright 2024 Ant Group Co., Ltd.
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

# Most of the code below is borrowed from XGBoost CallBack, with some adaptions.
# XGBoost also has Apache-2.0 licensed code.
# See: xgboost/python-package/xgboost/callback.py

import collections
import logging
from abc import ABC
from typing import Callable, cast, Dict, List, Optional, Sequence, Tuple, Union

import numpy

from secretflow.data import FedNdarray
from secretflow.data.vertical import VDataFrame

__all__ = [
    "TrainingCallback",
    "EarlyStopping",
    "EvaluationMonitor",
    "CallbackContainer",
    "CallBackCompatibleModel",
]

_Score = Union[float, Tuple[float, float]]
_ScoreList = Union[List[float], List[Tuple[float, float]]]


VData = Union[FedNdarray, VDataFrame]


class CallBackCompatibleModel(ABC):
    """A template class to define a compatible model with Callback"""

    def __init__(self, *args, **kwargs):
        pass

    def eval_set(self, evals, metric) -> List:
        pass

    def set_save_best(self):
        pass

    def set_best_iteration_score(self, iteration, score):
        pass


# pylint: disable=unused-argument
class TrainingCallback(ABC):
    """Interface for training callback."""

    EvalsLog = Dict[str, Dict[str, _ScoreList]]  # pylint: disable=invalid-name

    def __init__(self) -> None:
        pass

    def before_training(
        self, model: CallBackCompatibleModel
    ) -> CallBackCompatibleModel:
        """Run before training starts."""
        return model

    def after_training(self, model: CallBackCompatibleModel) -> CallBackCompatibleModel:
        """Run after training is finished."""
        return model

    def before_iteration(
        self, model: CallBackCompatibleModel, epoch: int, evals_log: EvalsLog
    ) -> bool:
        """Run before each iteration.  Returns True when training should stop. See
        :py:meth:`after_iteration` for details.

        """
        return False

    def after_iteration(
        self, model: CallBackCompatibleModel, epoch: int, evals_log: EvalsLog
    ) -> bool:
        """Run after each iteration.  Returns `True` when training should stop.

        Parameters
        ----------

        model : SS XGB Model or Sgb Model
        epoch :
            The current training iteration.
        evals_log :
            A dictionary containing the evaluation history:

            .. code-block:: python

                {"data_name": {"metric_name": [0.5, ...]}}

        """
        return False


class CallbackContainer:
    """A special internal callback for invoking a list of other callbacks."""

    def __init__(
        self,
        callbacks: Sequence[TrainingCallback],
        metric: Optional[Callable] = None,
        history: TrainingCallback.EvalsLog = None,
    ) -> None:
        self.callbacks = set(callbacks)
        for cb in callbacks:
            if not isinstance(cb, TrainingCallback):
                raise TypeError("callback must be an instance of `TrainingCallback`.")
        msg = (
            "metric must be callable object for monitoring.  For builtin metrics"
            ", passing them in training parameter invokes monitor automatically."
        )
        if metric is not None and not callable(metric):
            raise TypeError(msg)

        self.metric = metric
        try:
            self.history: TrainingCallback.EvalsLog = (
                collections.OrderedDict()
                if history is None
                else collections.OrderedDict(history)
            )
        except:
            assert False, f"{history} type {type(history)} is not supported"

    def before_training(
        self, model: CallBackCompatibleModel
    ) -> CallBackCompatibleModel:
        """Function called before training."""
        for c in self.callbacks:
            model = c.before_training(model=model)
            msg = "before_training should return the model"
            assert isinstance(model, CallBackCompatibleModel), msg
        return model

    def after_training(self, model: CallBackCompatibleModel) -> CallBackCompatibleModel:
        """Function called after training."""
        for c in self.callbacks:
            model = c.after_training(model=model)
            msg = "after_training should return the model"
            assert isinstance(model, CallBackCompatibleModel), msg

        return model

    def before_iteration(
        self,
        model: CallBackCompatibleModel,
        epoch: int,
    ) -> bool:
        """Function called before training iteration."""
        return any(
            c.before_iteration(model, epoch, self.history) for c in self.callbacks
        )

    def _update_history(
        self,
        score: List[Tuple[str, float]],
    ) -> None:
        for d in score:
            name: str = d[0]
            s: float = d[1]
            x = s
            splited_names = name.split("-")
            data_name = splited_names[0]
            metric_name = "-".join(splited_names[1:])
            if data_name not in self.history:
                self.history[data_name] = collections.OrderedDict()
            data_history = self.history[data_name]
            if metric_name not in data_history:
                data_history[metric_name] = cast(_ScoreList, [])
            metric_history = data_history[metric_name]
            cast(List[float], metric_history).append(cast(float, x))

    def after_iteration(
        self,
        model: CallBackCompatibleModel,
        epoch: int,
        evals: List[Tuple[VData, VData, str]],
    ) -> bool:
        """Function called after training iteration."""

        evals = [] if evals is None else evals
        for _, _, name in evals:
            assert name.find("-") == -1, "Dataset name should not contain `-`"
        metric_score = model.eval_set(evals, self.metric)
        self._update_history(metric_score)
        ret = any(c.after_iteration(model, epoch, self.history) for c in self.callbacks)
        return ret


# TODO(zoupeicheng.zpc): support learning rate scheduler


# pylint: disable=too-many-instance-attributes
class EarlyStopping(TrainingCallback):
    """Callback function for early stopping

    Parameters
    ----------
    rounds :
        Early stopping rounds.
    metric_name :
        Name of metric that is used for early stopping.
    data_name :
        Name of dataset that is used for early stopping.
    maximize :
        Whether to maximize evaluation metric.  None means auto (discouraged).
    save_best :
        Whether training should return the best model or the last model.
    min_delta :
        Minimum absolute change in score to be qualified as an improvement.

    Examples
    --------

    .. code-block:: python

        es = xgboost.callback.EarlyStopping(
            rounds=2,
            min_delta=1e-3,
            save_best=True,
            maximize=False,
            data_name="validation_0",
            metric_name="mlogloss",
        )
        clf = xgboost.XGBClassifier(tree_method="hist", device="cuda", callbacks=[es])

        X, y = load_digits(return_X_y=True)
        clf.fit(X, y, eval_set=[(X, y)])
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        rounds: int,
        metric_name: Optional[str] = None,
        data_name: Optional[str] = None,
        maximize: Optional[bool] = None,
        save_best: Optional[bool] = False,
        min_delta: float = 0.0,
    ) -> None:
        self.data = data_name
        self.metric_name = metric_name
        self.rounds = rounds
        self.save_best = save_best
        self.maximize = maximize
        self.stopping_history: TrainingCallback.EvalsLog = {}
        self._min_delta = min_delta
        if self._min_delta < 0:
            raise ValueError("min_delta must be greater or equal to 0.")

        self.current_rounds: int = 0
        self.best_scores: dict = {}
        self.starting_round: int = 0
        super().__init__()

    def before_training(
        self, model: CallBackCompatibleModel
    ) -> CallBackCompatibleModel:
        return model

    def _update_rounds(
        self,
        score: _Score,
        name: str,
        metric: str,
        model: CallBackCompatibleModel,
        epoch: int,
    ) -> bool:
        def get_s(value: _Score) -> float:
            """get score if it's cross validation history."""
            return value[0] if isinstance(value, tuple) else value

        def maximize(new: _Score, best: _Score) -> bool:
            """New score should be greater than the old one."""
            return numpy.greater(get_s(new) - self._min_delta, get_s(best))

        def minimize(new: _Score, best: _Score) -> bool:
            """New score should be lesser than the old one."""
            return numpy.greater(get_s(best) - self._min_delta, get_s(new))

        if self.maximize is None:
            maximize_metrics = ("roc_auc",)
            if any(metric.startswith(x) for x in maximize_metrics):
                self.maximize = True
            else:
                self.maximize = False

        if self.maximize:
            improve_op = maximize
        else:
            improve_op = minimize

        if not self.stopping_history:  # First round
            self.current_rounds = 0
            self.stopping_history[name] = {}
            self.stopping_history[name][metric] = cast(_ScoreList, [score])
            self.best_scores[name] = {}
            self.best_scores[name][metric] = [score]
            model.set_best_iteration_score(
                iteration=epoch,
                score=score,
            )
        elif not improve_op(score, self.best_scores[name][metric][-1]):
            # Not improved
            self.stopping_history[name][metric].append(score)  # type: ignore
            self.current_rounds += 1
        else:  # Improved
            self.stopping_history[name][metric].append(score)  # type: ignore
            self.best_scores[name][metric].append(score)
            record = self.stopping_history[name][metric][-1]
            model.set_best_iteration_score(
                iteration=epoch,
                score=record,
            )
            self.current_rounds = 0  # reset

        if self.current_rounds >= self.rounds:
            # Should stop
            return True
        return False

    def after_iteration(
        self,
        model: CallBackCompatibleModel,
        epoch: int,
        evals_log: TrainingCallback.EvalsLog,
    ) -> bool:
        epoch += self.starting_round  # training continuation
        msg = "Must have at least 1 validation dataset for early stopping."
        if len(evals_log.keys()) < 1:
            raise ValueError(msg)

        # Get data name
        if self.data:
            data_name = self.data
        else:
            # Use the last one as default.
            data_name = list(evals_log.keys())[-1]
        if data_name not in evals_log:
            raise ValueError(f"No dataset named: {data_name}")

        if not isinstance(data_name, str):
            raise TypeError(
                f"The name of the dataset should be a string. Got: {type(data_name)}"
            )
        data_log = evals_log[data_name]

        # Get metric name
        if self.metric_name:
            metric_name = self.metric_name
        else:
            # Use last metric by default.
            metric_name = list(data_log.keys())[-1]
        if metric_name not in data_log:
            raise ValueError(f"No metric named: {metric_name}")

        # The latest score
        score = data_log[metric_name][-1]
        return self._update_rounds(score, data_name, metric_name, model, epoch)

    def after_training(self, model: CallBackCompatibleModel) -> CallBackCompatibleModel:
        if not self.save_best:
            model.set_save_best(False)
            return model
        model.set_save_best(True)
        return model


class EvaluationMonitor(TrainingCallback):
    """Print the evaluation result at each iteration.

    Parameters
    ----------

    period :
        How many epoches between printing.
    """

    def __init__(self, period: int = 1) -> None:
        self.period = period
        assert period > 0
        # last error message, useful when early stopping and period are used together.
        self._latest: Optional[str] = None
        super().__init__()

    def _fmt_metric(self, data: str, metric: str, score: float) -> str:
        msg = f"\t{data + '-' + metric}:{score:.5f}"
        return msg

    def after_iteration(
        self,
        _: CallBackCompatibleModel,
        epoch: int,
        evals_log: TrainingCallback.EvalsLog,
    ) -> bool:
        if not evals_log:
            return False

        msg: str = f"[{epoch}]"

        for data, metric in evals_log.items():
            for metric_name, log in metric.items():
                if isinstance(log[-1], tuple):
                    score = log[-1][0]
                else:
                    score = log[-1]
                msg += self._fmt_metric(data, metric_name, score)
        msg += "\n"

        if (epoch % self.period) == 0 or self.period == 1:
            logging.info(msg)
            self._latest = None
        else:
            # There is skipped message
            self._latest = msg
        return False

    def after_training(self, model: CallBackCompatibleModel) -> CallBackCompatibleModel:
        if self._latest is not None:
            logging.info(self._latest)
        return model


class Checkpointing(TrainingCallback):
    """Save the model at each iteration, support continuing training."""

    def __init__(self, dump_function: Callable = None):
        super().__init__()
        self.dump_function = dump_function

    def after_iteration(
        self,
        model: CallBackCompatibleModel,
        epoch: int,
        evals_log: TrainingCallback.EvalsLog,
    ) -> bool:
        if self.dump_function is not None and callable(self.dump_function):
            self.dump_function(model, epoch, evals_log)
        else:
            logging.warning("no effective dump_function provided.")
        return False
