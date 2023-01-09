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


from dataclasses import dataclass, field
from typing import Dict, List

from secretflow.ml.nn.metrics import Metric


@dataclass
class History:
    local_history: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    """
    Examples:
        >>> {
                'alice': {'loss': [0.46011224], 'accuracy': [0.8639647]},
                'bob': {'loss': [0.46011224], 'accuracy': [0.8639647]},
            }
    """

    local_detailed_history: Dict[str, Dict[str, List[Metric]]] = field(
        default_factory=dict
    )
    """
    Examples:
        >>> {
                'alice': {
                    'mean': [Mean()]
                },
                'bob': {
                    'mean': [Mean()]
                },
            }
    """

    global_history: Dict[str, List[float]] = field(default_factory=dict)
    """
    Examples:
        >>> {
                'loss': [0.46011224],
                'accuracy': [0.8639647]
            }
    """

    global_detailed_history: Dict[str, List[Metric]] = field(default_factory=dict)
    """
    Examples:
        >>> {
                'loss': [Loss(name='loss')],
                'precision': [Precision(name='precision')],
            }
    """

    def record_local_history(self, party, metrics: List[Metric], stage='train'):
        if party not in self.local_history:
            self.local_history[party] = {}
            self.local_detailed_history[party] = {}
        for m in metrics:
            if stage == "train":
                t_key = m.name
            else:
                t_key = "_".join([stage, m.name])
            if t_key not in self.local_history[party]:
                self.local_history[party][t_key] = []
                self.local_detailed_history[party][t_key] = []
            self.local_history[party][t_key].append(m.result().numpy())
            self.local_detailed_history[party][t_key].append(m)

    def record_global_history(self, metrics: List[Metric], stage='train'):
        for m in metrics:
            if stage == "train":
                t_key = m.name
            else:
                t_key = "_".join([stage, m.name])
            if t_key not in self.global_history:
                self.global_history[t_key] = []
                self.global_detailed_history[t_key] = []
            self.global_history[t_key].append(m.result().numpy())
            self.global_detailed_history[t_key].append(m)


def metric_wrapper(func, *args, **kwargs):
    def wrapped_func():
        return func(*args, **kwargs)

    return wrapped_func


def optim_wrapper(func, *args, **kwargs):
    def wrapped_func(params):
        return func(params, *args, **kwargs)

    return wrapped_func
