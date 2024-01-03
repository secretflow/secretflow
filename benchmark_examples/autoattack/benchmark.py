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

import logging
import os

import click
import pandas as pd

try:
    import secretflow as sf

    print(sf.__version__)
except ImportError as e:
    print(
        "Cannot find secretflow module, "
        "maybe try use: "
        "export PYTHONPATH='/path/to/secretflow'"
    )
    raise e

import benchmark_examples.autoattack.utils.dispatch as dispatch
from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.main import run_case
from secretflow.utils.errors import NotSupportedError


class Benchmark:
    def __init__(self, benchmark_mode: str = 'auto'):
        """
        Benchmark
        Args:
            benchmark_mode:
                  auto: default, do all auto attack on all datasets/models/attacks.
                  train: only test train on all datasets/models/attacks.
                  attack: do attack on all datasets/models/attacks.
        """
        assert (
            benchmark_mode == 'auto'
            or benchmark_mode == 'train'
            or benchmark_mode == 'attack'
        ), f"got unexpect benchmark_mode {benchmark_mode}"
        self.benchmark_mode = benchmark_mode
        columns = ['datasets', 'models']
        self.candidates = None
        if benchmark_mode == 'train':
            self.candidates = ['train']
        elif benchmark_mode == 'attack':
            self.candidates = dispatch.ATTACKS
        else:
            self.candidates = [f"auto_{attack}" for attack in dispatch.ATTACKS]
        rows = len(dispatch.APPLICATIONS)
        dataum = {col: ['....'] * rows for col in columns + self.candidates}
        dataum['datasets'] = [
            application[0] for application in dispatch.APPLICATIONS.keys()
        ]
        dataum['models'] = [
            application[1] for application in dispatch.APPLICATIONS.keys()
        ]
        self.experiments = pd.DataFrame(dataum)
        self.print_candidates()

    def print_candidates(self):
        print("All Candidate Dataset/Model/Attask:")
        print(self.experiments.to_markdown())

    def run_case(self, ds, md, at):
        try:
            return run_case(ds, md, at, False)
        except NotSupportedError:
            logging.warning(f"attack not support.")
            return 'Attack-Unsupported'
        except ModuleNotFoundError:
            logging.warning(f"module not found:")
            return 'Alg-Unsupported'
        except ImportError:
            logging.warning(
                f"failed to import {ds}/{md}/{at} cases, maybe not implement, please manually check."
            )
            return 'None'
        except Exception as e:
            logging.error(f"failed to run experiments on {ds}/{md}/{at} ...")
            logging.error(e)
            return 'Error'

    def run(
        self,
    ):
        for ds_md_i in range(self.experiments.shape[0]):
            ds = self.experiments.loc[ds_md_i, 'datasets']
            md = self.experiments.loc[ds_md_i, 'models']
            for at in self.candidates:
                logging.info(f"Starting experiment on {ds}/{md}/{at} ...")
                ret = self.run_case(ds, md, at)
                self.experiments.at[ds_md_i, at] = ret
                logging.info(f"Finish experiment on {ds}/{md}/{at} ...")
                if ret != 'None':
                    self.print_candidates()


@click.command()
@click.argument('benchmark_mode', type=click.STRING, required=False, default='train')
@click.option(
    "--datasets_path",
    type=click.STRING,
    required=False,
    default=os.path.join(os.path.expanduser('~'), '.secretflow/datasets'),
)
@click.option(
    "--autoattack_storage_path",
    type=click.STRING,
    required=False,
    default=os.path.join(os.path.expanduser('~'), '.secretflow/workspace'),
)
def run(benchmark_mode, datasets_path, autoattack_storage_path):
    global_config.set_dataset_path(datasets_path)
    global_config.set_autoattack_path(autoattack_storage_path)
    benchmark = Benchmark(benchmark_mode=benchmark_mode)
    benchmark.run()


if __name__ == '__main__':
    run()
