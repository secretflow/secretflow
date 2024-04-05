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
import os.path
import time
from typing import List, Union

import click
import pandas as pd

from benchmark_examples.autoattack.utils.timer import readable_time

try:
    import secretflow as sf

    v = sf.__version__
except ImportError as e:
    logging.error(
        "Cannot find secretflow module, "
        "maybe try use: "
        "export PYTHONPATH='/path/to/secretflow'"
    )
    raise e

import benchmark_examples.autoattack.main as main
import benchmark_examples.autoattack.utils.dispatch as dispatch
from benchmark_examples.autoattack import global_config
from secretflow.utils.errors import NotSupportedError


class Benchmark:
    def __init__(
        self,
        benchmark_mode: str = 'auto',
        dataset: Union[List, str] = 'all',
        model: Union[List, str] = 'all',
        attack: Union[List, str] = 'all',
    ):
        """
        Benchmark
        Args:
            benchmark_mode:
                  auto: default, do all auto attack on all datasets/models/attacks.
                  train: only test train on all datasets/models/attacks.
                  attack: do attack on all datasets/models/attacks.
        """
        assert (
            benchmark_mode in ['auto', 'attack', 'train'] + dispatch.ATTACKS
        ), f"got unexpect benchmark_mode {benchmark_mode}"
        self.benchmark_mode = benchmark_mode
        columns = ['datasets', 'models']
        self.candidates = None
        participate_attacks = dispatch.ATTACKS
        if attack != 'all':
            attack = attack if isinstance(attack, list) else [attack]
            participate_attacks = [c for c in participate_attacks if c in attack]
        if benchmark_mode == 'train':
            self.candidates = ['train']
        elif benchmark_mode == 'attack':
            self.candidates = participate_attacks
        elif benchmark_mode == 'auto':
            self.candidates = [f"auto_{attack}" for attack in participate_attacks]
        else:
            assert (
                benchmark_mode in dispatch.ATTACKS
            ), f"got unexpect attack {benchmark_mode}"
            self.candidates = [f'{benchmark_mode}']

        applications = {}
        if dataset != 'all':
            dataset = dataset if isinstance(dataset, list) else [dataset]
        if model != 'all':
            model = model if isinstance(model, list) else [model]
        is_find = False
        for app, v in dispatch.APPLICATIONS.items():
            if (dataset == 'all' or app[0] in dataset) and (
                model == 'all' or app[1] in model
            ):
                applications[app] = v
                is_find = True
        assert (
            is_find
        ), f"provide app {dataset}:{model} is not in the implemented applications."
        rows = len(applications)
        datum = {col: ['....'] * rows for col in columns + self.candidates}
        apps = list(applications.keys())
        # Sort according to the order of user input dataset and model
        if isinstance(dataset, list):
            apps.sort(
                key=lambda a: dataset.index(a[0]) * 100
                + (model.index(a[1]) if isinstance(model, list) else 0)
            )
        datum['datasets'] = [application[0] for application in apps]
        datum['models'] = [application[1] for application in apps]
        self.experiments = pd.DataFrame(datum)
        self.print_candidates()
        if not os.path.exists(global_config.get_cur_experiment_result_path()):
            os.makedirs(global_config.get_cur_experiment_result_path())
        self.log_file = open(
            global_config.get_cur_experiment_result_path() + "/final_result.md", "a"
        )
        self.log_file_simple = open(
            global_config.get_cur_experiment_result_path() + "/final_result_simple.md",
            "a",
        )

    def __del__(self):
        try:
            self.log_file.close()
            self.log_file_simple.close()
        except Exception:
            pass

    def print_candidates(self):
        logging.warning("All candidate Datasets/Models/Attacks:")
        logging.warning(f"\n{self.experiments.to_markdown()}")

    @staticmethod
    def custom_result(result, metric_names) -> dict:
        """record 1 tune result with additinal origin application train metric (like auc)"""
        r = copy.deepcopy(result.config)
        for mn in metric_names:
            r[mn] = result.metrics.get(mn, "Not Found")
        for k, v in result.metrics.items():
            # if metrics start with 'app_' then it is an origin application train metric like auc.
            if 'app_' in k:
                r[k] = v
        r['error'] = '-' if result.error is None else "Error"
        return r

    def custom_results(self, results, metric_names: list):
        ret = {
            k: ['-'] * len(results)
            for k in self.custom_result(results[0], metric_names)
        }
        for i in range(len(results)):
            custom_result = self.custom_result(results[i], metric_names)
            for k, v in custom_result.items():
                if k not in ret:
                    ret[k] = ['-'] * len(results)
                ret[k][i] = v
        return pd.DataFrame(ret)

    def log_root_full_result(self, ds, md, at, r):
        # record to full result log
        self.log_file.write(f"\n\n\nExperiments {ds}, {md}, {at} autoattack results:\n")
        for br, mn, mnd in zip(r.best_results, r.metric_names, r.metric_modes):
            self.log_file.write(
                f"\nBest results for metric {mn} (mode={r.metric_modes}):\n"
            )
            self.log_file.write(br.metrics_dataframe.to_markdown())
            self.log_file.write("\n")
        self.log_file.write(f"\n\nResult Grid:\n")
        self.log_file.write(r.results.get_dataframe().to_markdown())
        if r.results.num_errors > 0:
            self.log_file.write(f"\nThere are {r.results.num_errors} Errors:\n")
            for error in r.results.errors:
                self.log_file.write(f"{error}\n")

    def log_root_simple_result(self, ds, md, at, r):
        # record to simple result log.
        self.log_file_simple.write(
            f"\n\n\nExperiments {ds}, {md}, {at} autoattack results:\n"
        )
        for br, mn, mnd in zip(r.best_results, r.metric_names, r.metric_modes):
            self.log_file_simple.write(
                f"\nBest results for metric {mn} (mode={r.metric_modes}):\n\n"
            )
            self.log_file_simple.write(
                f"{self.custom_results([br], [mn]).to_markdown()}"
            )
            self.log_file_simple.write("\n")
        self.log_file_simple.write(f"\n\nResult Grid:\n\n")
        self.log_file_simple.write(
            f"{self.custom_results(r.results, r.metric_names).to_markdown()}"
        )

    @staticmethod
    def log_case_result_csv(ds, md, at, r):
        log_file_path = (
            global_config.get_cur_experiment_result_path()
            + f"/{ds}_{md}_{at.lstrip('auto_')}"
        )
        # log all best result csv to .csv
        for br, mn, mnd in zip(r.best_results, r.metric_names, r.metric_modes):
            best_csv: pd.DataFrame = br.metrics_dataframe
            best_csv.to_csv(log_file_path + f"/best_result_{mn}_{mnd}.csv", index=False)
        # log the full result grid to .csv
        result_grid_df = r.results.get_dataframe()
        result_grid_df.to_csv(log_file_path + f"/full_result.csv", index=False)

    def log_results(self, ds, md, at, r: main.AutoAttackResult):
        if r is None or not isinstance(r, main.AutoAttackResult):
            return
        try:
            self.log_root_full_result(ds, md, at, r)
            self.log_root_simple_result(ds, md, at, r)
            self.log_case_result_csv(ds, md, at, r)
        except Exception as e:
            logging.error("Results are not been logged correctly, please check", e)

    def log_final_result(self):
        for log_file in [self.log_file_simple, self.log_file]:
            log_file.write(f"\n\n\nAll Experiments records:\n\n")
            log_file.write(f"{self.experiments.to_markdown()}")

    def run_case(self, dataset, model, attack: str):
        start = time.time()
        try:
            results = main.run_case(dataset, model, attack)
            ret = readable_time(time.time() - start)
            self.log_results(dataset, model, attack, results)
            if 'auto' in attack:
                if results.results.num_errors > 0:
                    ret = (
                        ret
                        + f"({len(results.results)}trails/{results.results.num_errors}failed)"
                    )
                else:
                    ret = ret + f"({len(results.results)}trails)"
            return ret
        except NotSupportedError:
            logging.warning(f"attack not support.")
            return '-'
        except ModuleNotFoundError:
            logging.warning(f"module not found:")
            return 'Alg-Unsupported'
        except ImportError as e:
            logging.warning(
                f"failed to import {dataset}/{model}/{attack} cases, maybe not implement, please manually check.",
                e,
            )
            return 'None'
        except Exception as e:
            logging.error(
                f"failed to run experiments on {dataset}/{model}/{attack} ...", e
            )
            return f'Error({readable_time(time.time() - start)})'

    def run(
        self,
    ):
        start_time = time.time()
        for ds_md_i in range(self.experiments.shape[0]):
            ds = self.experiments.loc[ds_md_i, 'datasets']
            md = self.experiments.loc[ds_md_i, 'models']
            for at in self.candidates:
                logging.info(f"Starting experiment on {ds}/{md}/{at} ...")
                self.experiments.at[ds_md_i, at] = 'Running'
                self.print_candidates()
                ret = self.run_case(ds, md, at)
                self.experiments.at[ds_md_i, at] = ret
                logging.info(f"Finish experiment on {ds}/{md}/{at} ...")
        logging.info(
            f"All experiments is finish, total time cost: {readable_time(time.time() - start_time)}s"
        )
        self.print_candidates()
        self.log_final_result()


@click.command()
@click.option(
    '--mode',
    type=click.STRING,
    required=False,
    default=None,
    help='Benchmark mode like "train/attack/auto", default to "train".',
)
@click.option(
    '--dataset',
    type=click.STRING,
    required=False,
    default=None,
    help='Dataset target like "all/bank/criteo",.etc, default to all.',
)
@click.option(
    '--model',
    type=click.STRING,
    required=False,
    default=None,
    help='Model target like "all/dnn/deepfm",.etc, default to "all".',
)
@click.option(
    '--attack',
    type=click.STRING,
    required=False,
    default=None,
    help='Attack target like "all/fia/lia",.etc, default to "all".',
)
@click.option(
    "--simple",
    is_flag=True,
    default=None,
    help='Whether to use simple testing for easy debugging.',
)
@click.option(
    "--datasets_path",
    type=click.STRING,
    required=False,
    default=None,
    help='Datasets load path, default to "~/.secretflow/datasets"',
)
@click.option(
    "--autoattack_storage_path",
    type=click.STRING,
    required=False,
    default=None,
    help='Autoattack results storage path, default to "~/.secretflow/datasets"',
)
@click.option(
    "--use_gpu",
    is_flag=True,
    required=False,
    default=None,
    help="Whether to use GPU, default to False",
)
@click.option(
    "--config",
    type=click.STRING,
    required=False,
    default=None,
    help="The config yaml files, you can put all configs here.",
)
@click.option(
    "--debug_mode",
    type=click.BOOL,
    required=False,
    help='Wheter to run secretflow on the debug mode.',
)
@click.option(
    "--ray_cluster_address",
    type=click.STRING,
    required=False,
    default=None,
    help="The existing ray cluster address to connect.",
)
@click.option(
    "--random_seed",
    type=click.STRING,
    required=False,
    default=None,
    help="To achieve reproducible.",
)
def run(
    mode,
    dataset,
    model,
    attack,
    simple,
    datasets_path,
    autoattack_storage_path,
    use_gpu,
    config,
    debug_mode,
    ray_cluster_address,
    random_seed,
):
    """Run your autoattack benchmark."""
    global_config.init_globalconfig(
        mode=mode,
        dataset=dataset,
        model=model,
        attack=attack,
        simple=simple,
        datasets_path=datasets_path,
        autoattack_storage_path=autoattack_storage_path,
        use_gpu=use_gpu,
        config=config,
        debug_mode=debug_mode,
        ray_cluster_address=ray_cluster_address,
        random_seed=random_seed,
    )
    benchmark = Benchmark(
        benchmark_mode=global_config.get_benchmrak_mode(),
        dataset=global_config.get_benchmark_dataset(),
        model=global_config.get_benchmark_model(),
        attack=global_config.get_benchmark_attack(),
    )
    benchmark.run()


if __name__ == '__main__':
    run()
