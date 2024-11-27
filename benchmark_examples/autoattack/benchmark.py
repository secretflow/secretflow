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
from typing import Callable, Dict, List

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
        enable_tune: bool = True,
        dataset: List | str = 'all',
        model: List | str = 'all',
        attack: List | str | None = 'all',
        defense: List | str | None = 'all',
        enable_log: bool = True,
        objective: Callable = main.objective_trainning,
    ):
        """
        Benchmark
        Args:
            benchmark_mode:
                  auto: default, do all auto attack on all datasets/models/attacks.
                  train: only test train on all datasets/models/attacks.
                  defense: test train with defense.
                  attack: do attack on all datasets/models/attacks.
        """
        self.enable_tune = enable_tune
        self.enable_log = enable_log
        self.objective = objective
        columns = ['datasets', 'models', 'defenses']
        self.candidates = None
        participate_attacks = dispatch.ATTACKS
        participate_attacks.insert(0, 'no_attack')
        if attack is None:
            self.candidates = ['train']
        if attack != 'all':
            attack = attack if isinstance(attack, list) else [attack]
            participate_attacks = [c for c in participate_attacks if c in attack]
            participate_attacks.sort(key=lambda c: attack.index(c))
        self.candidates = participate_attacks
        participate_defenses = dispatch.DEFENSES
        # no defense is also a scene when defense == all.
        participate_defenses.insert(0, "no_defense")
        if defense is None:
            participate_defenses = ['no_defense']
        if defense != 'all':
            defense = defense if isinstance(defense, list) else [defense]
            participate_defenses = [c for c in participate_defenses if c in defense]
            participate_defenses.sort(key=lambda c: defense.index(c))
        if dataset != 'all':
            dataset = dataset if isinstance(dataset, list) else [dataset]
        if model != 'all':
            model = model if isinstance(model, list) else [model]
        is_find = False
        apps = []
        for app in dispatch.APPLICATIONS.keys():
            if (dataset == 'all' or app[0] in dataset) and (
                model == 'all' or app[1] in model
            ):
                apps.append(app)
                is_find = True
        assert (
            is_find
        ), f"provide app {dataset}:{model} is not in the implemented applications."
        # app * defenses as the row, and attack as the col
        rows = len(apps) * len(participate_defenses)
        datum = {col: ['....'] * rows for col in columns + self.candidates}
        # Sort according to the order of user input dataset and model
        if isinstance(dataset, list):
            apps.sort(
                key=lambda a: dataset.index(a[0]) * 100
                + (model.index(a[1]) if isinstance(model, list) else 0)
            )
        extend_apps = [v for v in apps for _ in range(len(participate_defenses))]
        datum['datasets'] = [application[0] for application in extend_apps]
        datum['models'] = [application[1] for application in extend_apps]
        datum['defenses'] = participate_defenses * len(apps)
        self.experiments = pd.DataFrame(datum)
        self.print_candidates()
        if not os.path.exists(global_config.get_cur_experiment_result_path()):
            os.makedirs(global_config.get_cur_experiment_result_path())
        if self.enable_log:
            self.log_file = open(
                global_config.get_cur_experiment_result_path() + "/result.md", "a"
            )
            self.log_file_simple = open(
                global_config.get_cur_experiment_result_path() + "/result_simple.md",
                "a",
            )

    def __del__(self):
        try:
            if self.enable_log:
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

    def log_root_full_result(self, ds, md, at, df, r: main.AutoAttackResult):
        # record to full result log
        self.log_file.write(
            f"\n\n\nExperiments {ds}, {md}, {at}, {df} autoattack results:\n"
        )
        for br, mn, mnd in zip(
            r.best_results, r.metrics.keys(), r.metrics.values(), strict=True
        ):
            self.log_file.write(f"\nBest results for metric {mn} (mode={mnd}):\n")
            self.log_file.write(br.metrics_dataframe.to_markdown())
            self.log_file.write("\n")
        self.log_file.write(f"\n\nResult Grid:\n")
        self.log_file.write(r.results.get_dataframe().to_markdown())
        if r.results.num_errors > 0:
            self.log_file.write(f"\nThere are {r.results.num_errors} Errors:\n")
            for error in r.results.errors:
                self.log_file.write(f"{error}\n")

    def log_root_simple_result(self, ds, md, at, df, r: main.AutoAttackResult | Dict):
        # record to simple result log.
        self.log_file_simple.write(
            f"\n\n\nExperiment {ds}, {md}, {at}, {df} autoattack results:\n\n"
        )
        if isinstance(r, main.AutoAttackResult):
            for br, mn, mnd in zip(
                r.best_results, r.metrics.keys(), r.metrics.values(), strict=True
            ):
                self.log_file_simple.write(
                    f"\nBest results for metric {mn} (mode={mnd}):\n\n"
                )
                self.log_file_simple.write(
                    f"{self.custom_results([br], [mn]).to_markdown()}"
                )
                self.log_file_simple.write("\n")
            self.log_file_simple.write(f"\n\nResult Grid:\n\n")
            self.log_file_simple.write(
                f"{self.custom_results(r.results, list(r.metrics.keys())).to_markdown()}"
            )
        else:
            if global_config.need_monitor():
                resource_usage = r.pop('resource_usage')
                self.log_file_simple.write(f"resource usage: {resource_usage}\n\n")
            r = {k: str(v) for k, v in r.items()}
            self.log_file_simple.write(f"{pd.DataFrame(r,index=[0]).to_markdown()}")

    @staticmethod
    def log_case_result_csv(ds, md, at, df, r):
        log_file_path = (
            global_config.get_cur_experiment_result_path() + f"/{ds}_{md}_{at}_{df}"
        )
        # log all best result csv to .csv
        for br, mn, mnd in zip(
            r.best_results, r.metrics.keys(), r.metrics.values(), strict=True
        ):
            best_csv: pd.DataFrame = br.metrics_dataframe
            best_csv.to_csv(log_file_path + f"/best_result_{mn}_{mnd}.csv", index=False)
        # log the full result grid to .csv
        result_grid_df = r.results.get_dataframe()
        result_grid_df.to_csv(log_file_path + f"/full_result.csv", index=False)

    def log_results(self, ds, md, at, df, r: main.AutoAttackResult | Dict):
        try:
            if self.enable_log:
                if isinstance(r, main.AutoAttackResult):
                    # results is an AutoAttackResult object
                    self.log_root_full_result(ds, md, at, df, r)
                    self.log_root_simple_result(ds, md, at, df, r)
                    self.log_case_result_csv(ds, md, at, df, r)
                elif isinstance(r, Dict):
                    # results is a sigle experiments results.
                    self.log_root_simple_result(ds, md, at, df, r)
                else:
                    return
        except Exception as e:
            logging.error("Results are not been logged correctly, please check", e)

    def log_final_result(self):
        if self.enable_log:
            for log_file in [self.log_file_simple, self.log_file]:
                log_file.write(f"\n\n\nAll Experiments records:\n\n")
                log_file.write(f"{self.experiments.to_markdown()}")

    def run_case(self, dataset: str, model: str, attack: str, defense: str):
        start = time.time()
        try:
            results = main.run_case(
                dataset, model, attack, defense, self.enable_tune, self.objective
            )
            ret = readable_time(time.time() - start)
            self.log_results(dataset, model, attack, defense, results)
            if self.enable_tune:
                if results.results.num_errors > 0:
                    ret = (
                        ret
                        + f"({len(results.results)}trails/{results.results.num_errors}failed)"
                    )
                else:
                    ret = ret + f"({len(results.results)}trails)"
            return ret
        except NotSupportedError:
            logging.warning(f"Case {dataset}/{model}/{attack}/{defense} not supported.")
            return '-'
        except ModuleNotFoundError as e:
            logging.warning(f"module not found:", e)
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

    def case_valid_check(self, ds, md, at, df):
        try:
            main.case_valid_check(ds, md, at, df)
        except NotSupportedError:
            return False
        except Exception as ex:
            raise ex
        return True

    def run(
        self,
    ):
        start_time = time.time()
        for at in self.candidates:
            for ds_md_i in range(self.experiments.shape[0]):
                ds = self.experiments.loc[ds_md_i, 'datasets']
                md = self.experiments.loc[ds_md_i, 'models']
                df = self.experiments.loc[ds_md_i, 'defenses']
                logging.info(f"Starting experiment on {ds}/{md}/{at} ...")
                if self.case_valid_check(ds, md, at, df):
                    self.experiments.at[ds_md_i, at] = 'Running'
                    self.print_candidates()
                    ret = self.run_case(ds, md, at, df)
                    self.experiments.at[ds_md_i, at] = ret
                else:
                    logging.warning(f"Case {ds}/{md}/{at}/{df} not supported.")
                    self.experiments.at[ds_md_i, at] = '-'
                logging.info(f"Finish experiment on {ds}/{md}/{at} ...")
        logging.info(
            f"All experiments is finish, total time cost: {readable_time(time.time() - start_time)}s"
        )
        self.print_candidates()
        self.log_final_result()


@click.command(no_args_is_help=True)
@click.option(
    '--enable_tune',
    type=click.BOOL,
    required=False,
    is_flag=True,
    default=None,
    help='Benchmark mode like "train/attack/defense/auto", default to "train".',
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
    '--defense',
    type=click.STRING,
    required=False,
    default=None,
    help='Attack target like "grad_avg",.etc, default to "all".',
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
    "--enable_monitor",
    is_flag=True,
    required=False,
    default=None,
    help="Whether to enable resource monitor, default to False",
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
    enable_tune,
    dataset,
    model,
    attack,
    defense,
    simple,
    datasets_path,
    autoattack_storage_path,
    use_gpu,
    config,
    debug_mode,
    enable_monitor,
    ray_cluster_address,
    random_seed,
):
    """Run your autoattack benchmark."""
    global_config.init_globalconfig(
        enable_tune=enable_tune,
        dataset=dataset,
        model=model,
        attack=attack,
        defense=defense,
        simple=simple,
        datasets_path=datasets_path,
        autoattack_storage_path=autoattack_storage_path,
        use_gpu=use_gpu,
        config=config,
        debug_mode=debug_mode,
        enable_monitor=enable_monitor,
        ray_cluster_address=ray_cluster_address,
        random_seed=random_seed,
    )
    benchmark = Benchmark(
        enable_tune=global_config.is_enable_tune(),
        dataset=global_config.get_benchmark_dataset(),
        model=global_config.get_benchmark_model(),
        attack=global_config.get_benchmark_attack(),
        defense=global_config.get_benchmark_defense(),
    )
    benchmark.run()


if __name__ == '__main__':
    run()
