# Copyright 2023 Ant Group Co., Ltd.
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

import argparse
import json
import logging
import os
import time

from config_reader import load_config_dict, recursive_fill, test_config_generation
from pipelines import pipeline_map

from secretflow.utils.logging import LOG_FORMAT


def run_one_test(test_name, config_dict_one_test, args):
    test = test_config_generation(
        config_dict=config_dict_one_test, test_run=args.test_run
    )
    pipeline_configs = load_config_dict(config_dict_one_test["pipelines"])
    if args.single is None:
        for pipeline_name, pipeline_builder in pipeline_map.items():
            if pipeline_name in pipeline_configs:
                pipeline_case = pipeline_builder(
                    current_dir, pipeline_configs[pipeline_name]
                )
                test.add_pipeline_case(pipeline_case)
    else:
        if args.single in pipeline_configs:
            pipeline_name = args.single
            pipeline_builder = pipeline_map[args.single]
            pipeline_case = pipeline_builder(
                current_dir, pipeline_configs[pipeline_name]
            )
            test.add_pipeline_case(pipeline_case)

    report = test.run()

    logging.info(f"report {test_name}:\n.....\n{json.dumps(report)}\n......")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for benchmark')
    parser.add_argument('-c', '--config_path', help='Path to the configuration file')
    parser.add_argument(
        '-t',
        '--test_run',
        help='sample the features and provide a fast run',
        action="store_true",
        default=False,
    )
    parser.add_argument(
        '-s', '--single', help="select one single pipeline to run", default=None
    )
    args = parser.parse_args()
    current_dir = os.getcwd()
    config_path = os.path.join(current_dir, args.config_path)
    handler = logging.StreamHandler(open("test.log", "a"))
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    config_dict = load_config_dict(config_path)
    config_dict = recursive_fill(config_dict, current_dir)
    reports = {}
    for test_name, config_dict_test in config_dict.items():
        reports[test_name] = run_one_test(test_name, config_dict_test, args)

    timestamp = int(time.time())

    # Convert the timestamp to a human-readable format
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(timestamp))
    output_report_name = formatted_time + "_benchmark_result"
    if args.test_run:
        output_report_name += "_test_run"
    if args.single is not None:
        output_report_name += f"_single_{args.single}"

    output_report_name += ".json"
    output_p = os.path.join(current_dir, output_report_name)
    with open(output_p, "w") as f:
        json.dump(reports, f)
