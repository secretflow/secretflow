# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ultity functions for IO.
"""
import os
import json
import logging

##################################
#               IO               #
##################################


def check_dir(dir):
    if not os.path.exists(dir):
        print("Directory {} does not exist. Exit.".format(dir))
        exit(1)


def check_files(files):
    for f in files:
        if (f is not None) and not os.path.exists(f):
            print("File {} does not exist. Exit.".format(f))
            exit(1)


def ensure_dir(dir, verbose=True):
    if not os.path.exists(dir):
        if verbose:
            print("Directory {} do not exist; creating...".format(dir))
        os.makedirs(dir)


##################################
#  Input Argument Configuration  #
##################################


def save_config(args, verbose=True):
    # Ensure the directory exists
    model_id = args.model_id if len(args.model_id) > 1 else "0" + args.model_id
    method_ckpt_path = os.path.join(
        args.checkpoint_dir,
        "domain_" + "".join([domain[0] for domain in args.domains]),
        args.method + "_" + model_id,
    )
    ensure_dir(method_ckpt_path, verbose=True)

    # Save the input argument configuration
    config_filename = os.path.join(method_ckpt_path, "config.json")
    args_dict = vars(args)
    with open(config_filename, "w") as outfile:
        json.dump(vars(args), outfile, indent=2)
    if verbose:
        print("Config saved to file {}.".format(config_filename))

    # Print the input argument information
    print_config(args_dict)


def print_config(config):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    logging.info(info)
