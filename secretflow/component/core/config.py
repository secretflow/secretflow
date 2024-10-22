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

import json
import logging

import spu

from secretflow.error_system.exceptions import (
    NotSupportedError,
    SFTrainingHyperparameterError,
)
from secretflow.spec.extend.cluster_pb2 import SFClusterConfig


def _parse_runtime_config(key: str, raw: str):
    if key == "protocol":
        if raw == "REF2K":
            return spu.spu_pb2.REF2K
        elif raw == "SEMI2K":
            return spu.spu_pb2.SEMI2K
        elif raw == "ABY3":
            return spu.spu_pb2.ABY3
        elif raw == "CHEETAH":
            return spu.spu_pb2.CHEETAH
        else:
            raise ValueError(f"unsupported spu protocol: {raw}")
    elif key == "field":
        if raw == "FM32":
            return spu.spu_pb2.FM32
        elif raw == "FM64":
            return spu.spu_pb2.FM64
        elif raw == "FM128":
            return spu.spu_pb2.FM128
        else:
            raise ValueError(f"unsupported spu field: {raw}")
    elif key == "fxp_fraction_bits":
        return int(raw)
    else:
        raise ValueError(f"unsupported runtime config: {key}, raw {raw}")


def extract_device_config(config: SFClusterConfig):  # type: ignore
    if config is None:
        return None, None
    spu_configs = {}
    spu_addresses = {spu.name: spu for spu in config.public_config.spu_configs}

    heu_config = None

    for device in config.desc.devices:
        if len(set(device.parties)) != len(device.parties):
            raise ValueError(f"parties of device {device.name} are not unique.")
        if device.type.lower() == "spu":
            if device.name not in spu_addresses:
                raise ValueError(f"addresses of spu {device.name} is not available.")

            addresses = spu_addresses[device.name]

            # check parties
            if len(set(addresses.parties)) != len(addresses.parties):
                raise ValueError(
                    f"parties in addresses of device {device.name} are not unique."
                )

            if set(addresses.parties) != set(device.parties):
                raise ValueError(
                    f"parties in addresses of device {device.name} do not match those in desc."
                )

            spu_config_json = json.loads(device.config)
            cluster_def = {
                "nodes": [
                    {
                        "party": p,
                        "address": addresses.addresses[idx],
                        "listen_address": (
                            addresses.listen_addresses[idx]
                            if len(addresses.listen_addresses)
                            else ""
                        ),
                    }
                    for idx, p in enumerate(list(addresses.parties))
                ]
            }

            # parse runtime config
            if "runtime_config" in spu_config_json:
                cluster_def["runtime_config"] = {}
                SUPPORTED_RUNTIME_CONFIG_ITEM = [
                    "protocol",
                    "field",
                    "fxp_fraction_bits",
                ]
                raw_runtime_config = spu_config_json["runtime_config"]

                for k, v in raw_runtime_config.items():
                    if k not in SUPPORTED_RUNTIME_CONFIG_ITEM:
                        logging.warning(f"runtime config item {k} is not parsed.")
                    else:
                        rt = _parse_runtime_config(k, v)
                        cluster_def["runtime_config"][k] = rt

            spu_configs[device.name] = {"cluster_def": cluster_def}

            if "link_desc" in spu_config_json:
                spu_configs[device.name]["link_desc"] = spu_config_json["link_desc"]
        elif device.type == "heu":
            if heu_config is not None:
                raise SFTrainingHyperparameterError.sf_cluster_config_error(
                    "only support one heu config"
                )
            heu_config_json = json.loads(device.config)
            if not isinstance(heu_config_json, dict):
                raise SFTrainingHyperparameterError.sf_cluster_config_error(
                    f"heu config {device.config} should be a dict"
                )
            SUPPORTED_HEU_CONFIG_ITEM = ["mode", "schema", "key_size"]
            heu_config = {}
            for k in SUPPORTED_HEU_CONFIG_ITEM:
                if k not in heu_config_json:
                    raise SFTrainingHyperparameterError.sf_cluster_config_error(
                        f"missing {k} config in heu config {device.config}"
                    )
                heu_config[k] = heu_config_json.pop(k)
            if len(heu_config_json) != 0:
                raise SFTrainingHyperparameterError.sf_cluster_config_error(
                    f"unknown {list(heu_config_json.keys())} config in heu config {device.config}"
                )
        else:
            raise NotSupportedError.not_supported_device_type(device_type=device.type)

    return spu_configs, heu_config
