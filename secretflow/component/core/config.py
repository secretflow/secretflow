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

import copy
import json
import logging

import spu

from secretflow.spec.extend.cluster_pb2 import SFClusterConfig
from secretflow.utils.errors import InvalidArgumentError


def _parse_runtime_config(key: str, raw: str | dict, nodes: list[dict]):
    if key == "protocol":
        return raw
        # if raw == "REF2K":
        #     return spu.ProtocolKind.REF2K
        # elif raw == "SEMI2K":
        #     return spu.ProtocolKind.SEMI2K
        # elif raw == "ABY3":
        #     return spu.ProtocolKind.ABY3
        # elif raw == "CHEETAH":
        #     return spu.ProtocolKind.CHEETAH
        # else:
        #     raise InvalidArgumentError(
        #         "unsupported spu protocol", detail={"protocol": raw}
        #     )
    elif key == "field":
        return raw
        # if raw == "FM32":
        #     return spu.FieldType.FM32
        # elif raw == "FM64":
        #     return spu.FieldType.FM64
        # elif raw == "FM128":
        #     return spu.FieldType.FM128
        # else:
        #     raise InvalidArgumentError("unsupported spu field", detail={"field": raw})
    if key == "fxp_fraction_bits":
        return int(raw)
    elif key == "beaver_type":
        if raw.lower() == "tfp":
            return spu.RuntimeConfig.BeaverType.TrustedFirstParty.name
        elif raw.lower() == "ttp":
            return spu.RuntimeConfig.BeaverType.TrustedThirdParty.name
        else:
            raise InvalidArgumentError(
                "unsupported beaver type", detail={"beaver_type": raw}
            )
    elif key == "ttp_beaver_config":
        ttp_config = copy.deepcopy(raw)
        adjust_party = ttp_config.pop("adjust_party")
        for rank, node in enumerate(sorted(nodes, key=lambda x: x["party"])):
            if node['party'] == adjust_party:
                ttp_config["adjust_rank"] = rank
                return ttp_config
        raise InvalidArgumentError(
            "adjust_party not found in nodes", detail={"adjust_party": adjust_party}
        )
    else:
        raise InvalidArgumentError(
            f"unsupported runtime config", detail={"key": key, "value": raw}
        )


def extract_device_config(config: SFClusterConfig):
    if config is None:
        return None, None
    spu_configs = {}
    spu_addresses = {spu.name: spu for spu in config.public_config.spu_configs}

    heu_config = None

    for device in config.desc.devices:
        if len(set(device.parties)) != len(device.parties):
            raise InvalidArgumentError(
                f"parties of device {device.name} are not unique."
            )
        if device.type.lower() == "spu":
            if device.name not in spu_addresses:
                raise InvalidArgumentError(
                    f"addresses of spu {device.name} is not available."
                )

            addresses = spu_addresses[device.name]

            # check parties
            if len(set(addresses.parties)) != len(addresses.parties):
                raise InvalidArgumentError(
                    f"parties in addresses of device {device.name} are not unique."
                )

            if set(addresses.parties) != set(device.parties):
                raise InvalidArgumentError(
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
                    "beaver_type",
                    "ttp_beaver_config",
                ]
                raw_runtime_config = spu_config_json["runtime_config"]
                cluster_def["runtime_config"] = raw_runtime_config

                for k, v in raw_runtime_config.items():
                    if k not in SUPPORTED_RUNTIME_CONFIG_ITEM:
                        logging.warning(f"runtime config item {k} is not parsed.")
                    else:
                        rt = _parse_runtime_config(k, v, cluster_def["nodes"])
                        cluster_def["runtime_config"][k] = rt

            spu_configs[device.name] = {"cluster_def": cluster_def}

            if "link_desc" in spu_config_json:
                spu_configs[device.name]["link_desc"] = spu_config_json["link_desc"]
        elif device.type == "heu":
            if heu_config is not None:
                raise InvalidArgumentError("only support one heu config")
            heu_config_json = json.loads(device.config)
            if not isinstance(heu_config_json, dict):
                raise InvalidArgumentError(
                    f"heu config {device.config} should be a dict"
                )
            SUPPORTED_HEU_CONFIG_ITEM = ["mode", "schema", "key_size"]
            heu_config = {}
            for k in SUPPORTED_HEU_CONFIG_ITEM:
                if k not in heu_config_json:
                    raise InvalidArgumentError(
                        f"missing {k} config in heu config {device.config}"
                    )
                heu_config[k] = heu_config_json.pop(k)
            if len(heu_config_json) != 0:
                raise InvalidArgumentError(
                    f"unknown {list(heu_config_json.keys())} config in heu config {device.config}"
                )
        else:
            raise InvalidArgumentError(f"unsupported device type: {device.type}")

    return spu_configs, heu_config
