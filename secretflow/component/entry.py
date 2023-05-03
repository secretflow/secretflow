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

from secretflow.component.ml.linear.ss_sgd import ss_sgd_predict_comp, ss_sgd_train_comp
from secretflow.component.preprocessing.train_test_split import train_test_split_comp
from secretflow.component.psi.two_party_balanced import two_party_balanced_psi_comp
from secretflow.protos.component.comp_def_pb2 import CompListDef
from secretflow.protos.component.node_def_pb2 import NodeDef

ALL_COMPONENTS = [
    train_test_split_comp,
    two_party_balanced_psi_comp,
    ss_sgd_train_comp,
    ss_sgd_predict_comp,
]
COMP_LIST_NAME = 'experimental'
COMP_LIST_DOC_STRING = 'Some experimental componments. Not production ready.'
COMP_LIST_VERSION = '0.0.1'


def gen_key(domain: str, name: str, version: str) -> str:
    return f'{domain}/{name}:{version}'


def generate_comp_list():
    comp_list = CompListDef()
    comp_list.name = COMP_LIST_NAME
    comp_list.doc_string = COMP_LIST_DOC_STRING
    comp_list.version = COMP_LIST_VERSION
    comp_map = {}
    all_comp_defs = []
    for x in ALL_COMPONENTS:
        x_def = x.definition()
        comp_map[gen_key(x_def.domain, x_def.name, x_def.version)] = x
        all_comp_defs.append(x_def)

    all_comp_defs = sorted(all_comp_defs, key=lambda k: (k.domain, k.name, k.version))
    comp_list.comps.extend(all_comp_defs)
    return comp_list, comp_map


COMP_LIST, COMP_MAP = generate_comp_list()


def eval(instance: NodeDef, secretflow_cluster_config):
    key = gen_key(instance.domain, instance.name, instance.version)
    if key in COMP_MAP:
        comp = COMP_MAP[key]
        comp.eval(instance, secretflow_cluster_config)
    else:
        raise RuntimeError("component is not found.")
