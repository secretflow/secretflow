import os.path

import pandas as pd
import spu
from sklearn.metrics import roc_auc_score
import os
import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal, wait
from secretflow.ml.boost.sgb_v import (
    Sgb,
    get_classic_lightGBM_params,
    get_classic_XGB_params,
)

from secretflow.ml.boost.sgb_v.model import load_model
import pprint

# secretflow general configs
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7896'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7896'
pp = pprint.PrettyPrinter(depth=4)
print('The version of Secretflow:{}'.format(sf.__version__))

alice_ip = '127.0.0.1'
bob_ip = '127.0.0.1'
ip_party_map = {bob_ip: 'bob', alice_ip: 'alice'}

_system_config = {'lineage_pinning_enabled': False}
sf.shutdown()
# init cluster
sf.init(
    ['alice', 'bob'],
    address='local',
    _system_config=_system_config,
    object_store_memory=5 * 1024 * 1024 * 1024,
)

cluster_def ={
    'nodes': [
        {'party': 'alice', 'id': 'local:0', 'address': alice_ip + ':12945'},
        {'party': 'bob', 'id': 'local:1', 'address': bob_ip + ':12946'},
    ],
    'runtime_config': {
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
    },
}

heu_config = {
    'sk_keeper': {'party': 'alice'},
    'evaluators': [{'party': 'bob'}],
    'mode': 'PHEU',
    'he_parameters':{
        'schema': 'ASHE',
        'key_pair': {
            'generate':{
                'bit_size': 2048,
            },
        },
    },
    'encoding' : {
        'cleartext_type': 'DT_I32',
        'encoder': 'IntegerEncoder',
        'encoder_args': {'scale': 1000},
    },
}

alice = sf.PYU('alice')
bob = sf.PYU('bob')
heu = sf.HEU(heu_config, cluster_def['runtime_config']['field'])
# prepare for a vertical dataset

from sklearn.datasets import load_breast_cancer, load_digits, fetch_covtype, get_data_home
dataset = fetch_covtype()
X = dataset.data
y = dataset.target
# pd.DataFrame(data = dataset['data'], columns=dataset['feature_names']).to_csv('dataset.csv', index=False)
# print(dataset)
# print(dataset['feature_names'], dataset['target_names'])
x, y = dataset['data'], dataset['target']  # features & target
print(x.shape)
feature_data = FedNdarray(
    {
        alice: (alice(lambda: x[:, 27:])()),
        bob: (bob(lambda: x[:, :27])()),
    },
    partition_way=PartitionWay.VERTICAL,
)

label_data = FedNdarray(
    {alice: (alice(lambda : y)())},
    partition_way=PartitionWay.VERTICAL,
)
# print(type(alice(lambda :y)()))
# prepare for parameters
params = get_classic_XGB_params()
params['num_boost_round'] = 5
params['max_depth'] = 3
pp.pprint(params)

# run a Sgb model
sgb = Sgb(heu)
model = sgb.train(params, feature_data, label_data)

yhat = model.predict(feature_data)
yhat = reveal(yhat)
print(f'auc: {roc_auc_score(y, yhat)}')
