import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

from secretflow import reveal
from secretflow.preprocessing.scaler import StandardScaler
from secretflow.stats import SSVertPearsonR
from secretflow.utils.simulation.datasets import dataset, load_linear


@pytest.fixture(scope='module')
def prod_env_and_data(sf_production_setup_devices):
    vdata = load_linear(
        parts={
            sf_production_setup_devices.alice: (1, 11),
            sf_production_setup_devices.bob: (11, 21),
        }
    ).astype(np.float32)

    return sf_production_setup_devices, vdata


def scipy_pearsonr():
    ret = np.ones((20, 20))
    data = pd.read_csv(dataset('linear'), usecols=[f'x{i}' for i in range(1, 21)])

    for i in range(20):
        for j in range(i, 20):
            if i == j:
                ret[i, i] = 1
            else:
                p = pearsonr(data["x%d" % (i + 1)], data["x%d" % (j + 1)])
                ret[i, j] = p[0]
                ret[j, i] = p[0]

    return ret


def test_pearsonr(prod_env_and_data):
    env, data = prod_env_and_data
    for d in data.partitions.values():
        reveal(d.data)
    v_pearsonr = SSVertPearsonR(env.spu)
    scaler = StandardScaler()
    std_data = scaler.fit_transform(data)
    ss_pearsonr_1 = v_pearsonr.pearsonr(data, infeed_elements_limit=1000)
    ss_pearsonr_2 = v_pearsonr.pearsonr(std_data, False)
    expected = scipy_pearsonr()
    np.testing.assert_almost_equal(ss_pearsonr_1, expected, decimal=2)
    np.testing.assert_almost_equal(ss_pearsonr_2, expected, decimal=2)


# TODO(fengjun.feng): move the following to integration tests.

# if __name__ == '__main__':
#     # HOW TO RUN:
#     # 0. change args following <<< !!! >>> flag.
#     #    you need change input data path & train settings before run.
#     # 1. install requirements following INSTALLATION.md
#     # 2. set env
#     #    export PYTHONPATH=$PYTHONPATH:bazel-bin
#     # 3. run
#     #    python tests/stats/test_ss_pearsonr_v.py

#     # use aby3 in this example.
#     cluster = ABY3MultiDriverDeviceTestCase()
#     cluster.setUpClass()
#     # init log
#     logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#     # prepare data
#     start = time.time()
#     # read dataset.
#     v_data = create_df(
#         # load file 'dataset('linear')' as train dataset.
#         # <<< !!! >>> replace dataset path to your own local file.
#         dataset('linear'),
#         # split 1-10 columns to alice and 11-20 columns to bob
#         # <<< !!! >>> replace parts range to your own dataset's columns count.
#         parts={cluster.alice: (1, 11), cluster.bob: (11, 21)},
#         # split by vertical. DON'T change this.
#         axis=1,
#     )
#     wait([p.data for p in v_data.partitions.values()])
#     logging.info(f"IO times: {time.time() - start}s")

#     # run ss vif
#     v_pearsonr = SSVertPearsonR(cluster.spu)
#     corr = v_pearsonr.pearsonr(v_data)
#     logging.info(f"main time: {time.time() - start}")
#     logging.info(f"main auc: {corr}")
