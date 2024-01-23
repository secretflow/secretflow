import numpy as np
import pandas as pd
import sklearn
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

from secretflow.data import partition
from secretflow.data.vertical import VDataFrame
from secretflow.device.driver import reveal
from secretflow.stats import SSVertVIF
from secretflow.utils.simulation.datasets import dataset, load_linear


def _statsmodels_vif(data):
    cols = data.shape[1]
    std = sklearn.preprocessing.StandardScaler()
    data = std.fit_transform(data)
    ret = np.array([vif(data, i) for i in range(cols)])
    return ret


def _run_vif(env, vdata, data):
    v_vif = SSVertVIF(env.spu)
    ss_vif = v_vif.vif(vdata, infeed_elements_limit=1000)
    vif = _statsmodels_vif(data)
    # for nan/inf value in statsmodels' results, see NOTICE of SSVertVIF.
    ss_vif = np.select([ss_vif > 1000], [1000], ss_vif)
    vif = np.select([~np.isfinite(vif), vif > 1000], [1000, 1000], vif)
    err = np.absolute(ss_vif - vif) / np.maximum(vif, ss_vif)
    assert np.amax(err) < 0.5


def test_linear_vif(sf_production_setup_devices):
    vdata = load_linear(
        parts={
            sf_production_setup_devices.alice: (1, 11),
            sf_production_setup_devices.bob: (11, 21),
        }
    ).astype(np.float32)
    data = pd.read_csv(dataset('linear'), usecols=[f'x{i}' for i in range(1, 21)])
    _run_vif(sf_production_setup_devices, vdata, data)


def test_breast_cancer(sf_production_setup_devices):
    from sklearn.datasets import load_breast_cancer

    nd_data = load_breast_cancer()['data']

    alice_df = pd.DataFrame(nd_data[:, :15], copy=True)
    bob_df = pd.DataFrame(nd_data[:, 15:], copy=True)
    data = pd.DataFrame(nd_data, copy=True)
    vdata = VDataFrame(
        partitions={
            sf_production_setup_devices.alice: partition(
                sf_production_setup_devices.alice(lambda: alice_df)()
            ),
            sf_production_setup_devices.bob: partition(
                sf_production_setup_devices.bob(lambda: bob_df)()
            ),
        }
    )

    _run_vif(sf_production_setup_devices, vdata, data)


def test_const_col_data(sf_production_setup_devices):
    nd_data = reveal(
        sf_production_setup_devices.alice(lambda: np.random.random((10, 8)))()
    ).copy()
    # const value col
    nd_data[:, 2] = 1

    alice_df = pd.DataFrame(nd_data[:, :4], copy=True)
    bob_df = pd.DataFrame(nd_data[:, 4:], copy=True)
    data = pd.DataFrame(nd_data, copy=True)
    vdata = VDataFrame(
        partitions={
            sf_production_setup_devices.alice: partition(
                sf_production_setup_devices.alice(lambda: alice_df)()
            ),
            sf_production_setup_devices.bob: partition(
                sf_production_setup_devices.bob(lambda: bob_df)()
            ),
        }
    )

    _run_vif(sf_production_setup_devices, vdata, data)


def test_linear_col_data(sf_production_setup_devices):
    nd_data = reveal(
        sf_production_setup_devices.alice(lambda: np.random.random((10, 8)))()
    ).copy()
    # linear correlational col
    nd_data[:, 2] = nd_data[:, 6] * 2

    alice_df = pd.DataFrame(nd_data[:, :4], copy=True)
    bob_df = pd.DataFrame(nd_data[:, 4:], copy=True)
    data = pd.DataFrame(nd_data, copy=True)
    vdata = VDataFrame(
        partitions={
            sf_production_setup_devices.alice: partition(
                sf_production_setup_devices.alice(lambda: alice_df)()
            ),
            sf_production_setup_devices.bob: partition(
                sf_production_setup_devices.bob(lambda: bob_df)()
            ),
        }
    )

    _run_vif(sf_production_setup_devices, vdata, data)


# TODO(fengjun.feng): move the following to integration tests.

# if __name__ == '__main__':
#     # HOW TO RUN:
#     # 0. change args following <<< !!! >>> flag.
#     #    you need change input data path & train settings before run.
#     # 1. install requirements following INSTALLATION.md
#     # 2. set env
#     #    export PYTHONPATH=$PYTHONPATH:bazel-bin
#     # 3. run
#     #    python tests/stats/test_ss_vif_v.py

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
#     v_vif = SSVertVIF(cluster.spu)
#     ss_vif = v_vif.vif(v_data)
#     logging.info(f"main time: {time.time() - start}")
#     logging.info(f"main auc: {ss_vif}")
