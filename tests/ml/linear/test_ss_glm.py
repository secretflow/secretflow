import time

import logging
import numpy as np
from secretflow.device.driver import reveal, wait
from secretflow.ml.linear.ss_glm import SSGLM
from secretflow.ml.linear.ss_glm.core import get_dist
from sklearn.preprocessing import StandardScaler
from tests.basecase import ABY3MultiDriverDeviceTestCase
from secretflow.data import FedNdarray, PartitionWay
from secretflow.utils.simulation.datasets import load_linear, create_df, dataset
from sklearn.metrics import roc_auc_score


class TestVertBinning(ABY3MultiDriverDeviceTestCase):
    def _transform(self, data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def wait_io(self, inputs):
        wait_objs = list()
        for input in inputs:
            wait_objs.extend([input.partitions[d] for d in input.partitions])
        wait(wait_objs)

    def run_test(self, test_name, v_data, label_data, y, batch_size, link, dist):
        model = SSGLM(self.spu)
        start = time.time()
        model.fit_sgd(
            v_data,
            label_data,
            None,
            None,
            3,
            link,
            dist,
            1,
            1,
            0.3,
            iter_start_irls=1,
            batch_size=batch_size,
        )
        logging.info(f"{test_name} sgb train time: {time.time() - start}")
        start = time.time()
        spu_yhat = model.predict(v_data)
        yhat = reveal(spu_yhat)
        assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
        logging.info(f"{test_name} predict time: {time.time() - start}")
        deviance = get_dist(dist, 1, 1).deviance(yhat, y, None)
        logging.info(f"{test_name} deviance: {deviance}")

        model.fit_irls(
            v_data,
            label_data,
            None,
            None,
            3,
            link,
            dist,
            1,
            1,
        )
        logging.info(f"{test_name} irls train time: {time.time() - start}")
        start = time.time()
        spu_yhat = model.predict(v_data)
        yhat = reveal(spu_yhat)
        assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
        logging.info(f"{test_name} predict time: {time.time() - start}")
        deviance = get_dist(dist, 1, 1).deviance(yhat, y, None)
        logging.info(f"{test_name} deviance: {deviance}")

        fed_yhat = model.predict(v_data, to_pyu=self.alice)
        assert len(fed_yhat.partitions) == 1 and self.alice in fed_yhat.partitions
        yhat = reveal(fed_yhat.partitions[self.alice])
        assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
        deviance = get_dist(dist, 1, 1).deviance(yhat, y, None)
        logging.info(f"{test_name} deviance: {deviance}")

    def test_breast_cancer(self):
        start = time.time()
        from sklearn.datasets import load_breast_cancer

        ds = load_breast_cancer()
        x, y = self._transform(ds['data']), ds['target']

        v_data = FedNdarray(
            partitions={
                self.alice: self.alice(lambda: x[:, :15])(),
                self.bob: self.bob(lambda: x[:, 15:])(),
            },
            partition_way=PartitionWay.VERTICAL,
        )
        label_data = FedNdarray(
            partitions={self.alice: self.alice(lambda: y)()},
            partition_way=PartitionWay.VERTICAL,
        )

        self.wait_io([v_data, label_data])
        logging.info(f"IO times: {time.time() - start}s")

        self.run_test("breast_cancer", v_data, label_data, y, 128, 'Logit', 'Bernoulli')

    def test_linear(self):
        start = time.time()
        vdf = load_linear(parts={self.alice: (1, 11), self.bob: (11, 22)})
        label_data = vdf['y']
        v_data = vdf.drop(columns="y")
        y = reveal(label_data.partitions[self.bob].data).values
        self.wait_io([v_data.values, label_data.values])
        logging.info(f"IO times: {time.time() - start}s")

        self.run_test("linear", v_data, label_data, y, 32, 'Logit', 'Bernoulli')

    def test_gamma_data(self):
        start = time.time()
        import statsmodels.api as sm

        data = sm.datasets.scotland.load()

        x = self._transform(np.array(data.exog))
        y = np.array(data.endog)

        v_data = FedNdarray(
            partitions={
                self.alice: self.alice(lambda: x[:, :3])(),
                self.bob: self.bob(lambda: x[:, 3:])(),
            },
            partition_way=PartitionWay.VERTICAL,
        )
        label_data = FedNdarray(
            partitions={self.alice: self.alice(lambda: y)()},
            partition_way=PartitionWay.VERTICAL,
        )

        self.wait_io([v_data, label_data])
        logging.info(f"IO times: {time.time() - start}s")

        self.run_test("gamma", v_data, label_data, y, 32, 'Log', 'Gamma')


if __name__ == '__main__':
    # HOW TO RUN:
    # 0. change args following <<< !!! >>> flag.
    #    you need change input data path & train settings before run.
    # 1. install requirements following INSTALLATION.md
    # 2. set env
    #    export PYTHONPATH=$PYTHONPATH:bazel-bin
    # 3. run
    #    python tests/ml/linear/test_ss_glm.py

    # !!!!!!
    # This example contains two test: irls mode glm and sgd mode glm
    # need to be timed separately in benchmark testing.
    # !!!!!!

    # <<< !!! >>> uncomment next line if you need run this demo under MPU.
    # jax.config.update("jax_enable_x64", True)

    # use aby3 in this example.
    cluster = ABY3MultiDriverDeviceTestCase()
    cluster.setUpClass()
    # init log
    logging.getLogger().setLevel(logging.INFO)

    # prepare data
    start = time.time()
    # read dataset.
    vdf = create_df(
        # load file 'dataset('linear')' as train dataset.
        # <<< !!! >>> replace dataset path to your own local file.
        dataset('linear'),
        # split 1-11 columns to alice and 11-21 columns to bob which include y col.
        # <<< !!! >>> replace parts range to your own dataset's columns count.
        parts={cluster.alice: (1, 11), cluster.bob: (11, 22)},
        # split by vertical. DON'T change this.
        axis=1,
    )
    # split y out of dataset,
    # <<< !!! >>> change 'y' if label column name is not y in dataset.
    label_data = vdf["y"]
    # v_data remains all features.
    v_data = vdf.drop(columns="y")
    # <<< !!! >>> change cluster.bob if y not belong to bob.
    y = reveal(label_data.partitions[cluster.bob].data)
    wait([p.data for p in v_data.partitions.values()])
    logging.info(f"IO times: {time.time() - start}s")

    # <<< !!! >>> run irls mode glm
    model = SSGLM(cluster.spu)
    start = time.time()
    model.fit_irls(
        v_data,
        label_data,
        None,
        None,
        3,
        'Logit',
        'Bernoulli',
        1,
        1,
    )
    logging.info(f"main irls mode train time: {time.time() - start}")
    start = time.time()
    spu_yhat = model.predict(v_data)
    yhat = reveal(spu_yhat)
    logging.info(f"main predict time: {time.time() - start}")
    logging.info(f"main auc: {roc_auc_score(y, yhat)}")

    # <<< !!! >>> run sgd mode glm
    start = time.time()
    model.fit_sgd(
        v_data,
        label_data,
        None,
        None,
        3,
        'Logit',
        'Bernoulli',
        1,
        1,
        0.3,
        iter_start_irls=0,
        batch_size=128,
    )
    logging.info(f"main sgd mode train time: {time.time() - start}")
    start = time.time()
    spu_yhat = model.predict(v_data)
    yhat = reveal(spu_yhat)
    logging.info(f"main predict time: {time.time() - start}")
    logging.info(f"main auc: {roc_auc_score(y, yhat)}")
