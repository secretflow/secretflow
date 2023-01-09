import time

import sys
import logging
from secretflow.device.driver import reveal, wait
from secretflow.ml.linear.ss_sgd import SSRegression
from sklearn.preprocessing import StandardScaler
from tests.basecase import ABY3MultiDriverDeviceTestCase
from sklearn.metrics import roc_auc_score
from secretflow.data import FedNdarray, PartitionWay
from secretflow.utils.simulation.datasets import load_linear, create_df, dataset


class TestVertBinning(ABY3MultiDriverDeviceTestCase):
    def _transform(self, data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def wait_io(self, inputs):
        wait_objs = list()
        for input in inputs:
            wait_objs.extend([input.partitions[d] for d in input.partitions])
        wait(wait_objs)

    def run_test(self, test_name, v_data, label_data, y, batch_size):
        model = SSRegression(self.spu)
        start = time.time()
        model.fit(
            v_data,
            label_data,
            3,
            0.3,
            batch_size,
            't1',
            'logistic',
            'l2',
            0.5,
        )
        logging.info(f"{test_name} train time: {time.time() - start}")
        start = time.time()
        spu_yhat = model.predict(v_data, batch_size)
        yhat = reveal(spu_yhat)
        assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
        logging.info(f"{test_name} predict time: {time.time() - start}")
        logging.info(f"{test_name} auc: {roc_auc_score(y, yhat)}")

        fed_yhat = model.predict(v_data, batch_size, self.alice)
        assert len(fed_yhat.partitions) == 1 and self.alice in fed_yhat.partitions
        yhat = reveal(fed_yhat.partitions[self.alice])
        assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
        logging.info(f"{test_name} auc: {roc_auc_score(y, yhat)}")

    def test_breast_cancer(self):
        from sklearn.datasets import load_breast_cancer

        start = time.time()
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

        self.run_test("breast_cancer", v_data, label_data, y, 128)

    def test_linear(self):
        start = time.time()
        vdf = load_linear(parts={self.alice: (1, 11), self.bob: (11, 22)})
        label_data = vdf['y']
        v_data = vdf.drop(columns="y")
        y = reveal(label_data.partitions[self.bob].data)
        self.wait_io([v_data.values, label_data.values])
        logging.info(f"IO times: {time.time() - start}s")

        self.run_test("linear", v_data, label_data, y, 1024)


if __name__ == '__main__':
    # HOW TO RUN:
    # 0. change args following <<< !!! >>> flag.
    #    you need change input data path & train settings before run.
    # 1. install requirements following INSTALLATION.md
    # 2. set env
    #    export PYTHONPATH=$PYTHONPATH:bazel-bin
    # 3. run
    #    python tests/ml/linear/test_ss_lr.py

    # use aby3 in this example.
    cluster = ABY3MultiDriverDeviceTestCase()
    cluster.setUpClass()
    # init log
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # prepare data
    start = time.time()
    # read dataset.
    vdf = create_df(
        # load file 'dataset('linear')' as train dataset.
        # <<< !!! >>> replace dataset path to your own local file.
        dataset('linear'),
        # split 1-10 columns to alice and 11-21 columns to bob which include y col.
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

    # run ss lr
    model = SSRegression(cluster.spu)
    start = time.time()
    model.fit(
        v_data,
        label_data,
        # <<< !!! >>> change args to your test settings.
        3,  # epochs
        0.3,
        1024,  # batch_size
        't1',  # sig_type
        'logistic',  # reg_type
        'l2',  # penalty
        0.5,
    )
    logging.info(f"main train time: {time.time() - start}")
    start = time.time()
    spu_yhat = model.predict(v_data, 1024)
    yhat = reveal(spu_yhat)
    logging.info(f"main predict time: {time.time() - start}")
    logging.info(f"main auc: {roc_auc_score(y, yhat)}")
