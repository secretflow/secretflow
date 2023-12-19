import os

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from secretflow import reveal, wait
from secretflow.component.batch_reader import SimpleVerticalBatchReader


def test_works(sf_production_setup_devices):
    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]

    expected_row_cnt = x.shape[0]
    cols = {}
    paths = {
        'alice': os.path.join("tmp", "alice", "test_batch_reader", "alice.csv"),
        'bob': os.path.join("tmp", "bob", "test_batch_reader", "bob.csv"),
    }

    def create_alice_data(p, x, y):
        os.makedirs(
            os.path.dirname(p),
            exist_ok=True,
        )
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(p, index=False)

    wait(sf_production_setup_devices.alice(create_alice_data)(paths["alice"], x, y))

    def create_bob_data(p, x):
        os.makedirs(
            os.path.dirname(p),
            exist_ok=True,
        )

        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(p, index=False)

    wait(sf_production_setup_devices.bob(create_bob_data)(paths["bob"], x))

    cols = {
        "alice": [f"a{i}" for i in range(15)],
        "bob": [f"b{i}" for i in range(15)],
    }

    reader = SimpleVerticalBatchReader(
        paths=paths,
        batch_size=50,
        cols=cols,
    )

    row_cnt = 0
    for batches in reader:
        alice_df, bob_df = reveal(batches["alice"]), reveal(batches["bob"])

        assert alice_df.shape[0] == bob_df.shape[0]

        row_cnt += alice_df.shape[0]

        assert alice_df.shape[1] == 15

        assert bob_df.shape[1] == 15

        assert row_cnt == reader.total_read_cnt()

    assert expected_row_cnt == row_cnt
