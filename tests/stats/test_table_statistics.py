from secretflow.stats import table_statistics

from tests.basecase import DeviceTestCase
import pandas as pd
import secretflow as sf
from sklearn.datasets import load_iris
import tempfile
from secretflow.data.vertical import read_csv as v_read_csv


class TestTableStatistics(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        sf.shutdown()
        sf.init(['alice', 'bob'])
        alice = sf.PYU('alice')
        bob = sf.PYU('bob')
        iris = load_iris(as_frame=True)
        data = pd.concat([iris.data, iris.target], axis=1)
        data.iloc[1, 1] = None
        data.iloc[100, 1] = None

        # Restore target to its original name.
        data['target'] = data['target'].map(
            {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        )
        # Vertical partitioning.
        v_alice, v_bob = data.iloc[:, :2], data.iloc[:, 2:]

        # Save to temprary files.
        _, alice_path = tempfile.mkstemp()
        _, bob_path = tempfile.mkstemp()
        v_alice.to_csv(alice_path, index=False)
        v_bob.to_csv(bob_path, index=False)

        cls.df_v = v_read_csv({alice: alice_path, bob: bob_path})
        cls.df = data

    def test_table_statistics(self):
        """
        This test shows that table statistics works on both pandas and VDataFrame,
         i.e. all APIs align and the result is correct.
        """
        correct_summary = table_statistics(self.df)
        summary = table_statistics(self.df_v)
        result = summary.equals(correct_summary)
        if not result:
            n_rows = correct_summary.shape[0]
            n_cols = correct_summary.shape[1]
            assert n_rows == summary.shape[0], "row number mismatch"
            assert n_cols == summary.shape[1], "col number mismatch"
            for i in range(n_rows):
                for j in range(n_cols):
                    assert (
                        correct_summary.iloc[i, j] == summary.iloc[i, j]
                    ), "row {}, col {} mismatch".format(i, summary.columns[j])
