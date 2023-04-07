import pandas as pd
from sklearn.datasets import load_iris

from secretflow.data.base import Partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.stats import table_statistics
from tests.basecase import MultiDriverDeviceTestCase


class TestTableStatistics(MultiDriverDeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
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
        cls.df_v = VDataFrame(
            partitions={
                cls.alice: Partition(cls.alice(lambda: v_alice)()),
                cls.bob: Partition(cls.bob(lambda: v_bob)()),
            }
        )
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
