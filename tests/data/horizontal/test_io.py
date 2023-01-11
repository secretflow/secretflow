import os
import tempfile

import pandas as pd

from secretflow import reveal
from secretflow.data.base import Partition
from secretflow.data.horizontal import HDataFrame, read_csv, to_csv
from tests.basecase import MultiDriverDeviceTestCase


class TestHDataFrameIO(MultiDriverDeviceTestCase):
    @staticmethod
    def cleartmp(paths):
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_read_csv_and_to_csv_should_ok(self):
        # GIVEN
        _, path1 = tempfile.mkstemp()
        _, path2 = tempfile.mkstemp()
        file_uris = {self.alice: path1, self.bob: path2}
        df1 = pd.DataFrame(
            {'c1': ['A5', 'A1', 'A2', 'A6', 'A7', 'A9'], 'c2': [5, 1, 2, 6, 2, 4]}
        )

        df2 = pd.DataFrame({'c1': ['B3', 'B1', 'B9', 'B4'], 'c2': [3, 1, 9, 4]})

        df = HDataFrame(
            {
                self.alice: Partition(self.alice(lambda df: df)(df1)),
                self.bob: Partition(self.bob(lambda df: df)(df2)),
            }
        )

        # WHEN
        to_csv(df, file_uris, index=False)

        # THEN
        # Waiting a while for to_csv finish.
        import time

        time.sleep(5)
        actual_df = read_csv(file_uris)
        pd.testing.assert_frame_equal(
            reveal(actual_df.partitions[self.alice].data), df1
        )
        pd.testing.assert_frame_equal(reveal(actual_df.partitions[self.bob].data), df2)
        self.cleartmp([path1, path2])
