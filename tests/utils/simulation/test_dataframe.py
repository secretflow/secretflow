import pandas as pd

from secretflow import reveal
from secretflow.utils.errors import InvalidArgumentError
from secretflow.utils.simulation.data.dataframe import create_df
from tests.basecase import DeviceTestCase


class TestDataFrame(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.df = pd.read_csv('tests/datasets/iris/iris.csv')

    def test_create_hdataframe_should_ok_when_input_dataframe(self):
        # WHEN
        hdf = create_df(self.df, parts=[self.alice, self.bob, self.carol], axis=0)

        # THEN
        self.assertEqual(len(hdf.partitions), 3)
        pd.testing.assert_frame_equal(
            self.df, pd.concat([reveal(part.data) for part in hdf.partitions.values()])
        )

    def test_create_hdataframe_should_ok_when_input_file(self):
        # WHEN
        hdf = create_df(
            'tests/datasets/iris/iris.csv',
            parts=[self.alice, self.bob, self.carol],
            axis=0,
        )

        # THEN
        self.assertEqual(len(hdf.partitions), 3)
        pd.testing.assert_frame_equal(
            self.df, pd.concat([reveal(part.data) for part in hdf.partitions.values()])
        )

    def test_create_hdataframe_should_ok_when_specify_indexes(self):
        # WHEN
        hdf = create_df(
            self.df, parts={self.alice: (0, 50), self.bob: (50, 150)}, axis=0
        )

        # THEN
        self.assertEqual(len(hdf.partitions), 2)
        pd.testing.assert_frame_equal(
            self.df, pd.concat([reveal(part.data) for part in hdf.partitions.values()])
        )

    def test_create_hdataframe_should_ok_when_specify_percentage(self):
        # WHEN
        hdf = create_df(self.df, parts={self.alice: 0.3, self.bob: 0.7}, axis=0)

        # THEN
        self.assertEqual(len(hdf.partitions), 2)
        self.assertEqual(
            len(reveal(hdf.partitions[self.alice].data)), 0.3 * len(self.df)
        )
        self.assertEqual(len(reveal(hdf.partitions[self.bob].data)), 0.7 * len(self.df))
        pd.testing.assert_frame_equal(
            self.df, pd.concat([reveal(part.data) for part in hdf.partitions.values()])
        )

    def test_create_vdataframe_should_ok(self):
        # WHEN
        hdf = create_df(self.df, parts=[self.alice, self.bob, self.carol], axis=1)

        # THEN
        self.assertEqual(len(hdf.partitions), 3)
        pd.testing.assert_frame_equal(
            self.df,
            pd.concat([reveal(part.data) for part in hdf.partitions.values()], axis=1),
        )

    def test_create_vdataframe_should_ok_when_input_callable(self):
        # WHEN
        hdf = create_df(
            lambda: pd.read_csv('tests/datasets/iris/iris.csv'),
            parts=[self.alice, self.bob, self.carol],
            axis=1,
        )

        # THEN
        self.assertEqual(len(hdf.partitions), 3)
        pd.testing.assert_frame_equal(
            self.df,
            pd.concat([reveal(part.data) for part in hdf.partitions.values()], axis=1),
        )

    def test_create_vdataframe_should_error_when_illegal_source(self):
        with self.assertRaisesRegex(
            AssertionError, 'Callable source must return a pandas DataFrame'
        ):
            create_df(lambda: 1, parts=[self.alice, self.bob])

        with self.assertRaisesRegex(InvalidArgumentError, 'Unknown source type'):
            create_df({}, parts=[self.alice, self.bob])

    def test_create_vdataframe_should_error_when_illegal_parts(self):
        with self.assertRaisesRegex(
            AssertionError, 'Parts should not be none or empty!'
        ):
            create_df(self.df, parts=None)

        with self.assertRaisesRegex(AssertionError, 'Parts shall be list like of PYUs'):
            create_df(self.df, parts=[1, 2])

        with self.assertRaisesRegex(AssertionError, 'Keys of parts shall be PYU'):
            create_df(self.df, parts={1: 0.1})

        with self.assertRaisesRegex(AssertionError, 'Sum of percentages shall be 1.0.'):
            create_df(self.df, parts={self.alice: 0.1, self.bob: 2.0})

        with self.assertRaisesRegex(
            AssertionError, 'Not all dict values are percentages.'
        ):
            create_df(self.df, parts={self.alice: 0.1, self.bob: '3'})
