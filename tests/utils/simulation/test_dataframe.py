import pandas as pd
import pytest

from secretflow import reveal
from secretflow.utils.errors import InvalidArgumentError
from secretflow.utils.simulation.data.dataframe import create_df
from secretflow.utils.simulation.datasets import dataset


@pytest.fixture(scope='module')
def df():
    yield pd.read_csv(dataset('iris'))


def test_create_hdataframe_should_ok_when_input_dataframe(
    sf_production_setup_devices, df
):
    # WHEN
    hdf = create_df(
        df,
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        axis=0,
    )

    # THEN
    assert len(hdf.partitions) == 3
    pd.testing.assert_frame_equal(
        df, pd.concat([reveal(part.data) for part in hdf.partitions.values()])
    )


def test_create_hdataframe_should_ok_when_input_file(df, sf_production_setup_devices):
    # WHEN
    hdf = create_df(
        dataset('iris'),
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        axis=0,
    )

    # THEN
    assert len(hdf.partitions) == 3
    pd.testing.assert_frame_equal(
        df, pd.concat([reveal(part.data) for part in hdf.partitions.values()])
    )


def test_create_hdataframe_should_ok_when_specify_indexes(
    df, sf_production_setup_devices
):
    # WHEN
    hdf = create_df(
        df,
        parts={
            sf_production_setup_devices.alice: (0, 50),
            sf_production_setup_devices.bob: (50, 150),
        },
        axis=0,
    )

    # THEN
    assert len(hdf.partitions) == 2
    pd.testing.assert_frame_equal(
        df, pd.concat([reveal(part.data) for part in hdf.partitions.values()])
    )


def test_create_hdataframe_should_ok_when_specify_percentage(
    df,
    sf_production_setup_devices,
):
    # WHEN
    hdf = create_df(
        df,
        parts={
            sf_production_setup_devices.alice: 0.3,
            sf_production_setup_devices.bob: 0.7,
        },
        axis=0,
    )

    # THEN
    assert len(hdf.partitions) == 2
    assert len(
        reveal(hdf.partitions[sf_production_setup_devices.alice].data)
    ) == 0.3 * len(df)
    assert len(
        reveal(hdf.partitions[sf_production_setup_devices.bob].data)
    ) == 0.7 * len(df)

    pd.testing.assert_frame_equal(
        df, pd.concat([reveal(part.data) for part in hdf.partitions.values()])
    )


def test_create_vdataframe_should_ok(df, sf_production_setup_devices):
    # WHEN
    hdf = create_df(
        df,
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        axis=1,
    )

    # THEN
    assert len(hdf.partitions) == 3
    pd.testing.assert_frame_equal(
        df,
        pd.concat([reveal(part.data) for part in hdf.partitions.values()], axis=1),
    )


def test_create_vdataframe_should_ok_when_input_callable(
    df, sf_production_setup_devices
):
    # WHEN
    hdf = create_df(
        lambda: pd.read_csv(dataset('iris')),
        parts=[
            sf_production_setup_devices.alice,
            sf_production_setup_devices.bob,
            sf_production_setup_devices.carol,
        ],
        axis=1,
    )

    # THEN
    assert len(hdf.partitions) == 3
    pd.testing.assert_frame_equal(
        df,
        pd.concat([reveal(part.data) for part in hdf.partitions.values()], axis=1),
    )


def test_create_vdataframe_should_error_when_illegal_source(
    sf_production_setup_devices,
):
    with pytest.raises(
        AssertionError, match='Callable source must return a pandas DataFrame'
    ):
        create_df(
            lambda: 1,
            parts=[sf_production_setup_devices.alice, sf_production_setup_devices.bob],
        )

    with pytest.raises(InvalidArgumentError, match='Unknown source type'):
        create_df(
            {},
            parts=[sf_production_setup_devices.alice, sf_production_setup_devices.bob],
        )


def test_create_vdataframe_should_error_when_illegal_parts(
    df, sf_production_setup_devices
):
    with pytest.raises(AssertionError, match='Parts should not be none or empty!'):
        create_df(df, parts=None)

    with pytest.raises(AssertionError, match='Parts shall be list like of PYUs'):
        create_df(df, parts=[1, 2])

    with pytest.raises(AssertionError, match='Keys of parts shall be PYU'):
        create_df(df, parts={1: 0.1})

    with pytest.raises(AssertionError, match='Sum of percentages shall be 1.0.'):
        create_df(
            df,
            parts={
                sf_production_setup_devices.alice: 0.1,
                sf_production_setup_devices.bob: 2.0,
            },
        )

    with pytest.raises(AssertionError, match='Not all dict values are percentages.'):
        create_df(
            df,
            parts={
                sf_production_setup_devices.alice: 0.1,
                sf_production_setup_devices.bob: '3',
            },
        )
