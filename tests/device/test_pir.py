import os
import shutil

import pandas as pd
import pytest

import secretflow as sf
import tempfile

@pytest.fixture()
def set_up(sf_production_setup_devices):
    da = pd.DataFrame(
        {
            'id': ['453108121117357944', '425053662810955002'],
        }
    )

    db = pd.DataFrame(
        {
            'id': [
                '167019600627484111',
                '623049248851670423',
                '453108121117357944',
                '356284488573434088',
                '425053662810955002',
                '828403397565948434',
            ],
            'date': [
                '2008-10-29,',
                '2011-12-14',
                '2008-11-22',
                '2007-03-11',
                '2000-01-07',
                '2003-03-29',
            ],
            'age': ['36', '32', '46', '58', '44', '18'],
        }
    )

    da = sf.to(sf_production_setup_devices.alice, da)
    db = sf.to(sf_production_setup_devices.bob, db)
    yield sf_production_setup_devices, da, db


def test_pir(set_up):
    env, da, db = set_up
    with tempfile.TemporaryDirectory() as data_dir:
        input_path = {
            env.alice: f'{data_dir.name}/alice.csv',
            env.bob: f'{data_dir.name}/bob.csv',
        }

        sf.reveal(
            env.alice(lambda df, save_path: df.to_csv(save_path, index=False))(
                da, input_path[env.alice]
            )
        )
        sf.reveal(
            env.bob(lambda df, save_path: df.to_csv(save_path, index=False))(
                db, input_path[env.bob]
            )
        )
        key_columns = ['id']
        label_columns = ['date', 'age']

        oprf_key = "000102030405060708090a0b0c0d0e0ff0e0d0c0b0a090807060504030201000"
        oprf_key_path = f'{data_dir.name}/oprf_key.bin'
        with open(oprf_key_path, 'wb') as f:
            f.write(bytes.fromhex(oprf_key))

        setup_path = f'{data_dir.name}/setup_dir'

        env.spu.pir_setup(
            server='bob',
            input_path=input_path[env.bob],
            key_columns=key_columns,
            label_columns=label_columns,
            oprf_key_path=oprf_key_path,
            setup_path=setup_path,
            num_per_query=1,
            label_max_len=20,
        )

        pir_result_path = f'{data_dir.name}/pir_out.csv'

        alice_config = {
            'input_path': input_path[env.alice],
            'key_columns': key_columns,
            'output_path': pir_result_path,
        }

        bob_config = {
            'oprf_key_path': oprf_key_path,
            'setup_path': setup_path,
        }

        query_config = {
            env.alice: alice_config,
            env.bob: bob_config,
        }

        env.spu.pir_query(
            server='bob',
            config=query_config,
        )

        result_a = pd.DataFrame(
            {
                'id': [453108121117357944, 425053662810955002],
                'date': ['2008-11-22', '2000-01-07'],
                'age': [46, 44],
            }
        )
        pd.testing.assert_frame_equal(
            sf.reveal(env.alice(pd.read_csv)(pir_result_path)), result_a
        )
