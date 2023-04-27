import os
import shutil

import pandas as pd
import pytest
import spu

import secretflow as sf
import tempfile


def set_up(devices):
    da = pd.DataFrame(
        {
            'c1': ['K5', 'K1', 'K2', 'K6', 'K4', 'K3'],
            'c2': ['A5', 'A1', 'A2', 'A6', 'A4', 'A3'],
            'c3': [5, 1, 2, 6, 4, 3],
        }
    )

    db = pd.DataFrame(
        {
            'c1': ['K3', 'K1', 'K9', 'K4'],
            'c2': ['B3', 'A1', 'A9', 'A4'],
            'c3': [3, 1, 9, 4],
        }
    )

    db2 = pd.DataFrame(
        {
            'c1': ['K3', 'K1', 'K1', 'K4'],
            'c2': ['B3', 'A1', 'A1', 'A4'],
            'c3': ['C3', 'C1', 'D1', 'C4'],
            'c4': [3, 1, 9, 4],
        }
    )

    db3 = pd.DataFrame(
        {'c1': ['K7', 'K8', 'K9'], 'c2': ['A7', 'A8', 'A9'], 'c3': [7, 8, 9]}
    )

    db4 = pd.DataFrame(
        {
            'c11': ['K3', 'K1', 'K9', 'K4'],
            'c21': ['B3', 'A1', 'A9', 'A4'],
            'c31': [3, 1, 9, 4],
        }
    )

    da_aby3 = pd.DataFrame(
        {
            'c1': ['K5', 'K1', 'K2', 'K6', 'K4', 'K3'],
            'c2': ['A5', 'A1', 'A2', 'A6', 'B4', 'A3'],
            'c3': [5, 1, 2, 6, 4, 3],
        }
    )

    db_aby3 = pd.DataFrame(
        {
            'c1': ['K3', 'K1', 'K9', 'K4'],
            'c2': ['B3', 'A1', 'A9', 'A4'],
            'c3': [3, 1, 9, 4],
        }
    )

    dc_aby3 = pd.DataFrame(
        {
            'c1': ['K9', 'K4', 'K3', 'K1', 'k8'],
            'c2': ['A9', 'B4', 'B3', 'A1', 'k8'],
            'c3': [9, 4, 3, 1, 8],
        }
    )

    da_new = pd.DataFrame(
        {
            'id1': ['K100', 'K200', 'K200', 'K300', 'K400', 'K400', 'K500'],
            'item': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            'feature1': ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'],
        }
    )

    db_new = pd.DataFrame(
        {
            'id2': ['K200', 'K300', 'K400', 'K500', 'K600', 'K700'],
            'feature2': ['AA', 'BB', 'CC', 'DD', 'EE', 'FF'],
        }
    )

    data = {}

    data['da'] = sf.to(devices.alice, da)
    data['db'] = sf.to(devices.bob, db)
    data['db2'] = sf.to(devices.bob, db2)
    data['db3'] = sf.to(devices.bob, db3)
    data['dc'] = sf.to(devices.carol, db)
    data['db4'] = sf.to(devices.bob, db4)
    data['da_new'] = sf.to(devices.alice, da_new)
    data['db_new'] = sf.to(devices.bob, db_new)
    data['da_aby3'] = sf.to(devices.alice, da_aby3)
    data['db_aby3'] = sf.to(devices.bob, db_aby3)
    data['dc_aby3'] = sf.to(devices.carol, dc_aby3)

    return data


@pytest.fixture(scope='function')
def prod_env_and_model(sf_production_setup_devices):
    data = set_up(sf_production_setup_devices)
    yield sf_production_setup_devices, data


@pytest.fixture(scope='function')
def sim_env_and_model(sf_simulation_setup_devices):
    data = set_up(sf_simulation_setup_devices)
    yield sf_simulation_setup_devices, data


def _test_single_col(devices, data):
    da, db = devices.spu.psi_df('c1', [data['da'], data['db']], 'alice')

    expected = pd.DataFrame(
        {'c1': ['K1', 'K3', 'K4'], 'c2': ['A1', 'A3', 'A4'], 'c3': [1, 3, 4]}
    )
    pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)

    expected = pd.DataFrame(
        {'c1': ['K1', 'K3', 'K4'], 'c2': ['A1', 'B3', 'A4'], 'c3': [1, 3, 4]}
    )
    pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected)


def test_single_col_prod(prod_env_and_model):
    devices, data = prod_env_and_model
    _test_single_col(devices, data)


def test_single_col_sim(sim_env_and_model):
    devices, data = sim_env_and_model
    _test_single_col(devices, data)


def _test_multiple_col(devices, data):
    da, db = devices.spu.psi_df(['c1', 'c2'], [data['da'], data['db']], 'alice')

    expected = pd.DataFrame({'c1': ['K1', 'K4'], 'c2': ['A1', 'A4'], 'c3': [1, 4]})
    pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)
    pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected)


def test_multiple_col_prod(prod_env_and_model):
    devices, data = prod_env_and_model
    _test_multiple_col(devices, data)


def test_multiple_col_sim(sim_env_and_model):
    devices, data = sim_env_and_model
    _test_multiple_col(devices, data)


def _test_different_cols(devices, data):
    da, db = devices.spu.psi_df(
        {devices.alice: ['c1', 'c2'], devices.bob: ['c11', 'c21']},
        [data['da'], data['db4']],
        'alice',
    )

    expected_a = pd.DataFrame({'c1': ['K1', 'K4'], 'c2': ['A1', 'A4'], 'c3': [1, 4]})
    expected_b = pd.DataFrame({'c11': ['K1', 'K4'], 'c21': ['A1', 'A4'], 'c31': [1, 4]})
    pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected_a)
    pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected_b)


def test_different_cols_prod(prod_env_and_model):
    devices, data = prod_env_and_model
    _test_different_cols(devices, data)


def test_different_cols_sim(sim_env_and_model):
    devices, data = sim_env_and_model
    _test_different_cols(devices, data)


def _test_invalid_device(devices, data):
    with pytest.raises(AssertionError, match='not co-located'):
        da, dc = devices.spu.psi_df(['c1', 'c2'], [data['da'], data['dc']], 'alice')
        sf.reveal([da, dc])


def test_invalid_device_prod(prod_env_and_model):
    devices, data = prod_env_and_model
    _test_invalid_device(devices, data)


def test_invalid_device_sim(sim_env_and_model):
    devices, data = sim_env_and_model
    _test_invalid_device(devices, data)


# @unittest.skip('spu reset not works now FIXME @raofei')
# def duplicate_col(self):
#     with self.assertRaisesRegex(RuntimeError, 'found duplicated keys'):
#         da, db = self.spu.psi_df('c1', [self.da, self.db2], 'alice')
#         sf.reveal([da, db])

#     # reset spu to clear corrupted state
#     self.spu.reset()


# @unittest.skip('spu reset not works now FIXME @raofei')
# def missing_col(self):
#     with self.assertRaisesRegex(RuntimeError, "can't find feature names 'c4'"):
#         da, db = self.spu.psi_df(['c1', 'c4'], [self.da, self.db2], 'alice')
#         sf.reveal([da, db])

#     # reset spu to clear corrupted state
#     self.spu.reset()


def _test_no_intersection(devices, data):
    da, db = devices.spu.psi_df('c1', [data['da'], data['db3']], 'alice')
    expected = pd.DataFrame({'c1': [], 'c2': [], 'c3': []}).astype('object')
    pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)
    pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected)


def test_no_intersection_prod(prod_env_and_model):
    devices, data = prod_env_and_model
    _test_no_intersection(devices, data)


def test_no_intersection_sim(sim_env_and_model):
    devices, data = sim_env_and_model
    _test_no_intersection(devices, data)


def _test_no_broadcast(devices, data):
    # only alice can get result
    da, db = devices.spu.psi_df(
        'c1', [data['da'], data['db']], 'alice', 'KKRT_PSI_2PC', False, True, False
    )
    expected = pd.DataFrame(
        {'c1': ['K1', 'K3', 'K4'], 'c2': ['A1', 'A3', 'A4'], 'c3': [1, 3, 4]}
    )
    pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)
    # bob can not get result
    assert sf.reveal(db) is None


def test_no_broadcast_prod(prod_env_and_model):
    devices, data = prod_env_and_model
    _test_no_broadcast(devices, data)


def test_no_broadcast_sim(sim_env_and_model):
    devices, data = sim_env_and_model
    _test_no_broadcast(devices, data)


def _test_psi_csv(devices, data):
    with tempfile.TemporaryDirectory() as data_dir:
        input_path = {
            devices.alice: f'{data_dir.name}/alice.csv',
            devices.bob: f'{data_dir.name}/bob.csv',
        }
        output_path = {
            devices.alice: f'{data_dir.name}/alice_psi.csv',
            devices.bob: f'{data_dir.name}/bob_psi.csv',
        }

        sf.reveal(
            devices.alice(lambda df, save_path: df.to_csv(save_path, index=False))(
                data['da'], input_path[devices.alice]
            )
        )
        sf.reveal(
            devices.bob(lambda df, save_path: df.to_csv(save_path, index=False))(
                data['db'], input_path[devices.bob]
            )
        )

        devices.spu.psi_csv(['c1', 'c2'], input_path, output_path, 'alice')

        expected = pd.DataFrame({'c1': ['K1', 'K4'], 'c2': ['A1', 'A4'], 'c3': [1, 4]})

        pd.testing.assert_frame_equal(
            sf.reveal(devices.alice(pd.read_csv)(output_path[devices.alice])), expected
        )
        pd.testing.assert_frame_equal(
            sf.reveal(devices.bob(pd.read_csv)(output_path[devices.bob])), expected
        )


def test_psi_csv_prod(prod_env_and_model):
    devices, data = prod_env_and_model
    _test_psi_csv(devices, data)


def test_psi_csv_sim(sim_env_and_model):
    devices, data = sim_env_and_model
    _test_psi_csv(devices, data)


def _test_unbalanced_psi_csv(devices, data):
    with tempfile.TemporaryDirectory() as data_dir:
        input_path = {
            devices.alice: f'{data_dir.name}/alice.csv',
            devices.bob: f'{data_dir.name}/bob.csv',
        }

        sf.reveal(
            devices.alice(lambda df, save_path: df.to_csv(save_path, index=False))(
                data['da'], input_path[devices.alice]
            )
        )
        sf.reveal(
            devices.bob(lambda df, save_path: df.to_csv(save_path, index=False))(
                data['db'], input_path[devices.bob]
            )
        )

        # write secretkey.bin
        secret_key = "000102030405060708090a0b0c0d0e0ff0e0d0c0b0a090807060504030201000"
        secret_key_path = f'{data_dir.name}/secret_key.bin'
        with open(secret_key_path, 'wb') as f:
            f.write(bytes.fromhex(secret_key))

        offline_input_path = {
            devices.alice: 'fake.csv',
            devices.bob: f'{data_dir.name}/bob.csv',
        }

        server_party_name = 'bob'
        client_party_name = 'alice'

        offline_preprocess_path = f'{data_dir.name}/offline_preprocess_data.csv'

        # gen cache
        print("=====gen cache phase====")
        # gen cache phase
        gen_cache_config = spu.psi.BucketPsiConfig(
            input_params=spu.psi.InputParams(
                path=f'{data_dir.name}/bob.csv', select_fields=['c1', 'c2'], precheck=False
            ),
            output_params=spu.psi.OutputParams(
                path=f'{data_dir.name}/server_cache.bin', need_sort=False
            ),
            curve_type="CURVE_FOURQ",
            bucket_size=1000000,
            ecdh_secret_key_path=secret_key_path,
        )

        gen_cache_report = devices.bob(spu.psi.gen_cache_for_2pc_ub_psi)(gen_cache_config)
        print(f'gen_cache_report={sf.reveal(gen_cache_report)}')

        # transfer cache
        print("=====transfer cache phase====")

        transfer_cache_input_path = f'{data_dir.name}/server_cache.bin'
        transfer_cache_output_path = ''

        # tansfer cache phase
        devices.spu.psi_csv(
            key=[],
            input_path=transfer_cache_input_path,  # client no input, server input cache file path
            output_path=transfer_cache_output_path,  # client and server both no output
            receiver=client_party_name,  # client(small set data party) name
            protocol='ECDH_OPRF_UB_PSI_2PC_TRANSFER_CACHE',  # psi protocol
            precheck_input=False,  # will cost ext time if set True
            sort=False,  # set false, not used in tansfer_cache phase
            broadcast_result=False,  # set false, not used in tansfer_cache phase
            bucket_size=1000000,  # not used
            curve_type="CURVE_FOURQ",
            preprocess_path=offline_preprocess_path,  # output client cache
        )

        # shuffle online
        print("=====shuffle online phase====")

        shuffle_online_input_path = {
            devices.alice: f'{data_dir.name}/alice.csv',
            devices.bob: f'{data_dir.name}/bob.csv',
        }
        shuffle_online_output_path = {
            devices.alice: '',
            devices.bob: f'{data_dir.name}/bob_psi.csv',
        }
        shuffle_online_preprocess_path = {
            devices.alice: offline_preprocess_path,
            devices.bob: transfer_cache_input_path,
        }

        # shuffle online phase
        devices.spu.psi_csv(
            key=[],
            input_path=shuffle_online_input_path,  # client and server input file path
            output_path=shuffle_online_output_path,  # server got intersection output
            receiver=server_party_name,  # server(large set data party) name
            protocol='ECDH_OPRF_UB_PSI_2PC_SHUFFLE_ONLINE',  # psi protocol
            precheck_input=False,  # will cost ext time if set True
            sort=False,  # will cost ext time if set True
            broadcast_result=False,  # set false, not used in shuffle online subprotocol phase
            bucket_size=10000000,  # used for buffer compare intersection
            curve_type="CURVE_FOURQ",
            preprocess_path=shuffle_online_preprocess_path,  # client and server cache path
            ecdh_secret_key_path=secret_key_path,  # server's private key
        )

        # offline
        print("=====offline phase====")

        offline_output_path = {
            devices.alice: "dummy.csv",
            devices.bob: "dummy.csv",
        }

        # offline phase
        # streaming read server(large set) data, evaluate and send to client
        devices.spu.psi_csv(
            key=['c1', 'c2'],
            input_path=offline_input_path,  # client no input, server large set data input
            output_path=offline_output_path,  # client no output, server no output,
            receiver=client_party_name,  # client(small set data party) name
            protocol='ECDH_OPRF_UB_PSI_2PC_OFFLINE',  # psi protocol
            precheck_input=False,  # will cost ext time if set True
            sort=False,  # set false, not used in offline phase
            broadcast_result=False,  # offline must set broadcast_result False
            bucket_size=1000000,  # pre-read and shuffle data in bucket size
            curve_type="CURVE_FOURQ",
            preprocess_path=offline_preprocess_path,  # output client cache
            ecdh_secret_key_path=secret_key_path,  # server's ecc secret key
        )

        # online
        print("=====online phase====")
        online_input_path = {
            devices.alice: f'{data_dir.name}/alice.csv',
            devices.bob: f'{data_dir.name}/bob.csv',
        }
        online_output_path = {
            devices.alice: f'{data_dir.name}/alice_psi.csv',
            devices.bob: f'{data_dir.name}/bob_psi.csv',
        }

        cache_path = {
            devices.alice: offline_preprocess_path,
            devices.bob: transfer_cache_input_path,
        }

        devices.spu.psi_csv(
            key=['c1', 'c2'],
            input_path=online_input_path,  # client small set data input, server large set data input
            output_path=online_output_path,  # client small set data input, server no input
            receiver=client_party_name,  # if `broadcast_result=False`, only receiver can get output file.
            protocol='ECDH_OPRF_UB_PSI_2PC_ONLINE',  # psi protocol
            precheck_input=False,  # will cost ext time if set True
            sort=True,  # will cost ext time if set True
            broadcast_result=True,  # False, only client get psi result, True, bot get result
            bucket_size=300000000,  # read bucket_size once, and compute intersection
            curve_type="CURVE_FOURQ",
            preprocess_path=cache_path,  # client cache file path
            ecdh_secret_key_path=secret_key_path,  # server's ecc sercret key path
        )

        expected = pd.DataFrame({'c1': ['K1', 'K4'], 'c2': ['A1', 'A4'], 'c3': [1, 4]})

        pd.testing.assert_frame_equal(
            sf.reveal(devices.alice(pd.read_csv)(online_output_path[devices.alice])),
            expected,
        )


def test_unbalanced_psi_csv_prod(prod_env_and_model):
    devices, data = prod_env_and_model
    _test_unbalanced_psi_csv(devices, data)


def test_unbalanced_psi_csv_sim(sim_env_and_model):
    devices, data = sim_env_and_model
    _test_unbalanced_psi_csv(devices, data)


def test_single_col(prod_env_and_model):
    devices, data = prod_env_and_model
    da, db, dc = devices.spu2.psi_df(
        'c1',
        [data['da_aby3'], data['db_aby3'], data['dc_aby3']],
        'alice',
        protocol='ECDH_PSI_3PC',
    )

    expected = pd.DataFrame(
        {'c1': ['K1', 'K3', 'K4'], 'c2': ['A1', 'A3', 'B4'], 'c3': [1, 3, 4]}
    )
    pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)

    expected = pd.DataFrame(
        {'c1': ['K1', 'K3', 'K4'], 'c2': ['A1', 'B3', 'A4'], 'c3': [1, 3, 4]}
    )
    pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected)

    expected = pd.DataFrame(
        {'c1': ['K1', 'K3', 'K4'], 'c2': ['A1', 'B3', 'B4'], 'c3': [1, 3, 4]}
    )
    pd.testing.assert_frame_equal(sf.reveal(dc).reset_index(drop=True), expected)


def test_multiple_col(prod_env_and_model):
    devices, data = prod_env_and_model
    da, db, dc = devices.spu2.psi_df(
        ['c1', 'c2'],
        [data['da_aby3'], data['db_aby3'], data['dc_aby3']],
        protocol='ECDH_PSI_3PC',
        receiver='alice',
    )
    expected = pd.DataFrame({'c1': ['K1'], 'c2': ['A1'], 'c3': [1]})
    pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), expected)
    pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), expected)
    pd.testing.assert_frame_equal(sf.reveal(dc).reset_index(drop=True), expected)


def test_psi_join_df(prod_env_and_model):
    devices, data = prod_env_and_model
    select_keys = {
        devices.alice: ['id1'],
        devices.bob: ['id2'],
    }

    da, db = devices.spu.psi_join_df(
        select_keys, [data['da_new'], data['db_new']], 'bob', 'bob'
    )

    result_a = pd.DataFrame(
        {
            'id1': ['K200', 'K200', 'K300', 'K400', 'K400', 'K500'],
            'item': ['B', 'C', 'D', 'E', 'F', 'G'],
            'feature1': ['BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'],
        }
    )

    result_b = pd.DataFrame(
        {
            'id2': ['K200', 'K200', 'K300', 'K400', 'K400', 'K500'],
            'feature2': ['AA', 'AA', 'BB', 'CC', 'CC', 'DD'],
        }
    )

    pd.testing.assert_frame_equal(sf.reveal(da).reset_index(drop=True), result_a)
    pd.testing.assert_frame_equal(sf.reveal(db).reset_index(drop=True), result_b)


def test_psi_join_csv(prod_env_and_model):
    devices, data = prod_env_and_model
    with tempfile.TemporaryDirectory() as data_dir:
        input_path = {
            devices.alice: f'{data_dir.name}/alice.csv',
            devices.bob: f'{data_dir.name}/bob.csv',
        }
        output_path = {
            devices.alice: f'{data_dir.name}/alice_psi.csv',
            devices.bob: f'{data_dir.name}/bob_psi.csv',
        }

        sf.reveal(
            devices.alice(lambda df, save_path: df.to_csv(save_path, index=False))(
                data['da_new'], input_path[devices.alice]
            )
        )
        sf.reveal(
            devices.bob(lambda df, save_path: df.to_csv(save_path, index=False))(
                data['db_new'], input_path[devices.bob]
            )
        )

        select_keys = {
            devices.alice: ['id1'],
            devices.bob: ['id2'],
        }

        devices.spu.psi_join_csv(select_keys, input_path, output_path, 'alice', 'alice')

        result_a = pd.DataFrame(
            {
                'id1': ['K200', 'K200', 'K300', 'K400', 'K400', 'K500'],
                'item': ['B', 'C', 'D', 'E', 'F', 'G'],
                'feature1': ['BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'],
            }
        )

        result_b = pd.DataFrame(
            {
                'id2': ['K200', 'K200', 'K300', 'K400', 'K400', 'K500'],
                'feature2': ['AA', 'AA', 'BB', 'CC', 'CC', 'DD'],
            }
        )

        def check_df(filename, expect):
            df = pd.read_csv(filename)
            try:
                pd.testing.assert_frame_equal(df, expect)
                return True
            except AssertionError as e:
                print(e)
                return False

        pd.testing.assert_frame_equal(
            sf.reveal(devices.alice(pd.read_csv)(output_path[devices.alice])), result_a
        )
        pd.testing.assert_frame_equal(
            sf.reveal(devices.bob(pd.read_csv)(output_path[devices.bob])), result_b
        )
