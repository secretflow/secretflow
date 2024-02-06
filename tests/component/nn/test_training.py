from dataclasses import dataclass
from tempfile import mkdtemp

import pandas as pd
import pytest
import tensorflow as tf

import secretflow as sf
from secretflow.component.component import CompEvalContext
from secretflow.component.ml.nn.sl.compile import compile
from secretflow.component.ml.nn.sl.training import predictor, saver, trainer
from secretflow.component.ml.nn.sl.training.tensorflow import data, model
from secretflow.data import partition
from secretflow.data.vertical import VDataFrame

from .model_def import (
    LOSS_CLASS_CODE,
    LOSS_FUNC_CODE,
    MODELS_CODE,
    SUBCLASS_MULTIINPUT_MODEL_CODE,
)


@pytest.fixture(scope="function")
def mock_train_data(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob

    # fmt: off
    data = pd.DataFrame(
        data=[
            [0.015, 0.727, 1., 0.333, 0., 0.051, 0., 0., 0., 0.967, 0., 0.085, 0., 0., 0., 1., 1.],
            [0.235, 0.364, 1., 0.667, 0., 0.054, 0., 0., 0., 0.733, 0.273, 0.045, 0.02, 0.203, 0.12, 0., 0.],
            [0.191, 0.364, 1., 0.667, 0., 0.079, 1., 0., 0., 0.833, 0.091, 0.316, 0.102, 0., 0., 1., 1.],
            [0.25, 0.545, 0.5, 0.667, 0., 0.049, 1., 0., 0., 0.433, 0.727, 0.112, 0., 0.38, 0.08, 0.333, 0.],
            [0.294, 0.818, 0.5, 0.333, 0., 0.046, 1., 0., 0., 0.167, 0.727, 0.049, 0.02, 0., 0., 1., 0.],
            [0.441, 0.818, 0.5, 0.667, 0., 0.061, 0., 0., 0., 0.4, 0.091, 0.116, 0.041, 0., 0., 1., 1.],
            [0.191, 0.091, 0.5, 0.333, 0., 0.073, 1., 0., 0., 0.433, 0.818, 0.042, 0., 0., 0., 1., 1.],
            [0.206, 0.364, 0.5, 0.333, 0., 0.097, 1., 0., 0., 0.167, 0.727, 0.252, 0., 0.393, 0.08, 0., 1.],
            [0.618, 0., 0.5, 1., 0., 0.107, 1., 0., 0., 0.867, 0.364, 0.059, 0., 0.107, 0.04, 0.667, 1.],
            [0.162, 0.909, 0.5, 0., 0., 0.068, 0., 0., 0., 0.6, 0.909, 0.025, 0., 0., 0., 1., 0.],
            [0.206, 0.636, 0.5, 0.333, 0., 0.109, 1., 1., 0., 0.333, 0.727, 0.071, 0., 0.39, 0.16, 0., 0.],
            [0.235, 0.364, 1., 0.667, 0., 0.063, 1., 0., 0., 0.5, 0., 0.06, 0., 0.38, 0.04, 0., 0.],
            [0.162, 0.364, 0.5, 0.667, 0., 0.064, 1., 1., 1., 0.067, 0.545, 0.065, 0.061, 0., 0., 1., 0.],
            [0.721, 0.455, 0., 0.333, 0., 0.101, 0., 0., 0.5, 0.433, 0.455, 0.296, 0.02, 0., 0., 1., 1.],
            [0.868, 0.455, 0., 0., 0., 0.048, 0., 0., 0.5, 0.7, 0.909, 0.031, 0., 0., 0., 1., 1.],
            [0.588, 0.091, 0.5, 0.333, 0., 0.044, 1., 0., 1., 0.133, 0.727, 0.073, 0., 0., 0., 1., 0.],
        ],
        columns=[
            "age", "job", "marital", "education", "default", "balance", "housing", "loan",
            "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y",
        ],
    )
    # fmt: on

    alice_df = data[["age", "job", "marital", "education"]]

    bob_df = data[
        [
            "default",
            "balance",
            "housing",
            "loan",
            "contact",
            "day",
            "month",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "poutcome",
        ]
    ]

    y_df = data[["y"]]

    x = VDataFrame(
        partitions={
            alice: partition(alice(lambda x: x)(alice_df)),
            bob: partition(bob(lambda x: x)(bob_df)),
        }
    )
    y = VDataFrame(partitions={alice: partition(alice(lambda x: x)(y_df))})

    yield type("obj", (), {"x": x, "y": y})


def test_create_dataset_builder(sf_simulation_setup_devices):
    dataset_builder = data.create_dataset_builder([], model_input_scheme="tensor")
    assert dataset_builder is None

    dataset_builder = data.create_dataset_builder(
        pyus=[sf_simulation_setup_devices.alice, sf_simulation_setup_devices.bob],
        label_pyu=sf_simulation_setup_devices.alice,
        label_name="y",
        batch_size=2,
        epochs=2,
        model_input_scheme="tensor_dict",
    )

    assert len(dataset_builder) == 2

    df_alice = pd.DataFrame(
        {
            "a1": ["K5", "K1", "", "K6"],
            "a2": ["A5", "A1", "A2", "A6"],
            "a3": [5, 1, 2, 6],
            "y": [1, 2, 3, 4],
        }
    )

    df_bob = pd.DataFrame(
        {
            "b4": [10.2, 20.5, 0, -0.4],
            "b5": ["B3", "", "B9", "B4"],
            "b6": [3, 1, 9, 4],
        }
    )

    alice_builder = dataset_builder[sf_simulation_setup_devices.alice]
    bob_builder = dataset_builder[sf_simulation_setup_devices.bob]

    ds_alice = alice_builder([df_alice[["a1", "a2", "a3"]], df_alice[["y"]]])
    ds_bob = bob_builder([df_bob])

    assert ds_alice is not None and ds_bob is not None

    batch_alice, batch_y = next(iter(ds_alice))
    batch_bob = next(iter(ds_bob))

    assert len(batch_alice) == 3 and len(batch_bob) == 3
    assert len(batch_y) == 2


def test_create_model_builder(sf_simulation_setup_devices):
    configs = compile.compile_by_initiator(
        parties=[sf_simulation_setup_devices.alice, sf_simulation_setup_devices.bob],
        initiator=sf_simulation_setup_devices.alice,
        models=MODELS_CODE,
        learning_rate=0.01,
        loss="",
        custom_loss=LOSS_CLASS_CODE,
        optimizer="adam",
        optimizer_params="{}",
        metrics=["AUC", "acc"],
    )

    alice_config = configs[sf_simulation_setup_devices.alice]
    client_base_builder = model.create_model_builder(
        alice_config.client_base_path, alice_config
    )
    bob_config = configs[sf_simulation_setup_devices.bob]
    server_base_builder = model.create_model_builder(
        bob_config.server_base_path, bob_config
    )
    server_fuse_builder = model.create_model_builder(
        bob_config.server_fuse_path, bob_config
    )

    client_base = client_base_builder()
    server_base = server_base_builder()
    server_fuse = server_fuse_builder()

    assert (
        isinstance(client_base, tf.keras.Model)
        and isinstance(server_base, tf.keras.Model)
        and isinstance(server_fuse, tf.keras.Model)
    )
    print(configs)

    assert client_base.built and server_base.built and server_fuse.built

    client_base.summary()
    server_base.summary()
    server_fuse.summary()


def test_compile_by_self(sf_simulation_setup_devices):
    configs = compile.compile_by_self(
        parties=[sf_simulation_setup_devices.alice, sf_simulation_setup_devices.bob],
        models=MODELS_CODE,
        learning_rate=0.01,
        loss="",
        custom_loss=LOSS_FUNC_CODE,
        optimizer="adam",
        optimizer_params="",
        metrics=["AUC", "acc"],
    )

    assert (
        configs[sf_simulation_setup_devices.alice] is not None
        and configs[sf_simulation_setup_devices.bob] is not None
    )

    alice_config = configs[sf_simulation_setup_devices.alice]
    bob_config = configs[sf_simulation_setup_devices.bob]

    assert alice_config.loss_config is None and bob_config.loss_config is None

    alice_config.loss_config = "mock"
    bob_config.loss_config = "mock"
    assert all(vars(alice_config).values()) and all(vars(bob_config).values())


def test_compile_by_initiator(sf_simulation_setup_devices):
    configs = compile.compile_by_initiator(
        parties=[sf_simulation_setup_devices.alice, sf_simulation_setup_devices.bob],
        initiator=sf_simulation_setup_devices.alice,
        models=MODELS_CODE,
        learning_rate=0.01,
        loss="",
        custom_loss=LOSS_CLASS_CODE,
        optimizer="adam",
        optimizer_params="{}",
        metrics=["AUC", "acc"],
    )

    assert (
        configs[sf_simulation_setup_devices.alice] is not None
        and configs[sf_simulation_setup_devices.bob] is not None
    )

    alice_config = configs[sf_simulation_setup_devices.alice]
    bob_config = configs[sf_simulation_setup_devices.bob]

    assert alice_config.loss_config is None and bob_config.loss_config is None

    alice_config.loss_config = "mock"
    bob_config.loss_config = "mock"
    assert all(vars(alice_config).values()) and all(vars(bob_config).values())


@dataclass
class Params:
    loss: str
    custom_loss: str
    models: str
    model_input_scheme: str
    strategy: str
    strategy_params: str
    compressor: str
    compressor_params: str


@pytest.mark.parametrize(
    "params",
    [
        Params(
            "binary_crossentropy", "", MODELS_CODE, "tensor", "split_nn", "", "", ""
        ),
        Params(
            "",
            LOSS_FUNC_CODE,
            SUBCLASS_MULTIINPUT_MODEL_CODE,
            "tensor_dict",
            "pipeline",
            '{"pipeline_size": 2}',
            "mixed_compressor",
            """
            [
                {"name": "quantized_fp", "params": {"quant_bits": 16}},
                {"name": "topk_sparse", "params": {"sparse_rate": 0.1}}
            ]
            """,
        ),
    ],
    ids=["simple", "complex"],
)
def test_fit_save_load_predict(
    sf_simulation_setup_devices, mock_train_data, params: Params
):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob

    slmodel, history, model_configs = trainer.fit(
        ctx=CompEvalContext(initiator_party=str(alice)),
        x=mock_train_data.x,
        y=mock_train_data.y,
        val_x=mock_train_data.x,
        val_y=mock_train_data.y,
        models=params.models,
        epochs=5,
        learning_rate=0.01,
        batch_size=4,
        loss=params.loss,
        custom_loss=params.custom_loss,
        optimizer="adam",
        optimizer_params="",
        metrics=["AUC"],
        model_input_scheme=params.model_input_scheme,
        strategy=params.strategy,
        strategy_params=params.strategy_params,
        compressor=params.compressor,
        compressor_params=params.compressor_params,
    )

    print(history)

    assert history["val_auc_1"][-1] > 0.8

    tmpdirs = {
        alice: mkdtemp(),
        bob: mkdtemp(),
    }

    contents, metas = saver.save(
        slmodel=slmodel,
        label_pyu=alice,
        model_configs=model_configs,
        tmpdirs=tmpdirs,
    )

    assert len(contents) == 2 and len(metas) == 2

    real_contents = sf.reveal(contents)
    for c in real_contents:
        assert len(c) > 10000
    for meta in metas:
        assert meta["base"] is True

    tmpdirs2 = {
        alice: mkdtemp(),
        bob: mkdtemp(),
    }

    model_configs = saver.load(contents, metas, tmpdirs2)

    print(model_configs)

    assert (
        model_configs[alice].server_fuse_path is not None
        and model_configs[alice].server_base_path is not None
    )
    assert model_configs[bob].client_base_path is not None

    y_pred = predictor.predict(
        ctx=None,
        batch_size=2,
        feature_dataset=mock_train_data.x,
        model=model_configs,
        model_input_scheme=params.model_input_scheme,
    )

    y_pred_plain = sf.reveal(y_pred.partitions[alice])
    y_plain = sf.reveal(mock_train_data.y.partitions[alice].data).to_numpy()

    assert len(y_pred_plain) == len(y_plain)

    print(y_plain)
    print(y_pred_plain)

    acc = tf.reduce_mean(tf.metrics.binary_accuracy(y_plain, y_pred_plain))
    print(acc)

    assert acc > 0.6
