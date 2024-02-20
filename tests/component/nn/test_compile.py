import shutil
from pathlib import Path
from tempfile import mkdtemp

import tensorflow as tf
from tensorflow.python.trackable import autotrackable

from secretflow.component.ml.nn.sl.compile import compile
from secretflow.component.ml.nn.sl.compile.tensorflow import (
    loss,
    metric,
    model,
    optimizer,
)

from .model_def import LOSS_CLASS_CODE, LOSS_FUNC_CODE, MODELS_CODE


def test_compile_and_load_models():
    tmpdir = Path(mkdtemp())
    config = compile.build_model_paths(tmpdir)

    model.compile_models(
        MODELS_CODE,
        config.server_fuse_path,
        config.server_base_path,
        config.client_base_path,
    )

    assert (
        config.server_fuse_path.joinpath("saved_model.pb").is_file()
        and config.server_base_path.joinpath("saved_model.pb").is_file()
        and config.client_base_path.joinpath("saved_model.pb").is_file()
    )

    server_fuse, server_base, client_base = model.load_models(
        config.server_fuse_path, config.server_base_path, config.client_base_path
    )

    assert (
        isinstance(server_fuse, autotrackable.AutoTrackable)
        and isinstance(server_base, autotrackable.AutoTrackable)
        and isinstance(client_base, autotrackable.AutoTrackable)
    )

    shutil.rmtree(tmpdir)


def test_compile_and_load_loss_builtin():
    loss_func = loss.compile_loss("categorical_crossentropy")
    assert loss_func == "categorical_crossentropy"

    loss_func = loss.compile_loss("CategoricalCrossentropy")
    assert loss_func == "CategoricalCrossentropy"


def test_compile_and_load_loss_custom():
    tmpdir = Path(mkdtemp())

    # function custom loss
    config = compile.build_model_paths(tmpdir.joinpath("loss1"))
    loss.compile_loss(custom_code=LOSS_FUNC_CODE, loss_path=config.loss_path)
    assert config.loss_path.joinpath("saved_model.pb").is_file()

    loaded_loss = loss.load_loss(config.loss_path)
    assert isinstance(loaded_loss, autotrackable.AutoTrackable)

    # class custom loss
    config = compile.build_model_paths(tmpdir.joinpath("loss2"))
    loss.compile_loss(custom_code=LOSS_CLASS_CODE, loss_path=config.loss_path)
    assert config.loss_path.joinpath("saved_model.pb").is_file()

    loaded_loss = loss.load_loss(config.loss_path)
    assert isinstance(loaded_loss, autotrackable.AutoTrackable)

    shutil.rmtree(tmpdir)


def test_compile_metircs():
    metrics = metric.compile_metircs(["AUC", "acc"])
    assert metrics == ["AUC", "acc"]

    metric_objs = metric.get_metrics(metrics)
    for obj in metric_objs:
        assert callable(obj)


def test_compile_optimizer():
    opt = optimizer.compile_optimizer("adam", '{"beta_1": 0.95}', 0.1)

    assert (
        isinstance(opt, dict)
        and opt["class_name"] == "adam"
        and "beta_1" in opt["config"]
        and "learning_rate" in opt["config"]
    )

    opt_obj = optimizer.get_optimizer(opt)

    assert isinstance(opt_obj, tf.keras.optimizers.experimental.Adam)


def test_compile_all_safe_only():
    tmpdir = Path(mkdtemp())
    config = compile.do_compile_all(
        MODELS_CODE,
        0.1,
        None,
        LOSS_FUNC_CODE,
        "adam",
        "",
        ["AUC", "acc"],
        tmpdir,
        safe_only=True,
    )

    assert (
        config.client_base_path is None
        and config.server_base_path is None
        and config.server_fuse_path is None
        and config.loss_path is None
    )

    assert config.optimizer_config is not None and config.metrics_config is not None

    shutil.rmtree(tmpdir)


def test_compile_all():
    tmpdir = Path(mkdtemp())
    config = compile.do_compile_all(
        MODELS_CODE,
        0.1,
        None,
        LOSS_FUNC_CODE,
        "adam",
        "",
        ["AUC", "acc"],
        tmpdir,
        safe_only=False,
    )

    assert (
        config.client_base_path.joinpath("saved_model.pb").is_file()
        and config.server_base_path.joinpath("saved_model.pb").is_file()
        and config.server_fuse_path.joinpath("saved_model.pb").is_file()
        and config.loss_path.joinpath("saved_model.pb").is_file()
    )

    assert config.optimizer_config is not None and config.metrics_config is not None

    shutil.rmtree(tmpdir)
