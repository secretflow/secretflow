# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from secretflow.component.component import Component, IoType
from secretflow.data.vertical import read_csv
from secretflow.device.device.pyu import PYU
from secretflow.device.device.spu import SPU
from secretflow.device.driver import wait
from secretflow.ml.linear import LinearModel, SSRegression
from secretflow.protos.component.comp_def_pb2 import TableType

ss_sgd_train_comp = Component(
    "ss_sgd_train",
    domain="ml.linear",
    version="0.0.1",
    desc="""This method provides both linear and logistic regression
    linear models for vertical split dataset setting by using secret sharing
    with mini batch SGD training solver. SS-SGD is short for secret sharing SGD training.
    """,
)
ss_sgd_train_comp.int_param(
    name="epochs",
    desc="iteration rounds.",
    is_list=False,
    is_optional=True,
    default_value=1,
    allowed_values=None,
    lower_bound=1,
    upper_bound=None,
    lower_bound_inclusive=True,
)
ss_sgd_train_comp.float_param(
    name="learning_rate",
    desc="controls how much to change the model in one epoch.",
    is_list=False,
    is_optional=True,
    default_value=0.1,
)
ss_sgd_train_comp.int_param(
    name="batch_size",
    desc="how many samples use in one calculation.",
    is_list=False,
    is_optional=True,
    default_value=1024,
)
ss_sgd_train_comp.str_param(
    name="sig_type",
    desc="sigmoid approximation type.",
    is_list=False,
    is_optional=True,
    default_value="t1",
    allowed_values=["real", "t1", "t3", "t5", "df", "sr", "mix"],
)
ss_sgd_train_comp.str_param(
    name="reg_type",
    desc="Linear or Logistic regression",
    is_list=False,
    is_optional=True,
    default_value="logistic",
    allowed_values=["linear", "logistic"],
)
ss_sgd_train_comp.str_param(
    name="penalty",
    desc="The penalty (aka regularization term) to be used.",
    is_list=False,
    is_optional=True,
    default_value="None",
    allowed_values=["None", "l1", "l2"],
)
ss_sgd_train_comp.float_param(
    name="l2_norm",
    desc="L2 regularization term.",
    is_list=False,
    is_optional=True,
    default_value=0.5,
)
ss_sgd_train_comp.float_param(
    name="eps",
    desc="""If the W's change rate is less than this threshold,
            the model is considered to be converged,
            and the training stops early. 0 disable.""",
    is_list=False,
    is_optional=True,
    default_value=0.001,
)
ss_sgd_train_comp.table_io(
    io_type=IoType.INPUT,
    name="x",
    desc="features",
    types=[TableType.VERTICAL_PARTITIONING_TABLE],
    col_params=None,
)
ss_sgd_train_comp.table_io(
    io_type=IoType.INPUT,
    name="y",
    desc="label",
    types=[TableType.VERTICAL_PARTITIONING_TABLE],
    col_params=None,
)
ss_sgd_train_comp.model_io(
    io_type=IoType.OUTPUT, name="output", desc="output", types=["ss_sgb"]
)


@ss_sgd_train_comp.eval_fn
def ss_sgd_train_eval_fn(
    *,
    ctx,
    epochs,
    learning_rate,
    batch_size,
    sig_type,
    reg_type,
    penalty,
    l2_norm,
    eps,
    x,
    y,
    output,
):
    """This method provides both linear and logistic regression
    linear models for vertical split dataset setting by using secret sharing
    with mini batch SGD training solver. SS-SGD is short for secret sharing SGD training.
    """

    x_parties = x.table_metadata.vertical_partitioning.parties
    x_paths = x.table_metadata.vertical_partitioning.paths

    y_parties = y.table_metadata.vertical_partitioning.parties
    y_paths = y.table_metadata.vertical_partitioning.paths

    model_public_path = output.model_metadata.public_file_path
    model_parties = output.model_metadata.parties
    model_party_paths = output.model_metadata.party_dir_paths

    spu = SPU(
        ctx['spu'],
        link_desc={
            'connect_retry_times': 60,
            'connect_retry_interval_ms': 1000,
            'brpc_channel_protocol': "http",
            "brpc_channel_connection_type": "pooled",
            'recv_timeout_ms': 1200 * 1000,  # 1200s
            'http_timeout_ms': 1200 * 1000,  # 1200s
        },
    )

    reg = SSRegression(spu)
    pyus = {k: PYU(k) for k in ctx['pyu']}

    x_filepath = {pyus[k]: p for k, p in zip(x_parties, x_paths)}
    y_filepath = {pyus[k]: p for k, p in zip(y_parties, y_paths)}

    x = read_csv(x_filepath, no_header=True)
    y = read_csv(y_filepath, no_header=True)

    reg.fit(
        x=x,
        y=y,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        sig_type=sig_type,
        reg_type=reg_type,
        penalty=penalty,
        l2_norm=l2_norm,
        eps=eps,
    )

    model = reg.save_model()

    dir_path = {k: v for k, v in zip(model_parties, model_party_paths)}

    record = model.dump(dir_path=dir_path)

    with open(model_public_path, 'wb') as f:
        import cloudpickle as pickle

        pickle.dump(record, f)


ss_sgd_predict_comp = Component(
    "ss_sgd_predict",
    domain="ml.linear",
    version="0.0.1",
    desc="Predict using the model.",
)
ss_sgd_predict_comp.int_param(
    name="batch_size",
    desc="how many samples use in one calculation.",
    is_list=False,
    is_optional=True,
    default_value=1024,
)
ss_sgd_predict_comp.table_io(
    io_type=IoType.INPUT,
    name="x",
    desc="features",
    types=[TableType.VERTICAL_PARTITIONING_TABLE],
    col_params=None,
)
ss_sgd_predict_comp.model_io(
    io_type=IoType.INPUT, name="model", desc="model", types=["ss_sgb"]
)
ss_sgd_predict_comp.table_io(
    io_type=IoType.OUTPUT,
    name="y",
    desc="label",
    types=[TableType.INDIVIDUAL_TABLE],
    col_params=None,
)


@ss_sgd_predict_comp.eval_fn
def ss_sgd_predict_eval_fn(*, ctx, batch_size, x, model, y):
    x_parties = x.table_metadata.vertical_partitioning.parties
    x_paths = x.table_metadata.vertical_partitioning.paths

    model_public_path = model.model_metadata.public_file_path

    y_party = y.table_metadata.indivial.party
    y_path = y.table_metadata.indivial.path

    spu = SPU(
        ctx['spu'],
        link_desc={
            'connect_retry_times': 60,
            'connect_retry_interval_ms': 1000,
            'brpc_channel_protocol': "http",
            "brpc_channel_connection_type": "pooled",
            'recv_timeout_ms': 1200 * 1000,  # 1200s
            'http_timeout_ms': 1200 * 1000,  # 1200s
        },
    )
    pyus = {k: PYU(k) for k in ctx['pyu']}

    with open(model_public_path, 'rb') as f:
        import cloudpickle as pickle

        record = pickle.load(f)

    model = LinearModel.load(record=record, spu=spu)
    x_filepath = {pyus[k]: p for k, p in zip(x_parties, x_paths)}
    x = read_csv(x_filepath, no_header=True)

    reg = SSRegression(spu)
    reg.load_model(model)

    pyu = pyus[y_party]
    y = reg.predict(
        x=x,
        batch_size=batch_size,
        to_pyu=pyu,
    )

    def save_csv(x, path):
        import numpy

        numpy.savetxt(path, x, delimiter=",")

    wait(pyu(save_csv)(y.partitions[pyu], y_path))
