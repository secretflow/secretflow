"""split learning of bank marketing dataset

The following codes are demos only. 
It's **NOT for production** due to system security concerns,
please **DO NOT** use it directly in production.
"""

import os
from random import seed

from absl import app, flags

import secretflow as sf
from examples.ml.nn.sl_bank.funcs import create_base_model, create_fuse_model
from secretflow.data.split import train_test_split
from secretflow.data.vertical import read_csv
from secretflow.ml.nn.sl_model import SLModelTF

flags.DEFINE_float("iid_fraction", 1.0, "iid_fraction among clients")
flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_integer("num_clients", 2, "number of clients to partition")
flags.DEFINE_integer("batch_size", 512, "batch size of training")
flags.DEFINE_string("filename", "./data/bank_data_{}_of_{}", "filename template")
flags.DEFINE_integer('seed_number', 1, 'seed number for random')

FLAGS = flags.FLAGS


def main(_):
    os.environ['PYTHONHASHSEED'] = str(1)
    seed(FLAGS.seed_number)

    file_path = '../../../../tests/datasets/bank/vertical/'

    clients_ids = ['client_' + str(i) for i in range(FLAGS.num_clients)]

    sf.init(clients_ids, num_cpus=8, log_to_driver=True)
    device_list = []
    data_dict = {}
    for i in range(FLAGS.num_clients):
        device_list.append(sf.PYU(clients_ids[i]))
        data_dict[device_list[i]] = file_path + 'bank_data_{}_of_{}.csv'.format(
            i, FLAGS.num_clients
        )

    vdf = read_csv(data_dict)
    label = vdf['deposit']
    data = vdf.drop(columns='deposit', inplace=False)

    train_data, test_data = train_test_split(
        data, train_size=0.8, random_state=FLAGS.seed_number
    )
    train_label, test_label = train_test_split(
        label, train_size=0.8, random_state=FLAGS.seed_number
    )

    hidden_size = 64
    base_model_dict = {}  # 用户定义的已编译后的keras model
    dimensions = [len(columns) for columns in train_data.partition_columns.values()]
    for i in range(FLAGS.num_clients):
        input_size = dimensions[i]
        base_model_dict[device_list[i]] = create_base_model(input_size, hidden_size)

    # 定义融合模型
    model_fuse = create_fuse_model(
        input_dim=hidden_size, party_nums=FLAGS.num_clients, output_dim=1
    )

    sl_model = SLModelTF(
        base_model_dict=base_model_dict, device_y=device_list[-1], model_fuse=model_fuse
    )

    sl_model.fit(
        train_data,
        train_label,
        validation_data=(test_data, test_label),
        epochs=20,
        batch_size=128,
        shuffle=True,
        verbose=1,
        validation_freq=1,
    )
    # 创建base模型
    global_metric = sl_model.evaluate(test_data, test_label, batch_size=len(test_data))
    print(global_metric)


if __name__ == '__main__':
    app.run(main)
