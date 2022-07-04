"""
The following codes are demos only. 
It's **NOT for production** due to system security concerns,
please **DO NOT** use it directly in production.
"""

from absl import app, flags
from funcs import create_conv_model
from tensorflow.keras.optimizers import Adam

import secretflow as sf
from secretflow.data.ndarray import load
from secretflow.data.split import train_test_split
from secretflow.ml.nn.fl_model import FLModelTF
from secretflow.security.aggregation import PlainAggregator

flags.DEFINE_float("lr", 0.001, "learning rate")
FLAGS = flags.FLAGS


def main(_):
    sf.init(['alice', 'bob', 'charlie'], num_cpus=8, log_to_driver=True)
    alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')

    data_dict = {
        alice: '../../../../tests/datasets/mnist/horizontal/mnist_alice.npz',
        bob: '../../../../tests/datasets/mnist/horizontal/mnist_bob.npz',
    }

    fed_npz = load(data_dict, allow_pickle=False)
    image = fed_npz['image']
    label = fed_npz['label']

    random_seed = 1234
    train_image, test_image = train_test_split(
        data=image, train_size=0.8, random_state=random_seed
    )
    train_label, test_label = train_test_split(
        data=label, train_size=0.8, random_state=random_seed
    )

    num_classes = 10
    input_shape = (28, 28, 1)

    optimizer = Adam(learning_rate=FLAGS.lr)
    model = create_conv_model(input_shape, num_classes, optimizer)

    device_list = [alice, bob]
    plain_aggregator = PlainAggregator(charlie)

    fed_model = FLModelTF(
        device_list=device_list, model=model, aggregator=plain_aggregator
    )

    fed_model.fit(
        train_image,
        train_label,
        validation_data=(test_image, test_label),
        epochs=10,
        batch_size=128,
        aggregate_freq=1,
    )
    global_metric = fed_model.evaluate(test_image, test_label, batch_size=128)
    print(global_metric)


if __name__ == '__main__':
    app.run(main)
