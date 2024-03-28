import getopt
import sys

from attacks import unsplit_attack
from model_attack import create_unsplit_base_model, create_unsplit_server_model
from sl_model_attack import SLModel_attack

import secretflow as sf
from datasets import load_mnist


def main(with_training_decoder=False):
    sf.init(["alice", "bob"], address="local")
    alice, bob = sf.PYU("alice"), sf.PYU("bob")

    data_mnist_train, data_mnist_test = load_mnist(
        parts={alice: 0.5, bob: 0.5}, categorical_y=True
    )
    random_state = 1234

    hidden_size = 64

    # unused
    model_base_alice = create_unsplit_base_model(1, hidden_size, 1)

    model_base_bob = create_unsplit_base_model(1, hidden_size, 1)

    model_fuse = create_unsplit_server_model(
        input_shape=(28, 28, hidden_size), output_dim=10
    )

    base_model_dict = {
        alice: model_base_alice,
        bob: model_base_bob,
    }

    train_batch_size = 128

    sl_model = SLModel_attack(
        base_model_dict=base_model_dict,
        device_y=alice,
        strategy="split_nn",
        model_fuse=model_fuse,
        model_decoder=None,
    )

    train_data = data_mnist_train[0]
    train_label = data_mnist_train[1]

    test_data = data_mnist_test[0]
    test_label = data_mnist_test[1]

    sl_model.fit_unsplit(
        train_data,
        train_label,
        batch_size=train_batch_size,
        validation_data=(test_data, test_label),
        epochs=2,
        shuffle=False,
        verbose=1,
        validation_freq=1,
    )

    base_model_name = "base_model_unsplit_test"
    fuse_model_name = "fuse_model_unsplit_test"
    decoder_model_name = "decoder_model_unsplit_test"
    sl_model.save_model(
        "model/" + base_model_name,
        "model/" + fuse_model_name,
        "model/" + decoder_model_name,
        is_test=True,
        save_traces=True,
        save_format="h5",
        with_training_decoder=False,
    )

    base_model_name = "base_model_unsplit_test"
    unsplit_attack(base_model_name)


if __name__ == "__main__":
    main()
