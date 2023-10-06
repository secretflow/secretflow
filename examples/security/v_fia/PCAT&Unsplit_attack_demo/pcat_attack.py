import getopt
import sys
import secretflow as sf
from attacks import attack_AfterTrain, attack_WithTrain
from model_attack import (
    create_pcat_attack_decoder_model,
    create_pcat_base_model,
    create_server_model,
)
from sl_model_attack import SLModel_attack
from datasets import load_mnist_pcat_attack


def main(with_training_decoder=True):
    sf.init(["alice", "bob"], address="local")
    alice, bob = sf.PYU("alice"), sf.PYU("bob")
    data_mnist_train, data_mnist_test = load_mnist_pcat_attack(
        parts={alice: 300, bob: 3000},
        reshape_=True,
        categorical_y=True,
        normalized_x=False,
    )

    random_state = 1234
    hidden_size = 64

    model_base_alice = create_pcat_base_model(32, 3, hidden_size)
    model_base_bob = create_pcat_base_model(32, 3, hidden_size)

    model_fuse = create_server_model(input_shape=(8, 8, hidden_size), output_dim=10)

    if with_training_decoder:
        model_decoder = create_pcat_attack_decoder_model((8, 8, hidden_size))
    else:
        model_decoder = None

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
        model_decoder=model_decoder,
    )

    train_data = data_mnist_train[0]
    train_label = data_mnist_train[1]

    test_data = data_mnist_test[0]
    test_label = data_mnist_test[1]

    sl_model.fit_pcat(
        train_data,
        train_label,
        batch_size=train_batch_size,
        validation_data=(test_data, test_label),
        epochs=10,
        shuffle=False,
        verbose=1,
        validation_freq=1,
    )

    base_model_name = "base_model_pcat_test"
    fuse_model_name = "fuse_model_pcat_test"
    decoder_model_name = "decoder_model_test"
    sl_model.save_model(
        "model/" + base_model_name,
        "model/" + fuse_model_name,
        "model/" + decoder_model_name,
        is_test=True,
        save_traces=True,
        save_format="h5",
        with_training_decoder=with_training_decoder,
    )

    if with_training_decoder:
        attack_WithTrain(
            base_model_name=base_model_name, decoder_model_name=decoder_model_name
        )
    else:
        attack_AfterTrain(base_model_name=base_model_name)


if __name__ == "__main__":
    with_training_decoder = True
    opts, _ = getopt.getopt(sys.argv[1:], "-h-v:", ["help", "value="])
    for opt, value in opts:
        if opt in ("-h", "--help"):
            print(
                "**********************************\nYou should choose a way to run the program\n"
            )
            print("-h, --help   ——————Look for more information\n")
            print(
                "-v, --value   ——————Choose a way to start your project\n"
                "———————— wt : means stealing while training\n"
                "———————— at : means stealing after training!\n**********************************\n"
            )
            sys.exit(0)
        elif opt in ("-v", "--value"):
            if value == "wt":
                with_training_decoder = True
            elif value == "at":
                with_training_decoder = False
            else:
                print(
                    "**********************************illegal parameters, program exit**********************************\n"
                )
                sys.exit(0)
        else:
            print(
                "**********************************\nRun with default parameter: -v wt\n**********************************\n"
            )

    main(with_training_decoder=with_training_decoder)
