import logging

from secretflow.component.test_framework.test_case import PipelineCase, TestComp
from secretflow.component.test_framework.test_controller import TestController
from secretflow.utils.logging import LOG_FORMAT

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)

    aci_pipe = PipelineCase("aci_pipe")

    attrs = {
        "protocol": "PROTOCOL_ECDH",
        "receiver": "alice",
        "disable_alignment": False,
        "ecdh_curve": "CURVE_FOURQ",
        "input/receiver_input/key": ["id0"],
        "input/sender_input/key": ["id1"],
    }
    # 测试psi
    psi = TestComp("psi_test", "data_prep", "psi", "0.0.2", attrs)
    aci_pipe.add_comp(psi, ["DAGInput.alice", "DAGInput.bob"])

    attrs = {
        "input/in_ds/drop_features": ["alice1", "bob9"],
    }
    # 测试feature_filter
    feature_filter = TestComp("ff", "data_filter", "feature_filter", "0.0.1", attrs)
    aci_pipe.add_comp(feature_filter, ["psi_test.0"])

    attrs = {
        "train_size": 0.7,
        "test_size": 0.3,
        "random_state": 42,
        "shuffle": False,
    }
    # 测试train_test_split
    ds_split = TestComp("ds_split", "data_prep", "train_test_split", "0.0.1", attrs)
    aci_pipe.add_comp(ds_split, ["ff.0"])

    feature_selects = [f"alice{c}" for c in range(15)] + [f"bob{c}" for c in range(15)]
    feature_selects.remove("alice1")
    feature_selects.remove("bob9")

    attrs = {
        "epochs": 2,
        "learning_rate": 0.1,
        "batch_size": 512,
        "sig_type": "t1",
        "reg_type": "logistic",
        "input/train_dataset/label": ["y"],
        "input/train_dataset/feature_selects": feature_selects,
    }
    # 测试ss_sgd_train
    sslr = TestComp("sslr_train", "ml.train", "ss_sgd_train", "0.0.1", attrs)
    aci_pipe.add_comp(sslr, ["ds_split.0"])

    attrs = {
        "batch_size": 32,
        "receiver": "alice",
        "save_ids": True,
        "save_label": True,
    }
    # 测试ss_sgd_predict
    sslr = TestComp("sslr_pred", "ml.predict", "ss_sgd_predict", "0.0.1", attrs)
    aci_pipe.add_comp(sslr, ["sslr_train.0", "ds_split.1"])

    # TODO: add others comp

    test = TestController(aci_mode=True)
    test.add_pipeline_case(aci_pipe)

    test.run(True)
