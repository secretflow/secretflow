import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, Precision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

import secretflow as sf
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
from secretflow.ml.nn.sl.attack.torch.label_inferece_attack import (
    LabelInferenceAttacker,
)
from secretflow.ml.nn.utils import TorchModel
from tests.ml.nn.attack.data_util import (
    CIFAR10Labeled,
    CIFAR10Unlabeled,
    label_index_split,
)
from tests.ml.nn.attack.model_def import (
    BottomModelForCifar10,
    BottomModelPlus,
    TopModelForCifar10,
)


def data_builder(file_path, batch_size):
    def prepare_data():
        n_labeled = 40
        num_classes = 10

        def get_transforms():
            transform_ = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            return transform_

        transforms_ = get_transforms()

        base_dataset = datasets.CIFAR10(file_path, train=True)

        train_labeled_idxs, train_unlabeled_idxs = label_index_split(
            base_dataset.targets, int(n_labeled / num_classes), num_classes
        )
        train_labeled_dataset = CIFAR10Labeled(
            file_path, train_labeled_idxs, train=True, transform=transforms_
        )
        train_unlabeled_dataset = CIFAR10Unlabeled(
            file_path, train_unlabeled_idxs, train=True, transform=transforms_
        )
        train_complete_dataset = CIFAR10Labeled(
            file_path, None, train=True, transform=transforms_
        )
        test_dataset = CIFAR10Labeled(
            file_path, train=False, transform=transforms_, download=True
        )
        print(
            "#Labeled:",
            len(train_labeled_idxs),
            "#Unlabeled:",
            len(train_unlabeled_idxs),
        )

        labeled_trainloader = torch.utils.data.DataLoader(
            train_labeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        unlabeled_trainloader = torch.utils.data.DataLoader(
            train_unlabeled_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        dataset_bs = batch_size * 10
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=dataset_bs, shuffle=False, num_workers=0
        )
        train_complete_trainloader = torch.utils.data.DataLoader(
            train_complete_dataset,
            batch_size=dataset_bs,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        return (
            labeled_trainloader,
            unlabeled_trainloader,
            test_loader,
            train_complete_trainloader,
        )

    return prepare_data


def correct_counter(output, target, batch_size, topk=(1, 5)):
    tensor_target = torch.Tensor(target)
    dataset = torch.utils.data.TensorDataset(tensor_target)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    correct_counts = [0] * len(topk)
    for idx, tt in enumerate(dataloader):
        for i, k in enumerate(topk):
            _, pred = output[idx].topk(k, 1, True, True)
            correct_k = torch.eq(pred, tt[0].view(-1, 1)).sum().float().item()
            correct_counts[i] += correct_k

    print('correct_counts: ', correct_counts)
    return correct_counts


def create_attacker_builder(file_path, batch_size, model_save_path):
    def attacker_builder():
        def create_model(ema=False):
            bottom_model = BottomModelForCifar10()
            model = BottomModelPlus(bottom_model)

            if ema:
                for param in model.parameters():
                    param.detach_()

            return model

        model = create_model(ema=False)
        ema_model = create_model(ema=True)

        data_buil = data_builder(file_path=file_path, batch_size=batch_size)

        attacker = LabelInferenceAttacker(
            model, ema_model, 10, data_buil, save_model_path=model_save_path
        )
        return attacker

    return attacker_builder


def test_sl_and_lia(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    device_y = bob

    tmp_dir = tempfile.TemporaryDirectory()
    lia_path = tmp_dir.name
    import logging

    logging.warning('lia_path: ' + lia_path)

    # first, train a sl model and save model
    # prepare data
    data_file_path = lia_path + '/data_download'
    model_save_path = lia_path + '/lia_model'
    train = True
    train_dataset = datasets.CIFAR10(
        data_file_path, train, transform=ToTensor(), download=True
    )
    # dataloader transformed train_dataset, so we here call dataloader before we get data and label
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=False
    )

    train_np = np.array(train_loader.dataset)
    train_data = np.array([t[0].numpy() for t in train_np])
    train_label = np.array([t[1] for t in train_np])

    # put into FedNdarray
    fed_data = FedNdarray(
        partitions={
            alice: alice(lambda x: x[:, :, :, 0:16])(train_data),
            bob: bob(lambda x: x[:, :, :, 16:32])(train_data),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    label = bob(lambda x: x)(train_label)

    # model configure
    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=5e-4)
    base_model = TorchModel(
        model_fn=BottomModelForCifar10,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average='micro'
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average='micro'
            ),
        ],
    )

    fuse_model = TorchModel(
        model_fn=TopModelForCifar10,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average='micro'
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average='micro'
            ),
        ],
    )

    base_model_dict = {
        alice: base_model,
        bob: base_model,
    }

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=device_y,
        model_fuse=fuse_model,
        dp_strategy_dict=None,
        compressor=None,
        simulation=True,
        random_seed=1234,
        backend='torch',
        strategy='split_nn',
    )

    callback_dict = {alice: create_attacker_builder(file_path=data_file_path, batch_size=16, model_save_path=model_save_path)}

    history = sl_model.fit(
        fed_data,
        label,
        validation_data=(fed_data, label),
        epochs=1,
        batch_size=128,
        shuffle=False,
        random_seed=1234,
        dataset_builder=None,
        callbacks=callback_dict
    )
    print(history)

    pred_bs = 128
    result = sl_model.predict(fed_data, batch_size=pred_bs, verbose=1)
    cor_count = bob(correct_counter)(result, label, batch_size=pred_bs, topk=(1, 4))
    sf.wait(cor_count)
