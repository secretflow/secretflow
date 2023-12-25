# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import shutil
from collections import Counter
from os.path import join
from pathlib import Path

import IPython.display as ipd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn, optim
from torchmetrics import Accuracy, Precision
from torchvision import datasets, transforms

import secretflow as sf
from secretflow.ml.nn import FLModel
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
from secretflow.ml.nn.utils import BaseModule, TorchModel
from secretflow.security.aggregation import SecureAggregator

# ATTENTION: need install audio backend
# pip install -r https://raw.githubusercontent.com/MicrosoftDocs/pytorchfundamentals/main/audio-pytorch/install-packages.txt
default_dir = os.getcwd()
folder = 'data'
print(f'Data directory will be: {default_dir}/{folder}')

if os.path.isdir(folder):
    print("Data folder exists.")
else:
    print("Creating folder.")
    os.mkdir(folder)


trainset_speechcommands = torchaudio.datasets.SPEECHCOMMANDS(
    f'./{folder}/', download=True
)


os.chdir(f'./{folder}/SpeechCommands/speech_commands_v0.02/')
labels = [name for name in os.listdir('.') if os.path.isdir(name)]
# back to default directory
os.chdir(default_dir)
print(f'Total Labels: {len(labels)} \n')
print(f'Label Names: {labels}')


filename = "./data/SpeechCommands/speech_commands_v0.02/yes/00f0204f_nohash_0.wav"
waveform, sample_rate = torchaudio.load(uri=filename, num_frames=3)
print(f'waveform tensor with 3 frames:  {waveform} \n')
waveform, sample_rate = torchaudio.load(uri=filename, num_frames=3, frame_offset=2)
print(f'waveform tensor with 2 frame_offsets: {waveform} \n')
waveform, sample_rate = torchaudio.load(uri=filename)
print(f'waveform tensor:  {waveform}')


def plot_audio(filename):
    waveform, sample_rate = torchaudio.load(filename)

    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))

    plt.figure()
    plt.plot(waveform.t().numpy())

    return waveform, sample_rate


filename = "./data/SpeechCommands/speech_commands_v0.02/yes/00f0204f_nohash_0.wav"
waveform, sample_rate = plot_audio(filename)
ipd.Audio(waveform.numpy(), rate=sample_rate)


filename = "./data/SpeechCommands/speech_commands_v0.02/no/0b40aa8e_nohash_0.wav"
waveform, sample_rate = plot_audio(filename)
ipd.Audio(waveform.numpy(), rate=sample_rate)


def load_audio_files(path: str, label: str):
    dataset = []
    walker = sorted(str(p) for p in Path(path).glob(f'*.wav'))

    for i, file_path in enumerate(walker):
        path, filename = os.path.split(file_path)
        speaker, _ = os.path.splitext(filename)
        speaker_id, utterance_number = speaker.split("_nohash_")
        utterance_number = int(utterance_number)

        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)
        dataset.append([waveform, sample_rate, label, speaker_id, utterance_number])

    return dataset


trainset_speechcommands_yes = load_audio_files(
    './data/SpeechCommands/speech_commands_v0.02/yes', 'yes'
)
trainset_speechcommands_no = load_audio_files(
    './data/SpeechCommands/speech_commands_v0.02/no', 'no'
)


print(f'Length of yes dataset: {len(trainset_speechcommands_yes)}')
print(f'Length of no dataset: {len(trainset_speechcommands_no)}')


trainloader_yes = torch.utils.data.DataLoader(
    trainset_speechcommands_yes, batch_size=1, shuffle=True, num_workers=0
)


trainloader_no = torch.utils.data.DataLoader(
    trainset_speechcommands_no, batch_size=1, shuffle=True, num_workers=0
)


yes_waveform = trainset_speechcommands_yes[0][0]
yes_sample_rate = trainset_speechcommands_yes[0][1]
print(f'Yes Waveform: {yes_waveform}')
print(f'Yes Sample Rate: {yes_sample_rate}')
print(f'Yes Label: {trainset_speechcommands_yes[0][2]}')
print(f'Yes ID: {trainset_speechcommands_yes[0][3]} \n')

no_waveform = trainset_speechcommands_no[0][0]
no_sample_rate = trainset_speechcommands_no[0][1]
print(f'No Waveform: {no_waveform}')
print(f'No Sample Rate: {no_sample_rate}')
print(f'No Label: {trainset_speechcommands_no[0][2]}')
print(f'No ID: {trainset_speechcommands_no[0][3]}')


def show_waveform(waveform, sample_rate, label):
    print(
        "Waveform: {}\nSample rate: {}\nLabels: {} \n".format(
            waveform, sample_rate, label
        )
    )
    new_sample_rate = sample_rate / 10

    # Resample applies to a single channel, we resample first channel here
    channel = 0
    waveform_transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(
        waveform[channel, :].view(1, -1)
    )

    print(
        "Shape of transformed waveform: {}\nSample rate: {}".format(
            waveform_transformed.size(), new_sample_rate
        )
    )

    plt.figure()
    plt.plot(waveform_transformed[0, :].numpy())


show_waveform(yes_waveform, yes_sample_rate, 'yes')


show_waveform(no_waveform, no_sample_rate, 'no')


def show_spectrogram(waveform_classA, waveform_classB):
    yes_spectrogram = torchaudio.transforms.Spectrogram()(waveform_classA)
    print("\nShape of yes spectrogram: {}".format(yes_spectrogram.size()))

    no_spectrogram = torchaudio.transforms.Spectrogram()(waveform_classB)
    print("Shape of no spectrogram: {}".format(no_spectrogram.size()))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Features of {}".format('no'))
    plt.imshow(yes_spectrogram.log2()[0, :, :].numpy(), cmap='viridis')

    plt.subplot(1, 2, 2)
    plt.title("Features of {}".format('yes'))
    plt.imshow(no_spectrogram.log2()[0, :, :].numpy(), cmap='viridis')


show_spectrogram(yes_waveform, no_waveform)


def show_melspectrogram(waveform, sample_rate):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)
    print("Shape of spectrogram: {}".format(mel_spectrogram.size()))

    plt.figure()
    plt.imshow(mel_spectrogram.log2()[0, :, :].numpy(), cmap='viridis')


show_melspectrogram(yes_waveform, yes_sample_rate)


def show_mfcc(waveform, sample_rate):
    mfcc_spectrogram = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)
    print("Shape of spectrogram: {}".format(mfcc_spectrogram.size()))

    plt.figure()
    fig1 = plt.gcf()
    plt.imshow(mfcc_spectrogram.log2()[0, :, :].numpy(), cmap='viridis')

    plt.figure()
    plt.plot(mfcc_spectrogram.log2()[0, :, :].numpy())
    plt.draw()


show_mfcc(no_waveform, no_sample_rate)


def create_spectrogram_images(trainloader, label_dir):
    # make directory
    directory = f'./data/spectrograms/{label_dir}/'
    if os.path.isdir(directory):
        print("Data exists for", label_dir)
    else:
        os.makedirs(directory, mode=0o777, exist_ok=True)

        for i, data in enumerate(trainloader):
            waveform = data[0]
            sample_rate = data[1][0]
            label = data[2]
            ID = data[3]

            # create transformed waveforms
            spectrogram_tensor = torchaudio.transforms.Spectrogram()(waveform)

            fig = plt.figure()
            plt.imsave(
                f'./data/spectrograms/{label_dir}/spec_img{i}.png',
                spectrogram_tensor[0].log2()[0, :, :].numpy(),
                cmap='viridis',
            )


def create_mfcc_images(trainloader, label_dir):
    # make directory
    os.makedirs(f'./data/mfcc_spectrograms/{label_dir}/', mode=0o777, exist_ok=True)

    for i, data in enumerate(trainloader):
        waveform = data[0]
        sample_rate = data[1][0]
        label = data[2]
        ID = data[3]

        mfcc_spectrogram = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)

        plt.figure()
        fig1 = plt.gcf()
        plt.imshow(mfcc_spectrogram[0].log2()[0, :, :].numpy(), cmap='viridis')
        plt.draw()
        fig1.savefig(f'./data/mfcc_spectrograms/{label_dir}/spec_img{i}.png', dpi=100)

        # spectorgram_train.append([spectrogram_tensor, label, sample_rate, ID])


create_spectrogram_images(trainloader_yes, 'yes')
create_spectrogram_images(trainloader_no, 'no')


data_path = './data/spectrograms'  # looking in subfolder train

yes_no_dataset = datasets.ImageFolder(
    root=data_path,
    transform=transforms.Compose([transforms.Resize((201, 81)), transforms.ToTensor()]),
)
print(yes_no_dataset)


class_map = yes_no_dataset.class_to_idx

print("\nClass category and index of the images: {}\n".format(class_map))


# split data to test and train
# use 80% to train
train_size = int(0.8 * len(yes_no_dataset))
test_size = len(yes_no_dataset) - train_size
yes_no_train_dataset, yes_no_test_dataset = torch.utils.data.random_split(
    yes_no_dataset, [train_size, test_size]
)

print("Training size:", len(yes_no_train_dataset))
print("Testing size:", len(yes_no_test_dataset))


# labels in training set
train_classes = [label for _, label in yes_no_train_dataset]
Counter(train_classes)


train_dataloader = torch.utils.data.DataLoader(
    yes_no_train_dataset, batch_size=15, num_workers=2, shuffle=True
)

test_dataloader = torch.utils.data.DataLoader(
    yes_no_test_dataset, batch_size=15, num_workers=2, shuffle=True
)


td = train_dataloader.dataset[0][0][0][0]
print(td)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(51136, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


model = CNNet().to(device)


# cost function used to determine best parameters
cost = torch.nn.CrossEntropyLoss()

# used to create optimal parameters
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create the training function


def train(dataloader, model, loss, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


# Create the validation/test function


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            test_loss += cost(pred, Y).item()
            correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    print(f'\nTest Error:\nacc: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}\n')


epochs = 15

for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train(train_dataloader, model, cost, optimizer)
    test(test_dataloader, model)
print('Done!')


model.eval()
test_loss, correct = 0, 0
class_map = ['no', 'yes']

with torch.no_grad():
    for batch, (X, Y) in enumerate(test_dataloader):
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        print(
            "Predicted:\nvalue={}, class_name= {}\n".format(
                pred[0].argmax(0), class_map[pred[0].argmax(0)]
            )
        )
        print("Actual:\nvalue={}, class_name= {}\n".format(Y[0], class_map[Y[0]]))
        break

print(data_path)

print(os.listdir(data_path))

subdataset_yes_path = os.path.join(data_path, 'yes')
print(os.listdir(subdataset_yes_path)[:20])


parties_list = ['alice', 'bob']

dataset_name = "spectrograms"
parties_path_list = []
split_dataset_path = os.path.join('.', 'fl-data', dataset_name)

for party in parties_list:
    party_path = os.path.join('.', 'fl-data', dataset_name, party)
    os.makedirs(party_path, exist_ok=True)
    parties_path_list.append(party_path)


parties_path_list


commands = os.listdir(data_path)


if 'README.md' in commands:
    commands.remove('README.md')
elif '.DS_Store' in commands:
    commands.remove('.DS_Store')


print(commands)


parties_num = len(parties_list)
for command in commands:
    command_path = join(data_path, command)
    for party_path in parties_path_list:
        party_command_path = join(party_path, command)
        print(party_command_path)
        os.makedirs(party_command_path, exist_ok=True)

    index = 0
    for wav_name in os.listdir(command_path):
        wav_path = join(command_path, wav_name)
        target_dir_path = join(
            '.', 'fl-data', dataset_name, parties_list[index % parties_num], command
        )
        shutil.copy(wav_path, target_dir_path)
        # if you want to watch the progress of copying the files, please uncomment the following line
        # print(f'copy {wav_path}-->{target_dir_path}')
        index += 1

for command in commands:
    for party_path in parties_path_list:
        command_path = join(party_path, command)
        file_num = len(os.listdir(command_path))
        print(f'{command_path} : {file_num}')


print('The version of SecretFlow: {}'.format(sf.__version__))

sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], address="local", log_to_driver=False)
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')


def create_dataset_builder(
    batch_size=32,
    train_split=0.8,
    shuffle=True,
    random_seed=1234,
):
    def dataset_builder(data_path, stage="train"):
        """ """
        import math

        import numpy as np
        from torch.utils.data import DataLoader
        from torch.utils.data.sampler import SubsetRandomSampler
        from torchvision import datasets, transforms

        # Define dataset
        yes_no_dataset = datasets.ImageFolder(
            root=data_path,
            transform=transforms.Compose(
                [transforms.Resize((201, 81)), transforms.ToTensor()]
            ),
        )

        dataset_size = len(yes_no_dataset)
        # Define sampler

        indices = list(range(dataset_size))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        split = int(np.floor(train_split * dataset_size))
        train_indices, val_indices = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # Define databuilder
        train_loader = DataLoader(
            yes_no_dataset, batch_size=batch_size, sampler=train_sampler
        )
        valid_loader = DataLoader(
            yes_no_dataset, batch_size=batch_size, sampler=valid_sampler
        )

        # Return
        if stage == "train":
            train_step_per_epoch = math.ceil(split / batch_size)

            return train_loader, train_step_per_epoch
        elif stage == "eval":
            eval_step_per_epoch = math.ceil((dataset_size - split) / batch_size)
            return valid_loader, eval_step_per_epoch

    return dataset_builder


# prepare dataset dict
data_builder_dict = {
    alice: create_dataset_builder(
        batch_size=32,
        train_split=0.8,
        shuffle=False,
        random_seed=1234,
    ),
    bob: create_dataset_builder(
        batch_size=32,
        train_split=0.8,
        shuffle=False,
        random_seed=1234,
    ),
}


class ConvNet(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(51136, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


device_list = [alice, bob]
aggregator = SecureAggregator(charlie, [alice, bob])


num_classes = 2
loss_fn = nn.CrossEntropyLoss
optim_fn = optim_wrapper(optim.Adam, lr=1e-3)

model_def = TorchModel(
    model_fn=ConvNet,
    loss_fn=loss_fn,
    optim_fn=optim_fn,
    metrics=[
        metric_wrapper(
            Accuracy, task="multiclass", num_classes=num_classes, average='micro'
        ),
        metric_wrapper(
            Precision, task="multiclass", num_classes=num_classes, average='micro'
        ),
    ],
)

fed_model = FLModel(
    device_list=device_list,
    model=model_def,
    aggregator=aggregator,
    backend="torch",
    strategy="fed_avg_w",
    random_seed=1234,
)


parties_path_list


data = {
    alice: parties_path_list[0],
    bob: parties_path_list[1],
}


history = fed_model.fit(
    data,
    None,
    validation_data=data,
    epochs=15,
    batch_size=32,
    aggregate_freq=2,
    sampler_method="batch",
    random_seed=1234,
    dp_spent_step_freq=1,
    dataset_builder=data_builder_dict,
)


print(history.global_history.keys())

plt.plot(history.global_history['multiclassaccuracy'])
plt.plot(history.global_history['val_multiclassaccuracy'])
plt.title('FLModel accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()
