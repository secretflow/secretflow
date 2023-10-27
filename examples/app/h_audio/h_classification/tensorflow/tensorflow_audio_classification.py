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


# # 基于 TensorFlow 在 SecretFlow 中实现水平联邦音频分类任务
#
# ## 引言
# 本教程基于 TensorFlow 的 [Simple audio recognition: Recognizing keywords](https://tensorflow.google.cn/tutorials/audio/simple_audio) 而改写，通过本教程，您将了解到现有的基于 TensorFlow 的示例如何快速地迁移到 SecretFlow 隐语的联邦学习框架之下，实现模型的联邦学习化。
#
# 本教程基于 TensorFlow 的而改写，通过本教程，您将了解到现有的基于 TensorFlow 的示例如何可以快速地迁移到 SecretFlow 隐语的联邦学习框架之下，实现模型的联邦学习化。


# ## 单机模式
#
# ### 小节引言
# 本小节的代码主要来自于 [Simple audio recognition: Recognizing keywords](https://tensorflow.google.cn/tutorials/audio/simple_audio)  ，
# 主要讲解如何在 TensorFlow 下构建一个基础的自动语音识别（[automatic speech recognition](https://en.wikipedia.org/wiki/Speech_recognition) , ASR）模型
# 识别 10 个不同的词。 在本次示例，教程通过在数据集 [Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands)
# ([Warden, 2018](https://arxiv.org/abs/1804.03209)) 的部分数据上训练模型。 为了教程的简洁，本小节仅仅简要介绍了一下各部分的功能；
# 对于实现的具体解析，请读者移步参考[原教程](https://tensorflow.google.cn/tutorials/audio/simple_audio)。


# ### 环境建立


# !pip install -U -q tensorflow tensorflow_datasets


import os
import pathlib
import shutil
from os.path import join

import numpy as np
import seaborn as sns
import tensorflow as tf
from IPython import display
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models

import secretflow as sf
from secretflow.ml.nn import FLModel
from secretflow.security.aggregation import SecureAggregator

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


# ### 导入迷你版 Speech Commands 数据集


DATASET_PATH = 'data/mini_speech_commands'

data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
    tf.keras.utils.get_file(
        'mini_speech_commands.zip',
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir='.',
        cache_subdir='data',
    )


commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
print('Commands:', commands)


train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset='both',
)

label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)


train_ds.element_spec


def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)


test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)


for example_audio, example_labels in train_ds.take(1):
    print(example_audio.shape)
    print(example_labels.shape)


label_names[[1, 1, 3, 0]]


plt.figure(figsize=(16, 10))
rows = 3
cols = 3
n = rows * cols
for i in range(n):
    plt.subplot(rows, cols, i + 1)
    audio_signal = example_audio[i]
    plt.plot(audio_signal)
    plt.title(label_names[example_labels[i]])
    plt.yticks(np.arange(-1.2, 1.2, 0.2))
    plt.ylim([-1.1, 1.1])


# ### 将波形转换为频谱图


def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


for i in range(3):
    label = label_names[example_labels[i]]
    waveform = example_audio[i]
    spectrogram = get_spectrogram(waveform)

    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)
    print('Audio playback')
    display.display(display.Audio(waveform, rate=16000))


def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label.title())
plt.show()


def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)


for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break


rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

for i in range(n):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(example_spectrograms[i].numpy(), ax)
    ax.set_title(label_names[example_spect_labels[i].numpy()])

plt.show()


# ### 构建和训练模型


train_spectrogram_ds = (
    train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)


input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential(
    [
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ]
)

model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)


EPOCHS = 10
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)


metrics = history.history
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plt.subplot(1, 2, 2)
plt.plot(
    history.epoch,
    100 * np.array(metrics['accuracy']),
    100 * np.array(metrics['val_accuracy']),
)
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')


# ### 评估模型性能


model.evaluate(test_spectrogram_ds, return_dict=True)


# #### 展示混淆矩阵


y_pred = model.predict(test_spectrogram_ds)


y_pred = tf.argmax(y_pred, axis=1)


y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0)


confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    confusion_mtx, xticklabels=label_names, yticklabels=label_names, annot=True, fmt='g'
)
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()


# #### 在声音文件上进行推理


x = data_dir / 'no/01bb6a2a_nohash_0.wav'
x = tf.io.read_file(str(x))
x, sample_rate = tf.audio.decode_wav(
    x,
    desired_channels=1,
    desired_samples=16000,
)
x = tf.squeeze(x, axis=-1)
waveform = x
x = get_spectrogram(x)
x = x[tf.newaxis, ...]

prediction = model(x)
x_labels = ['no', 'yes', 'down', 'go', 'left', 'up', 'right', 'stop']
plt.bar(x_labels, tf.nn.softmax(prediction[0]))
plt.title('No')
plt.show()

display.display(display.Audio(waveform, rate=16000))


# ## 联邦模式
#
# ### 小节引言
# 通过单机模式，我们已经学会到，如何在单机模式下使用  ASR 模型实现语音分类，本节我们将看到如何将单机模型如何快速和低成本地迁移到 SecretFlow 隐语的联邦学习框架之下。


# ### 数据划分
#
# 首先为了模拟联邦学习多方参与的场景设定，我们先人为进行一下数据集划分。为方便演示，我们对数据按参与方进行均匀划分。
# 我们假定联邦学习的数据拥有方是 **alice** 和 **bob**


dataset_name = 'mini_speech_commands'
dataset_path = os.path.join('.', 'data', dataset_name)
parties_list = ['alice', 'bob']
parties_path_list = []


# 可以看到目前我们的数据集所保存的目录


os.path.abspath(dataset_path)


split_dataset_path = os.path.join('.', 'fl-data', dataset_name)

for party in parties_list:
    party_path = os.path.join('.', 'fl-data', dataset_name, party)
    os.makedirs(party_path, exist_ok=True)
    parties_path_list.append(party_path)


# 由上述结果，我们可以看到，语音分类的训练数据集主要位于 **".data/mini_speech_commands/"** ；我们的参与方数据文件夹分别为 **'./fl-data/mini_speech_commands/alice'** 和 **'./fl-data/mini_speech_commands/bob'**；


parties_path_list


commands = os.listdir(dataset_path)


if 'README.md' in commands:
    commands.remove('README.md')
elif '.DS_Store' in commands:
    commands.remove('.DS_Store')


# 可以看到数据集拥有 8 个标签，所以我们分别在对应的参与方数据文件夹下建立对应的子文件夹


commands


parties_num = len(parties_list)
for command in commands:
    command_path = join(dataset_path, command)
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


# 查看参与方的各个文件夹下所拥有的文件数目


for command in commands:
    for party_path in parties_path_list:
        command_path = join(party_path, command)
        file_num = len(os.listdir(command_path))
        print(f'{command_path} : {file_num}')


# ### 隐语环境初始化


# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], address="local", log_to_driver=False)

alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')


# ### 封装 DataBuilder
#
# 在迁移过程，对于数据集的预处理方式，联邦学习模式和单机模式是一样的，我们不再重复。为了完成迁移适配过程，我们只需要参考
# [在 SecretFlow 中使用自定义 DataBuilder（TensorFlow）](https://github.com/secretflow/secretflow/blob/main/docs/tutorial/CustomDataLoaderTF.ipynb)
# 封装我们自定义 DataBuilder 即可。现在，参考原教程，我们封装对应的DataBuilder，所以我们也不需要额外写很多代码。


def create_dataset_builder(
    batch_size=32,
):
    def dataset_builder(folder_path, stage="train"):
        import tensorflow as tf

        dataset = tf.keras.utils.audio_dataset_from_directory(
            directory=folder_path,
            batch_size=batch_size,
            validation_split=0.2,
            seed=0,
            output_sequence_length=16000,
            subset='both',
        )
        # dataset split
        train_dataset = dataset[0]
        eval_dataset = dataset[1]
        # audio preprocess
        train_dataset = train_dataset.map(squeeze, tf.data.AUTOTUNE)
        eval_dataset = eval_dataset.map(squeeze, tf.data.AUTOTUNE)

        train_dataset = make_spec_ds(train_dataset)
        eval_dataset = make_spec_ds(eval_dataset)

        # cache
        train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
        eval_dataset = eval_dataset.cache().prefetch(tf.data.AUTOTUNE)

        # split process
        if stage == "train":
            train_step_per_epoch = len(train_dataset)
            return train_dataset, train_step_per_epoch
        elif stage == "eval":
            eval_step_per_epoch = len(eval_dataset)
            return eval_dataset, eval_step_per_epoch

    return dataset_builder


# ### 构建 dataset_builder_dict


data_builder_dict = {
    alice: create_dataset_builder(
        batch_size=32,
    ),
    bob: create_dataset_builder(
        batch_size=32,
    ),
}


# ### 定义网络结构
#
# 得益于隐语优异的设计，我们只需要将单机模式下定义的网络结构，进行适当的封装即可，这里为了便于演示，我们去除原来的网络结构中依赖数据集的正则化层`norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))`


def create_audio_classification_model(input_shape, num_classes, name='model'):
    def create_model():
        # Create model

        # Instantiate the `tf.keras.layers.Normalization` layer.
        # norm_layer = layers.Normalization()
        # Fit the state of the layer to the spectrograms
        # with `Normalization.adapt`.
        # delete: norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                # Downsample the input.
                layers.Resizing(32, 32),
                # Normalize.
                # delete: norm_layer,
                layers.Conv2D(32, 3, activation='relu'),
                layers.Conv2D(64, 3, activation='relu'),
                layers.MaxPooling2D(),
                layers.Dropout(0.25),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(num_classes),
            ]
        )

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )
        return model

    return create_model


# ### 定义 TensorFlow 后端的 FLModel
device_list = [alice, bob]
aggregator = SecureAggregator(charlie, [alice, bob])

# prepare model
num_classes = 8
input_shape = (124, 129, 1)

# keras model
model = create_audio_classification_model(input_shape, num_classes)


fed_model = FLModel(
    device_list=device_list,
    model=model,
    aggregator=aggregator,
    backend="tensorflow",
    strategy="fed_avg_w",
    random_seed=1234,
)


# ### 给出参与方数据集路径


parties_path_list


data = {
    alice: parties_path_list[0],
    bob: parties_path_list[1],
}


# ### 训练联邦模型


history = fed_model.fit(
    data,
    None,
    validation_data=data,
    epochs=20,
    batch_size=32,
    aggregate_freq=2,
    sampler_method="batch",
    random_seed=1234,
    dp_spent_step_freq=1,
    dataset_builder=data_builder_dict,
)
# ### 可视化训练结果
print(history.global_history.keys())
# Draw accuracy values for training & validation
plt.plot(history.global_history['accuracy'])
plt.plot(history.global_history['val_accuracy'])
plt.title('FLModel accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()


# Draw loss for training & validation
plt.plot(history.global_history['loss'])
plt.plot(history.global_history['val_loss'])
plt.title('FLModel loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()


# ## 小结
# 通过本教程，您将看到，如何将 TensorFlow 的语音分类模型快速迁移到 SecretFlow 隐语 的联邦学习框架之下，实现语音分类模型的联邦学习。
