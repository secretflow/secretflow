# # 基于 TensorFlow 在 SecretFlow 中实现水平联邦 RNN 生成音乐任务
# ## 引言
# 本教程基于 TensorFlow 的 [使用 RNN 生成音乐](https://tensorflow.google.cn/tutorials/audio/music_generation?hl=zh-cn) 而改写，通过本教程，您将了解到现有的基于 TensorFlow 的示例如何快速地迁移到 SecretFlow 隐语的联邦学习框架之下，实现模型的联邦学习化。本教程建议在 Python 版本为`3.8.10`下运行。


# ## 单机模式
# ### 小节引言
# 本小节的代码主要来自于 [使用 RNN 生成音乐](https://tensorflow.google.cn/tutorials/audio/music_generation?hl=zh-cn) ，
# 主要讲解如何使用简单的 RNN 生成音符。您将使用来自 [MAESTRO 数据集](https://magenta.tensorflow.org/datasets/maestro) 的钢琴 MIDI 文件集合来训练模型。
# 给定一系列音符，您的模型将学习预测序列中的下一个音符。可以通过重复调用模型来生成更长的音符序列。 为了教程的简洁，本小节仅仅简要介绍了一下各部分的功能；
# 对于实现的具体解析，请读者移步参考[原教程](https://tensorflow.google.cn/tutorials/audio/music_generation?hl=zh-cn)。
# ### 安装依赖

"""
when you run this script for the first time, you should install the dependencies below

!sudo apt install -y fluidsynth


%pip install --upgrade pyfluidsynth


%pip install pretty_midi
"""

import collections
import glob
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
from IPython import display
from matplotlib import pyplot as plt

import secretflow as sf
from secretflow.ml.nn import FLModel
from secretflow.security.aggregation import SecureAggregator

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
_SAMPLING_RATE = 16000


# ### 下载 Maestro 数据集


data_dir = pathlib.Path('data/maestro-v2.0.0')
if not data_dir.exists():
    tf.keras.utils.get_file(
        'maestro-v2.0.0-midi.zip',
        origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
        extract=True,
        cache_dir='.',
        cache_subdir='data',
    )


filenames = glob.glob(str(data_dir / '**/*.mid*'))
print('Number of files:', len(filenames))


# ### 处理 MIDI 文件


sample_file = filenames[1]
print(sample_file)


pm = pretty_midi.PrettyMIDI(sample_file)


def display_audio(pm: pretty_midi.PrettyMIDI, seconds=30):
    waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
    # Take a sample of the generated waveform to mitigate kernel resets
    waveform_short = waveform[: seconds * _SAMPLING_RATE]
    return display.Audio(waveform_short, rate=_SAMPLING_RATE)


display_audio(pm)


print('Number of instruments:', len(pm.instruments))
instrument = pm.instruments[0]
instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
print('Instrument name:', instrument_name)


# ### 提取音符


for i, note in enumerate(instrument.notes[:10]):
    note_name = pretty_midi.note_number_to_name(note.pitch)
    duration = note.end - note.start
    print(
        f'{i}: pitch={note.pitch}, note_name={note_name},' f' duration={duration:.4f}'
    )


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


raw_notes = midi_to_notes(sample_file)
raw_notes.head()


get_note_names = np.vectorize(pretty_midi.note_number_to_name)
sample_note_names = get_note_names(raw_notes['pitch'])
sample_note_names[:10]


def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
    if count:
        title = f'First {count} notes'
    else:
        title = f'Whole track'
        count = len(notes['pitch'])
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    _ = plt.title(title)


plot_piano_roll(raw_notes, count=100)


plot_piano_roll(raw_notes)


def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    sns.histplot(notes, x="pitch", bins=20)

    plt.subplot(1, 3, 2)
    max_step = np.percentile(notes['step'], 100 - drop_percentile)
    sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))

    plt.subplot(1, 3, 3)
    max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
    sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))


plot_distributions(raw_notes)


# ### 创建 MIDI 文件


def notes_to_midi(
    notes: pd.DataFrame,
    out_file: str,
    instrument_name: str,
    velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm


example_file = 'example.midi'
example_pm = notes_to_midi(
    raw_notes, out_file=example_file, instrument_name=instrument_name
)


display_audio(example_pm)


# ### 创建训练数据集


num_files = 5
all_notes = []
for f in filenames[:num_files]:
    notes = midi_to_notes(f)
    all_notes.append(notes)

all_notes = pd.concat(all_notes)


n_notes = len(all_notes)
print('Number of notes parsed:', n_notes)


key_order = ['pitch', 'step', 'duration']
train_notes = np.stack([all_notes[key] for key in key_order], axis=1)


notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
notes_ds.element_spec


def create_sequences(
    dataset: tf.data.Dataset,
    seq_length: int,
    vocab_size=128,
) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    seq_length = seq_length + 1

    # Take 1 extra for the labels
    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize note pitch
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    # Split the labels
    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(key_order)}

        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


seq_length = 25
vocab_size = 128
seq_ds = create_sequences(notes_ds, seq_length, vocab_size)
seq_ds.element_spec


for seq, target in seq_ds.take(1):
    print('sequence shape:', seq.shape)
    print('sequence elements (first 10):', seq[0:10])
    print()
    print('target:', target)


batch_size = 64
buffer_size = n_notes - seq_length  # the number of items in the dataset
train_ds = (
    seq_ds.shuffle(buffer_size)
    .batch(batch_size, drop_remainder=True)
    .cache()
    .prefetch(tf.data.experimental.AUTOTUNE)
)


train_ds.element_spec


# ### 创建并训练模型


def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


input_shape = (seq_length, 3)
learning_rate = 0.005

inputs = tf.keras.Input(input_shape)
x = tf.keras.layers.LSTM(128)(inputs)

outputs = {
    'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
    'step': tf.keras.layers.Dense(1, name='step')(x),
    'duration': tf.keras.layers.Dense(1, name='duration')(x),
}

model = tf.keras.Model(inputs, outputs)

loss = {
    'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'step': mse_with_positive_pressure,
    'duration': mse_with_positive_pressure,
}

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(loss=loss, optimizer=optimizer)

model.summary()


losses = model.evaluate(train_ds, return_dict=True)
losses


model.compile(
    loss=loss,
    loss_weights={
        'pitch': 0.05,
        'step': 1.0,
        'duration': 1.0,
    },
    optimizer=optimizer,
)


model.evaluate(train_ds, return_dict=True)


callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./training_checkpoints/ckpt_{epoch}', save_weights_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=5, verbose=1, restore_best_weights=True
    ),
]


epochs = 20

history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
)


plt.plot(history.epoch, history.history['loss'], label='total loss')
plt.show()


# ### 生成音符


def predict_next_note(
    notes: np.ndarray, keras_model: tf.keras.Model, temperature: float = 1.0
):
    """Generates a note as a tuple of (pitch, step, duration), using a trained sequence model."""

    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)


temperature = 2.0
num_predictions = 120

sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

# The initial sequence of notes; pitch is normalized similar to training
# sequences
input_notes = sample_notes[:seq_length] / np.array([vocab_size, 1, 1])

generated_notes = []
prev_start = 0
for _ in range(num_predictions):
    pitch, step, duration = predict_next_note(input_notes, model, temperature)
    start = prev_start + step
    end = start + duration
    input_note = (pitch, step, duration)
    generated_notes.append((*input_note, start, end))
    input_notes = np.delete(input_notes, 0, axis=0)
    input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    prev_start = start

generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, 'start', 'end'))


generated_notes.head(10)


out_file = 'output.mid'
out_pm = notes_to_midi(
    generated_notes, out_file=out_file, instrument_name=instrument_name
)
display_audio(out_pm)


plot_piano_roll(generated_notes)


plot_distributions(generated_notes)


# ## 联邦模式
# ### 小节引言
# 通过单机模式，我们已经学会到，如何在单机模式下使用 RNN 生成音符，本节我们将看到如何将单机模型如何快速和低成本地迁移到 SecretFlow 隐语的联邦学习框架之下。

# ### 隐语环境初始化
# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], address="local", log_to_driver=False)
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')


# ### 封装 DataBuilder
# 在迁移过程，对于数据集的预处理方式，联邦学习模式和单机模式是一样的，我们不再重复。为了完成迁移适配过程，
# 我们只需要参考[在 SecretFlow 中使用自定义 DataBuilder（TensorFlow）](https://github.com/secretflow/secretflow/blob/main/docs/tutorial/CustomDataLoaderTF.ipynb)
# 封装我们自定义 DataBuilder 即可。现在，参考原教程，我们封装对应的DataBuilder，所以我们也不需要额外写很多代码。


def create_dataset_builder(
    batch_size=32,
):
    def dataset_builder(filenames, stage="train"):
        import tensorflow as tf

        num_files = 5
        all_notes = []
        for f in filenames[:num_files]:
            notes = midi_to_notes(f)
            all_notes.append(notes)

        all_notes = pd.concat(all_notes)
        n_notes = len(all_notes)

        key_order = ['pitch', 'step', 'duration']
        train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

        notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

        seq_length = 25
        vocab_size = 128
        seq_ds = create_sequences(notes_ds, seq_length, vocab_size)

        buffer_size = n_notes - seq_length  # the number of items in the dataset
        train_ds = (
            seq_ds.shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        if stage == "train":
            train_dataset = train_ds
            # train_step_per_epoch = math.ceil(n_notes / batch_size)
            train_step_per_epoch = 160
            return train_dataset, train_step_per_epoch
        elif stage == "eval":
            eval_dataset = train_ds
            eval_step_per_epoch = 160
            return eval_dataset, eval_step_per_epoch

    return dataset_builder


# ### 定义网络结构
# 得益于隐语优异的设计，我们只需要将单机模式下定义的网络结构，进行适当的封装即可。具体到本教程，我们只需要对单机模式的模型进行封装即可。


def create_audio_generate_model(input_shape, name='model'):
    def create_model():
        # Create model

        inputs = tf.keras.Input(input_shape)
        x = tf.keras.layers.LSTM(128)(inputs)

        outputs = {
            'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
            'step': tf.keras.layers.Dense(1, name='step')(x),
            'duration': tf.keras.layers.Dense(1, name='duration')(x),
        }

        model = tf.keras.Model(inputs, outputs)

        loss = {
            'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'step': mse_with_positive_pressure,
            'duration': mse_with_positive_pressure,
        }
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        model.compile(loss=loss, optimizer=optimizer)

        return model

    return create_model


# ### 定义 TensorFlow 后端的 FLModel


device_list = [alice, bob]
aggregator = SecureAggregator(charlie, [alice, bob])

# prepare model
input_shape = (seq_length, 3)

# keras model
model = create_audio_generate_model(input_shape)


fed_model = FLModel(
    device_list=device_list,
    model=model,
    aggregator=aggregator,
    backend="tensorflow",
    strategy="fed_avg_w",
    random_seed=1234,
)


data_builder_dict = {
    alice: create_dataset_builder(
        batch_size=32,
    ),
    bob: create_dataset_builder(
        batch_size=32,
    ),
}


# ### 给出参与方数据集路径
# #### 查看数据集结构


len(filenames)


filenames[:10]


# 可以看到，我们的数据集路径保存在变量 **filenames** 里，并且一共是 **1282** 个文件，所以我们直接将这份列表一分为二，模拟联邦学习的数据拥有方是 **alice** 和 **bob**


len_dataset = len(filenames)
alice_filenames_dataset = filenames[: int(len_dataset / 2)]
bob_filenames_dataset = filenames[int(len_dataset / 2) :]


# 分别查看**alice** 和 **bob**的数据集数目


print("the number of samples of Alice: {}".format(len(alice_filenames_dataset)))
print("the number of samples of Bob: {}".format(len(bob_filenames_dataset)))


data = {
    alice: filenames,
    bob: filenames,
}


# ### 训练联邦学习模型


history = fed_model.fit(
    data,
    None,
    validation_data=data,
    epochs=50,
    batch_size=64,
    aggregate_freq=2,
    sampler_method="batch",
    random_seed=1234,
    dp_spent_step_freq=1,
    dataset_builder=data_builder_dict,
)


# ### 可视化训练历史
print(history.global_history.keys())


# Draw loss values for training & validation
plt.plot(history.global_history['loss'])
plt.plot(history.global_history['val_loss'])
plt.title('FLModel loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()


# Draw loss values for training & validation
plt.plot(history.global_history['pitch_loss'])
plt.plot(history.global_history['val_pitch_loss'])
plt.title('FLModel pitch loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()


# Draw loss values for training & validation
plt.plot(history.global_history['step_loss'])
plt.plot(history.global_history['val_step_loss'])
plt.title('FLModel step loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()


# 至此，我们很好地完成了联邦学习模型的训练。


# ## 小结
#
# 通过本教程，您将看到，如何将 TensorFlow 的生成音乐模型快速迁移到 SecretFlow 隐语 的联邦学习框架之下，实现生成音乐模型的联邦学习。
