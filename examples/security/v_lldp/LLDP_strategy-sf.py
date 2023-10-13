
import secretflow as sf
import matplotlib.pyplot as plt
from typing import List
import math
import numpy as np
sf.shutdown()
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
import pandas as pd
from secretflow.utils.simulation.datasets import dataset

df = pd.read_csv(dataset('bank_marketing'), sep=';')
alice_data = df[["age", "job", "marital", "education", "y"]]
alice_data
bob_data = df[["default", "balance", "housing", "loan", "contact",
             "day","month","duration","campaign","pdays","previous","poutcome"]]
bob_data
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
# spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))
from secretflow.utils.simulation.datasets import load_bank_marketing

# Alice has the first four features,
# while bob has the left features
data = load_bank_marketing(parts={alice: (0, 4), bob: (4, 16)}, axis=1)
# Alice holds the label.
label = load_bank_marketing(parts={alice: (16, 17)}, axis=1)
data['age'].partitions[alice].data
from secretflow.preprocessing.scaler import MinMaxScaler
from secretflow.preprocessing.encoder import LabelEncoder
encoder = LabelEncoder()
data['job'] = encoder.fit_transform(data['job'])
data['marital'] = encoder.fit_transform(data['marital'])
data['education'] = encoder.fit_transform(data['education'])
data['default'] = encoder.fit_transform(data['default'])
data['housing'] = encoder.fit_transform(data['housing'])
data['loan'] = encoder.fit_transform(data['loan'])
data['contact'] = encoder.fit_transform(data['contact'])
data['poutcome'] = encoder.fit_transform(data['poutcome'])
data['month'] = encoder.fit_transform(data['month'])
label = encoder.fit_transform(label)
print(f"label= {type(label)},\ndata = {type(data)}")
scaler = MinMaxScaler()

data = scaler.fit_transform(data)
from secretflow.data.split import train_test_split
random_state = 1234
train_data,test_data = train_test_split(data, train_size=0.8, random_state=random_state)
train_label,test_label = train_test_split(label, train_size=0.8, random_state=random_state)
def create_base_model(input_dim, output_dim,  name='base_model'):
    # Create model
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers
        import tensorflow as tf
        model = keras.Sequential(
            [
                keras.Input(shape=input_dim),
                layers.Dense(100,activation ="relu" ),
                layers.Dense(output_dim, activation="relu"),
            ]
        )
        # Compile model
        model.summary()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=["accuracy",tf.keras.metrics.AUC()])
        return model
    return create_model
# prepare model
hidden_size = 64
# get the number of features of each party.
# When the input data changes, the network automatically adjusts to the input data
alice_input_feature_num = train_data.values.partition_shape()[alice][1]
bob_input_feature_num = train_data.values.partition_shape()[bob][1]

model_base_alice = create_base_model(alice_input_feature_num, hidden_size)
model_base_bob = create_base_model(bob_input_feature_num, hidden_size)
model_base_alice()
model_base_bob()
def create_fuse_model(input_dim, output_dim, party_nums, name='fuse_model'):
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers
        import tensorflow as tf
        # input
        input_layers = []
        for i in range(party_nums):
            input_layers.append(keras.Input(input_dim,))

        merged_layer = layers.concatenate(input_layers)
        fuse_layer = layers.Dense(64, activation='relu')(merged_layer)
        output = layers.Dense(output_dim, activation='sigmoid')(fuse_layer)

        model = keras.Model(inputs=input_layers, outputs=output)
        model.summary()

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=["accuracy",tf.keras.metrics.AUC()])
        return model
    return create_model
model_fuse = create_fuse_model(
    input_dim=hidden_size, party_nums=2, output_dim=1)
model_fuse()
model_fuse_data_list =model_fuse().get_weights()
print(model_fuse_data_list)
delta = math.exp(-3)
epsilon = [80, 80, 40, 40, 30, 30]##卷积层不加噪，后三层加噪
# data_list_l = data[:4]
# data_list_r = data[4:]
for i in range(len(model_fuse_data_list)):
    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon[i]
    # print("sigma:", sigma)
    noise = np.random.normal(0, sigma, model_fuse_data_list[i].shape)
    # add_noise_data[i]=data_list_r[i] + noise
    model_fuse_data_list[i] = model_fuse_data_list[i] + noise
model_fuse().set_weights(model_fuse_data_list)
model_fuse()

base_model_dict = {
    alice: model_base_alice,
    bob:   model_base_bob
}
from secretflow.security.privacy import DPStrategy, GaussianEmbeddingDP, LabelDP

# Define DP operations
train_batch_size = 128
# gaussian_embedding_dp = GaussianEmbeddingDP(
#     noise_multiplier=0.5,
#     l2_norm_clip=1.0,
#     batch_size=train_batch_size,
#     num_samples=train_data.values.partition_shape()[alice][0],
#     is_secure_generator=False,
# )
# # print(train_data.values.partition_shape()[alice][0])
# dp_strategy_alice = DPStrategy(embedding_dp=gaussian_embedding_dp)
# label_dp = LabelDP(eps=64.0)
# dp_strategy_bob = DPStrategy(label_dp=label_dp)
# dp_strategy_dict = {alice: dp_strategy_alice, bob: dp_strategy_bob}
# dp_spent_step_freq = 10
sl_model = SLModel(
    base_model_dict=base_model_dict,
    device_y=alice,
    model_fuse=model_fuse,)
    # dp_strategy_dict=dp_strategy_dict,)

sf.reveal(test_data.partitions[alice].data), sf.reveal(test_label.partitions[alice].data)
sf.reveal(train_data.partitions[alice].data), sf.reveal(train_label.partitions[alice].data)
history =  sl_model.fit(train_data,
             train_label,
             validation_data=(test_data,test_label),
             epochs=10,
             batch_size=train_batch_size,
             shuffle=True,
             verbose=1,
             validation_freq=1,)
             # dp_spent_step_freq=dp_spent_step_freq,)
print(history)
print(history.keys())
# Plot the change of loss during training
plt.plot(history['train_loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc='upper right')
plt.show()
# Plot the change of accuracy during training
plt.plot(history['train_accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
# Plot the Area Under Curve(AUC) of loss during training
plt.plot(history['train_auc_1'])
plt.plot(history['val_auc_1'])
plt.title('Model Area Under Curve')
plt.ylabel('Area Under Curve')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
global_metric = sl_model.evaluate(test_data, test_label, batch_size=128)
print(global_metric)



