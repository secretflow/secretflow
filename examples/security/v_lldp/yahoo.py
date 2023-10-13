# load_ext autoreload
# autoreload 2

import secretflow as sf
import tempfile
import tensorflow as tf
import math
import pandas as pd

print('The version of SecretFlow:{}', format(sf.__version__))

sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], address="local", log_to_driver=False)
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')

'''data_builder_dict = {
    alice: create_alice_dataset_builder(batch_size=32,),
    bob: create_bob_dataset_builder(batch_size=32),
}
'''
# _temp_dir = tempfile.mktemp()
# path_to_
path = '/media/whb/Elements/secretflow/yahoo_answers_csv/train.csv'
user_data = pd.read_csv(path, nrows=10, header=None, encoding="utf-8") # 只取十行，在此修改
user_data.columns = ['type', 'question', 'attention', 'answers'] # 在原有数据集上命名了四个属性，type代表标签
# print(user_data)
# 删除指定列data_set

# alice: attention, answers
alice_df = user_data.drop(columns=user_data.columns[[0, 1]])
# bob: type, question
bob_df = user_data.drop(columns=user_data.columns[[2, 3]])

# print(alice)
# print(bob)

alice_df['attention'] = alice_df['attention'].astype("string")
alice_dict = dict(alice_df)
data_set = tf.data.Dataset.from_tensor_slices(alice_dict).batch(32).repeat(1)
# print(data_set)


# print(bob)
label = bob_df['type']
data = bob_df.drop(columns='type')
##可以作为输入传入SLModel进行建模

def _parse_bob(row_sample, label):
    import tensorflow as tf

    y_t = label
    y = tf.expand_dims(
        tf.where(
            y_t > 3,
            tf.ones_like(y_t, dtype=tf.float32),
            tf.zeros_like(y_t, dtype=tf.float32),
        ),
        axis=1
    )
    return row_sample, y


bob_dict = tuple([dict(data), label])
data_set = tf.data.Dataset.from_tensor_slices(bob_dict).batch(32).repeat(1)
data_set = data_set.map(_parse_bob)
next(iter(data_set))

# print(next(iter(data_set)))


# warp DataBuilder

# alice
def create_dataset_builder_alice(
        batch_size=128,
        repeat_count=5,
):
    def dataset_builder(x):
        import pandas as pd
        import tensorflow as tf

        x = [dict(t) if isinstance(t, pd.DataFrame) else t for t in x]
        x = x[0] if len(x) == 1 else tuple(x)
        data_set = (
            tf.data.Dataset.from_tensor_slices(x).batch(batch_size).repeat(repeat_count)
        )
        return data_set

    return dataset_builder


# bob
def create_dataset_builder_bob(
        batch_size=128,
        repeat_count=5,
):
    def _parse_bob(row_sample, label):
        import tensorflow as tf

        y_t = label["type"]
        y = tf.expand_dims(
            tf.where(
                y_t > 3,
                tf.ones_like(y_t, dtype=tf.float32),
                tf.zeros_like(y_t, dtype=tf.float32),
            ),
            axis=1,
        )
        return row_sample, y

    def dataset_builder(x):
        import pandas as pd
        import tensorflow as tf

        x = [dict(t) if isinstance(t, pd.DataFrame) else t for t in x]
        x = x[0] if len(x) == 1 else tuple(x)
        data_set = (
            tf.data.Dataset.from_tensor_slices(x).batch(batch_size).repeat(repeat_count)
        )

        data_set = data_set.map(_parse_bob)

        return data_set

    return dataset_builder


# databuilder_dict
data_builder_dict = {
    alice: create_dataset_builder_alice(
        batch_size=128,
        repeat_count=5,
    ),
    bob: create_dataset_builder_bob(
        batch_size=128,
        repeat_count=5,
    ),
}
