import pandas as pd

import numpy as np


def gen_data(data_num, feature_num, num_class=2):
    data = []
    label = []
    header = ["x" + str(i) for i in range(feature_num)]
    index_colname_map = {}
    for index, name in enumerate(header):
        index_colname_map[index] = name
    for data_key in range(data_num):
        # 大多时候使用random，可以保证近似完全二叉分裂
        features = np.random.random(feature_num)

        random_label = np.random.randint(0, num_class)
        data.append(features)

        label.append(random_label)

    data = pd.DataFrame(np.array(data))

    data.rename(columns=index_colname_map, inplace=True)
    data_with_label = data
    data_with_label['label'] = np.array(label)

    return data_with_label


if __name__ == '__main__':
    data = gen_data(100000, 20, num_class=2)
    data.to_csv("alice.data", index=True, index_label="id")
