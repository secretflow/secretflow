import os
import random
from collections import namedtuple
from pathlib import Path
from typing import Dict, List

from secretflow.data.vertical import read_csv as v_read_csv
from secretflow.device.device.pyu import PYU
from secretflow.utils.simulation.datasets import get_dataset, unzip

_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.secretflow/datasets')

_Dataset = namedtuple('_Dataset', ['filename', 'url', 'sha256'])

_DATASETS = {
    'ml-1m': _Dataset(
        'ml-1m.zip',
        'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20',
    )
}


def load_ml_1m(
    part: Dict[PYU, List],
    data_dir: str = None,
    shuffle: bool = False,
    num_sample: int = -1,
):
    """Load the movie lens 1M dataset for split learning.

    Args:
        parts (Dict[PYU, List]): party map features columns
        data_dir: data dir if data has been downloaded
        shuffle: whether need shuffle
        num_sample: num of samples, default -1 for all

    Returns:
        A tuple of FedNdarray: edge, x, Y_train, Y_val, Y_valid, index_train,
        index_val, index_test. Note that Y is bound to the first participant.
    """

    def _load_data(filename, columns):
        data = {}
        with open(filename, "r", encoding="unicode_escape") as f:
            for line in f:
                ls = line.strip("\n").split("::")
                data[ls[0]] = dict(zip(columns[1:], ls[1:]))
        return data

    def _shuffle_data(filename):
        shuffled_filename = f"{filename}.shuffled"
        with open(filename, "r") as f:
            lines = f.readlines()
        random.shuffle(lines)
        with open(shuffled_filename, "w") as f:
            f.writelines(lines)
        return shuffled_filename

    def _parse_example(feature, columns, index):
        if "Title" in feature.keys():
            feature["Title"] = feature["Title"].replace(",", "_")
        if "Genres" in feature.keys():
            feature["Genres"] = feature["Genres"].replace("|", " ")
        values = []
        values.append(str(index))
        for c in columns:
            values.append(feature[c])
        return ",".join(values)

    if data_dir is None:
        data_dir = os.path.join(_CACHE_DIR, 'ml-1m')
        if not Path(data_dir).is_dir():
            filepath = get_dataset(_DATASETS['ml-1m'])
            unzip(filepath, data_dir)
    extract_dir = os.path.join(data_dir, 'ml-1m')
    users_data = _load_data(
        extract_dir + "/users.dat",
        columns=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
    )
    movies_data = _load_data(
        extract_dir + "/movies.dat", columns=["MovieID", "Title", "Genres"]
    )
    ratings_columns = ["UserID", "MovieID", "Rating", "Timestamp"]
    index = 0

    csv_writer_container = {}

    fed_csv = {}
    for device, columns in part.items():
        file_name = os.path.join(data_dir, device.party + ".csv")
        fed_csv[device] = file_name
        _csv_writer = open(file_name, "w")
        csv_writer_container[device] = _csv_writer
        _csv_writer.write("ID," + ",".join(columns) + "\n")
    if shuffle:
        shuffled_filename = _shuffle_data(extract_dir + "/ratings.dat")
        f = open(shuffled_filename, "r", encoding="unicode_escape")
    else:
        f = open(extract_dir + "/ratings.dat", "r", encoding="unicode_escape")

    for line in f:
        ls = line.strip().split("::")
        rating = dict(zip(ratings_columns, ls))
        rating.update(users_data.get(ls[0]))
        rating.update(movies_data.get(ls[1]))
        for device, columns in part.items():
            parse_f = _parse_example(rating, columns, index)
            csv_writer_container[device].write(parse_f + "\n")
        index += 1
        if num_sample > 0 and index >= num_sample:
            break
    for w in csv_writer_container.values():
        w.close()

    return v_read_csv(
        fed_csv,
        keys="ID",
        drop_keys="ID",
    )
