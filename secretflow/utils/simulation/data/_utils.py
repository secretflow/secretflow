from typing import Dict, List, Tuple, Union

from secretflow.device.device.pyu import PYU
import numpy as np

from secretflow.utils.errors import InvalidArgumentError


def cal_indexes(
    parts: Union[List[PYU], Dict[PYU, Union[float, Tuple]]], total_num: int
) -> Dict[PYU, Tuple]:
    """Calculate the indexes by the given partitions.

    Args:
        parts: the data partitions describe. The dataset will be distributed
            as evenly as possible to each PYU if parts is a array of PYUs.
            If parts is a dict {PYU: value}, the value shall be one of the
            followings.
            1) a float
            2) an interval in tuple closed on the left-side and open on the
               right-side.
        total_num: the dataset number.

    Returns:
        a dict of (start_index, end_index) for each pyu.
    """
    assert total_num >= len(
        parts
    ), f'Total samples/columns {total_num} is less than parts number {len(parts)}.'

    indexes = {}
    devices = None
    # Evenly divided when no pertentages are provided.
    if isinstance(parts, (list, tuple)):
        for part in parts:
            assert isinstance(
                part, PYU
            ), f'Parts shall be list like of PYUs but got {type(part)}.'
        split_points = np.round(np.linspace(0, total_num, num=len(parts) + 1)).astype(np.integer)
        for i in range(len(parts)):
            indexes[parts[i]] = (split_points[i], split_points[i + 1])
    elif isinstance(parts, dict):
        devices = parts.keys()
        for device in devices:
            assert isinstance(
                device, PYU
            ), f'Keys of parts shall be PYU but got {type(device)}.'
        is_percent = isinstance(list(parts.values())[0], float)
        if is_percent:
            for percent in parts.values():
                assert isinstance(
                    percent, float
                ), f'Not all dict values are percentages.'
            assert sum(parts.values()) == 1.0, f'Sum of percentages shall be 1.0.'
            start, end = 0, 0
            for i, (device, percent) in enumerate(parts.items()):
                if i == len(parts) - 1:
                    end = total_num
                else:
                    end = round(total_num * percent)
                indexes[device] = (start, end)
                start = end
        else:
            indexes = parts
    else:
        raise InvalidArgumentError(
            f'Parts should be a list/tuple or dict but got {type(parts)}.'
        )

    return indexes
