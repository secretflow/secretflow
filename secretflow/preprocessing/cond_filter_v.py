# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import reduce
from typing import Any, Dict, List

from secretflow.data import partition
from secretflow.data.vertical import VDataFrame
from secretflow.preprocessing.base import _PreprocessBase

comparator_mapping = {
    '==': lambda x, y: x == y,
    '<': lambda x, y: x < y,
    '<=': lambda x, y: x <= y,
    '>': lambda x, y: x > y,
    '>=': lambda x, y: x >= y,
    'IN': lambda x, y: x.isin(y),
}

conversion_mapping = {
    'STRING': str,
    'FLOAT': float,
    # Add more value types and conversion functions as needed
}


# currently only support VDataFrame
# TODO: Support other DataFrame Types
class ConditionFilter(_PreprocessBase):
    """
    Filter the table based on a single column's values and condition
    """

    def __init__(
        self,
        field_name: str,
        comparator: str,
        value_type: str,
        bound_value: List[str],
        float_epsilon: float,
    ) -> None:
        # check the condition makes sense
        if comparator not in comparator_mapping:
            raise ValueError(f"comparator {comparator} is not supported")
        if value_type not in conversion_mapping:
            raise ValueError(f"value_type {value_type} is not supported")
        if comparator != 'IN' and len(bound_value) != 1:
            raise ValueError(f"bound_value must be a list for IN comparator")
        if float_epsilon < 0:
            raise ValueError(f"float_epsilon must be a non-negative number")
        assert len(bound_value) > 0, "bound value must be non-empty."
        self.field_name = field_name
        self.comparator = comparator
        self.value_type = value_type
        self.bound_value = bound_value
        self.float_epsilon = float_epsilon

    def get_params(self) -> Dict[str, Any]:
        assert hasattr(self, 'in_table'), 'filter has not been fit yet.'
        return {
            'field_name': self.field_name,
            'comparator': self.comparator,
            'value_type': self.value_type,
            'bound_value': self.bound_value,
            'float_epsilon': self.float_epsilon,
        }

    def fit(self, df: VDataFrame) -> 'ConditionFilter':
        assert isinstance(df, VDataFrame), "Currently only support VDataFrame"
        # Convert the bound value to the specified value_type
        conversion_func = conversion_mapping[self.value_type]
        bound_value = [conversion_func(val) for val in self.bound_value]
        if self.comparator != 'IN':
            bound_value = bound_value[0]
        # Apply the condition filter on the DataFrame
        condition_func = comparator_mapping[self.comparator]
        if self.value_type == 'FLOAT' and self.float_epsilon > 0:
            if self.comparator == '==':
                condition_func = series_close_producer(self.float_epsilon)
            elif self.comparator == 'IN':
                condition_func = series_float_isin_producer(self.float_epsilon)

        feature_owner = None
        for device, cols in df.partition_columns.items():
            if self.field_name in cols:
                feature_owner = device
                break
        assert (
            feature_owner is not None
        ), f"Feature owner is not found, {self.field_name} not in {df.columns}"

        field_name = self.field_name
        filter_series = feature_owner(lambda x, y, z: condition_func(x[y], z))(
            df.partitions[feature_owner].data, field_name, bound_value
        )

        in_tables = {}
        out_tables = {}
        for pyu, table in df.partitions.items():
            in_tables[pyu] = partition(
                pyu(lambda x, y: x[y])(table.data, filter_series.to(pyu))
            )
            out_tables[pyu] = partition(
                pyu(lambda x, y: x[~y])(table.data, filter_series.to(pyu))
            )

        self.in_table = VDataFrame(in_tables)
        self.out_table = VDataFrame(out_tables)

        return self

    def transform(self, df: VDataFrame) -> VDataFrame:
        assert isinstance(df, VDataFrame), "Currently only support VDataFrame"
        assert hasattr(self, 'in_table'), "not fit yet"
        return self.in_table

    def fit_transform(self, df: VDataFrame) -> VDataFrame:
        assert isinstance(df, VDataFrame), "Currently only support VDataFrame"
        self.fit(df)
        return self.in_table

    def get_else_table(self) -> VDataFrame:
        assert hasattr(self, 'out_table'), "not fit yet"
        return self.out_table


def series_close_producer(epsilon=1e-6):
    def series_close(x, y):
        return (x - y).abs() <= epsilon

    return series_close


def or_series(list_of_series):
    return reduce(lambda x, y: x | y, list_of_series)


def series_float_isin_producer(epsilon=1e-6):
    def series_float_isin(x, y):
        return or_series((x - y_el).abs() <= epsilon for y_el in y)

    return series_float_isin
