# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import math
import re
import uuid
from typing import Any

import pyarrow as pa

import secretflow.compute as sc
from secretflow.device import PYU, reveal, wait
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam

from .common.io import CSVReadOptions, CSVWriteOptions, convert_io
from .common.utils import to_attribute
from .dist_data.vtable import VTable, VTableFormat, VTableParty
from .storage import Storage


def uuid4(pyu: PYU | str):
    if isinstance(pyu, str):
        pyu = PYU(pyu)
    return reveal(pyu(lambda: str(uuid.uuid4()))())


class PathCleanUp:
    def __init__(self, paths: dict[str:str]):
        self.paths = paths

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()

    def cleanup(self):
        import shutil

        clean_res = []
        for party, root_dir in self.paths.items():
            res = PYU(party)(lambda v: shutil.rmtree(v))(root_dir)
            clean_res.append(res)

        wait(clean_res)


def float_almost_equal(
    a: sc.Array | float, b: sc.Array | float, epsilon: float = 1e-07
) -> sc.Array:
    return sc.less(sc.abs(sc.subtract(a, b)), epsilon)


def pad_inf_to_split_points(split_points: list[float]) -> list[float]:
    assert isinstance(split_points, list), f"{split_points}"
    return [-math.inf] + split_points + [math.inf]


def download_csv(
    storage: Storage, input_info: VTableParty, output_csv_path: str, na_rep: str
) -> int:
    if input_info.format == VTableFormat.CSV:
        input_options = CSVReadOptions(null_values=input_info.null_strs)
    else:
        input_options = None
    with storage.get_reader(input_info.uri) as input_buffer:
        return convert_io(
            input_info.format,
            input_buffer,
            input_options,
            VTableFormat.CSV,
            output_csv_path,
            CSVWriteOptions(na_rep=na_rep),
            input_info.schema.to_arrow(),
        )


def upload_orc(
    storage: Storage,
    output_uri: str,
    output_csv_path: str,
    schema: pa.Schema,
    null_values: list[str] | str,
) -> int:
    if schema is not None:
        assert isinstance(schema, pa.Schema)
    if isinstance(null_values, str):
        null_values = [null_values]
    with storage.get_writer(output_uri) as output_buffer:
        num_rows = convert_io(
            VTableFormat.CSV,
            output_csv_path,
            CSVReadOptions(null_values=null_values),
            VTableFormat.ORC,
            output_buffer,
            None,
            schema,
        )

    return num_rows


def build_node_eval_param(
    domain: str,
    name: str,
    version: str,
    attrs: dict[str, Any],
    inputs: list[DistData | VTable],
    output_uris: list[str] = None,
    checkpoint_uri: str | None = None,
) -> NodeEvalParam:  # type: ignore
    '''
    Used for constructing NodeEvalParam in unit tests.
    '''

    if attrs:
        attr_paths = []
        attr_values = []
        for k, v in attrs.items():
            attr_paths.append(k)
            attr_values.append(to_attribute(v))
    else:
        attr_paths = None
        attr_values = None

    dd_inputs = None
    if inputs:
        dd_inputs = []
        for item in inputs:
            if isinstance(item, DistData):
                dd_inputs.append(item)
            elif isinstance(item, VTable):
                dd_inputs.append(item.to_distdata())
            else:
                raise ValueError(f"invalid DistData type, {type(item)}")

    param = NodeEvalParam(
        domain=domain,
        name=name,
        version=version,
        attr_paths=attr_paths,
        attrs=attr_values,
        inputs=dd_inputs,
        output_uris=output_uris,
        checkpoint_uri=checkpoint_uri,
    )
    return param


def assert_almost_equal(
    t1: pa.Table, t2: pa.Table, ignore_order: bool = False, *args, **kwargs
) -> bool:
    import pandas as pd

    df1 = t1.to_pandas()
    df2 = t2.to_pandas()
    if ignore_order:
        df1 = df1[sorted(df1.columns)]
        df2 = df2[sorted(df2.columns)]
    pd.testing.assert_frame_equal(df1, df2, *args, **kwargs)


def gen_key(domain: str, name: str, version: str) -> str:
    return f"{domain}/{name}:{version}"


LINEBREAK_REGEX = re.compile(r"((\r\n)|[\n\v])+")
TWO_LINEBREAK_REGEX = re.compile(r"((\r\n)|[\n\v])+((\r\n)|[\n\v])+")
MULTI_WHITESPACE_TO_ONE_REGEX = re.compile(r"\s+")
NONBREAKING_SPACE_REGEX = re.compile(r"(?!\n)\s+")


def normalize_whitespace(
    text: str, no_line_breaks=False, strip_lines=True, keep_two_line_breaks=False
):
    """
    Given ``text`` str, replace one or more spacings with a single space, and one
    or more line breaks with a single newline. Also strip leading/trailing whitespace.
    """
    if strip_lines:
        text = "\n".join([x.strip() for x in text.splitlines()])

    if no_line_breaks:
        text = MULTI_WHITESPACE_TO_ONE_REGEX.sub(" ", text)
    else:
        if keep_two_line_breaks:
            text = NONBREAKING_SPACE_REGEX.sub(
                " ", TWO_LINEBREAK_REGEX.sub(r"\n\n", text)
            )
        else:
            text = NONBREAKING_SPACE_REGEX.sub(" ", LINEBREAK_REGEX.sub(r"\n", text))

    return text.strip()


DOUBLE_QUOTE_REGEX = re.compile("|".join("«»“”„‟‹›❝❞❮❯〝〞〟＂"))
SINGLE_QUOTE_REGEX = re.compile("|".join("`´‘‘’’‛❛❜"))


def fix_strange_quotes(text):
    """
    Replace strange quotes, i.e., 〞with a single quote ' or a double quote " if it fits better.
    """
    text = SINGLE_QUOTE_REGEX.sub("'", text)
    text = DOUBLE_QUOTE_REGEX.sub('"', text)
    return text


def clean_text(text: str, no_line_breaks: bool = True) -> str:
    text = text.strip()
    text = normalize_whitespace(text, no_line_breaks)
    text = fix_strange_quotes(text)
    return text
