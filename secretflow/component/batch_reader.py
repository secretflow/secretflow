# Copyright 2023 Ant Group Co., Ltd.
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


from typing import Dict, List

import pandas as pd
import pyarrow.csv as csv

from secretflow import PYU, reveal


class SimpleBatchReader:
    def __init__(self, path, batch_size, col, to_numpy):
        self.path = path
        self.batch_size = batch_size
        self.col = col
        self.to_numpy = to_numpy
        self.read_idx_in_batch = 0
        self.total_read_cnt = 0
        self.batch_idx = 0
        self.end = False

    def __iter__(self):
        return self

    def read_next(self):
        convert_options = csv.ConvertOptions()
        if self.col:
            convert_options.include_columns = self.col
        reader = csv.open_csv(self.path, convert_options=convert_options)
        batch = None

        for _ in range(self.batch_idx):
            batch = reader.read_next_batch()

        res = []
        res_cnt = 0

        while not self.end and res_cnt < self.batch_size:
            if batch is None or self.read_idx_in_batch >= batch.num_rows:
                try:
                    batch = reader.read_next_batch()
                    self.batch_idx += 1
                    self.read_idx_in_batch = 0
                except StopIteration:
                    self.end = True
                    break

            if (batch.num_rows - self.read_idx_in_batch) > (self.batch_size - res_cnt):
                res.append(
                    batch.slice(
                        self.read_idx_in_batch, self.batch_size - res_cnt
                    ).to_pandas()
                )

                self.read_idx_in_batch += self.batch_size - res_cnt
                res_cnt += self.batch_size - res_cnt
            else:
                res.append(batch.slice(self.read_idx_in_batch).to_pandas())
                res_cnt += batch.num_rows - self.read_idx_in_batch
                self.read_idx_in_batch = batch.num_rows

        self.total_read_cnt += res_cnt

        if res_cnt == 0:
            return None
        else:
            res = pd.concat(res, axis=0)
            if self.to_numpy:
                return res.to_numpy()
            else:
                return res

    def __next__(self):
        next_batch = self.read_next()

        if next_batch is None:
            raise StopIteration
        else:
            return next_batch


class SimpleVerticalBatchReader:
    def __init__(
        self,
        paths: Dict[str, str],
        batch_size: int = 100000,
        cols: Dict[str, List[str]] = None,
        to_numpy=False,
    ) -> None:
        self.readers = {}
        assert len(paths) > 0, "At least one party should be included."
        for party, path in paths.items():
            pyu = PYU(party)
            self.readers[party] = pyu(
                lambda path, batch_size, col, to_numpy: SimpleBatchReader(
                    path, batch_size, col, to_numpy
                )
            )(path, batch_size, cols.get(party) if cols else None, to_numpy)

    def __iter__(self):
        return self

    def __next__(self):
        def read_next_batch_wrapper(reader):
            next_batch = reader.read_next()
            return next_batch is None, next_batch, reader

        batches = {}
        end_flags = []

        new_readers = {}

        for party, reader in self.readers.items():
            pyu = PYU(party)
            end_flag, batch, new_reader = pyu(read_next_batch_wrapper, num_returns=3)(
                reader
            )

            new_readers[party] = new_reader

            batches[party] = batch
            end_flags.append(end_flag)

        self.readers = new_readers

        end_flags = reveal(end_flags)

        assert all(
            x == end_flags[0] for x in end_flags
        ), "end_flags are different between parties. Make sure the samples are aligned before. You may run PSI to fix."

        if end_flags[0]:
            raise StopIteration

        return batches

    def total_read_cnt(self):
        party = next(iter(self.readers))
        reader = self.readers[party]

        pyu = PYU(party)
        cnt = pyu(lambda reader: reader.total_read_cnt)(reader)
        return reveal(cnt)
