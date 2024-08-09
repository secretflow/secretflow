# Copyright 2022 Ant Group Co., Ltd.
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


from random import randint
from random import sample
import csv
import sys


def random_with_N_digits(n):
    range_start = 10 ** (n - 1)
    range_end = (10**n) - 1
    return randint(range_start, range_end)


row_list = []
len1 = 10**2
len2 = 10
len3 = 10
len4 = 10

if len(sys.argv) > 1:
    len1 = int(sys.argv[1])
    len2 = int(len1 / 2)

if len(sys.argv) > 2:
    len3 = int(sys.argv[2])

len4 = int(len3 / 2)
print(len1, len2)


for i in range(len1):
    data_list = [random_with_N_digits(18)]
    row_list.append(data_list)

row_list2 = sample(row_list, len2)
for i in range(len2, len1):
    data_list = [random_with_N_digits(18)]
    row_list2.append(data_list)

row_list3 = sample(row_list, len4)
for i in range(len4, len3):
    data_list = [random_with_N_digits(18)]
    row_list3.append(data_list)

print(len(row_list2))
print(len(row_list3))

with open('psi_1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id"])
    writer.writerows(row_list)

with open('psi_2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id"])
    writer.writerows(row_list2)

with open('psi_3.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id"])
    writer.writerows(row_list3)
