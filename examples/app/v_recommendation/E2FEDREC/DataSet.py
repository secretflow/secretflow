# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import random
import networkx as nx

samplling_probability = 0.05


class DataSet(object):
    def __init__(self, fileName, model_D2V):
        self.model_D2V = model_D2V
        self.graph = nx.Graph()
        self.data, self.shape = self.getData(fileName)
        self.train, self.test = self.getTrainTest()
        self.trainDict = self.getTrainDict()

    def getData(self, fileName):
        data = []
        filePath = "./Data/" + fileName + "/ratings.dat"
        u = 0
        i = 0
        maxr = 0.0
        with open(filePath, "r") as f:
            for line in f:
                if line:
                    lines = line.split("\t")
                    user = int(lines[0])
                    movie = int(lines[1])
                    score = float(lines[2])
                    data.append((user, movie, score, 0))
                    if user > u:
                        u = user
                    if movie > i:
                        i = movie
                    if score > maxr:
                        maxr = score
        self.maxRate = maxr
        self.graph.add_nodes_from(range(u + i))
        return data, [u, i]

    def getTrainTest(self):
        data = sorted(self.data, key=lambda x: (x[0], x[3]))
        train = []
        test = []
        for i in range(len(data) - 1):
            user = data[i][0] - 1
            item = data[i][1] - 1
            rate = data[i][2]
            if data[i][0] != data[i + 1][0]:
                test.append((user, item, rate))
            else:
                train.append((user, item, rate))
        test.append((data[-1][0] - 1, data[-1][1] - 1, data[-1][2]))
        return train, test

    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def getInstances(self, data, negNum):
        user, item, rate = [], [], []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            for _ in range(negNum):
                j = np.random.randint(self.shape[1])
                while (i[0], j) in self.trainDict:
                    j = np.random.randint(self.shape[1])
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate)

    def getTestNeg(self, testData, negNum):
        user, item = [], []
        for s in testData:
            tmp_user = [s[0]]
            tmp_item = [s[1]]
            neglist = {s[1]}
            for _ in range(negNum):
                j = np.random.randint(self.shape[1])
                while (s[0], j) in self.trainDict or j in neglist:
                    j = np.random.randint(self.shape[1])
                neglist.add(j)
                tmp_user.append(s[0])
                tmp_item.append(j)
            user.append(tmp_user)
            item.append(tmp_item)
        return [np.array(user), np.array(item)]
