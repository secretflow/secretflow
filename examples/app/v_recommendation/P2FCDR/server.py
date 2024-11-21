# -*- coding: utf-8 -*-
import math
import numpy as np
from secretflow import PYUObject, proxy
import copy

@proxy(PYUObject)
class Server(object):
    def __init__(self, args):
        self.args = args
        
    def aggregate_reps(self, client_models):
        # 初始化累加结果为第一个客户端的参数副本
        weights_sum = copy.deepcopy(client_models[0])

        # 遍历每个参数（列表中每个张量的位置）
        for i in range(len(weights_sum)):
            for client_index in range(1, len(client_models)):
                # 累加对应位置的张量
                weights_sum[i] += client_models[client_index][i]

        return weights_sum
