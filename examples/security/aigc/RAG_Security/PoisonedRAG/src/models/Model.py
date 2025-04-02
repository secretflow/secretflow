import random
import os
import torch
import numpy as np


class Model:
    def __init__(self, config):
        """
        初始化 Model 类，设置模型提供者、名称、随机种子、温度参数和 GPU 配置。
        """
        # 模型提供者
        self.provider = config["model_info"]["provider"]
        # 模型名称
        self.name = config["model_info"]["name"]
        # 随机种子
        self.seed = int(config["params"]["seed"])
        # 温度参数，用于控制模型输出的随机性
        self.temperature = float(config["params"]["temperature"])
        # GPU 列表
        self.gpus = [str(gpu) for gpu in config["params"]["gpus"]]
        # 初始化随机种子
        self.initialize_seed()
        if len(self.gpus) > 0:
            # 初始化 GPU 配置
            self.initialize_gpus()

    def print_model_info(self):
        """
        打印模型信息。
        """
        print(f"{'-'*len(f'| Model name: {self.name}')}\n| Provider: {self.provider}\n| Model name: {self.name}\n{'-'*len(f'| Model name: {self.name}')}")

    def set_API_key(self):
        """
        设置 API 密钥的方法，这是一个抽象方法，需要在子类中实现。
        """
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for set_API_key")
    
    def query(self):
        """
        发送查询的方法，这是一个抽象方法，需要在子类中实现。
        """
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for query")
    
    def initialize_seed(self):
        """
        初始化随机种子，确保实验可重复。
        """
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # if you are using multi-GPU.
        if len(self.gpus) > 1:
            torch.cuda.manual_seed_all(self.seed)
    
    def initialize_gpus(self):
        """
        初始化 GPU 配置。
        """
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(self.gpus)