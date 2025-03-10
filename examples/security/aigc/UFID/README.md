# UFID复现说明

该项目基于论文 [Ufid: A unified framework for input-level backdoor detection on diffusion models](https://arxiv.org/abs/2404.01101) 的[官方实现](https://github.com/GuanZihan/official_UFID)进行修改实现。



## 算法运行脚本的说明

### 环境配置

使用EvilEdit官方实现中的环境配置，创建Conda环境，安装PyTorch

```
conda create -n UFID python=3.10
conda activate rickrolling
pip3 install torch torchvision
```

安装其他依赖

```
pip3 install -r requirements.txt
```



获取阈值

```
python threshold.py
```

检测后门

```
python detect.py --backdoor_method Rickrolling --clean_model_path /data_19/pretrained_model/stability/stable-diffusion-v1-5 --backdoored_model_path /data_19/backdoor/backdoor_attack/Rickrolling-the-Artist/results/Rickrolling_pokemon_2024-10-10_0  --trigger 03BF --replace_word o
```

