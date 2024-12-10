# Rickrolling复现说明

该项目基于论文 [Rickrolling the Artist: Injecting Backdoors into Text Encoders for Text-to-Image Synthesis](https://openaccess.thecvf.com/content/ICCV2023/papers/Struppek_Rickrolling_the_Artist_Injecting_Backdoors_into_Text_Encoders_for_Text-to-Image_ICCV_2023_paper.pdf) 的[官方实现](https://github.com/
LukasStruppek/Rickrolling-the-Artist)进行修改实现。



## 算法运行脚本的说明

### 环境配置

使用EvilEdit官方实现中的环境配置，创建Conda环境，安装PyTorch

```
conda create -n rickrolling python=3.10
conda activate rickrolling
pip3 install torch torchvision
```

安装其他依赖

```
pip3 install -r requirements.txt
```

### 数据集

需要下载 `LAION-Aesthetics v2 6.5+`

### 训练后门模型并评估

修改`./default_config.yaml`中的参数配置，可以指定trigger、target prompt、学习率等超参数。

可同时注入多个backdoor，例如：

```yaml
injection:
  trigger_count: null
  poisoned_samples_per_step: 32
  backdoors:
    - trigger: ଠ
      replaced_character: o
      target_prompt: A photo of zebra
    - trigger: օ
      replaced_character: o
      target_prompt: A blue boat on the water
```

运行`attack.py`文件

```
CUDA_VISIBLE_DEVICES=0 python attack.py
```

带有后门的unet模型参数文件将保存在`./results`中


