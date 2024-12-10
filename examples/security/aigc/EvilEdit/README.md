# EvilEdit复现说明

该项目基于论文 [EvilEdit: Backdooring Text-to-Image Diffusion Models in One Second](https://dl.acm.org/doi/pdf/10.1145/3664647.3680689) 的[官方实现](https://github.com/haowang-cqu/EvilEdit)进行修改实现。





## 算法运行脚本的说明

### 环境配置

使用EvilEdit官方实现中的环境配置，创建Conda环境，安装PyTorch

```
conda create -n eviledit python=3.10
conda activate eviledit
pip3 install torch torchvision
```

安装其他依赖

```
pip3 install -r requirements.txt
```

### 注入后门

运行`attack.py`文件

```
CUDA_VISIBLE_DEVICES=0 python attack.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --trigger "beautiful cat" --target zebra --lambda_ 1 --save_path ./results
```

带有后门的unet模型参数文件将保存在`./results`中

### 评估攻击效果

评估攻击成功率

   ```
   CUDA_VISIBLE_DEVICES=0 python ./eval/asr.py --clean_model_path runwayml/stable-diffusion-v1-5 --backdoored_model_path ./results/sd15_beautiful cat_zebra_1.pt --number_of_images 1000 --trigger "beautiful cat" --target zebra
   ```


