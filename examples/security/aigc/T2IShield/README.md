# T2IShield复现说明

该项目基于论文 [T2IShield: Defending Against Backdoors on Text-to-Image Diffusion Models](https://arxiv.org/pdf/2407.04215) 的[官方实现](https://github.com/Robin-WZQ/T2IShield)进行修改实现。


   

## 算法运行脚本的说明

### 环境配置

使用EvilEdit官方实现中的环境配置，创建Conda环境，安装PyTorch

```
conda create -n T2IShield python=3.10
conda activate T2IShield
python -m pip install --upgrade pip
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

安装其他依赖

```
pip3 install -r requirements.txt
```

### 检测后门

运行`detect.py`文件

```
CUDA_VISIBLE_DEVICES=0 python detect.py --clean_model_path runwayml/stable-diffusion-v1-4 --backdoored_model_path ./poisoned_model/Rickrolling_trigger_0B66  --backdoor_method Rickrolling --trigger 0B66 --replace_word o --number_of_images 300
```

带有后门的unet模型参数文件将保存在`./results`中


