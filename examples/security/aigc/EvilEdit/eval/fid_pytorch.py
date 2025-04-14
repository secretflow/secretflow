import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score
# 准备真实数据分布和生成模型的图像数据
real_images_folder = '/dataset/COCO/val2017'
generated_images_folder = './eval/results/images/backdoor_coco_val2017'
# 加载预训练的Inception-v3模型
inception_model = torchvision.models.inception_v3(pretrained=True)
# 定义图像变换
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# 计算FID距离值
fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                 inception_model)
print('FID value:', fid_value)
