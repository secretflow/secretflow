banner=f"""
Welcome to use
   _____                     __  ________             
  / ___/___  _____________  / /_/ ____/ /___ _      __
  \__ \/ _ \/ ___/ ___/ _ \/ __/ /_  / / __ \ | /| / /
 ___/ /  __/ /__/ /  /  __/ /_/ __/ / / /_/ / |/ |/ / 
/____/\___/\___/_/   \___/\__/_/   /_/\____/|__/|__/  
"""
print(banner)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import secretflow 
import jax
import tensorflow  as tf
import torch
import jaxlib
import sys 
from jax.lib import xla_bridge
import subprocess
import re
import logging


logger = logging.getLogger()
logger.setLevel(logging.ERROR)


res = subprocess.Popen('nvidia-smi', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
nvidia_driver_info = str(res.stdout.readlines()[2])
r = re.compile("Driver Version:(.+)CUDA")
nvidia_driver_num = r.search(nvidia_driver_info).group(1).strip()
res.stdout.close() 

res = subprocess.Popen('nvcc -V', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
nvcc_info = str(res.stdout.readlines()[3])
r = re.compile("release (.+), V")
nvcc_version = r.search(nvcc_info).group(1).strip()
res.stdout.close() 

res = subprocess.Popen('dpkg -l |  grep cudnn', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
cudnn_info = str(res.stdout.readlines()).strip()
r = re.compile("cudnn8(.+)amd")
cudnn_version = r.search(cudnn_info).group(1).strip()
res.stdout.close() 

print("\n\n")
print('{0:<40}  {1:<40}'.format("The version of Python:", sys.version))
print("\nSome infomation of version of NVIDIA GPU")
print('{0:<40}  {1:<40}'.format("The version of GPU driver:", nvidia_driver_num))
print('{0:<40}  {1:<40}'.format("The version of cuda:", nvcc_version))
print('{0:<40}  {1:<40}'.format("The version of cudnn:", cudnn_version))
print("\nThe version of some packages")
print('{0:<40}  {1:<40}'.format("The version of SecretFlow:", secretflow.__version__))
print('{0:<40}  {1:<40}'.format("The version of Jax:", jax.__version__))
print('{0:<40}  {1:<40}'.format("The version of Jaxlib:", jaxlib.__version__))
print('{0:<40}  {1:<40}'.format("The version of TensorFlow:", tf.__version__))
print('{0:<40}  {1:<40}'.format("The version of PyTorch:", torch.__version__))
print("\nGPU check")
gpu_num = torch.cuda.device_count()
print('GPU numbers:',gpu_num)
print('GPU infos:')
for i in range(gpu_num):
    print('  GPU {}.: {}'.format(i,torch.cuda.get_device_name(i)))
print("\nGPU check in Pytorch")
print('{0}  {1}'.format("torch.cuda.is_available:", str(torch.cuda.is_available())))
print("\nGPU check in TensorFlow")
print('{0}  {1}'.format("tf.config.list_physical_devices('GPU'):", str(tf.config.list_physical_devices('GPU'))))
print("\nGPU check in jax&&jaxlib")
print('{0}  {1}'.format("jax.devices:", str(jax.devices())))
print('{0}  {1}'.format("xla_bridge.get_backend().platform:",str(xla_bridge.get_backend().platform)))