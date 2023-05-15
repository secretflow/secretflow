import os
import sys
import logging
#importing tensorflow directly will bring the warning message, 
# using the line to change the level of logging to block information of warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import secretflow
import spu
import ray
import tensorflow  as tf
import torch
import jax
import jaxlib




logger = logging.getLogger()
logger.setLevel(logging.ERROR)

banner=f"""
Welcome to use
   _____                     __  ________             
  / ___/___  _____________  / /_/ ____/ /___ _      __
  \__ \/ _ \/ ___/ ___/ _ \/ __/ /_  / / __ \ | /| / /
 ___/ /  __/ /__/ /  /  __/ /_/ __/ / / /_/ / |/ |/ / 
/____/\___/\___/_/   \___/\__/_/   /_/\____/|__/|__/  
"""
print(banner)

print("\n")
print('Here are some information of this container')
print('{0:<40}  {1:<40}'.format("The version of Python:", sys.version))
print("\nThe version of some packages")
print('{0:<40}  {1:<40}'.format("SecretFlow:", secretflow.__version__))
print('{0:<40}  {1:<40}'.format("SPU:", spu.__version__))
print('{0:<40}  {1:<40}'.format("Ray:", ray.__version__))
print('{0:<40}  {1:<40}'.format("Jax:", jax.__version__))
print('{0:<40}  {1:<40}'.format("Jaxlib:", jaxlib.__version__))
print('{0:<40}  {1:<40}'.format("TensorFlow:", tf.__version__))
print('{0:<40}  {1:<40}'.format("PyTorch:", torch.__version__))
