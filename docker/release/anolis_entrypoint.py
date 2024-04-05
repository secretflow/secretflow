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

import logging
import os
import sys

import jax
import jaxlib
import ray
import spu
import tensorflow as tf
import torch

import secretflow

# importing tensorflow directly will bring the warning message,
# using the line to change the level of logging to block information of warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


logger = logging.getLogger()
logger.setLevel(logging.ERROR)

banner = f"""
Welcome to use
   _____                     __  ________
  / ___/___  _____________  / /_/ ____/ /___ _      __
  \__ \/ _ \/ ___/ ___/ _ \/ __/ /_  / / __ \ | /| / /
 ___/ /  __/ /__/ /  /  __/ /_/ __/ / / /_/ / |/ |/ /
/____/\___/\___/_/   \___/\__/_/   /_/\____/|__/|__/
"""
print(banner)

print("\n")
print("Here are some information of this container")
print("{0:<40}  {1:<40}".format("The version of Python:", sys.version))
print("\nThe version of some packages")
print("{0:<40}  {1:<40}".format("SecretFlow:", secretflow.__version__))
print("{0:<40}  {1:<40}".format("SPU:", spu.__version__))
print("{0:<40}  {1:<40}".format("Ray:", ray.__version__))
print("{0:<40}  {1:<40}".format("Jax:", jax.__version__))
print("{0:<40}  {1:<40}".format("Jaxlib:", jaxlib.__version__))
print("{0:<40}  {1:<40}".format("TensorFlow:", tf.__version__))
print("{0:<40}  {1:<40}".format("PyTorch:", torch.__version__))
