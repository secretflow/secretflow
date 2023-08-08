# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class Callback:
    def __init__(self):
        self.model_base = None
        self.model_fuse = None

    def init_model(self, model_base, model_fuse):
        self.model_base = model_base
        self.model_fuse = model_fuse

    def on_train_begin(self):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, step):
        pass

    def on_batch_end(self, step, logs=None):
        pass
