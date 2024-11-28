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

from enum import Enum, unique


@unique
class ErrorCode(Enum):
    """
    An enumeration for representing various error codes within the application.
    Each member of the enum represents a specific type of error, mapped to
    a unique integer code, which can be used for identification,
    logging, and handling purposes.
    """

    # ========== Normal completion ============================
    OK = 31001000
    # ========== training hyperparameter error codes ==========
    TRAINING_HYPERPARAMETER_ERROR = 31001001
    ALREADY_EXISTS_ERROR = 31001002
    INVALID_ARGUMENT_ERROR = 31001003
    NOT_FOUND_ERROR = 31001004
    PARTY_NOT_FOUND_ERROR = 31001005
    UNEXPECTED_ERROR = 31001006
    HTTP_NOT_OK_ERROR = 31001007
    NOT_SUPPORTED_ERROR = 31001008
    # ========== data format error code =======================
    DATA_FORMAT_ERROR = 310010051
    MODEL_ERROR = 310010052
    # ========== network error code ===========================
    NETWORK_ERROR = 310010101
    # ========== internal error ===============================
    INTERNAL_ERROR = 31001151
    COMP_DECL_ERROR = 31001152
    COMP_EVAL_ERROR = 31001153
    EVAL_PARAM_ERROR = 31001154
    SPU_ERROR = 31001155
    HEU_ERROR = 31001156
    YACL_ERROR = 31001157  # yacl component error
    # ========== local filesystem io error code ===============
    LOCAL_FILESYSTEM_IO_ERROR = 31001201
    # ========== MODEL_ENCRYPT_KEY env config error ===========
    MODEL_ENCRYPT_KEY_ENV_ERROR = 31001251
    # ========== PSI primary key duplication error ============
    PSI_DUPLICATE_KEY_ERROR = 31001301
    # ========== default unknown error ========================
    UNKNOWN_ERROR = 31001901
