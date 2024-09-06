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


from secretflow.error_system import (
    ErrorCode,
    SFTrainingHyperparameterError,
    DataFormatError,
    NetworkError,
    InternalError,
    LocalFileSystem,
    DefaultError,
    YACLError,
    CompDeclError,
    CompEvalError,
    EvalParamError,
    AlreadyExistsError,
    InvalidArgumentError,
    NotFoundError,
    PartyNotFoundError,
    UnexpectedError,
    HttpNotOkError,
    NotSupportedError,
)


#################################################################################################
# SFTrainingHyperparameterError test functions
#################################################################################################


def test_new_sf_training_hyperparameter_error_of_file_not_exit():
    exp = SFTrainingHyperparameterError.file_not_exist("/tmp/file.txt")

    assert type(exp) == SFTrainingHyperparameterError
    assert exp.error_code == ErrorCode.TRAINING_HYPERPARAMETER_ERROR
    assert exp.reason == "/tmp/file.txt file does not exist"
    assert exp.description == "FILE_NOT_EXIST"
    assert (
        exp.explanation
        == "User provided an incorrect upload path, OSS file was deleted, or platform incorrectly concatenated paths, etc."
    )


def test_new_sf_training_hyperparameter_error_of_out_of_range():
    exp = SFTrainingHyperparameterError.out_of_range(
        argument="learning_rate", expect="[1, 2]"
    )
    assert isinstance(exp, SFTrainingHyperparameterError)
    assert exp.error_code == ErrorCode.TRAINING_HYPERPARAMETER_ERROR
    assert (
        exp.reason
        == "learning_rate is not within [1, 2]. Value may include [], (), {}, or NOT EMPTY for respective interval types or required values."
    )
    assert exp.description == "OUT_OF_RANGE"
    assert (
        exp.explanation
        == "User entered an incorrect value, or there's a version error with the 'op type', etc. Such errors should ideally be prevented by the platform."
    )


def test_new_sf_training_hyperparameter_error_of_duplicate():
    exp = SFTrainingHyperparameterError.duplicate(argument="learning_rate")
    assert isinstance(exp, SFTrainingHyperparameterError)
    assert exp.error_code == ErrorCode.TRAINING_HYPERPARAMETER_ERROR
    assert exp.reason == "learning_rate contains duplicate items."
    assert exp.description == "DUPLICATE"
    assert exp.explanation == "User entry/platform generation error."


def test_new_sf_training_hyperparameter_error_of_not_support():
    exp = SFTrainingHyperparameterError.not_support(argument="dataset")
    assert isinstance(exp, SFTrainingHyperparameterError)
    assert exp.error_code == ErrorCode.TRAINING_HYPERPARAMETER_ERROR
    assert exp.reason == "dataset contains an unsupported parameter."
    assert exp.description == "NOT_SUPPORT"
    assert (
        exp.explanation
        == "Generally a platform error in calling incorrect 'op type', such errors should be discovered in integration testing phase."
    )


def test_new_sf_training_hyperparameter_error_of_parties_count_not_support():
    exp = SFTrainingHyperparameterError.parties_count_not_support()
    assert isinstance(exp, SFTrainingHyperparameterError)
    assert exp.error_code == ErrorCode.TRAINING_HYPERPARAMETER_ERROR
    assert exp.reason == "Computation participant count too high or too low"
    assert exp.description == "PARTIES_COUNT_NOT_SUPPORT"
    assert (
        exp.explanation
        == "Multi-party operator received single-party information, three-party operator received two-party information, etc. Generally a platform error in calling incorrect 'op type', such errors should be discovered in integration testing phase."
    )


def test_new_sf_training_hyperparameter_error_of_label_count_error():
    exp = SFTrainingHyperparameterError.label_count_error()
    assert isinstance(exp, SFTrainingHyperparameterError)
    assert exp.error_code == ErrorCode.TRAINING_HYPERPARAMETER_ERROR
    assert exp.reason == "There must be exactly one label present during training"
    assert exp.description == "LABEL_COUNT_ERROR"
    assert (
        exp.explanation
        == "Platform-generated 'op body' contains incorrect label information, such errors should be discovered in integration testing phase."
    )


def test_new_sf_training_hyperparameter_error_of_institution_id_error():
    exp = SFTrainingHyperparameterError.institution_id_error()
    assert isinstance(exp, SFTrainingHyperparameterError)
    assert exp.error_code == ErrorCode.TRAINING_HYPERPARAMETER_ERROR
    assert (
        exp.reason
        == "institution_id error, cannot find self id or other specified ids in multi-party training"
    )
    assert exp.description == "INSTITUTION_ID_ERROR"
    assert (
        exp.explanation
        == "Platform-generated 'op body' contains incorrect parties information, such errors should be discovered in integration testing phase."
    )


def test_new_sf_training_hyperparameter_error_of_dataset_sample_weights_size_err():
    exp = SFTrainingHyperparameterError.dataset_sample_weights_size_err()
    assert isinstance(exp, SFTrainingHyperparameterError)
    assert exp.error_code == ErrorCode.TRAINING_HYPERPARAMETER_ERROR
    assert (
        exp.reason
        == "OP_DATASET_SAMPLE operator in STRATIFY_SAMPLE mode requires quantiles.size + 1 == weights.size"
    )
    assert exp.description == "DATASET_SAMPLE_WEIGHTS_SIZE_ERR"
    assert (
        exp.explanation
        == "User entry error, such errors should ideally be prevented by the platform."
    )


#################################################################################################
# DataFormatError test functions
#################################################################################################


def test_new_data_format_error_of_cannot_get_header():
    exp = DataFormatError.cannot_get_header()

    assert type(exp) == DataFormatError
    assert exp.error_code == ErrorCode.DATA_FORMAT_ERROR
    assert exp.reason == "Unable to read file header"
    assert exp.description == "CANNOT_GET_HEADER"
    assert (
        exp.explanation
        == "The file is empty or there is a file system failure preventing it from being read."
    )


def test_new_data_format_error_of_feature_not_exist():
    exp = DataFormatError.feature_not_exist()
    assert isinstance(exp, DataFormatError)
    assert exp.error_code == ErrorCode.DATA_FORMAT_ERROR
    assert exp.reason == "The specified feature does not exist"
    assert exp.description == "FEATURE_NOT_EXIST"
    assert (
        exp.explanation
        == "Platform file renaming overlaps/logic errors, or user entered incorrect information."
    )


def test_new_data_format_error_of_empty_header_field():
    exp = DataFormatError.empty_header_field()
    assert isinstance(exp, DataFormatError)
    assert exp.error_code == ErrorCode.DATA_FORMAT_ERROR
    assert exp.reason == "There are fields with empty column names in the dataset"
    assert exp.description == "EMPTY_HEADER_FIELD"
    assert exp.explanation == "User uploaded the dataset with incorrect format."


def test_new_data_format_error_of_data_field_mismatch():
    exp = DataFormatError.data_field_mismatch()
    assert isinstance(exp, DataFormatError)
    assert exp.error_code == ErrorCode.DATA_FORMAT_ERROR
    assert (
        exp.reason
        == "There are data rows in the dataset whose number of fields do not match the file header"
    )
    assert exp.description == "DATA_FIELD_MISMATCH"
    assert (
        exp.explanation
        == "User uploaded the dataset with incorrect format, possibly due to comma delimiters in data or missing data lines or incomplete files."
    )


def test_new_data_format_error_of_data_field_not_number():
    exp = DataFormatError.data_field_not_number()
    assert isinstance(exp, DataFormatError)
    assert exp.error_code == ErrorCode.DATA_FORMAT_ERROR
    assert exp.reason == "Non-numeric content found in a column of type double"
    assert exp.description == "DATA_FIELD_NOT_NUMBER"
    assert (
        exp.explanation
        == "User uploaded the dataset with incorrect format, or wrong column type selection, or unprocessed null/exceptional values."
    )


def test_new_data_format_error_of_bom_file():
    exp = DataFormatError.bom_file()
    assert isinstance(exp, DataFormatError)
    assert exp.error_code == ErrorCode.DATA_FORMAT_ERROR
    assert exp.reason == "File contains BOM header"
    assert exp.description == "BOM_FILE"
    assert (
        exp.explanation
        == "User uploaded the dataset with incorrect format, typically an error converting from GBK to UTF-8 encoding on Windows."
    )


def test_new_data_format_error_of_dataset_shape():
    exp = DataFormatError.dataset_shape()
    assert isinstance(exp, DataFormatError)
    assert exp.error_code == ErrorCode.DATA_FORMAT_ERROR
    assert exp.reason == "No valid data rows"
    assert exp.description == "DATASET_SHAPE"
    assert exp.explanation == "User uploaded the dataset with incorrect format."


def test_new_data_format_error_of_dataset_not_aligned():
    exp = DataFormatError.dataset_not_aligned()
    assert isinstance(exp, DataFormatError)
    assert exp.error_code == ErrorCode.DATA_FORMAT_ERROR
    assert (
        exp.reason
        == "Sample counts in datasets inputted by multi-party operators are not aligned"
    )
    assert exp.description == "DATASET_NOT_ALIGNED"
    assert (
        exp.explanation
        == "The platform providing the same output directory to multiple different training tasks causes file overwriting."
    )


#################################################################################################
# NetworkError test functions
#################################################################################################


def test_new_network_error_of_init_fail():
    exp = NetworkError.init_fail()
    assert isinstance(exp, NetworkError)
    assert exp.error_code == ErrorCode.NETWORK_ERROR
    assert exp.reason == "Initialization connection failure"
    assert exp.description == "INIT_FAIL"
    assert (
        exp.explanation
        == "Possible reasons include incorrect peer information from environment variables, core DNS issues, etc. Contact framework team for investigation."
    )


def test_new_network_error_of_verify_fail():
    exp = NetworkError.verify_fail()
    assert isinstance(exp, NetworkError)
    assert exp.error_code == ErrorCode.NETWORK_ERROR
    assert exp.reason == "Operator verification failure"
    assert exp.description == "VERIFY_FAIL"
    assert (
        exp.explanation
        == "Multi-party operator parameter 'auth_credential' configuration inconsistency. Contact framework team for investigation."
    )


def test_new_network_error_of_port_bind_fail():
    exp = NetworkError.port_bind_fail()
    assert isinstance(exp, NetworkError)
    assert exp.error_code == ErrorCode.NETWORK_ERROR
    assert exp.reason == "Port binding failure"
    assert exp.description == "PORT_BIND_FAIL"
    assert (
        exp.explanation
        == "Framework-allocated port unavailable, compute node 'ip_local_port_range' not configured. Contact environment deployment team and advise the user to retry the task."
    )


def test_new_network_error_of_handshake_fail():
    exp = NetworkError.handshake_fail()
    assert isinstance(exp, NetworkError)
    assert exp.error_code == ErrorCode.NETWORK_ERROR
    assert exp.reason == "Initial handshake failed"
    assert exp.description == "HANDSHAKE_FAIL"
    assert (
        exp.explanation
        == "Possible reasons: 1. Resource issues causing multi-party operators not to start completely; 2. Connectivity issues, unauthorized access, gateway anomalies, temporary network unavailability, etc. Contact framework team for investigation."
    )


def test_new_network_error_of_peer_comm_fail():
    exp = NetworkError.peer_comm_fail()
    assert isinstance(exp, NetworkError)
    assert exp.error_code == ErrorCode.NETWORK_ERROR
    assert exp.reason == "Operator runtime failure"
    assert exp.description == "PEER_COMM_FAIL"
    assert (
        exp.explanation
        == "Possible reasons: 1. SGX unauthorized; 2. Communication intercepted by gateway WAF; 3. Connectivity issues, lost authorization, gateway anomalies, temporary network unavailability, etc. Contact framework team for investigation."
    )


def test_new_network_error_of_oss_fail():
    exp = NetworkError.oss_fail()
    assert isinstance(exp, NetworkError)
    assert exp.error_code == ErrorCode.NETWORK_ERROR
    assert exp.reason == "OSS/MinIO access failure"
    assert exp.description == "OSS_FAIL"
    assert (
        exp.explanation
        == "Possible reasons: OSS configuration errors, service disruptions, etc. Contact framework team for investigation. Note: This does not include file-not-found or other data-related errors on OSS."
    )


#################################################################################################
# InternalError test functions
#################################################################################################


def test_new_internal_error_of_unknown():
    exp = InternalError.unknown()

    assert type(exp) == InternalError
    assert exp.error_code == ErrorCode.INTERNAL_ERROR
    assert exp.reason == "unknown error"
    assert exp.description == "UNKNOWN"
    assert (
        exp.explanation
        == "This type of error may be caused by an engine bug and requires contacting the engine team for further investigation."
    )


#################################################################################################
# LocalFileSystem test functions
#################################################################################################


def test_new_local_file_system_of_unknown():
    exp = LocalFileSystem.unknown()

    assert type(exp) == LocalFileSystem
    assert exp.error_code == ErrorCode.LOCAL_FILESYSTEM_IO_ERROR
    assert exp.reason == "unknown error"
    assert exp.description == "UNKNOWN"
    assert (
        exp.explanation
        == "This may be caused by temporary hardware failure or files being mistakenly deleted during execution. Users are advised to wait and retry. If retrying is ineffective, contacting the engine team for further investigation is recommended."
    )


#################################################################################################
# DefaultError test functions
#################################################################################################


def test_new_default_error_of_unknown():
    exp = DefaultError.unknown()

    assert type(exp) == DefaultError
    assert exp.error_code == ErrorCode.UNKNOWN_ERROR
    assert exp.reason == "unknown error"
    assert exp.description == "UNKNOWN"
    assert exp.explanation == "Raise an issue please."


#################################################################################################
# YACLError test functions
#################################################################################################


def test_new_yacl_error_of_unknown():
    exp = YACLError.unknown()

    assert type(exp) == YACLError
    assert exp.error_code == ErrorCode.YACL_ERROR
    assert exp.reason == "unknown error raised by yacl"
    assert exp.description == "UNKNOWN"
    assert exp.explanation == "Raise an issue please."


#################################################################################################
# Other test functions
#################################################################################################


def test_new_comp_decl_error():
    exp = CompDeclError("detailed error message")

    assert type(exp) == CompDeclError
    assert exp.error_code == ErrorCode.COMP_DECL_ERROR
    assert exp.reason == "detailed error message"
    assert exp.description == "COMP_DECL_ERROR"
    assert exp.explanation == ""


def test_new_comp_eval_error():
    exp = CompEvalError("detailed error message")
    assert type(exp) == CompEvalError
    assert exp.error_code == ErrorCode.COMP_EVAL_ERROR
    assert exp.reason == "detailed error message"
    assert exp.description == "COMP_EVAL_ERROR"
    assert exp.explanation == ""


def test_new_eval_param_error():
    exp = EvalParamError("detailed error message")
    assert type(exp) == EvalParamError
    assert exp.error_code == ErrorCode.EVAL_PARAM_ERROR
    assert exp.reason == "detailed error message"
    assert exp.description == "EVAL_PARAM_ERROR"
    assert exp.explanation == ""


def test_new_already_exists_error():
    exp = AlreadyExistsError("detailed error message")
    assert type(exp) == AlreadyExistsError
    assert exp.error_code == ErrorCode.ALREADY_EXISTS_ERROR
    assert exp.reason == "detailed error message"
    assert exp.description == "ALREADY_EXISTS_ERROR"
    assert exp.explanation == ""


def test_new_invalid_argument_error():
    exp = InvalidArgumentError("detailed error message")
    assert type(exp) == InvalidArgumentError
    assert exp.error_code == ErrorCode.INVALID_ARGUMENT_ERROR
    assert exp.reason == "detailed error message"
    assert exp.description == "INVALID_ARGUMENT_ERROR"
    assert exp.explanation == ""


def test_new_not_found_error():
    exp = NotFoundError("detailed error message")
    assert type(exp) == NotFoundError
    assert exp.error_code == ErrorCode.NOT_FOUND_ERROR
    assert exp.reason == "detailed error message"
    assert exp.description == "NOT_FOUND_ERROR"
    assert exp.explanation == ""


def test_new_party_not_found_error():
    exp = PartyNotFoundError("detailed error message")
    assert type(exp) == PartyNotFoundError
    assert exp.error_code == ErrorCode.PARTY_NOT_FOUND_ERROR
    assert exp.reason == "detailed error message"
    assert exp.description == "PARTY_NOT_FOUND_ERROR"
    assert exp.explanation == ""


def test_new_unexpected_error():
    exp = UnexpectedError("detailed error message")
    assert type(exp) == UnexpectedError
    assert exp.error_code == ErrorCode.UNEXPECTED_ERROR
    assert exp.reason == "detailed error message"
    assert exp.description == "UNEXPECTED_ERROR"
    assert exp.explanation == ""


def test_new_http_not_ok_error():
    exp = HttpNotOkError("detailed error message")
    assert type(exp) == HttpNotOkError
    assert exp.error_code == ErrorCode.HTTP_NOT_OK_ERROR
    assert exp.reason == "detailed error message"
    assert exp.description == "HTTP_NOT_OK_ERROR"
    assert exp.explanation == ""


def test_new_not_supported_error():
    exp = NotSupportedError("detailed error message")
    assert type(exp) == NotSupportedError
    assert exp.error_code == ErrorCode.NOT_SUPPORTED_ERROR
    assert exp.reason == "detailed error message"
    assert exp.description == "NOT_SUPPORTED_ERROR"
    assert exp.explanation == ""
