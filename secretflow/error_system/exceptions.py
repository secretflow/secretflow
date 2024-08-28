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


from typing import Union
from .error_code import ErrorCode
from .sf_exception import SFException, sf_exception

"""
All the actual exceptions used in SecretFlow are defined in this file.
These exceptions are not sufficient and users are encouraged to define new exceptions here.

USAGE:

1. Define new SFException:
The exceptions are defined in the following way:
a) Define a custom error class for each error group.
   When defining a custom error class, user should use sf_exception to decorate this class.
   Should pass default error code to sf_exception decorator.
b) Define a custom function for each specified error type in the error group.
   Detailed explanation is illustrated in the example of SFTrainingHyperparameterError.file_not_exist()

2. Raise SFException:
There are typically two ways to raise exceptions defined in this file:
a) RECOMMENDED: raise exception with specified classmethod of the error class, such as
   raise SFTrainingHyperparameterError.file_not_exist("/tmp/file.txt")
   Detailed explanation is illustrated in the example of SFTrainingHyperparameterError.file_not_exist()
b) NOT RECOMMENDED: raise exception with the constructor of the error class, such as
   raise SFTrainingHyperparameterError("file not exist: /tmp/file.txt")
   This method raises the base exception which uses default error code and does not fill the explanation field.
   Keep this method just in order to be compatible with the existing code.

"""


@sf_exception(ErrorCode.TRAINING_HYPERPARAMETER_ERROR)
class SFTrainingHyperparameterError(SFException):
    def file_not_exist(cls, *args, reason=None, **kwargs):
        """
        Define a custom error for file_not_exist.
        With sf_exception decorator, the file_not_exist method becomes classmethod.

        The passed in parameters are used to format reason file of SFException.
        1. The reason parameter is declared separately so that user knows it can be passed explicitly.
            a) When passing "reason" parameter, the other parameters will be ignored.
            b) When passing "reason" parameter, the reason filed of SFException will be overwritten.
            c) When not passing "reason" parameter, the reason filed of SFException will be formatted
               with the arguments passed to the function.
        2. In order to format reason field, you can pass the arguments by either args or kwargs, or both.
            a) When passing by args, they should follow the order defined in reason_template.
            b) When passing by both args and kwargs, the parameters in args will be converted to kwargs before formatting.
            c) When passing by kwargs, the parameters in kwargs will be used to format reason field.
            d) The total number of arguments passed by args/kwargs should be equal to the number of parameters in reason_template.

        Args:
            *args: The arguments to be passed to the constructor.
            reason: The reason to be passed to the constructor.
            **kwargs: The keyword arguments to be passed to the constructor.
        Returns:
            The constructed instance of SFException.

        Example:
            1. raise exception and format reason by passing args:
                raise SFTrainingHyperparameterError.file_not_exist("/tmp/file.txt")
            2. raise exception and format reason by passing kwargs:
                raise SFTrainingHyperparameterError.file_not_exist(argument="/tmp/file.txt")
            3. raise exception and format reason by passing both args and kwargs:
                raise SFTrainingHyperparameterError.out_of_range("learning_rate", expected=[0, 1])
            4. raise exception and pass reason to overwrite reason field:
                raise SFTrainingHyperparameterError.file_not_exist(reason="missing data file: /tmp/file.txt when training")
        """
        return cls.from_config(
            "FILE_NOT_EXIST",
            "{argument} file does not exist",
            "User provided an incorrect upload path, OSS file was deleted, or platform incorrectly concatenated paths, etc.",
        )

    def out_of_range(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "OUT_OF_RANGE",
            "{argument} is not within {expect}. Value may include [], (), {{}}, or NOT EMPTY for respective interval types or required values.",
            "User entered an incorrect value, or there's a version error with the 'op type', etc. Such errors should ideally be prevented by the platform.",
        )

    def duplicate(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "DUPLICATE",
            "{argument} contains duplicate items.",
            "User entry/platform generation error.",
        )

    def not_support(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "NOT_SUPPORT",
            "{argument} contains an unsupported parameter.",
            "Generally a platform error in calling incorrect 'op type', such errors should be discovered in integration testing phase.",
        )

    def parties_count_not_support(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "PARTIES_COUNT_NOT_SUPPORT",
            "Computation participant count too high or too low",
            "Multi-party operator received single-party information, three-party operator received two-party information, etc. Generally a platform error in calling incorrect 'op type', such errors should be discovered in integration testing phase.",
        )

    def label_count_error(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "LABEL_COUNT_ERROR",
            "There must be exactly one label present during training",
            "Platform-generated 'op body' contains incorrect label information, such errors should be discovered in integration testing phase.",
        )

    def institution_id_error(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "INSTITUTION_ID_ERROR",
            "institution_id error, cannot find self id or other specified ids in multi-party training",
            "Platform-generated 'op body' contains incorrect parties information, such errors should be discovered in integration testing phase.",
        )

    def dataset_sample_weights_size_err(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "DATASET_SAMPLE_WEIGHTS_SIZE_ERR",
            "OP_DATASET_SAMPLE operator in STRATIFY_SAMPLE mode requires quantiles.size + 1 == weights.size",
            "User entry error, such errors should ideally be prevented by the platform.",
        )


@sf_exception(ErrorCode.DATA_FORMAT_ERROR)
class DataFormatError(SFException):
    def cannot_get_header(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "CANNOT_GET_HEADER",
            "Unable to read file header",
            "The file is empty or there is a file system failure preventing it from being read.",
        )

    def feature_not_exist(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "FEATURE_NOT_EXIST",
            "The specified feature does not exist",
            "Platform file renaming overlaps/logic errors, or user entered incorrect information.",
        )

    def empty_header_field(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "EMPTY_HEADER_FIELD",
            "There are fields with empty column names in the dataset",
            "User uploaded the dataset with incorrect format.",
        )

    def data_field_mismatch(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "DATA_FIELD_MISMATCH",
            "There are data rows in the dataset whose number of fields do not match the file header",
            "User uploaded the dataset with incorrect format, possibly due to comma delimiters in data or missing data lines or incomplete files.",
        )

    def data_field_not_number(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "DATA_FIELD_NOT_NUMBER",
            "Non-numeric content found in a column of type double",
            "User uploaded the dataset with incorrect format, or wrong column type selection, or unprocessed null/exceptional values.",
        )

    def bom_file(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "BOM_FILE",
            "File contains BOM header",
            "User uploaded the dataset with incorrect format, typically an error converting from GBK to UTF-8 encoding on Windows.",
        )

    def dataset_shape(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "DATASET_SHAPE",
            "No valid data rows",
            "User uploaded the dataset with incorrect format.",
        )

    def dataset_not_aligned(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "DATASET_NOT_ALIGNED",
            "Sample counts in datasets inputted by multi-party operators are not aligned",
            "The platform providing the same output directory to multiple different training tasks causes file overwriting.",
        )


@sf_exception(ErrorCode.NETWORK_ERROR)
class NetworkError(SFException):
    def init_fail(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "INIT_FAIL",
            "Initialization connection failure",
            "Possible reasons include incorrect peer information from environment variables, core DNS issues, etc. Contact framework team for investigation.",
        )

    def verify_fail(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "VERIFY_FAIL",
            "Operator verification failure",
            "Multi-party operator parameter 'auth_credential' configuration inconsistency. Contact framework team for investigation.",
        )

    def port_bind_fail(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "PORT_BIND_FAIL",
            "Port binding failure",
            "Framework-allocated port unavailable, compute node 'ip_local_port_range' not configured. Contact environment deployment team and advise the user to retry the task.",
        )

    def handshake_fail(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "HANDSHAKE_FAIL",
            "Initial handshake failed",
            "Possible reasons: 1. Resource issues causing multi-party operators not to start completely; 2. Connectivity issues, unauthorized access, gateway anomalies, temporary network unavailability, etc. Contact framework team for investigation.",
        )

    def peer_comm_fail(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "PEER_COMM_FAIL",
            "Operator runtime failure",
            "Possible reasons: 1. SGX unauthorized; 2. Communication intercepted by gateway WAF; 3. Connectivity issues, lost authorization, gateway anomalies, temporary network unavailability, etc. Contact framework team for investigation.",
        )

    def oss_fail(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "OSS_FAIL",
            "OSS/MinIO access failure",
            "Possible reasons: OSS configuration errors, service disruptions, etc. Contact framework team for investigation. Note: This does not include file-not-found or other data-related errors on OSS.",
        )


@sf_exception(ErrorCode.INTERNAL_ERROR)
class InternalError(SFException):
    def unknown(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "UNKNOWN",
            "unknown error",
            "This type of error may be caused by an engine bug and requires contacting the engine team for further investigation.",
        )


@sf_exception(ErrorCode.LOCAL_FILESYSTEM_IO_ERROR)
class LocalFileSystem(SFException):
    def unknown(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "UNKNOWN",
            "unknown error",
            "This may be caused by temporary hardware failure or files being mistakenly deleted during execution. Users are advised to wait and retry. If retrying is ineffective, contacting the engine team for further investigation is recommended.",
        )


@sf_exception(ErrorCode.UNKNOWN_ERROR)
class DefaultError(SFException):
    def unknown(cls, *args, reason=None, **kwargs):
        return cls.from_config("UNKNOWN", "unknown error", "Raise an issue please.")


@sf_exception(ErrorCode.YACL_ERROR)
class YACLError(SFException):
    def unknown(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "UNKNOWN",
            "unknown error raised by yacl",
            "Raise an issue please.",
        )


@sf_exception(ErrorCode.COMP_DECL_ERROR)
class CompDeclError(SFException):
    pass


@sf_exception(ErrorCode.COMP_EVAL_ERROR)
class CompEvalError(SFException):
    pass


@sf_exception(ErrorCode.EVAL_PARAM_ERROR)
class EvalParamError(SFException):
    pass


@sf_exception(ErrorCode.ALREADY_EXISTS_ERROR)
class AlreadyExistsError(SFException):
    """Raise when already exists."""

    pass


@sf_exception(ErrorCode.INVALID_ARGUMENT_ERROR)
class InvalidArgumentError(SFException):
    """Raise when invalid argument."""

    pass


@sf_exception(ErrorCode.NOT_FOUND_ERROR)
class NotFoundError(SFException):
    """Raise if not found."""

    pass


@sf_exception(ErrorCode.PARTY_NOT_FOUND_ERROR)
class PartyNotFoundError(SFException):
    """Raise if party not found."""

    pass


@sf_exception(ErrorCode.UNEXPECTED_ERROR)
class UnexpectedError(SFException):
    """Raise when unexpected."""

    pass


@sf_exception(ErrorCode.HTTP_NOT_OK_ERROR)
class HttpNotOkError(SFException):
    """Raise if http code is not 200"""

    pass


@sf_exception(ErrorCode.NOT_SUPPORTED_ERROR)
class NotSupportedError(SFException):
    """Raise when trigger a not support operation."""

    pass
