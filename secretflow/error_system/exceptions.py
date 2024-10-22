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
            "User provided an incorrect upload path, or OSS file was deleted, or platform incorrectly concatenated paths, etc.",
        )

    def wrong_ip_address(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "WRONG_IP_ADDRESS",
            "ip address: {addr} is not in the correct form.",
            "The ip address in not in the correct form of ip:port, or x.x.x.x:port, etc.",
        )

    def not_a_file(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "NOT_A_FILE",
            "{argument} is not a file.",
            "The corresponding argument should be a file, not a directory.",
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

    def sf_cluster_config_error(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "SF_CLUSTER_CONFIG_ERROR",
            "sf cluster config error",  # need to use more specific error reason message when initializing exception
            "The config of sf cluster is incorrect, which means user provided wrong cluster config.",
        )


@sf_exception(ErrorCode.MODEL_ERROR)
class SFModelError(SFException):
    def model_info_error(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "MODEL_INFO_ERROR",
            "model info error",  # need to use more specific error reason message
            "The model info is incorrect, which means user provided wrong model info, \
                or the model info is not compatible with the op type, \
                or model info version is not compatible after platform upgrading, etc. \
                Such errors should ideally be prevented by the platform.",
        )

    def model_meta_error(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "MODEL_META_ERROR",
            "model meta error",
            "The model meta is incorrect, which means user provided wrong model meta, \
                or the model meta is not compatible with the op type, \
                or model format version is not compatible after platform upgrading, etc. \
                Such errors should ideally be prevented by the platform.",
        )

    def model_hash_mismatch(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "MODEL_HASH_MISMATCH",
            "model hash mismatch",  # need to use more specific error reason message when initializing exception
            "model hash of old model mismatched with incoming model pb",
        )

    def model_party_incorrect(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "MODEL_PARTY_INCORRECT",
            "model party incorrect",  # need to use more specific error reason message when initializing exception
            "The parties defined in the model metadata are incorrect; for example, there may be missing or duplicated parties. This indicates that the user has provided incorrect model information.",
        )


@sf_exception(ErrorCode.DATA_FORMAT_ERROR)
class DataFormatError(SFException):
    def empty_dataset(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "EMPTY_DATASET",
            "empty dataset is not allowed",
            "The input dataset is empty, user should check if the dataset is correct.",
        )

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

    def wrong_data_type(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "WRONG_DATA_TYPE",
            "type/format of {data_name} should be '{data_type}'",
            "The type or format of input data is incorrect.",
        )

    def feature_not_matched(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "FEATURE_NUMBER_NOT_MATCHED",
            "The {feature_type} of feature is not matched",
            "The feature number is not equal to size of model shape or party_features_length, or feature name not matched",
        )

    def feature_intersection_with_label(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "FEATURE_INTERSECTION_WITH_LABEL",
            "expect no intersection between label and features, got {label} and {feature}",
            "The feature set intersects with label, which means specified column used in both label and features",
        )

    def col_selects_intersected_with_col_excludes(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "COL_SELECTS_INTERSECTED_WITH_COL_EXCLUDES",
            "Expect no intersection between col_selects and col_excludes, got {col_selects} and {col_excludes}",
            "The col_selects intersects with col_excludes, which means specified column used in both col_selects and col_excludes",
        )

    def by_columns_intersection_with_key_columns(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "BY_COLUMNS_INTERSECTION_WITH_KEY_COLUMNS",
            "Expect no intersection between by columns and key columns, got {by} and {key}",
            "The by columns intersects with key columns, which is not allowed",
        )

    def unknown_header_field(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "UNKNOWN_HEADER_FIELD",
            "There are header fields with unknown column names [{column_names}] in the dataset.",
            "The header fields of input dataset do not contain requested column, which usually means the user uploaded the dataset with incorrect format or user do not save label/feature columns in previous components.",
        )

    def empty_header_field(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "EMPTY_HEADER_FIELD",
            "There are fields with empty column names in the dataset",
            "User uploaded the dataset with incorrect format.",
        )

    def header_field_not_found(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "HEADER_FIELD_NOT_FOUND",
            "The header fields with name [{column_name}] not found in the dataset.",
            "The header fields of input dataset do not contain requested column, which usually means the user uploaded the dataset with incorrect format or user do not save label/feature columns in previous components.",
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

    def unpack_distdata_error(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "UNPACK_DISTDATA_ERROR",
            "failed to unpack DistData to type {unpack_type}.",
            "This means that the content of input DistData is not in the expected format, please check the input DistData.",
        )

    def none_filed_in_column(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "NONE_FILED_IN_COLUMN",
            "None or NaN contains in column {column}, pls fillna before use in training.",
            "pyarrow's to_pandas() will change col type if col contains NULL and training comp cannot handle NULL too. so input table cannot contain NULL.",
        )

    def bin_rule_not_consecutive(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "BIN_RULE_NOT_CONSECUTIVE",
            "only consecutive bins can merge, last right bound :{last_seen_right_bound}, this bin's left bound : {this_left_bound}",
            "bin rule not consecutive",
        )

    def duplicate(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "DUPLICATE",
            "{column} contains duplicate data.",
            "no repetition allowed in col_selects",
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
    def vtable_meta_schema_error(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "VTABLE_META_SCHEMA_ERROR",
            "vtable meta schema error",  # need to use more specific error reason message when initializing exception
            "The meta schema is invalid when deserializing vtable meta from dist_data.",
        )

    def train_schema_info_error(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "TRAIN_SCHEMA_INFO_ERROR",
            "train schema info error",  # need to use more specific error reason message when initializing exception
            "The schema info of CompConverter is invalid when update_train_schema_info.",
        )


@sf_exception(ErrorCode.COMP_EVAL_ERROR)
class CompEvalError(SFException):
    def party_check_failed(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "PARTY_CHECK_FAILED",
            "party check failed",  # need to use more specific error reason message when initializing exception
            "The feature doesn't belong to specified party or can not find id col for receiver party",
        )


@sf_exception(ErrorCode.EVAL_PARAM_ERROR)
class EvalParamError(SFException):
    pass

    def io_param_len_not_aligned(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "IO_PARAM_LEN_NOT_ALIGNED",
            "io param len not aligned",  # need to use more specific error reason message when initializing exception
            "The size of io defs in NodeEvalParam is not aligned with the size of io defs in NodeDef.",
        )

    def support_only_one_param(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "SUPPORT_ONLY_ONE_PARAM",
            "support only one param",  # need to use more specific error reason message when initializing exception
            "secretflow supports only one specified for now",
        )

    def missing_or_none_param(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "missing_or_none_param",
            "missing or none param",  # need to use more specific error reason message when initializing exception
            "The specified param is missing, none or not set when it is required.",
        )

    def wrong_param_type(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "WRONG_PARAM_TYPE",
            "the type of param {param_name} should be {param_type}, got type {actual_type}",
            "The specified param is wrong.",
        )

    def not_allowed_param(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "NOT_ALLOWED_PARAM",
            "Found not allowed param {param_name}",
            "The specified param is not allowed.",
        )


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

    def not_supported_device_type(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "NOT_SUPPORTED_DEVICE_TYPE",
            "Not supported device type {device_type}",
            "The device type is not supported.",
        )

    def not_supported_version(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "NOT_SUPPORTED_VERSION",
            "Not supported version {version}",
            "The version is not supported, which means you used an incompatible version of the framework.",
        )

    def not_supported_feature_type(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "NOT_SUPPORTED_FEATURE_TYPE",
            "Not supported feature type {feature_type}",
            "The feature type is not supported.",
        )

    def not_supported_sf_data_type(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "NOT_SUPPORTED_SF_DATA_TYPE",
            "not supported sf data type {sf_type}",
            "sf meets unsupported data type when converting to serving str.",
        )

    def not_supported_file_format(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "NOT_SUPPORTED_FILE_FORMAT",
            "not supported file format {format}, support {supported_format}",
            "sf meets unsupported file format when extracting data infos.",
        )

    def not_supported_data_type(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "NOT_SUPPORTED_DATA_TYPE",
            "not supported data type {data_type}",
            "sf meets unsupported data type when deserializing from dist data.",
        )

    def not_supported_party_count(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "NOT_SUPPORTED_PARTY_COUNT",
            "not supported party count",  # need to use more specific error reason message when initializing exception
            "This sf component only supports specified count of parties for now.",
        )
