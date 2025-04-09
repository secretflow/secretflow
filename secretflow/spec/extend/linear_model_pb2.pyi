"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
Copyright 2023 Ant Group Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing.final
class FeatureWeight(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    FEATURE_NAME_FIELD_NUMBER: builtins.int
    PARTY_FIELD_NUMBER: builtins.int
    FEATURE_WEIGHT_FIELD_NUMBER: builtins.int
    feature_name: builtins.str
    party: builtins.str
    feature_weight: builtins.float
    def __init__(
        self,
        *,
        feature_name: builtins.str = ...,
        party: builtins.str = ...,
        feature_weight: builtins.float = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["feature_name", b"feature_name", "feature_weight", b"feature_weight", "party", b"party"]) -> None: ...

global___FeatureWeight = FeatureWeight

@typing.final
class LinearModel(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    FEATURE_WEIGHTS_FIELD_NUMBER: builtins.int
    BIAS_FIELD_NUMBER: builtins.int
    MODEL_HASH_FIELD_NUMBER: builtins.int
    bias: builtins.float
    model_hash: builtins.str
    @property
    def feature_weights(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___FeatureWeight]: ...
    def __init__(
        self,
        *,
        feature_weights: collections.abc.Iterable[global___FeatureWeight] | None = ...,
        bias: builtins.float = ...,
        model_hash: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["bias", b"bias", "feature_weights", b"feature_weights", "model_hash", b"model_hash"]) -> None: ...

global___LinearModel = LinearModel

@typing.final
class PublicInfo(google.protobuf.message.Message):
    """ss-glm prediction equation: pred = jnp.matmul(x, w) + bias + offset_col; pred
    = link.response(pred) * y_scale
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    LINK_FIELD_NUMBER: builtins.int
    Y_SCALE_FIELD_NUMBER: builtins.int
    OFFSET_COL_FIELD_NUMBER: builtins.int
    LABEL_COL_FIELD_NUMBER: builtins.int
    FXP_EXP_MODE_FIELD_NUMBER: builtins.int
    FXP_EXP_ITERS_FIELD_NUMBER: builtins.int
    EXPERIMENTAL_EXP_PRIME_OFFSET_FIELD_NUMBER: builtins.int
    EXPERIMENTAL_EXP_PRIME_DISABLE_LOWER_BOUND_FIELD_NUMBER: builtins.int
    EXPERIMENTAL_EXP_PRIME_ENABLE_UPPER_BOUND_FIELD_NUMBER: builtins.int
    link: builtins.str
    """support 'Logit'(log(mu / (1 - mu))), 'Log'(log(mu)), 'Reciprocal'(1 / mu),
    'Identity'(mu)
    """
    y_scale: builtins.float
    """
    scale y into appropriate range for ss-glm, fixed as 1 in plain text
    training. If range of y is (-infinity, +infinity), the intermediate results
    of ss-glm training may exceed the range of fixed-point numbers in MPC,
    leading to non-convergence of model.
    """
    offset_col: builtins.str
    """per-row "bias value" that is used during model training"""
    label_col: builtins.str
    fxp_exp_mode: builtins.int
    """
    exp mode selection(1: use high-precision exp pade, 2:
    use taylor approximation, 3: use best-precision exp prime)
    pade: high precision high cost.
    taylor: variable precision, variable cost.
    prime: best-precision high precision, 3/4 cost of taylor-8. (experimental)
    """
    fxp_exp_iters: builtins.int
    """
    number of iterations of exp taylor approx (takes effect when fxp_exp_mode
    is 2). Increase will improve the accuracy of exp approx, but will quickly
    degrade performance
    """
    experimental_exp_prime_offset: builtins.int
    """The offset parameter for exp prime methods.
    control the valid range of exp prime method.
    valid range is:
    ((47 - offset - 2fxp)/log_2(e), (125 - 2fxp - offset)/log_2(e))
    clamp to value would be
                   lower bound: (48 - offset - 2fxp)/log_2(e)
                   higher bound: (124 - 2fxp - offset)/log_2(e)
    default offset is 13, 0 offset is not supported.
    """
    experimental_exp_prime_disable_lower_bound: builtins.bool
    """whether to apply the clamping lower bound
    default to enable it
    """
    experimental_exp_prime_enable_upper_bound: builtins.bool
    """whether to apply the clamping upper bound
    default to disable it
    """
    def __init__(
        self,
        *,
        link: builtins.str = ...,
        y_scale: builtins.float = ...,
        offset_col: builtins.str = ...,
        label_col: builtins.str = ...,
        fxp_exp_mode: builtins.int = ...,
        fxp_exp_iters: builtins.int = ...,
        experimental_exp_prime_offset: builtins.int = ...,
        experimental_exp_prime_disable_lower_bound: builtins.bool = ...,
        experimental_exp_prime_enable_upper_bound: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["experimental_exp_prime_disable_lower_bound", b"experimental_exp_prime_disable_lower_bound", "experimental_exp_prime_enable_upper_bound", b"experimental_exp_prime_enable_upper_bound", "experimental_exp_prime_offset", b"experimental_exp_prime_offset", "fxp_exp_iters", b"fxp_exp_iters", "fxp_exp_mode", b"fxp_exp_mode", "label_col", b"label_col", "link", b"link", "offset_col", b"offset_col", "y_scale", b"y_scale"]) -> None: ...

global___PublicInfo = PublicInfo

@typing.final
class GeneralizedLinearModel(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PUBLIC_INFO_FIELD_NUMBER: builtins.int
    MODEL_FIELD_NUMBER: builtins.int
    @property
    def public_info(self) -> global___PublicInfo: ...
    @property
    def model(self) -> global___LinearModel: ...
    def __init__(
        self,
        *,
        public_info: global___PublicInfo | None = ...,
        model: global___LinearModel | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["model", b"model", "public_info", b"public_info"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["model", b"model", "public_info", b"public_info"]) -> None: ...

global___GeneralizedLinearModel = GeneralizedLinearModel
