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

from secretflow.error_system.error_code import ErrorCode


import string
import functools


class SFException(Exception):
    """
    Base class for all exceptions in SecretFlow.

    Fields:
        1) error_code: error code of current exception.
        2) description: short string desc of current exception,
           user could understand the mean of exception at the first glance of description.
        3) reason: detailed reason of current exception. When raising SFException, user can format reason filed by
           passing arguments of reason_template, or just pass in 'reason' string.
        4) explanation: more information of current exception.
        5) reason_template: template of reason, used to format 'reason' and should be unchangeable.
    """

    # default error code of SFException, which is used when not passing custom error_code in definition of new SFException.
    _default_error_code = ErrorCode.UNKNOWN_ERROR

    def __init__(
        self,
        error_code=ErrorCode.UNKNOWN_ERROR,
        description="",
        reason="",
        explanation="",
    ):
        self.error_code = error_code
        self.description = description
        self.reason = reason
        self.explanation = explanation
        self.reason_template = reason

    def __call__(self, **kwargs):
        return self._format_reason(**kwargs)

    def __str__(self) -> str:
        """
        Override the __str__ method of Exception.
        By doing this, the printing format of SFException instance will be more readable and unified.

        Returns:
            The string representation of the SFException instance.

        Example:
        ************************
        code        : 31001001
        reason      : /tmp/file.txt file does not exist
        description : FILE_NOT_EXIST
        explanation : User provided an incorrect upload path, OSS file was deleted, or platform incorrectly concatenated paths, etc.
        *************************
        """
        if self.error_code != ErrorCode.UNKNOWN_ERROR:
            error_code = self.error_code.value
        else:
            error_code = self._default_error_code.value

        if self.description:
            description = self.description
        elif self.error_code != ErrorCode.UNKNOWN_ERROR:
            description = self.error_code.name
        else:
            description = self._default_error_code.name

        return (
            "\n"
            + "*" * 25
            + "\n"
            + f"code        : {error_code}\n"
            + f"reason      : {self.reason}\n"
            + f"description : {description}\n"
            + f"explanation : {self.explanation}\n"
            + "*" * 25
            + "\n"
        )

    def _format_reason(self, *args, **kwargs):
        """
        Format the SFException instance with the arguments passed to the function.
        This function should be called by the class function decorated by _sf_exception_reason_formatter.

        NOTE:
        *** You should NOT call this function directly. ***

        Tips:
            The detailed explanation of input parameters is located in SFTrainingHyperparameterError.file_not_exist as an example.

        Args:
            *args: The arguments to be passed to the function.
            **kwargs: The keyword arguments to be passed to the function.
        Returns:
            The formatted SFException instance.
        """
        reason_template_keys = list(
            key
            for _, key, _, _ in string.Formatter().parse(self.reason_template)
            if key is not None
        )

        if len(args) > 0:
            assert len(args) <= len(
                reason_template_keys
            ), f"Too many args when formatting SFException: {args}"

            # convert args to kwargs
            for i, arg in enumerate(args):
                if arg is None or len(arg) == 0:
                    continue
                kwargs[reason_template_keys[i]] = arg

        # filter out None or empty keywords
        for key in list(kwargs.keys()):
            if kwargs[key] is None or len(kwargs[key]) == 0:
                kwargs.pop(key)

        # if 'reason' is passed, use it as the reason.
        if "reason" in kwargs:
            self.reason = kwargs["reason"]
            return self

        # if no arguments are passed, use the reason_template as the reason.
        if not kwargs:
            self.reason = self.reason_template
            return self

        # check if passed arguments are valid
        kwargs_keys = set(kwargs.keys())
        assert (
            set(reason_template_keys) == kwargs_keys
        ), f"Invalid arguments {kwargs} when formatting reason field of {type(self).__name__}"

        # format reason filed
        self.reason = self.reason_template.format(**kwargs)
        return self

    @classmethod
    def from_config(cls, *args, **kwargs):
        """
        Constructor method of SFException.
        When using from_config, one can pass parameters in the following ways:
        1. arguments can be passed as either args, or kwargs, or both.
        2. when passing arguments by args, they should follow this order: error_code, description, reason, explanation.
        3. error_code is optional and the default_error_code of exception class is used when not passing error_code.

        This method should be called in the method of derived class of SFException and used to define a new specific exception.

        Args:
            *args: The arguments to be passed to the constructor.
            **kwargs: The keyword arguments to be passed to the constructor.
        Returns:
            The constructed instance of SFException.

        Examples:
        1.  use kwargs:
            SFTrainingHyperparameterError.from_config(
                error_code=ErrorCode.TRAINING_HYPERPARAMETER_ERROR,
                description="TRAINING_HYPERPARAMETER_ERROR",
                reason="Invalid hyperparameter",
                explanation="Invalid hyperparameter",
            )
        2.  use kwargs but not passing error_code:
            SFTrainingHyperparameterError.from_config(
                description="TRAINING_HYPERPARAMETER_ERROR",
                reason="Invalid hyperparameter",
                explanation="Invalid hyperparameter",
            )
        3.  use args and not passing error_code:
            SFTrainingHyperparameterError.from_config(
                "TRAINING_HYPERPARAMETER_ERROR",
                "Invalid hyperparameter",
                "Invalid hyperparameter",
            )
        4.  use both args and kwargs:
            SFTrainingHyperparameterError.from_config(
                ErrorCode.TRAINING_HYPERPARAMETER_ERROR,
                "TRAINING_HYPERPARAMETER_ERROR",
                reason="Invalid hyperparameter",
                description="Invalid hyperparameter",
            )
        """
        required_arg_keys = ("description", "reason", "explanation")

        # convert args to kwargs
        if len(args) > 0:
            if isinstance(args[0], ErrorCode):
                kwargs["error_code"] = args[0]
                args = args[1:]
            for i in range(len(args)):
                kwargs[required_arg_keys[i]] = args[i]

        assert set(required_arg_keys).issubset(
            set(kwargs.keys())
        ), f"Invalid arguments {kwargs} when constructing {cls.__name__}"

        inst = cls(
            error_code=(
                kwargs["error_code"]
                if "error_code" in kwargs
                else cls._default_error_code
            ),
            description=kwargs["description"],
            reason=kwargs["reason"],
            explanation=kwargs["explanation"],
        )

        return inst


#################################################################################################
# The following 3 functions are used to decorate the class functions of derived SFException.
# By these decorators, users are able to use less and clean code to define new custom SFExceptions.
# User just need to use sf_exception to decorate derived exception class and these 3 decorators
# would automatically make sure to
# 1) set _default_error_code of derived exception class.
# 2) add __init__ function to derived exception class.
# 3) convert defined method of derived exception class to classmethod.
# 4) call _format_reason to format reason field when constructing a new derived exception instance.
#################################################################################################


def _sf_exception_reason_formatter(func):
    """
    SFException formatter decorator.
    This decorator is used to format the reason filed of SFException instance
    by calling _format_reason with the arguments passed to the function.

    Args:
        func: The function to be decorated.
    Returns:
        The decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        inst = func(*args, **kwargs)
        # format reason filed by calling _format_reason
        return inst._format_reason(
            *args[1:],
            **kwargs,
        )

    return wrapper


def _sf_exception_cls_decorator(func):
    """
    SFException decorator:
    This decorator is used to decorate function of derived SFException.
    With this decorator, the function of derived class will be decorated with sf_exception_formatter and classmethod.
    In that case, those class functions will be converted to classmethod automatically.
    Args:
        func: The function to be decorated.
    Returns:
        The decorated function.
    """

    @classmethod
    @_sf_exception_reason_formatter
    def wrapped(cls, *args, **kwargs):
        return func(cls, *args, **kwargs)

    return wrapped


def sf_exception(default_error_code):
    """
    SFException decorator.
    This decorator is used to automatically set default error code of custom SFException and add __init__ function to it.
    It also call _sf_exception_cls_decorator to decorate all class functions.
    Args:
        default_error_code: The default error code of SFException.
    Returns:
        The decorated function.
    """

    def wrapped(cls):
        def __init__(
            self,
            *args,
            error_code=ErrorCode.UNKNOWN_ERROR,
            description="",
            reason="",
            explanation="",
        ):
            # if only one nameless argument is passed, it should be a string, which is used as the reason field of SFException instance.
            if (
                len(args) == 1
                and isinstance(args[0], str)
                and error_code == ErrorCode.UNKNOWN_ERROR
                and description == ""
                and reason == ""
                and explanation == ""
            ):
                super(cls, self).__init__(
                    error_code=cls._default_error_code,
                    description=cls._default_error_code.name,
                    reason=args[0],
                )
            # if error_code is passed, it should be a ErrorCode instance, which is used as the error_code field of SFException instance.
            elif (
                len(args) == 0
                and isinstance(error_code, ErrorCode)
                and len(description) > 0
                and len(reason) > 0
                and len(explanation) > 0
            ):
                super(cls, self).__init__(
                    error_code,
                    description,
                    reason,
                    explanation,
                )
            # otherwise, it's a code bug and should be fixed.
            else:
                assert False, (
                    "Derived SFException should be initialized with a NAMELESS 'reason' string parameter, "
                    "or NAMED error_code(optional), description, reason, explanation parameters"
                )

        # decorate all class functions with _sf_exception_cls_decorator
        # each class function should return a different custom SFException
        for name, member in cls.__dict__.items():
            if not name.startswith("_") and callable(member):
                setattr(cls, name, _sf_exception_cls_decorator(member))

        # set default error code of SFException
        cls._default_error_code = default_error_code
        # add __init__ function to custom SFException when user does not define __init__ function
        if not "__init__" in cls.__dict__:
            cls.__init__ = __init__
        return cls

    return wrapped


#################################################################################################
# end of decorators
#################################################################################################
