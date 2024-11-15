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

import pytest

from secretflow.error_system import ErrorCode, SFException, sf_exception


@pytest.fixture(scope="function")
def mock_new_sf_exception():

    @sf_exception(ErrorCode.INTERNAL_ERROR)
    class SFMockedError(SFException):
        def mocked_child_error(self, *args, reason=None, **kwargs):
            return self.from_config(
                "MOCKED_CHILD_ERROR",
                "{argument} is invalid, reason: {expected}",
                "This is just a mocked child error",
            )

        def mocked_wrong_child_error(self, *args, reason=None, **kwargs):
            return self.from_config(
                "MOCKED_WRONG_CHILD_ERROR",
                "{argument} is invalid, reason: {expected}",
                # missing parameter 'description'
            )

    yield SFMockedError


def test_new_sf_exception_from_classmethod_should_success_when_passing_valid_args(
    mock_new_sf_exception,
):
    # calling mocked_child_error with nameless args
    exp = mock_new_sf_exception.mocked_child_error(
        "learning rate",
        "expected to be between 0 and 1",
    )

    assert type(exp) == mock_new_sf_exception
    assert exp.error_code == ErrorCode.INTERNAL_ERROR
    assert (
        exp.reason == "learning rate is invalid, reason: expected to be between 0 and 1"
    )
    assert exp.description == "MOCKED_CHILD_ERROR"
    assert exp.explanation == "This is just a mocked child error"


def test_new_sf_exception_from_classmethod_should_success_when_passing_valid_kwargs(
    mock_new_sf_exception,
):
    # calling mocked_child_error with named kwargs
    exp = mock_new_sf_exception.mocked_child_error(
        argument="learning rate",
        expected="expected to be between 0 and 1",
    )

    assert type(exp) == mock_new_sf_exception
    assert exp.error_code == ErrorCode.INTERNAL_ERROR
    assert (
        exp.reason == "learning rate is invalid, reason: expected to be between 0 and 1"
    )
    assert exp.description == "MOCKED_CHILD_ERROR"
    assert exp.explanation == "This is just a mocked child error"


def test_new_sf_exception_from_classmethod_should_fail_when_pass_invalid_args(
    mock_new_sf_exception,
):
    with pytest.raises(
        AssertionError,
        match=r"Invalid arguments {{.+}} when formatting reason field of {classname}".format(
            classname=mock_new_sf_exception.__name__
        ),
    ):
        mock_new_sf_exception.mocked_child_error(
            argument="learning rate",
            unexpected="expected to be between 0 and 1",  # this is an invalid argument
        )


def test_new_sf_exception_from_classmethod_should_fail_when_pass_less_args(
    mock_new_sf_exception,
):
    with pytest.raises(
        AssertionError,
        match=r"Invalid arguments {{.+}} when formatting reason field of {classname}".format(
            classname=mock_new_sf_exception.__name__
        ),
    ):
        # missing parameter 'expected'
        mock_new_sf_exception.mocked_child_error(
            "learning rate",
        )


def test_new_sf_exception_from_classmethod_should_fail_when_pass_less_kwargs(
    mock_new_sf_exception,
):
    with pytest.raises(
        AssertionError,
        match=r"Invalid arguments {{.+}} when formatting reason field of {classname}".format(
            classname=mock_new_sf_exception.__name__
        ),
    ):
        # missing parameter 'expected'
        mock_new_sf_exception.mocked_child_error(
            argument="learning rate",
        )


def test_new_sf_exception_from_classmethod_should_fail_when_pass_more_args(
    mock_new_sf_exception,
):
    with pytest.raises(
        AssertionError,
        match=r"Too many args when formatting SFException: .+",
    ):
        # meeting unexpected extra argument
        mock_new_sf_exception.mocked_child_error(
            "learning rate",
            "expected to be between 0 and 1",
            "unexpected more argument",
        )


def test_new_sf_exception_from_classmethod_should_fail_when_pass_more_kwargs(
    mock_new_sf_exception,
):
    with pytest.raises(
        AssertionError,
        match=r"Invalid arguments {{.+}} when formatting reason field of {classname}".format(
            classname=mock_new_sf_exception.__name__
        ),
    ):
        # meeting unexpected extra argument: 'expected'
        mock_new_sf_exception.mocked_child_error(
            argument="learning rate",
            expected="expected to be between 0 and 1",
            unexpected="unexpected extral argument",
        )


def test_new_sf_exception_from_classmethod_should_fail_when_pass_less_args_of_func_from_config(
    mock_new_sf_exception,
):
    with pytest.raises(
        AssertionError,
        match=r"Invalid arguments {{.+}} when constructing {classname}".format(
            classname=mock_new_sf_exception.__name__
        ),
    ):
        mock_new_sf_exception.mocked_wrong_child_error(
            "learning rate",
            "expected to be between 0 and 1",
        )


def test_new_sf_exception_from_constructor_should_success(mock_new_sf_exception):
    exp = mock_new_sf_exception("detailed reason of mocked sf exception")
    assert type(exp) == mock_new_sf_exception
    assert exp.error_code == ErrorCode.INTERNAL_ERROR
    assert exp.reason == "detailed reason of mocked sf exception"
    assert exp.description == "INTERNAL_ERROR"
    assert exp.explanation == ""


def test_new_sf_exception_from_constructor_should_fail_when_pass_unexpected_args(
    mock_new_sf_exception,
):
    exception_msg = (
        r"Derived SFException should be initialized with a NAMELESS 'reason' string parameter, "
        r"or NAMED error_code\(optional\), description, reason, explanation parameters"
    )

    with pytest.raises(AssertionError, match=exception_msg):
        mock_new_sf_exception(
            "detailed reason string of mocked sf exception",
            "this is an unexpected argument",
        )


def test_new_sf_exception_from_constructor_should_fail_when_pass_unexpected_kwargs(
    mock_new_sf_exception,
):
    exception_msg = (
        r"Derived SFException should be initialized with a NAMELESS 'reason' string parameter, "
        r"or NAMED error_code\(optional\), description, reason, explanation parameters"
    )

    with pytest.raises(AssertionError, match=exception_msg):
        mock_new_sf_exception(
            "detailed reason string of mocked sf exception",
            reason="this is an unexpected argument",
        )


def test_format_sf_exception_should_success(mock_new_sf_exception):
    exp = mock_new_sf_exception.mocked_child_error(
        argument="learning rate",
        expected="expected to be between 0 and 1",
    )
    formatted_msg = (
        "\n"
        "*************************\n"
        "code        : 31001151\n"
        "reason      : learning rate is invalid, reason: expected to be between 0 and 1\n"
        "description : MOCKED_CHILD_ERROR\n"
        "explanation : This is just a mocked child error\n"
        "*************************\n"
    )
    assert str(exp) == formatted_msg


def test_raise_sf_exception_should_success(mock_new_sf_exception):
    with pytest.raises(mock_new_sf_exception):
        raise mock_new_sf_exception.mocked_child_error(
            argument="learning rate",
            expected="expected to be between 0 and 1",
        )
