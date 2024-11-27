# Error System (experimental)
This folder contains the error system for the project.
This is a system that allows us to clearly define errors, their reasoning and how they are handled.

## How to define new SFException

### 1. add new function which returns a new child exception to existing SFException class
```python
@sf_exception(ErrorCode.TRAINING_HYPERPARAMETER_ERROR)
class SFTrainingHyperparameterError(SFException):
    ...

    def newly_added_error(cls, *args, reason=None, **kwargs):
        return cls.from_config(
            "NEWLY_ADDED_ERROR",
            "{argument} is invalid",
            "description of the error",
        )

```

### 2. define new SFException class
1) add new exception code to error_code.py
```python
class ErrorCode(Enum):
    ...
    NEWLY_ADDED_ERROR_CODE = 31001501

```
2) add new exception class
```python
@sf_exception(ErrorCode.NEWLY_ADDED_ERROR_CODE)
class NewlyAddedError(SFException):
    ...
    # add any function which returns new child exception here
```

## How to raise a SFException

### 1. RECOMMENDED: use the newly added error function to raise the exception
format the reason use predifined reason_template
```python
raise SFTrainingHyperparameterError.newly_added_error(argument="learning rate")
```
or overwrite the reason by passing full reason message
```python
raise SFTrainingHyperparameterError.newly_added_error(reason="learning rate '{rate}' of training hyperparameter is invalid".format(rate=12.3))
```

### 2. NOT RECOMMENDED: directly raise newly added exception class
```python
raise NewlyAddedError("error reason message")
```
> Keep this method just in order to be compatible with the existing code.

## SFException output format
Take SFTrainingHyperparameterError.file_not_exist as an example, it will output the following log:
```log
************************
code        : 31001001
reason      : /tmp/file.txt file does not exist
description : FILE_NOT_EXIST
explanation : User provided an incorrect upload path, OSS file was deleted, or platform incorrectly concatenated paths, etc.
*************************
```