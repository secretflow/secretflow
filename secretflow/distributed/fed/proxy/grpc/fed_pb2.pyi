from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SfFedProxySendData(_message.Message):
    __slots__ = ["data", "seq_id", "job_name"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    SEQ_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_NAME_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    seq_id: int
    job_name: str
    def __init__(self, data: _Optional[bytes] = ..., seq_id: _Optional[int] = ..., job_name: _Optional[str] = ...) -> None: ...

class SfFedProxySendDataResponse(_message.Message):
    __slots__ = ["code", "result"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    code: int
    result: str
    def __init__(self, code: _Optional[int] = ..., result: _Optional[str] = ...) -> None: ...
