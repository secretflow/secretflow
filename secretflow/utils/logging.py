from logging import _checkLevel

_LOGGING_LEVEL = 'INFO'

LOG_FORMAT = '%(asctime)s,%(msecs)d %(levelname)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'


def set_logging_level(level: str):
    level = level.upper()
    _checkLevel(level)
    global _LOGGING_LEVEL
    _LOGGING_LEVEL = level


def get_logging_level():
    global _LOGGING_LEVEL
    return _LOGGING_LEVEL
