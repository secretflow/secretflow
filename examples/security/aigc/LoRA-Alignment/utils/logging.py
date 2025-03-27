import logging
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("Rich")

__all__ = ["Logger", "logging"]


class Logger:

    def __init__(self, logger=None):
        self.logger = logger

    def info(self, msg):
        self.logger.info(msg) if self.logger else print(msg)

    def error(self, msg):
        self.logger.error(msg) if self.logger else print(msg)

    def warn(self, msg):
        self.logger.warn(msg) if self.logger else print(msg)
