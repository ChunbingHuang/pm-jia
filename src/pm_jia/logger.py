"""
Logger configuration.
"""

import logging
import os

from dotenv import load_dotenv

from src.pm_jia.config import GeneralConfig, LoggerConfig

load_dotenv()


def setup_logger(logger_name: str = __name__, cli_mode: bool = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting and handling.

    Args:
        logger_name: Name for the logger
        cli_mode: If True, suppress INFO/DEBUG console logs

    Returns:
        logging.Logger: Configured logger instance
    """
    log_file_name = LoggerConfig.name
    level = LoggerConfig.level
    format = LoggerConfig.format
    date_format = LoggerConfig.date_format

    if cli_mode is None:
        cli_mode = os.getenv("CLI_MODE", "false").lower() == "true"

    log_dir = GeneralConfig.project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    logger.handlers.clear()

    if LoggerConfig.save_to_file:
        file_formatter = logging.Formatter(
            format,
            datefmt=date_format,
        )
        file_handler = logging.FileHandler(log_dir / f"{log_file_name}.log")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # In CLI mode, only show WARNING and ERROR messages to console
    console_formatter = logging.Formatter(format, datefmt=date_format)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    if cli_mode:
        console_handler.setLevel(logging.WARNING)

    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    logger = setup_logger(__name__)
    logger.info("Hello, world!")
