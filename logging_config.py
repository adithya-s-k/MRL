"""Logging configuration for MRL framework."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Set up logging for MRL.

    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
        log_file: Path to log file (optional, logs to stderr by default)

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Get or create MRL logger
    logger = logging.getLogger("MRL")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (will be prefixed with 'MRL.')

    Returns:
        Logger instance
    """
    return logging.getLogger(f"MRL.{name}")


# Default logger setup
_default_logger = None


def get_default_logger() -> logging.Logger:
    """Get the default MRL logger, initializing if needed."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger
