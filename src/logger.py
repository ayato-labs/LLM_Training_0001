import datetime
import sys
from pathlib import Path

from loguru import logger

# Create logs directory if not exists
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logger
logger.remove()

# Console output (human readable)
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Structured JSON file output (for machine parsing / traceability)
logger.add(
    "logs/app.json",
    serialize=True,
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
)

# Timestamped log file (one per run)
_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logger.add(
    f"logs/run_{_run_timestamp}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="50 MB",
)


def log_exceptions(func):
    """Decorator to log exceptions with full traceback."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Unhandled exception in {func.__name__}: {e}")
            raise e

    return wrapper
