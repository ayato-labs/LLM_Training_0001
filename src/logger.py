import sys
from loguru import logger
from pathlib import Path

# Create logs directory if not exists
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logger
logger.remove()  # Remove default handler
# Add console output (human readable)
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")
# Add JSON file output (structured)
logger.add("logs/app.json", serialize=True, level="DEBUG", rotation="10 MB")

def log_exceptions(func):
    """Decorator to log exceptions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {e}")
            raise e
    return wrapper
