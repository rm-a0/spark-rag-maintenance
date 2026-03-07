import sys
from loguru import logger
from pathlib import Path

def configure_logging(logs_dir: Path, run_name: str = "info"):
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> - {message}",
    )

    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        logs_dir / f"{run_name}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )

def get_logger(name: str):
    return logger.bind(name=name)