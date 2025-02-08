"""Logging configuration."""

from pathlib import Path

_file_path = Path() / "var" / "log"


def create_logging_dir() -> None:
    """Create the logging directory."""
    _file_path = Path() / "var" / "log"
    _file_path.mkdir(parents=True, exist_ok=True)


config = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "NOTSET",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "NOTSET",
            "formatter": "simple",
            "filename": f"{_file_path}/app.log",
            "mode": "w",
        },
    },
    "loggers": {
        "": {"handlers": ["console"], "level": "INFO"},
        "sddip": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
