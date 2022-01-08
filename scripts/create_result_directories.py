from pathlib import Path
from sddip import config


for dir in config.result_directories:
    Path(dir).mkdir(parents=True, exist_ok=True)