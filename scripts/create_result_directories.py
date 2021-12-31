import sys
sys.path.append('../sddip')

from pathlib import Path
from sddip.sddip import config


for dir in config.result_directories:
    Path(dir).mkdir(parents=True, exist_ok=True)