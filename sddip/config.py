"""Configuration file containing paths required by the `sddip`
package.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


# Directories
APP_DIR = Path(__file__).parent

DATA_DIR = APP_DIR / ".." / "data"

TEST_CASES_DIR = DATA_DIR / "01_test_cases"
RESULTS_DIR = DATA_DIR / "02_results"
LOGS_DIR = DATA_DIR / "03_logs"

# Load Profile
LOAD_PROFILE_DIR = TEST_CASES_DIR / "supplementary" / "load_profiles"
H0_LOAD_PROFILE_FILE = LOAD_PROFILE_DIR / "h0_summer_workday.txt"


if __name__ == "__main__":
    logger.info("Current directory: %s", os.getcwd())
    logger.info("App directory: %s", APP_DIR)

    data_dirs = [TEST_CASES_DIR, RESULTS_DIR, LOGS_DIR]
    dir_labels = ["Test cases", "Results", "Logs"]

    for data_dir, label in zip(data_dirs, dir_labels, strict=False):
        if os.path.isdir(data_dir):
            logger.info("%s directory exists.", label)
        else:
            logger.warning("%s directory does not exists.", label)
