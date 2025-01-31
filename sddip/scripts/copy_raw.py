"""Copy raw data files to test case directories."""

import shutil
from pathlib import Path

test_cases = [
    "WB5",
    "case6ww",
    "case14",
    "case118",
]

cwd = Path.cwd()
test_dir = cwd / "data" / "01_test_cases"

for case in test_cases:
    test_case_dir = test_dir / case
    raw_dir = test_case_dir / "raw"
    raw_files = list(raw_dir.glob("*"))

    for test in test_case_dir.glob(r"t*"):
        for f in raw_files:
            shutil.copy2(f, test / f.name)
