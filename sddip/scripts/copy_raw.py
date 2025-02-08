"""Copy raw data files to test case directories."""

import shutil
from pathlib import Path


def move_files(source: Path, target: Path) -> None:
    for f in source.glob("*"):
        shutil.copy2(f, target / f.name)


if __name__ == "__main__":
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
            move_files(raw_dir, test)
