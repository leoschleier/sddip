"""Copy raw data files to test case directories."""
import shutil
from pathlib import Path

test_cases = [
        "WB5", "case6ww", "case14", "case118",
]

cwd = Path().resolve()
test_dir = cwd / "data" / "01_test_cases"

for case in test_cases:
    test_case_dir = test_dir / case
    breakpoint()
    raw_dir = test_case_dir / "raw"
    raw_files = list(raw_dir.glob("*"))

    print([f.name for f in raw_files])

    for test in test_case_dir.glob(r"t*"):
        for f in raw_files:
            print("Copying", f, "to", test / f.name)
            shutil.copy2(f, test / f.name)


