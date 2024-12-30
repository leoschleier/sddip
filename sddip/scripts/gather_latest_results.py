import shutil
from collections.abc import Iterable
from pathlib import Path

from sddip import config


def main() -> None:
    n = 1

    log_files = config.LOGS_DIR.glob("*.txt")
    runtime_log_dirs = config.LOGS_DIR.glob("./log_*")
    bm_logs_dirs = config.LOGS_DIR.glob("./BM_log_*")
    results_dirs = config.RESULTS_DIR.glob("./results_*")

    latest_log_files = get_last(n, log_files)
    latest_runtime_log_dirs = get_last(n, runtime_log_dirs)
    latest_bm_log_dirs = get_last(n, bm_logs_dirs)
    latest_results_dirs = get_last(n, results_dirs)

    working_dir = Path.cwd()
    temp_dir = working_dir / "temp"

    for log, rt, bm, res in zip(
        latest_log_files,
        latest_runtime_log_dirs,
        latest_bm_log_dirs,
        latest_results_dirs,
        strict=False,
    ):
        test_case, stages, realizations = get_test_case_info(log)
        target_dir = temp_dir / test_case / f"t{stages}_n{realizations}"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log, target_dir / "logs.txt")
        shutil.copytree(rt, target_dir / "runtime", dirs_exist_ok=True)
        shutil.copytree(bm, target_dir / "bundle_method", dirs_exist_ok=True)
        shutil.copytree(res, target_dir / "results", dirs_exist_ok=True)


def get_last(n: int, l: Iterable) -> list:
    last_n = list(l)[-n:]
    if type(last_n) == list:
        return last_n
    return [last_n]


def get_test_case_info(log_file: Path):
    with open(log_file, encoding="utf-8") as file:
        content = file.read()
        for line in content.split("\n"):
            if "Test case:" in line:
                run_info = line.split(": ")[-1].split(",")
                test_case = run_info[0]
                stages = run_info[1].split("=")[-1]
                realizations = run_info[2].split("=")[-1]

                return test_case, stages, realizations

    msg = "No test case info found."
    raise ValueError(msg)


if __name__ == "__main__":
    main()
