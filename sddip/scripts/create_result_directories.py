from pathlib import Path

from sddip import config


def main() -> None:
    for dir in [config.RESULTS_DIR, config.LOGS_DIR]:
        Path(dir).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
