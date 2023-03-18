from pathlib import Path
from .. import config


def main():
    for dir in [config.RESULTS_DIR, config.LOGS_DIR]:
        Path(dir).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
