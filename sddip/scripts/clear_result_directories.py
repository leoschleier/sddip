import logging
import os
import shutil

from sddip import config

logger = logging.getLogger(__name__)


def main() -> None:
    for rd in [config.RESULTS_DIR, config.LOGS_DIR]:
        for filename in os.listdir(rd):
            file_path = os.path.join(rd, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as ex:
                logger.exception(
                    "Failed to delete %s. Reason: %s", file_path, ex
                )


if __name__ == "__main__":
    main()
