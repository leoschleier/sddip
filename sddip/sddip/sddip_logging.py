import json
import os
from datetime import datetime
from time import time

import gurobipy as gp

from .. import config


class LogManager:
    def __init__(self):
        pass

    def create_log_dir(self, dir_label: str) -> str:
        now_str = datetime.today().strftime("%Y%m%d%H%M%S")
        dir_name = f"{dir_label}_{now_str}"
        log_dir = config.LOGS_DIR / dir_name
        os.makedirs(log_dir)

        return log_dir

    def create_subdirectory(self, dir_name: str, sub_dir_name: str):
        dir_to_create = dir_name / sub_dir_name
        os.makedirs(dir_to_create)


class RuntimeLogger:
    def __init__(self, log_dir: str):
        self.runtime_file_path = os.path.join(log_dir, "runtime.json")
        self.runtime_dict = {}
        self.global_start_time = None

    def start(self):
        self.global_start_time = time()

    def log_task_end(self, task_name: str, task_start_time: float):
        task_runtime = time() - task_start_time
        self.runtime_dict.update({task_name: task_runtime})

    def log_experiment_end(self):
        self.log_task_end("global_runtime", self.global_start_time)
        json.dump(
            self.runtime_dict, open(self.runtime_file_path, "w"), indent=4
        )


class GurobiLogger:
    def __init__(self, log_dir: str):
        self.gurobi_log_dir = os.path.join(log_dir, "gurobi")
        os.makedirs(self.gurobi_log_dir)

    def log_model(self, model: gp.Model, label: str):
        model_file_path = os.path.join(
            self.gurobi_log_dir, f"{label}_model.lp"
        )
        solution_file_path = os.path.join(
            self.gurobi_log_dir, f"{label}_model.sol"
        )

        model.write(model_file_path)
        model.write(solution_file_path)
