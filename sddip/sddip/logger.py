import os
import json
import gurobipy as gp
from time import time
from datetime import datetime

from .. import config


class LogManager:
    def __init__(self):
        pass

    def create_log_dir(self, dir_label: str) -> str:
        start_time_str = datetime.today().strftime("%Y_%m_%d__%H_%M_%S")
        dir_name = f"{dir_label}_{start_time_str}"
        log_dir = os.path.join(config.solver_log_dir, dir_name)
        os.mkdir(log_dir)

        return log_dir

    def create_subdirectory(self, dir_name: str, sub_dir_name: str):
        os.mkdir(os.path.join(dir_name, sub_dir_name))


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
        os.mkdir(self.gurobi_log_dir)

    def log_model(self, model: gp.Model, label: str):
        model_file_path = os.path.join(
            self.gurobi_log_dir, f"{label}_model.lp"
        )
        solution_file_path = os.path.join(
            self.gurobi_log_dir, f"{label}_model.sol"
        )

        model.write(model_file_path)
        model.write(solution_file_path)
