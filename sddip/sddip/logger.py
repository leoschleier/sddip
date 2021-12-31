import os
import json
import config
from time import time
from datetime import datetime

class LogManager:
    def __init__(self):
        pass
        

    def create_log_dir(self, dir_label:str)->str:
        start_time_str = datetime.today().strftime("%Y_%m_%d__%H_%M_%S")
        dir_name = f"{dir_label}_{start_time_str}"
        log_dir = os.path.join(config.solver_log_dir, dir_name)
        os.mkdir(log_dir)

        return log_dir

class RuntimeLogger:
    def __init__(self, log_dir: str):
        self.runtime_filepath = os.path.join(log_dir, "runtime.json")
        self.runtime_dict = {}
        self.global_start_time = None
        
    def start(self):
        self.global_start_time = time()
        
    def log_task_end(self, task_name: str, task_start_time: float):
        task_runtime = time() - task_start_time
        self.runtime_dict.update({task_name: task_runtime})
        
    def log_experiment_end(self):
        self.log_task_end('global_runtime', self.global_start_time)
        json.dump(self.runtime_dict, 
                  open(self.runtime_filepath, 'w'), indent=4)