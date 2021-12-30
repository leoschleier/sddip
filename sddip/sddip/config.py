from pathlib import Path
import os

config_directory = Path(__file__).parent
data_dir = os.path.join(config_directory, "../../data")

test_cases_dir = os.path.join(data_dir, "01_test_cases")
solver_log_dir = os.path.join(data_dir, "02_solver_log")
solutions_dir = os.path.join(data_dir, "03_solutions")

data_directories = [test_cases_dir, solver_log_dir, solutions_dir]
directory_labels = ["Test case", "Solver log", "Solutions"]

if __name__ == '__main__':
    print("Current directory: {}".format(os.getcwd()))
    print("Config directory: {}".format(config_directory))
    
    for dir, label in zip(data_directories, directory_labels):
        if os.path.isdir(dir):
            print("{} directory exists.".format(label))
        else:
            print("{} directory does not exists.".format(label))
    