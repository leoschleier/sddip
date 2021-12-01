from pathlib import Path
import os

data_dir = Path('../data')

test_cases_dir = os.path.join(data_dir, "01_test_cases")
solver_log_dir = os.path.join(data_dir, "02_solver_log")
solutions_dir = os.path.join(data_dir, "03_solutions")

result_dirs = [solver_log_dir, solutions_dir]