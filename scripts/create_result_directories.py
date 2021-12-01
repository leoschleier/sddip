from pathlib import Path

result_dirs =  ['../data/02_solver_log', '../data/03_solutions']

for dir in result_dirs:
    Path(dir).mkdir(parents=True, exist_ok=True)