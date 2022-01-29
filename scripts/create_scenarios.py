import os
import pandas as pd
from sddip import scenarios
from sddip import config


# TODO Select parameters for scenario generation
test_case_raw_dir = "WB2/raw"
n_stages = 6
n_realizations_per_stage = 3


# Parameter retrieval
test_case_raw_dir = os.path.join(config.test_cases_dir, test_case_raw_dir)

bus_file_raw = os.path.join(test_case_raw_dir, "bus_data.txt")
scenario_file_path = os.path.join(test_case_raw_dir, "scenario_data.txt")

bus_df = pd.read_csv(bus_file_raw, delimiter="\s+")

demands = bus_df["Pd"].values.tolist()

n_buses = len(demands)

demand_buses = [b for b in range(len(demands)) if demands[b] != 0]

max_value_targets = [d for d in demands if d != 0]


# Generate scenarios
sc_generator = scenarios.ScenarioGenerator(n_stages, n_realizations_per_stage)

scenario_df = sc_generator.generate_scenario_dataframe(
    n_buses, demand_buses, max_value_targets
)


scenario_df.to_csv(scenario_file_path, sep="\t", index=False)
