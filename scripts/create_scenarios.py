import os
import pandas as pd
from sddip import scenarios
from sddip import config


# TODO Select parameters for scenario generation
test_case_raw_dir = "WB2/raw"
n_stages = 3
n_realizations_per_stage = 2


# Parameter retrieval
test_case_raw_dir = os.path.join(config.test_cases_dir, test_case_raw_dir)

bus_file_path = os.path.join(test_case_raw_dir, "bus_data.txt")
renewables_file_path = os.path.join(test_case_raw_dir, "ren_data.txt")
scenario_file_path = os.path.join(test_case_raw_dir, "scenario_data.txt")


bus_df = pd.read_csv(bus_file_path, delimiter="\s+")
renewables_df = pd.read_csv(renewables_file_path, delimiter="\s+")

demands = bus_df["Pd"].values.tolist()

re_base_frac = renewables_df["base_frac"].values.tolist()

n_buses = len(demands)


demand_buses = [b for b in range(n_buses) if demands[b] != 0]

max_demand_value_targets = [d for d in demands if d != 0]


renewables_buses = [b for b in range(n_buses) if re_base_frac[b] != 0]

re_base_values = [frac * sum(demands) for frac in re_base_frac if frac != 0]


# Generate scenarios
sc_generator = scenarios.ScenarioGenerator(n_stages, n_realizations_per_stage)

# Demand scenarios
demand_scenario_df = sc_generator.generate_demand_scenario_dataframe(
    n_buses, demand_buses, max_demand_value_targets
)

# Renewables scenarios
renewables_scenario_df = sc_generator.generate_renewables_scenario_dataframe(
    n_buses, renewables_buses, re_base_values, max_relative_variation=0.9
)
renewables_scenario_df.drop(["t", "n", "p"], axis=1, inplace=True)


scenario_df = pd.concat([demand_scenario_df, renewables_scenario_df], axis=1)

scenario_df.to_csv(scenario_file_path, sep="\t", index=False)
