import os
import pandas as pd
from sddip import scenarios
from sddip import config


def main():
    # TODO Select parameters for scenario generation
    test_case = "case30"
    n_stages = 6
    n_realizations_per_stage = 12

    scenario_str = (
        f"t{str(n_stages).zfill(2)}_n{str(n_realizations_per_stage).zfill(2)}"
    )

    # Parameter retrieval
    test_case_raw_dir = os.path.join(config.test_cases_dir, test_case, "raw")
    test_case_scenario_dir = os.path.join(
        config.test_cases_dir, test_case, scenario_str
    )

    if not os.path.exists(test_case_scenario_dir):
        os.mkdir(test_case_scenario_dir)

    bus_file_path = os.path.join(test_case_raw_dir, "bus_data.txt")
    renewables_file_path = os.path.join(test_case_raw_dir, "ren_data.txt")
    scenario_file_path = os.path.join(test_case_scenario_dir, "scenario_data.txt")

    bus_df = pd.read_csv(bus_file_path, delimiter="\s+")
    renewables_df = pd.read_csv(renewables_file_path, delimiter="\s+")

    demands = bus_df["Pd"].values.tolist()

    re_max_frac = renewables_df["max_frac"].values.tolist()

    n_buses = len(demands)

    demand_buses = [b for b in range(n_buses) if demands[b] != 0]

    max_demand_value_targets = [2 * d for d in demands if d != 0]

    renewables_buses = [b for b in range(n_buses) if re_max_frac[b] != 0]

    re_base_values = [frac * sum(demands) for frac in re_max_frac if frac != 0]

    # Generate scenarios
    sc_generator = scenarios.ScenarioGenerator(n_stages, n_realizations_per_stage)

    # Demand scenarios
    demand_scenario_df = sc_generator.generate_demand_scenario_dataframe(
        n_buses, demand_buses, max_demand_value_targets, 0.2
    )

    # Renewables scenarios
    min_values = [0] * len(renewables_buses)
    max_values = [frac * sum(demands) for frac in re_max_frac if frac != 0]
    start_values = [0.1 * m for m in max_values]
    step_sizes = [1 / 3 * m for m in max_values]

    renewables_scenario_df = sc_generator.generate_renewables_scenario_dataframe(
        n_buses,
        renewables_buses,
        start_values,
        step_sizes,
        min_values,
        max_values,
        0.3,
        0.2,
    )
    renewables_scenario_df.drop(["t", "n", "p"], axis=1, inplace=True)

    scenario_df = pd.concat([demand_scenario_df, renewables_scenario_df], axis=1)

    scenario_df.to_csv(scenario_file_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
