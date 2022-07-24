import os
import math
import pandas as pd
from ..sddip import config


def main():
    # TODO Select parameters for data generation
    test_case = "case6ww"

    t = 12
    n = 6

    scenario_str = f"t{str(t).zfill(2)}_n{str(n).zfill(2)}"

    # Ramp rate
    # (%/100 of rated capacity per unit time)
    ramp_rate = 0.1
    # Min up/-down time
    up_down_factor = 4

    # Parameter retrieval
    test_case_raw_dir = os.path.join(config.test_cases_dir, test_case, "raw")
    test_case_scenario_dir = os.path.join(
        config.test_cases_dir, test_case, scenario_str
    )

    gen_file = os.path.join(test_case_raw_dir, "gen_data.txt")
    scenario_file = os.path.join(test_case_scenario_dir, "scenario_data.txt")
    supplementary_file_path = os.path.join(test_case_scenario_dir, "gen_sup_data.txt")

    gen_df = pd.read_csv(gen_file, delimiter="\s+")
    scenario_df = pd.read_csv(scenario_file, delimiter="\s+")

    rated_capacities = gen_df["Pmax"].values.tolist()
    buses = gen_df["bus"].values.tolist()

    # Generate ramp rate limits
    r_up = [ramp_rate * p_max for p_max in rated_capacities]
    r_down = r_up

    # Generate min up-/down-time
    n_gens = len(rated_capacities)
    n_stages = scenario_df["t"].max()
    n_restrictive_periods = int(math.ceil(1 / up_down_factor * n_stages))

    min_up_time = [5] * n_gens
    min_down_time = min_up_time

    # Export results
    supplementary_dict = {
        "bus": buses,
        "R_up": r_up,
        "R_down": r_down,
        "UT": min_up_time,
        "DT": min_down_time,
    }

    supplementary_df = pd.DataFrame.from_dict(supplementary_dict)

    supplementary_df.to_csv(supplementary_file_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
