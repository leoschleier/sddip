from pathlib import Path

import pandas as pd

from sddip.sddip import scenarios


def create_scenario_data(
    t: int, n: int, test_case_dir: Path, base_case_dir: Path | None = None
) -> None:
    """Create scenario data for the given test case.

    Args:
        t: The number of stages.
        n: The number of scenarios.
        test_case_dir: The path to the test case directory.
        base_case_dir: The path to the base case directory.

    """
    base_case_dir = base_case_dir or test_case_dir

    bus_file_path = base_case_dir / "bus_data.txt"
    renewables_file_path = base_case_dir / "ren_data.txt"

    for f in [bus_file_path, renewables_file_path]:
        if not f.exists():
            msg = f"Path '{f.resolve().absolute()}' does not exist."
            raise ValueError(msg)

    test_case_dir.mkdir(exist_ok=True, parents=True)

    scenario_file_path = test_case_dir / "scenario_data.txt"

    bus_df = pd.read_csv(bus_file_path, delimiter=r"\s+")
    renewables_df = pd.read_csv(renewables_file_path, delimiter=r"\s+")

    demands = bus_df["Pd"].values.tolist()

    re_max_frac = renewables_df["max_frac"].values.tolist()

    n_buses = len(demands)

    demand_buses = [b for b in range(n_buses) if demands[b] != 0]

    max_demand_value_targets = [2 * d for d in demands if d != 0]

    renewables_buses = [b for b in range(n_buses) if re_max_frac[b] != 0]

    # Generate scenarios
    sc_generator = scenarios.ScenarioGenerator(t, n)

    # Demand scenarios
    demand_scenario_df = sc_generator.generate_demand_scenario_dataframe(
        n_buses, demand_buses, max_demand_value_targets, 0.2
    )

    # Renewables scenarios
    min_values = [0] * len(renewables_buses)
    max_values = [frac * sum(demands) for frac in re_max_frac if frac != 0]
    start_values = [0.1 * m for m in max_values]
    step_sizes = [1 / 3 * m for m in max_values]

    renewables_scenario_df = (
        sc_generator.generate_renewables_scenario_dataframe(
            n_buses,
            renewables_buses,
            start_values,
            step_sizes,
            min_values,
            max_values,
            0.3,
            0.2,
        )
    )
    renewables_scenario_df = renewables_scenario_df.drop(
        ["t", "n", "p"], axis=1
    )

    scenario_df = pd.concat(
        [demand_scenario_df, renewables_scenario_df], axis=1
    )

    scenario_df.to_csv(scenario_file_path, sep="\t", index=False)
