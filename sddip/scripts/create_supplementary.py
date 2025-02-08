from pathlib import Path

import pandas as pd


def create_supplementary_data(source_dir: Path, target_dir: Path) -> None:
    """Create supplementary data for the given test case."""
    if not target_dir.exists():
        msg = f"Path '{target_dir.resolve().absolute()}' does not exist."
        raise ValueError(msg)

    # Ramp rate
    # (%/100 of rated capacity per unit time)
    ramp_rate = 0.2
    # Min up/-down time

    # Parameter retrieval
    gen_file = source_dir / "gen_data.txt"
    supplementary_file_path = target_dir / "gen_sup_data.txt"

    gen_df = pd.read_csv(gen_file, delimiter=r"\s+")

    rated_capacities = gen_df["Pmax"].values.tolist()
    buses = gen_df["bus"].values.tolist()

    # Generate ramp rate limits
    r_up = [ramp_rate * p_max for p_max in rated_capacities]
    r_down = r_up

    # Generate min up-/down-time
    n_gens = len(rated_capacities)

    min_up_time = [2] * n_gens
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
