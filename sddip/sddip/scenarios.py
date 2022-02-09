from typing import Tuple
import numpy as np
import pandas as pd
import random as rdm

from sddip import config


class ScenarioGenerator:

    h0_load_profile = pd.read_csv(config.h0_load_profile_file, delimiter="\t")

    def __init__(self, n_stages, n_realizations_per_stage):
        if n_stages < 2:
            raise ValueError("Number of stages must be greater than 1.")
        if n_realizations_per_stage < 2:
            raise ValueError("Number of realizations per stage must be greater than 1.")
        self.h0_profile = ScenarioGenerator.h0_load_profile.h0.values.tolist()
        self.n_stages = n_stages
        self.n_realizations_per_stage = n_realizations_per_stage
        self.n_total_realizations = (n_stages - 1) * n_realizations_per_stage + 1

        self.reduction_factor = int(len(self.h0_load_profile) / n_stages)

    def generate_demand_scenario_dataframe(
        self,
        n_buses: int,
        demand_buses: list,
        max_value_targets: list,
        max_relative_variation: float = 0.1,
    ) -> pd.DataFrame:

        if not len(demand_buses) == len(max_value_targets):
            raise ValueError(
                "Number of maximum target values must equal the number of demand buses."
            )

        reduced_profile = self.reduce_profile(self.h0_profile, self.reduction_factor)

        base_profiles = [
            self.scale_profile(reduced_profile, max_value)
            for max_value in max_value_targets
        ]

        scenario_data = {"t": [1], "n": [1], "p": [1]}

        demand_bus_keys, no_demand_bus_keys = self.create_bus_keys(
            n_buses, demand_buses, "Pd"
        )

        # Set loads for the first (deterministic) stage
        for b in no_demand_bus_keys:
            scenario_data[b] = [0] * self.n_total_realizations

        for b in range(len(demand_buses)):
            scenario_data[demand_bus_keys[b]] = [
                self.get_rdm_variation(base_profiles[b][0], max_relative_variation)
            ]

        # Set loads for stages >1
        for t in range(1, self.n_stages):
            for n in range(1, self.n_realizations_per_stage + 1):
                scenario_data[f"t"].append(t + 1)
                scenario_data["n"].append(n)
                scenario_data["p"].append(1 / self.n_realizations_per_stage)
                for b in range(len(demand_buses)):
                    scenario_data[demand_bus_keys[b]].append(
                        self.get_rdm_variation(
                            base_profiles[b][t], max_relative_variation
                        )
                    )

        return pd.DataFrame(scenario_data)

    def generate_renewables_scenario_dataframe(
        self,
        n_buses: int,
        renewables_buses: list,
        base_generation: list,
        max_relative_variation: float,
    ):
        if not len(renewables_buses) == len(base_generation):
            raise ValueError(
                "Number of base generation entries must equal the number of renewables buses."
            )

        scenario_data = {"t": [1], "n": [1], "p": [1]}

        renewables_bus_keys, no_renewables_bus_keys = self.create_bus_keys(
            n_buses, renewables_buses, "Re"
        )

        # Set loads for the first (deterministic) stage
        for b in no_renewables_bus_keys:
            scenario_data[b] = [0] * self.n_total_realizations

        generation_prev = []
        for b in range(len(renewables_buses)):
            gen = self.get_rdm_variation(base_generation[b], max_relative_variation)
            scenario_data[renewables_bus_keys[b]] = [gen]
            generation_prev.append(gen)

        # Set loads for stages >1
        for t in range(1, self.n_stages):
            for n in range(1, self.n_realizations_per_stage + 1):
                scenario_data[f"t"].append(t + 1)
                scenario_data["n"].append(n)
                scenario_data["p"].append(1 / self.n_realizations_per_stage)
                for b in range(len(renewables_buses)):
                    gen = self.get_rdm_variation(
                        generation_prev[b], max_relative_variation
                    )
                    scenario_data[renewables_bus_keys[b]].append(gen)
                    generation_prev[b] = gen

        return pd.DataFrame(scenario_data)

    def create_bus_keys(self, n_buses: int, active_buses: list, label: str) -> Tuple:
        active_bus_keys = []
        inactive_bus_keys = []

        for b in range(n_buses):
            bus_key = f"{label}{b+1}"
            if b in active_buses:
                active_bus_keys.append(bus_key)
            else:
                inactive_bus_keys.append(bus_key)

        return (active_bus_keys, inactive_bus_keys)

    def get_rdm_variation(
        self, base_value: float, max_relative_variation: float
    ) -> float:
        return base_value + base_value * rdm.uniform(
            -max_relative_variation, max_relative_variation
        )

    def reduce_profile(self, values: list, reduction_factor: int) -> list:
        if len(values) % reduction_factor != 0:
            raise ValueError(
                "Number of values to be reduced must be divisible by the reduction factor."
            )

        values = np.array(values)

        return list(np.mean(values.reshape(-1, reduction_factor), axis=1))

    def scale_profile(self, values: list, max_value_target: float) -> list:
        values = np.array(values)

        max_value = np.amax(values)

        scaling_factor = max_value_target / max_value

        return list(values * scaling_factor)


class ScenarioSampler:
    def __init__(self, n_stages: int, n_realizations_per_stage: int):
        self.n_stages = n_stages
        self.n_realizations_per_stage = n_realizations_per_stage

    def generate_samples(self, n_samples: int) -> list:
        samples = []
        for _ in range(n_samples):
            sample = [
                rdm.randint(0, self.n_realizations_per_stage - 1)
                for _ in range(self.n_stages - 1)
            ]
            sample.insert(0, 0)
            samples.append(sample)

        return samples
