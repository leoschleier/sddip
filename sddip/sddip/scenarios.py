import random as rdm

import natsort
import numpy as np
import pandas as pd

from sddip import config


class ScenarioGenerator:
    h0_load_profile = pd.read_csv(config.H0_LOAD_PROFILE_FILE, delimiter="\t")

    def __init__(self, n_stages, n_realizations_per_stage) -> None:
        if n_stages < 2:
            msg = "Number of stages must be greater than 1."
            raise ValueError(msg)
        if n_realizations_per_stage < 2:
            msg = "Number of realizations per stage must be greater than 1."
            raise ValueError(msg)
        self.h0_profile = ScenarioGenerator.h0_load_profile.h0.values.tolist()
        self.n_stages = n_stages
        self.n_realizations_per_stage = n_realizations_per_stage
        self.n_total_realizations = (
            n_stages - 1
        ) * n_realizations_per_stage + 1

        self.reduction_factor = int(len(self.h0_load_profile) / n_stages)

    def generate_demand_scenario_dataframe(
        self,
        n_buses: int,
        demand_buses: list,
        max_value_targets: list,
        max_relative_variation: float,
    ) -> pd.DataFrame:
        if len(demand_buses) != len(max_value_targets):
            msg = "Number of maximum target values must equal the number of demand buses."
            raise ValueError(msg)

        reduced_profile = self.reduce_profile(
            self.h0_profile, self.reduction_factor
        )

        base_profiles = [
            self.scale_profile(reduced_profile, max(max_value - 10, 0))
            for max_value in max_value_targets
        ]

        demand_bus_keys, no_demand_bus_keys = self.create_bus_keys(
            n_buses, demand_buses, "Pd"
        )

        scenario_data: dict[str, list[float]] = {}
        # Set loads for buses without demand
        for b in no_demand_bus_keys:
            scenario_data[b] = [0] * self.n_total_realizations

        # Set loads for the first (deterministic) stage
        for b in range(len(demand_buses)):
            scenario_data[demand_bus_keys[b]] = [
                self.get_rdm_variation(
                    base_profiles[b][0], max_relative_variation
                )
            ]

        probabilities: dict[str, list[float]] = {"t": [1], "n": [1], "p": [1]}

        # Set loads for stages >1
        for t in range(1, self.n_stages):
            for n in range(1, self.n_realizations_per_stage + 1):
                probabilities["t"].append(t + 1)
                probabilities["n"].append(n)
                probabilities["p"].append(1 / self.n_realizations_per_stage)
                for b in range(len(demand_buses)):
                    scenario_data[demand_bus_keys[b]].append(
                        self.get_rdm_variation(
                            base_profiles[b][t], max_relative_variation
                        )
                    )

        scenario_data = dict(natsort.natsorted(scenario_data.items()))
        scenario_data.update(probabilities)

        return pd.DataFrame(scenario_data)

    def generate_renewables_scenario_dataframe(
        self,
        n_buses: int,
        renewables_buses: list,
        start_values: list,
        step_sizes: list,
        min_values: list,
        max_values: list,
        threshold: float,
        max_relative_variation: float,
    ) -> pd.DataFrame:
        if len(renewables_buses) != len(start_values):
            msg = "Number of list entries must equal the number of renewables buses."
            raise ValueError(msg)

        scenario_data = {"t": [1], "n": [1], "p": [1]}

        renewables_bus_keys, no_renewables_bus_keys = self.create_bus_keys(
            n_buses, renewables_buses, "Re"
        )

        # Set geneartion for buses without renewables
        for b in no_renewables_bus_keys:
            scenario_data[b] = [0] * self.n_total_realizations

        # Generate base profiles
        base_profiles = []
        for b in range(len(renewables_buses)):
            profile = self.random_walk(
                self.n_stages,
                start_values[b],
                step_sizes[b],
                min_values[b],
                max_values[b],
                threshold,
            )
            base_profiles.append(profile)

        # Set generation for the first (deterministic) stage
        for b in range(len(renewables_buses)):
            scenario_data[renewables_bus_keys[b]] = [
                self.get_rdm_variation(
                    base_profiles[b][0], max_relative_variation
                )
            ]

        # Set loads for stages >1
        for t in range(1, self.n_stages):
            for n in range(1, self.n_realizations_per_stage + 1):
                scenario_data["t"].append(t + 1)
                scenario_data["n"].append(n)
                scenario_data["p"].append(1 / self.n_realizations_per_stage)
                for b in range(len(renewables_buses)):
                    scenario_data[renewables_bus_keys[b]].append(
                        self.get_rdm_variation(
                            base_profiles[b][t], max_relative_variation
                        )
                    )

        return pd.DataFrame(scenario_data)

    def random_walk(
        self,
        n_steps: int,
        start_value: float,
        step_size: float,
        min_value: float,
        max_value: float,
        threshold: float = 0.5,
    ) -> list:
        values = []
        prev_value = start_value

        for _ in range(n_steps):
            new_value = 0
            probability = rdm.random()

            if probability >= threshold:
                new_value = prev_value + step_size
            else:
                new_value = prev_value - step_size

            if new_value > max_value:
                new_value = max_value
            elif new_value < min_value:
                new_value = min_value

            values.append(new_value)
            prev_value = new_value

        return values

    def create_bus_keys(
        self, n_buses: int, active_buses: list, label: str
    ) -> tuple:
        active_bus_keys = []
        inactive_bus_keys = []

        for b in range(n_buses):
            bus_key = f"{label}{b + 1}"
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
            msg = "Number of values to be reduced must be divisible by the reduction factor."
            raise ValueError(msg)

        values = np.array(values)

        return list(np.mean(values.reshape(-1, reduction_factor), axis=1))

    def scale_profile(self, values: list, max_value_target: float) -> list:
        values = np.array(values)

        max_value = np.amax(values)

        scaling_factor = max_value_target / max_value

        return list(values * scaling_factor)


class ScenarioSampler:
    def __init__(self, n_stages: int, n_realizations_per_stage: int) -> None:
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
