import os

import numpy as np
import pandas as pd

from sddip import utils, config


class Parameters:
    def __init__(
        self,
        test_case_name: str,
        sub_directory: str = "raw",
        bus_file="bus_data.txt",
        branch_file="branch_data.txt",
        gen_file="gen_data.txt",
        gen_cost_file="gen_cost_data.txt",
        gen_sup_file="gen_sup_data.txt",
        renewables_file="ren_data.txt",
        storage_file="storage_data.txt",
        scenario_file="scenario_data.txt",
    ):
        test_data_dir = os.path.join(test_case_name, sub_directory)
        test_data_dir = os.path.join(config.test_cases_dir, test_data_dir)

        data_importer = DataImporter(test_data_dir)

        # DataFrames
        self.bus_df = data_importer.dataframe_from_csv(bus_file)
        self.branch_df = data_importer.dataframe_from_csv(branch_file)
        self.gen_df = data_importer.dataframe_from_csv(gen_file)
        self.gen_cost_df = data_importer.dataframe_from_csv(gen_cost_file)
        self.gen_sup_df = data_importer.dataframe_from_csv(gen_sup_file)
        self.ren_df = data_importer.dataframe_from_csv(renewables_file)
        self.storage_df = data_importer.dataframe_from_csv(storage_file)
        self.scenario_df = data_importer.dataframe_from_csv(scenario_file)

        # Structural data
        self.ptdf = None
        self.n_lines = None
        self.n_buses = None
        self.n_gens = None
        self.gens_at_bus = None
        self.n_storages = None
        self.storages_at_bus = None

        # Cost data
        self.gc = None
        self.suc = None
        self.sdc = None
        self.cost_coeffs = None

        # Power generation limits
        self.pg_min = None
        self.pg_max = None
        # Generator ramp rates
        self.r_up = None
        self.r_down = None
        # Min up- and down-times
        self.min_up_time = None
        self.min_down_time = None
        self.backsight_periods = None
        # Storage charge/discharge rate limits
        self.rc_max = None
        self.rdc_min = None
        # Maximum state of charge
        self.soc_max = None
        # Charge/discharge efficiencies
        self.eff_c = None
        self.eff_dc = None

        # Line capacity
        self.pl_max = None

        # Stochastic problem parameters
        self.n_stages = None
        self.n_realizations_per_stage = None

        # Nodal probability
        self.prob = None
        # Power demand
        self.p_d = None
        # Renewable generation
        self.re = None
        # Cut constraints lower bound
        self.cut_lb = None
        # Frist stage trial points
        self.init_x_trial_point = None
        self.init_y_trial_point = None
        self.init_x_bs_trial_point = None
        self.init_soc_trial_point = None

        self.initialize()

    def initialize(self):
        """Triggers the initialization of all parameters based on the corresponding data frames
        """
        self._calc_ptdf()
        self._init_deterministic_parameters()
        self._init_stochastic_parameters()
        self._init_initial_trial_points()

    def _calc_ptdf(self):
        """Calculates the Power Transmission Distribution Factor and infers the number of buses and lines
        """
        nodes = self.bus_df.bus_i.values.tolist()
        edges = self.branch_df[["fbus", "tbus"]].values.tolist()

        graph = utils.Graph(nodes, edges)

        ref_bus = self.bus_df.loc[self.bus_df.type == 3].bus_i.values[0]

        a_inc = graph.incidence_matrix()
        b_l = (
            -self.branch_df.x / (self.branch_df.r ** 2 + self.branch_df.x ** 2)
        ).tolist()
        b_diag = np.diag(b_l)

        m1 = b_diag.dot(a_inc)
        m2 = a_inc.T.dot(b_diag).dot(a_inc)

        m1 = np.delete(m1, ref_bus - 1, 1)
        m2 = np.delete(m2, ref_bus - 1, 0)
        m2 = np.delete(m2, ref_bus - 1, 1)

        ptdf = m1.dot(np.linalg.inv(m2))

        self.ptdf = np.insert(ptdf, ref_bus - 1, 0, axis=1)

        self.n_lines, self.n_buses = self.ptdf.shape

    def _init_deterministic_parameters(self):
        """Initializes all deterministic parameters
        """
        self.gen_cost_df
        self.gen_df
        self.branch_df

        self.gc = np.array(self.gen_cost_df.c1)
        self.suc = np.array(self.gen_cost_df.startup)
        self.sdc = np.array(self.gen_cost_df.shutdown)
        # TODO Adjust penalty for slack variables
        self.penalty = 5000

        self.cost_coeffs = (
            self.gc.tolist()
            + self.suc.tolist()
            + self.sdc.tolist()
            + [self.penalty] * 2
        )

        self.pg_min = np.array(self.gen_df.Pmin)
        self.pg_max = np.array(self.gen_df.Pmax)
        self.pl_max = np.array(self.branch_df.rateA)

        self.n_gens = len(self.gc)

        # TODO Add ramp rate limits
        self.r_up = self.gen_sup_df["R_up"].values.tolist()
        self.r_down = self.gen_sup_df["R_down"].values.tolist()

        # TODO add min up and down times to probelm data
        self.min_up_time = self.gen_sup_df["UT"].values.tolist()
        self.min_down_time = self.gen_sup_df["DT"].values.tolist()
        self.backsight_periods = [
            max(ut, dt) for ut, dt in zip(self.min_up_time, self.min_down_time)
        ]

        # Lists of generators at each bus
        #
        # Example: [[0,1], [], [2]]
        # Generator 1 & 2 are located at bus 1
        # No Generator is located at bus 2
        # Generator 3 is located at bus 3
        gen_buses = self.gen_df.bus.values.tolist()
        gens_at_bus = [[] for _ in range(self.n_buses)]
        g = 0
        for b in gen_buses:
            gens_at_bus[b - 1].append(g)
            g += 1
        self.gens_at_bus = gens_at_bus

        # Storages
        storage_buses = self.storage_df.bus.values.tolist()
        self.n_storages = len(storage_buses)
        self.storages_at_bus = [[] for _ in range(self.n_buses)]
        s = 0
        for b in storage_buses:
            self.storages_at_bus[b - 1].append(s)
            s += 1

        self.rc_max = self.storage_df["Rc"].values.tolist()
        self.rdc_max = self.storage_df["Rdc"].values.tolist()
        self.soc_max = self.storage_df["SOC"].values.tolist()

        self.eff_c = self.storage_df["Effc"].values.tolist()
        self.eff_dc = self.storage_df["Effdc"].values.tolist()

    def _init_stochastic_parameters(self):
        """Initializes all stochastic parameters
        """
        scenario_df = self.scenario_df

        self.n_realizations_per_stage = scenario_df.groupby("t")["n"].nunique().tolist()
        self.n_stages = len(self.n_realizations_per_stage)

        prob = []
        p_d = []
        re = []

        for t in range(self.n_stages):
            stage_df = scenario_df[scenario_df["t"] == t + 1]
            p_d.append(
                stage_df[
                    scenario_df.columns[
                        scenario_df.columns.to_series().str.contains("Pd")
                    ]
                ].values.tolist()
            )
            re.append(
                stage_df[
                    scenario_df.columns[
                        scenario_df.columns.to_series().str.contains("Re")
                    ]
                ].values.tolist()
            )
            prob.append(stage_df["p"].values.tolist())

        self.prob = prob
        self.p_d = p_d
        self.re = re

        self.cut_lb = [0] * self.n_stages

    def _init_initial_trial_points(self):
        """Initializes the first stage trial points
        """
        self.init_x_trial_point = [0] * self.n_gens
        self.init_y_trial_point = [0] * self.n_gens
        self.init_x_bs_trial_point = [
            [0] * n_periods for n_periods in self.backsight_periods
        ]
        self.init_soc_trial_point = [0.5 * soc for soc in self.soc_max]


class DataImporter:
    def __init__(self, data_directory: str = None):
        self.data_directory = data_directory if data_directory else ""

    def dataframe_from_csv(
        self, file_path: str, delimiter: str = "\s+"
    ) -> pd.DataFrame:
        path = os.path.join(self.data_directory, file_path)
        df = pd.read_csv(path, sep=delimiter)
        return df
