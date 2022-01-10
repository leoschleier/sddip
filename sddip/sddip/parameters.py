import os

import numpy as np
import pandas as pd

from sddip import utils, config


class Parameters:
    def __init__(self, test_case_name: str):
        raw_dir = os.path.join(test_case_name, "raw")
        test_case_raw_dir = os.path.join(config.test_cases_dir, raw_dir)
        # Data files
        bus_file_raw = os.path.join(test_case_raw_dir, "bus_data.txt")
        branch_file_raw = os.path.join(test_case_raw_dir, "branch_data.txt")
        gen_file_raw = os.path.join(test_case_raw_dir, "gen_data.txt")
        gen_cost_file_raw = os.path.join(test_case_raw_dir, "gen_cost_data.txt")
        scenario_data_file = os.path.join(test_case_raw_dir, "scenario_data.txt")
        # DataFrames
        self.bus_df = pd.read_csv(bus_file_raw, delimiter="\s+")
        self.branch_df = pd.read_csv(branch_file_raw, delimiter="\s+")
        self.gen_df = pd.read_csv(gen_file_raw, delimiter="\s+")
        self.gen_cost_df = pd.read_csv(gen_cost_file_raw, delimiter="\s+")
        self.scenario_df = pd.read_csv(scenario_data_file, delimiter="\s+")
        # Parameter initialization
        self.calc_ptdf()
        self.init_deterministic_parameters()
        self.init_stochastic_parameters()
        self.init_initial_trial_points()

    def calc_ptdf(self):
        bus_df = self.bus_df
        branch_df = self.branch_df

        nodes = bus_df.bus_i.values.tolist()
        edges = branch_df[["fbus", "tbus"]].values.tolist()

        graph = utils.Graph(nodes, edges)

        ref_bus = bus_df.loc[bus_df.type == 3].bus_i.values[0]

        a_inc = graph.incidence_matrix()
        b_l = (-branch_df.x / (branch_df.r ** 2 + branch_df.x ** 2)).tolist()
        b_diag = np.diag(b_l)

        m1 = b_diag.dot(a_inc)
        m2 = a_inc.T.dot(b_diag).dot(a_inc)

        m1 = np.delete(m1, ref_bus - 1, 1)
        m2 = np.delete(m2, ref_bus - 1, 0)
        m2 = np.delete(m2, ref_bus - 1, 1)

        ptdf = m1.dot(np.linalg.inv(m2))

        self.ptdf = np.insert(ptdf, ref_bus - 1, 0, axis=1)

        self.n_lines, self.n_buses = self.ptdf.shape

    def init_deterministic_parameters(self):
        gen_cost_df = self.gen_cost_df
        gen_df = self.gen_df
        branch_df = self.branch_df

        self.gc = np.array(gen_cost_df.c1)
        self.suc = np.array(gen_cost_df.startup)
        self.sdc = np.array(gen_cost_df.startup)
        # TODO Adjust penalty for slack variables
        self.penalty = 10000

        self.cost_coeffs = (
            self.gc.tolist()
            + self.suc.tolist()
            + self.sdc.tolist()
            + [self.penalty] * 2
        )

        self.pg_min = np.array(gen_df.Pmin)
        self.pg_max = np.array(gen_df.Pmax)
        self.pl_max = np.array(branch_df.rateA)

        self.n_gens = len(self.gc)

        # TODO Add ramp rate limits
        self.rg_up_max = np.full(self.n_gens, 1000)
        self.rg_down_max = np.full(self.n_gens, 1000)

        # Lists of generators at each bus
        #
        # Example: [[0,1], [], [2]]
        # Generator 1 & 2 are located at bus 1
        # No Generator is located at bus 2
        # Generator 3 is located at bus 3
        gens_at_bus = [[] for _ in range(self.n_buses)]
        g = 0
        for b in gen_df.bus.values:
            gens_at_bus[b - 1].append(g)
            g += 1
        self.gens_at_bus = gens_at_bus

    def init_stochastic_parameters(self):
        scenario_df = self.scenario_df

        self.n_nodes_per_stage = scenario_df.groupby("t")["n"].nunique().tolist()
        self.n_stages = len(self.n_nodes_per_stage)

        prob = []
        p_d = []

        for t in range(self.n_stages):
            stage_df = scenario_df[scenario_df["t"] == t + 1]
            p_d.append(
                stage_df[
                    scenario_df.columns[
                        scenario_df.columns.to_series().str.contains("Pd")
                    ]
                ].values.tolist()
            )
            prob.append(stage_df["p"].values.tolist())

        self.prob = prob
        self.p_d = p_d

        # TODO Cut lower bound
        self.cut_lb = [0] * self.n_stages

    def init_initial_trial_points(self):
        self.init_x_trial_point = [0] * self.n_gens
        self.init_y_trial_point = [0] * self.n_gens
