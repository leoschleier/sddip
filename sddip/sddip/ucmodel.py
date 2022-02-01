from abc import ABC, abstractmethod
import gurobipy as gp
import numpy as np
from scipy import linalg


class ModelBuilder(ABC):
    def __init__(
        self,
        n_buses: int,
        n_lines: int,
        n_generators: int,
        generators_at_bus: list,
        backsight_periods: list,
    ) -> None:
        self.n_buses = n_buses
        self.n_lines = n_lines
        self.n_generators = n_generators
        self.generators_at_bus = generators_at_bus
        self.backsight_periods = backsight_periods
        self.model = gp.Model("MILP: Unit commitment")
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("InFeasTol", 10 ** (-9))
        self.model.setParam("NumericFocus", 3)

        # Commitment decision
        self.x = []
        # Dispatch decision
        self.y = []
        # Generator state backsight variables
        # Given current stage t, x_bs[g][k] is the state of generator g at stage (t-k-1)
        self.x_bs = []
        # Copy variables
        self.z_x = []
        self.z_y = []
        self.z_x_bs = []
        # Startup decsision
        self.s_up = []
        # Shutdown decision
        self.s_down = []
        # Expected value function approximation
        self.theta = None
        # Positive slack
        self.ys_p = None
        # Negative slack
        self.ys_n = None

        # Objective
        self.objective_terms = None
        # Balance constraints
        self.balance_constraints = None
        # Copy constraints
        self.copy_constraints_x = None
        self.copy_constraints_y = None
        # Cut constraints
        self.cut_constraints = None
        # Cut lower bound
        self.cut_lower_bound = None

        self.initialize_variables()
        self.initialize_copy_variables()

    def initialize_variables(self):
        for g in range(self.n_generators):
            self.x.append(self.model.addVar(vtype=gp.GRB.BINARY, name="x_%i" % (g + 1)))
            self.y.append(
                self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="y_%i" % (g + 1))
            )
            self.x_bs.append(
                [
                    self.model.addVar(
                        vtype=gp.GRB.BINARY, name="x_bs%i_%i" % (g + 1, k + 1),
                    )
                    for k in range(self.backsight_periods[g])
                ]
            )
            self.s_up.append(
                self.model.addVar(vtype=gp.GRB.BINARY, name="s_up_%i" % (g + 1))
            )
            self.s_down.append(
                self.model.addVar(vtype=gp.GRB.BINARY, name="s_down_%i" % (g + 1))
            )
        self.theta = self.model.addVar(
            vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="theta"
        )
        self.ys_p = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="ys_p")
        self.ys_n = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="ys_n")
        self.model.update()

    def initialize_copy_variables(self):
        for g in range(self.n_generators):
            self.z_x.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS,
                    lb=-gp.GRB.INFINITY,
                    name="z_x_%i" % (g + 1),
                )
            )
            self.z_y.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS,
                    lb=-gp.GRB.INFINITY,
                    name="z_y_%i" % (g + 1),
                )
            )
            self.z_x_bs.append(
                [
                    self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        lb=-gp.GRB.INFINITY,
                        name="z_x_bs%i_%i" % (g + 1, k + 1),
                    )
                    for k in range(self.backsight_periods[g])
                ]
            )
        self.model.update()

    def add_objective(self, coefficients: list):
        coefficients = coefficients + [1]
        variables = (
            self.y + self.s_up + self.s_down + [self.ys_p, self.ys_n, self.theta]
        )
        self.objective_terms = gp.LinExpr(coefficients, variables)
        self.model.setObjective(self.objective_terms)
        self.update_model()

    def add_balance_constraints(self, total_demand: float):
        self.balance_constraints = self.model.addConstr(
            gp.quicksum(self.y) + self.ys_p - self.ys_n == total_demand, "balance"
        )
        self.update_model()

    def add_generator_constraints(self, min_generation: list, max_generation: list):
        self.model.addConstrs(
            (
                self.y[g] >= min_generation[g] * self.x[g]
                for g in range(self.n_generators)
            ),
            "min-generation",
        )
        self.model.addConstrs(
            (
                self.y[g] <= max_generation[g] * self.x[g]
                for g in range(self.n_generators)
            ),
            "max-generation",
        )
        self.update_model()

    def add_power_flow_constraints(self, ptdf, max_line_capacities: list, demand: list):
        line_flows = [
            gp.quicksum(
                ptdf[l, b]
                * (
                    gp.quicksum(self.y[g] for g in self.generators_at_bus[b])
                    - demand[b]
                )
                for b in range(self.n_buses)
            )
            for l in range(self.n_lines)
        ]
        self.model.addConstrs(
            (line_flows[l] <= max_line_capacities[l] for l in range(self.n_lines)),
            "power-flow(1)",
        )
        self.model.addConstrs(
            (-line_flows[l] <= max_line_capacities[l] for l in range(self.n_lines)),
            "power-flow(2)",
        )
        self.update_model()

    def add_startup_shutdown_constraints(self):
        self.model.addConstrs(
            (self.x[g] - self.z_x[g] <= self.s_up[g] for g in range(self.n_generators)),
            "up-down(1)",
        )
        self.model.addConstrs(
            (
                self.x[g] - self.z_x[g] == self.s_up[g] - self.s_down[g]
                for g in range(self.n_generators)
            ),
            "up-down(2)",
        )
        self.update_model()

    def add_ramp_rate_constraints(self, max_rate_up: list, max_rate_down: list):
        self.model.addConstrs(
            (
                self.y[g] - self.z_y[g] <= max_rate_up[g]
                for g in range(self.n_generators)
            ),
            "rate-up",
        )
        self.model.addConstrs(
            (
                self.z_y[g] - self.y[g] <= max_rate_down[g]
                for g in range(self.n_generators)
            ),
            "rate-down",
        )

    def add_up_down_time_constraints(self, min_up_times: list, min_down_times: list):
        self.model.addConstrs(
            (
                gp.quicksum(self.z_x_bs[g]) >= min_up_times[g] * self.s_down[g]
                for g in range(self.n_generators)
            ),
            "up-time",
        )

        self.model.addConstrs(
            (
                len(self.z_x_bs[g]) - gp.quicksum(self.z_x_bs[g])
                >= min_down_times[g] * self.s_up[g]
                for g in range(self.n_generators)
            ),
            "down-time",
        )

        self.model.addConstrs(
            (
                self.z_x_bs[g][k] == self.x_bs[g][k]
                for g in range(self.n_generators)
                for k in range(self.backsight_periods[g])
            ),
            "backsight",
        )

    # TODO Adjust copy constraints to suit up- and down-time constraints
    @abstractmethod
    def add_copy_constraints(self, x_trial_point: list, y_trial_point):
        pass

    def add_cut_lower_bound(self, lower_bound: float):
        self.cut_lower_bound = self.model.addConstr(
            (self.theta >= lower_bound), "cut-lb"
        )

    def add_cut_constraints(
        self,
        cut_intercepts: list,
        cut_gradients: list,
        binary_multipliers: list,
        big_m: float = 10 ** 18,
        sos: bool = False,
    ):
        r = 0
        for intercept, gradient, multipliers in zip(
            cut_intercepts, cut_gradients, binary_multipliers
        ):
            self.add_cut(intercept, gradient, multipliers, r, big_m, sos)
            r += 1
        self.update_model()

    def add_cut(
        self,
        cut_intercept: float,
        cut_gradient: list,
        y_binary_multipliers: np.array,
        id: int,
        big_m: float = 10 ** 18,
        sos: bool = False,
    ):

        x_binary_multipliers = linalg.block_diag(*[1] * len(self.x))
        n_total_backsight_variables = sum(self.backsight_periods)
        x_bs_binary_multipliers = linalg.block_diag(*[1] * n_total_backsight_variables)

        binary_multipliers = linalg.block_diag(
            x_binary_multipliers, y_binary_multipliers, x_bs_binary_multipliers
        )

        n_var_approximations, n_binaries = binary_multipliers.shape

        ny = self.model.addVars(
            n_binaries, vtype=gp.GRB.CONTINUOUS, lb=0, name=f"ny_{id}"
        )
        my = self.model.addVars(
            n_binaries, vtype=gp.GRB.CONTINUOUS, lb=0, name=f"my_{id}"
        )
        eta = self.model.addVars(
            n_var_approximations,
            vtype=gp.GRB.CONTINUOUS,
            lb=-gp.GRB.INFINITY,
            name=f"eta_{id}",
        )
        lmda = self.model.addVars(
            n_binaries, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name=f"lambda_{id}"
        )

        state_vars = (
            self.x
            + self.y
            + [variable for bs_vars in self.x_bs for variable in bs_vars]
        )

        w = self.model.addVars(n_binaries, vtype=gp.GRB.BINARY, name=f"w_{id}")
        u = self.model.addVars(n_binaries, vtype=gp.GRB.BINARY, name=f"u_{id}")

        # TODO Define Big-Ms
        m2 = [big_m] * n_binaries
        m4 = [big_m] * n_binaries

        # Cut constraint
        self.model.addConstr(
            (
                self.theta
                >= cut_intercept
                + gp.quicksum(lmda[i] * cut_gradient[i] for i in range(n_binaries))
            ),
            f"cut_{id}",
        )

        # KKT conditions
        self.model.addConstrs(
            (
                0
                == -cut_gradient[j]
                - ny[j]
                + my[j]
                + gp.quicksum(
                    binary_multipliers[i, j] * eta[i]
                    for i in range(n_var_approximations)
                )
                for j in range(n_binaries)
            ),
            f"KKT(1)_{id}",
        )

        self.model.addConstrs(
            (
                0
                == gp.quicksum(
                    binary_multipliers[i, j] * lmda[j] for j in range(n_binaries)
                )
                - state_vars[i]
                for i in range(n_var_approximations)
            ),
            f"KKT(2)_{id}",
        )

        if sos:
            for i in range(n_binaries):
                self.model.addGenConstrIndicator(w[i], True, ny[i] == 0)
                self.model.addGenConstrIndicator(u[i], True, my[i] == 0)
            # self.model.addConstrs(
            #     ((w[i] == 1) >> (ny[i] == 0) for i in range(n_binaries)), f"KKT(4)_{id}"
            # )
            # self.model.addConstrs(
            #     ((u[i] == 1) >> (my[i] == 0) for i in range(n_binaries)), f"KKT(6)_{id}"
            # )
        else:
            self.model.addConstrs(
                (ny[i] <= m2[i] * (1 - w[i]) for i in range(n_binaries)), f"KKT(4)_{id}"
            )
            self.model.addConstrs(
                (my[i] <= m4[i] * (1 - u[i]) for i in range(n_binaries)), f"KKT(6)_{id}"
            )

        self.model.addConstrs(
            (lmda[i] <= w[i] for i in range(n_binaries)), f"KKT(3)_{id}"
        )

        self.model.addConstrs(
            (lmda[i] - 1 >= -u[i] for i in range(n_binaries)), f"KKT(5)_{id}"
        )

    def remove(self, gurobi_objects):
        # Remove if gurobi_objects not None or not empty
        if gurobi_objects:
            self.model.remove(gurobi_objects)
            self.update_model

    def update_model(self):
        self.model.update()

    def disable_output(self):
        self.model.setParam("OutputFlag", 0)

    def enable_output(self):
        self.model.setParam("OutputFlag", 1)


class ForwardModelBuilder(ModelBuilder):
    def __init__(
        self,
        n_buses: int,
        n_lines: int,
        n_generators: int,
        generators_at_bus: list,
        backsight_periods: list,
    ) -> None:
        super().__init__(
            n_buses, n_lines, n_generators, generators_at_bus, backsight_periods
        )

    def add_copy_constraints(
        self, x_trial_point: list, y_trial_point: list, x_bs_trial_point: list[list]
    ):
        self.copy_constraints_y = self.model.addConstrs(
            (self.z_x[g] == x_trial_point[g] for g in range(self.n_generators)),
            "copy-x",
        )
        self.copy_constraints_x = self.model.addConstrs(
            (self.z_y[g] == y_trial_point[g] for g in range(self.n_generators)),
            "copy-y",
        )
        self.copy_constraints_x_bs = self.model.addConstrs(
            (
                self.z_x_bs[g][k] == x_bs_trial_point[g][k]
                for g in range(self.n_generators)
                for k in range(self.backsight_periods[g])
            ),
            "copy-x-bs",
        )
        self.update_model()


class BackwardModelBuilder(ModelBuilder):
    def __init__(
        self,
        n_buses: int,
        n_lines: int,
        n_generators: int,
        generators_at_bus: list,
        backsight_periods: list,
    ) -> None:
        super().__init__(
            n_buses, n_lines, n_generators, generators_at_bus, backsight_periods
        )

        self.n_x_trial_binaries = None
        self.n_y_trial_binaries = None
        self.n_x_bs_trial_binaries = None

        self.relaxed_terms = []

        # Copy variable for binary variables
        self.x_bin_copy_vars = []
        self.y_bin_copy_vars = []
        self.x_bs_bin_copy_vars = []

        # Copy constraints
        self.copy_constraints_x = None
        self.copy_constraints_y = None

    def add_relaxation(
        self,
        x_binary_trial_point: list,
        y_binary_trial_point: list,
        x_bs_binary_trial_point: list[list],
    ):
        self.bin_copy_vars = []
        self.n_x_trial_binaries = len(x_binary_trial_point)
        self.n_y_trial_binaries = len(y_binary_trial_point)
        self.n_x_bs_trial_binaries = [
            len(trial_point) for trial_point in x_bs_binary_trial_point
        ]

        for j in range(self.n_x_trial_binaries):
            self.x_bin_copy_vars.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS,
                    lb=0,
                    ub=1,
                    name="x_bin_copy_var_%i" % (j + 1),
                )
            )

        for j in range(self.n_y_trial_binaries):
            self.y_bin_copy_vars.append(
                self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS,
                    lb=0,
                    ub=1,
                    name="y_bin_copy_var_%i" % (j + 1),
                )
            )

        for n_vars in self.n_x_bs_trial_binaries:
            self.x_bs_bin_copy_vars.append(
                [
                    self.model.addVar(
                        vtype=gp.GRB.CONTINUOUS,
                        lb=0,
                        ub=1,
                        name="x_bs_bin_copy_var_%i" % (j + 1),
                    )
                    for _ in range(n_vars)
                ]
            )

        self.relax(x_binary_trial_point, y_binary_trial_point, x_bs_binary_trial_point)

    def relax(
        self,
        x_binary_trial_point: list,
        y_binary_trial_point: list,
        x_bs_binary_trial_point: list[list],
    ):
        self.check_bin_copy_vars_not_empty()

        self.relaxed_terms += [
            x_binary_trial_point[j] - self.x_bin_copy_vars[j]
            for j in range(self.n_x_trial_binaries)
        ]

        self.relaxed_terms += [
            y_binary_trial_point[j] - self.y_bin_copy_vars[j]
            for j in range(self.n_y_trial_binaries)
        ]

        self.relaxed_terms += [
            x_bs_binary_trial_point[g][k] - self.x_bs_bin_copy_vars[g][k]
            for g in range(len(self.n_x_bs_trial_binaries))
            for k in range(self.n_x_bs_trial_binaries[g])
        ]

    def add_copy_constraints(self, y_binary_trial_multipliers: np.array):
        self.check_bin_copy_vars_not_empty()

        n_y_var_approximations, n_y_binaries = y_binary_trial_multipliers.shape

        self.copy_constraints_y = self.model.addConstrs(
            (
                self.z_y[i]
                == gp.quicksum(
                    y_binary_trial_multipliers[i, j] * self.y_bin_copy_vars[j]
                    for j in range(n_y_binaries)
                )
                for i in range(n_y_var_approximations)
            ),
            "copy-y",
        )

        self.copy_constraints_x = self.model.addConstrs(
            (
                self.z_x[i] == self.x_bin_copy_vars[i]
                for i in range(self.n_x_trial_binaries)
            ),
            "copy-x",
        )

        self.copy_constraints_x_bs = self.model.addConstrs(
            (
                self.z_x_bs[g][k] == self.x_bs_bin_copy_vars[g][k]
                for g in range(len(self.n_x_bs_trial_binaries))
                for k in range(self.n_x_bs_trial_binaries[g])
            ),
            "copy-x-bs",
        )

        self.update_model()

    def check_bin_copy_vars_not_empty(self):
        if not (
            self.x_bin_copy_vars and self.y_bin_copy_vars and self.x_bs_bin_copy_vars
        ):
            raise ValueError("Copy variable does not exist. Call add_relaxation first.")
