import logging
import gurobipy as gp

from .ucmodeldynamic import BackwardModelBuilder

logger = logging.getLogger(__name__)


class ClassicalModel(BackwardModelBuilder):
    def __init__(
        self,
        n_buses: int,
        n_lines: int,
        n_generators: int,
        n_storages: int,
        generators_at_bus: list,
        storages_at_bus: list,
        backsight_periods: list,
        lp_relax: bool = False,
    ) -> None:
        super().__init__(
            n_buses,
            n_lines,
            n_generators,
            n_storages,
            generators_at_bus,
            storages_at_bus,
            backsight_periods,
            lp_relax,
        )
        self.lp_relax = lp_relax

    def binary_approximation(self, y_bin_multipliers, soc_bin_multipliers):

        self.y_bin_states = []
        self.soc_bin_states = []
        n_y_bin_vars = [len(bin_mult) for bin_mult in y_bin_multipliers]
        n_soc_bin_vars = [len(bin_mult) for bin_mult in soc_bin_multipliers]

        var_type = gp.GRB.CONTINUOUS if self.lp_relax else gp.GRB.BINARY

        g = 1
        for n_y in n_y_bin_vars:
            self.y_bin_states.append(
                self.model.addVars(
                    n_y, vtype=var_type, lb=0, ub=1, name=f"y_bin_{g}"
                )
            )
            g += 1

        s = 1
        for n_soc in n_soc_bin_vars:
            self.soc_bin_states.append(
                self.model.addVars(
                    n_soc, vtype=var_type, lb=0, ub=1, name=f"soc_bin_{s}"
                )
            )
            s += 1

        self.y_bin_states_flattened = [
            y_bin_var
            for y_tuple_dict in self.y_bin_states
            for y_bin_var in y_tuple_dict.values()
        ]

        self.soc_bin_states_flattened = [
            soc_bin_var
            for soc_tuple_dict in self.soc_bin_states
            for soc_bin_var in soc_tuple_dict.values()
        ]

        self.model.addConstrs(
            (
                gp.LinExpr(
                    y_bin_multipliers[g], self.y_bin_states[g].select("*")
                )
                == self.y[g]
                for g in range(self.n_generators)
            ),
            name="y_bin_appr",
        )

        self.model.addConstrs(
            (
                gp.LinExpr(
                    soc_bin_multipliers[s], self.soc_bin_states[s].select("*")
                )
                == self.soc[s]
                for s in range(self.n_storages)
            ),
            name="soc_bin_appr",
        )
        self.update_model()

    def add_sddip_copy_constraints(
        self,
        x_binary_trial_point: list,
        y_binary_trial_point: list,
        x_bs_binary_trial_point: list[list],
        soc_binary_trial_point: list,
    ):

        self.add_relaxation(
            x_binary_trial_point,
            y_binary_trial_point,
            x_bs_binary_trial_point,
            soc_binary_trial_point,
        )
        self.sddip_copy_constrs = []
        for term in self.relaxed_terms:
            self.sddip_copy_constrs.append(
                self.model.addConstr(term == 0, "sddip-copy")
            )

    def relax_sddip_copy_constraints(
        self,
        x_binary_trial_point: list,
        y_binary_trial_point: list,
        x_bs_binary_trial_point: list[list],
        soc_binary_trial_point: list,
    ):
        self.add_relaxation(
            x_binary_trial_point,
            y_binary_trial_point,
            x_bs_binary_trial_point,
            soc_binary_trial_point,
        )

    def add_cut_constraints(
        self,
        cut_intercepts: list,
        cut_gradients: list,
    ):

        state_variables = (
            self.x
            + self.y_bin_states_flattened
            + [var for gen_bs in self.x_bs for var in gen_bs]
            + self.soc_bin_states_flattened
        )

        id = 0
        for intercept, gradient in zip(cut_intercepts, cut_gradients):
            # Cut constraint
            self.model.addConstr(
                (
                    self.theta
                    >= intercept + gp.LinExpr(gradient, state_variables)
                ),
                f"cut_{id}",
            )
            id += 1

    def add_benders_cuts(
        self, cut_intercepts: list, cut_gradients: list, trial_points: list
    ):

        state_variables = (
            self.x
            + self.y_bin_states_flattened
            + [var for gen_bs in self.x_bs for var in gen_bs]
            + self.soc_bin_states_flattened
        )

        n_state_variables = len(state_variables)

        for intercept, gradient, trial_point in zip(
            cut_intercepts, cut_gradients, trial_points
        ):
            if not n_state_variables == len(trial_point):
                logger.warning("Trial point: %s", trial_point)
                raise ValueError(
                    "Number of state variables must be equal to the number of trial points."
                )

            self.model.addConstr(
                (
                    self.theta
                    >= intercept
                    + gp.quicksum(
                        gradient[i] * (state_variables[i] - trial_point[i])
                        for i in range(n_state_variables)
                    )
                ),
                f"cut",
            )
