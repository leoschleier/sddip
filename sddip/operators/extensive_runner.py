import logging
from time import time

import gurobipy as gp

from sddip.sddip import parameters, sddip_logging, tree

logger = logging.getLogger(__name__)


def main() -> None:
    test_case_name = "case6ww"
    n_stages = 6
    n_realizations = 3

    log_manager = sddip_logging.LogManager()
    log_dir = log_manager.create_log_dir(f"{test_case_name}_ext")
    runtime_logger = sddip_logging.RuntimeLogger(log_dir)

    runtime_logger.start()

    logger.info("Building the model...")

    params = parameters.Parameters(test_case_name, n_stages, n_realizations)

    scenario_tree_construction_start_time = time()
    scenario_tree = tree.ScenarioTree(params.n_realizations_per_stage)
    runtime_logger.log_task_end(
        "scenario_tree_construction", scenario_tree_construction_start_time
    )
    logger.info(scenario_tree)

    ####################################################################
    # Variable initialization
    ####################################################################
    model_building_start_time = time()

    model = gp.Model("MSUC")

    x = {}
    y = {}
    s_up = {}
    s_down = {}
    ys_charge = {}
    ys_discharge = {}
    u = {}
    soc = {}
    socs_p = {}
    socs_n = {}
    ys_p = {}
    ys_n = {}
    delta = {}

    for t in range(params.n_stages):
        for node in scenario_tree.get_stage_nodes(t):
            n = node.index
            for g in range(params.n_gens):
                x[t, n, g] = model.addVar(
                    vtype=gp.GRB.BINARY, name=f"x_{t+1}_{n+1}_{g+1}"
                )
                y[t, n, g] = model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0, name=f"y_{t+1}_{n+1}_{g+1}"
                )
                s_up[t, n, g] = model.addVar(
                    vtype=gp.GRB.BINARY, name=f"s_up_{t+1}_{n+1}_{g+1}"
                )
                s_down[t, n, g] = model.addVar(
                    vtype=gp.GRB.BINARY, name=f"s_down_{t+1}_{n+1}_{g+1}"
                )
            for s in range(params.n_storages):
                ys_charge[t, n, s] = model.addVar(
                    vtype=gp.GRB.CONTINUOUS,
                    lb=0,
                    name=f"ys_c_{t+1}_{n+1}_{s+1}",
                )
                ys_discharge[t, n, s] = model.addVar(
                    vtype=gp.GRB.CONTINUOUS,
                    lb=0,
                    name=f"ys_dc_{t+1}_{n+1}_{s+1}",
                )
                u[t, n, s] = model.addVar(
                    vtype=gp.GRB.BINARY, name=f"u_{t+1}_{n+1}_{s+1}"
                )
                soc[t, n, s] = model.addVar(
                    vtype=gp.GRB.CONTINUOUS,
                    lb=0,
                    name=f"soc_{t+1}_{n+1}_{s+1}",
                )
                socs_p[t, n, s] = model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0, name=f"socs_p_{n+1}_{s+1}"
                )
                socs_n[t, n, s] = model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0, name=f"socs_n_{n+1}_{s+1}"
                )
            ys_p[t, n] = model.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=0, name=f"ys_p_{t+1}_{n+1}"
            )
            ys_n[t, n] = model.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=0, name=f"ys_n_{t+1}_{n+1}"
            )
            delta[t, n] = model.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=0, name=f"delta_{t+1}_{n+1}"
            )
            # model.addConstr(delta[t, n] == 0)

    model.update()

    ####################################################################
    # Objective and constraints
    ####################################################################

    # Objective
    logger.info("Adding objective...")

    conditional_probabilities = []
    p = 1
    for n in range(scenario_tree.n_stages):
        p = p * 1 / params.n_realizations_per_stage[n]
        conditional_probabilities.append(p)

    obj_gen = gp.quicksum(
        conditional_probabilities[t]
        * (
            params.gc[g] * y[t, n, g]
            + params.suc[g] * s_up[t, n, g]
            + params.sdc[g] * s_down[t, n, g]
        )
        for t in range(params.n_stages)
        for n in range(scenario_tree.n_nodes_per_stage[t])
        for g in range(params.n_gens)
    )

    penalty_gen = gp.quicksum(
        conditional_probabilities[t]
        * params.penalty
        * (ys_p[t, n] + ys_n[t, n] + delta[t, n])
        for t in range(params.n_stages)
        for n in range(scenario_tree.n_nodes_per_stage[t])
    )

    penalty_stge = gp.quicksum(
        conditional_probabilities[t]
        * params.penalty
        * (socs_n[t, n, s] + socs_p[t, n, s])
        for t in range(params.n_stages)
        for n in range(scenario_tree.n_nodes_per_stage[t])
        for s in range(params.n_storages)
    )

    obj = obj_gen + penalty_gen + penalty_stge

    model.setObjective(obj)

    # Balance constraints
    logger.info("Adding balance constraints...")

    model.addConstrs(
        (
            gp.quicksum(y[t, n.index, g] for g in range(params.n_gens))
            + gp.quicksum(
                params.eff_dc[s] * ys_discharge[t, n.index, s]
                - ys_charge[t, n.index, s]
                for s in range(params.n_storages)
            )
            + ys_p[t, n.index]
            - ys_n[t, n.index]
            == gp.quicksum(params.p_d[t][n.realization])
            - gp.quicksum(params.re[t][n.realization])
            for t in range(params.n_stages)
            for n in scenario_tree.get_stage_nodes(t)
        ),
        "balance",
    )

    # Generator constraints
    logger.info("Adding generation constraints...")

    model.addConstrs(
        (
            y[t, n, g] >= params.pg_min[g] * x[t, n, g] - delta[t, n]
            for g in range(params.n_gens)
            for t in range(params.n_stages)
            for n in range(scenario_tree.n_nodes_per_stage[t])
        ),
        "min-generation",
    )

    model.addConstrs(
        (
            y[t, n, g] <= params.pg_max[g] * x[t, n, g] + delta[t, n]
            for g in range(params.n_gens)
            for t in range(params.n_stages)
            for n in range(scenario_tree.n_nodes_per_stage[t])
        ),
        "max-generation",
    )

    # Storage constraints
    logger.info("Adding storage constraints...")

    model.addConstrs(
        (
            ys_charge[t, n, s] <= params.rc_max[s] * u[t, n, s]
            for s in range(params.n_storages)
            for t in range(params.n_stages)
            for n in range(scenario_tree.n_nodes_per_stage[t])
        ),
        "max-charge-rate",
    )

    model.addConstrs(
        (
            ys_discharge[t, n, s] <= params.rdc_max[s] * (1 - u[t, n, s])
            for s in range(params.n_storages)
            for t in range(params.n_stages)
            for n in range(scenario_tree.n_nodes_per_stage[t])
        ),
        "max-discharge-rate",
    )

    model.addConstrs(
        (
            soc[t, n, s] <= params.soc_max[s] + delta[t, n]
            for s in range(params.n_storages)
            for t in range(params.n_stages)
            for n in range(scenario_tree.n_nodes_per_stage[t])
        ),
        "max-soc",
    )

    # SOC transfer
    logger.info("Adding SOC constraints...")

    # t=0
    soc_init = params.init_soc_trial_point
    model.addConstrs(
        (
            soc[0, 0, s]
            == soc_init[s]
            + params.eff_c[s] * ys_charge[0, 0, s]
            - ys_discharge[0, 0, s]
            + socs_p[0, 0, s]
            - socs_n[0, 0, s]
            for s in range(params.n_storages)
        ),
        "soc",
    )
    # t>0
    for t in range(1, params.n_stages):
        for node in scenario_tree.get_stage_nodes(t):
            n = node.index
            a_n = node.parent.index
            model.addConstrs(
                (
                    soc[t, n, s]
                    == soc[t - 1, a_n, s]
                    + params.eff_c[s] * ys_charge[t, n, s]
                    - ys_discharge[t, n, s]
                    + socs_p[t, n, s]
                    - socs_n[t, n, s]
                    for s in range(params.n_storages)
                ),
                "soc",
            )
    # t=T
    t = params.n_stages - 1
    model.addConstrs(
        (
            soc[t, n.index, s] >= soc_init[s] - delta[t, n.index]
            for s in range(params.n_storages)
            for n in scenario_tree.get_stage_nodes(t)
        ),
        "soc-final",
    )

    # Power flow constraints
    logger.info("Adding power flow constraints...")

    for t in range(params.n_stages):
        for node in scenario_tree.get_stage_nodes(t):
            n = node.index
            line_flows = [
                gp.quicksum(
                    params.ptdf[l, b]
                    * (
                        gp.quicksum(y[t, n, g] for g in params.gens_at_bus[b])
                        + gp.quicksum(
                            params.eff_dc[s] * ys_discharge[t, n, s]
                            - ys_charge[t, n, s]
                            for s in params.storages_at_bus[b]
                        )
                        - params.p_d[t][node.realization][b]
                        + params.re[t][node.realization][b]
                    )
                    for b in range(params.n_buses)
                )
                for l in range(params.n_lines)
            ]
            model.addConstrs(
                (
                    line_flows[l] <= params.pl_max[l] + delta[t, n]
                    for l in range(params.n_lines)
                ),
                "power-flow(1)",
            )
            model.addConstrs(
                (
                    -line_flows[l] <= params.pl_max[l] + delta[t, n]
                    for l in range(params.n_lines)
                ),
                "power-flow(2)",
            )

    # Startup shutdown constraints
    logger.info("Adding start-up and shut-down constraints...")

    # t=0
    x_init = [0] * params.n_gens
    model.addConstrs(
        (
            x[0, 0, g] - x_init[g] <= s_up[0, 0, g] + delta[0, 0]
            for g in range(params.n_gens)
        ),
        "startup",
    )
    model.addConstrs(
        (
            x_init[g] - x[0, 0, g] <= s_down[0, 0, g] + delta[0, 0]
            for g in range(params.n_gens)
        ),
        "shutdown",
    )
    # t>0
    for t in range(1, params.n_stages):
        for node in scenario_tree.get_stage_nodes(t):
            n = node.index
            a_n = node.parent.index
            model.addConstrs(
                (
                    x[t, n, g] - x[t - 1, a_n, g]
                    <= s_up[t, n, g] + delta[t, n]
                    for g in range(params.n_gens)
                ),
                "startup",
            )
            model.addConstrs(
                (
                    x[t - 1, a_n, g] - x[t, n, g]
                    <= s_down[t, n, g] + delta[t, n]
                    for g in range(params.n_gens)
                ),
                "shutdown",
            )

    # Ramp rate constraints
    logger.info("Adding ramp rate constraints...")

    # t=0
    y_init = [0] * params.n_gens
    model.addConstrs(
        (
            y[0, 0, g] - y_init[g]
            <= params.r_up[g] * x_init[g]
            + params.r_su[g] * s_up[0, 0, g]
            + delta[0, 0]
            for g in range(params.n_gens)
        ),
        "rate-up",
    )
    model.addConstrs(
        (
            y_init[g] - y[0, 0, g]
            <= params.r_down[g] * x[0, 0, g]
            + params.r_sd[g] * s_down[0, 0, g]
            + delta[0, 0]
            for g in range(params.n_gens)
        ),
        "rate-down",
    )
    # t>0
    for t in range(1, params.n_stages):
        for node in scenario_tree.get_stage_nodes(t):
            n = node.index
            a_n = node.parent.index
            model.addConstrs(
                (
                    y[t, n, g] - y[t - 1, a_n, g]
                    <= params.r_up[g] * x[t - 1, a_n, g]
                    + params.r_su[g] * s_up[t, n, g]
                    + delta[t, n]
                    for g in range(params.n_gens)
                ),
                "rate-up",
            )
            model.addConstrs(
                (
                    y[t - 1, a_n, g] - y[t, n, g]
                    <= params.r_down[g] * x[t, n, g]
                    + params.r_sd[g] * s_down[t, n, g]
                    + delta[t, n]
                    for g in range(params.n_gens)
                ),
                "rate-down",
            )

    # Minimum up- and down-time constraints
    logger.info("Adding up- and down-time constraints...")

    for g in range(params.n_gens):
        for t in range(1, params.min_up_time[g]):
            for node in scenario_tree.get_stage_nodes(t):
                n = node.index
                ancestors = node.get_ancestors()
                model.addConstr(
                    (
                        gp.quicksum(x[m.stage, m.index, g] for m in ancestors)
                        >= t * s_down[t, n, g] - delta[t, n]
                    ),
                    "min-uptime",
                )

        for t in range(params.min_up_time[g], params.n_stages):
            for node in scenario_tree.get_stage_nodes(t):
                n = node.index
                ancestors = node.get_ancestors(params.min_up_time[g])
                model.addConstr(
                    (
                        gp.quicksum(x[m.stage, m.index, g] for m in ancestors)
                        >= params.min_up_time[g] * s_down[t, n, g]
                        - delta[t, n]
                    ),
                    "min-uptime",
                )

        for t in range(1, params.min_down_time[g]):
            for node in scenario_tree.get_stage_nodes(t):
                n = node.index
                ancestors = node.get_ancestors()
                model.addConstr(
                    (
                        gp.quicksum(
                            (1 - x[m.stage, m.index, g]) for m in ancestors
                        )
                        >= t * s_up[t, n, g] - delta[t, n]
                    ),
                    "min-downtime",
                )

        for t in range(params.min_down_time[g], params.n_stages):
            for node in scenario_tree.get_stage_nodes(t):
                n = node.index
                ancestors = node.get_ancestors(params.min_down_time[g])
                model.addConstr(
                    (
                        gp.quicksum(
                            (1 - x[m.stage, m.index, g]) for m in ancestors
                        )
                        >= params.min_down_time[g] * s_up[t, n, g]
                        - delta[t, n]
                    ),
                    "min-downtime",
                )

    # model.update()

    runtime_logger.log_task_end("model_building", model_building_start_time)

    ####################################################################
    # Solving procedure
    ####################################################################
    model.setParam("OutputFlag", 1)
    model.setParam("TimeLimit", 5 * 60 * 60)

    logger.info("Solving process started...")
    model_solving_start_time = time()
    model.optimize()

    # model.computeIIS()
    # model.write("model.ilp")
    # model.display()
    # model.logger.infoAttr("X")
    # model.write("model.lp")

    runtime_logger.log_task_end("model_solving", model_solving_start_time)

    slack_variables = [ys_p, ys_n, socs_p, socs_n, delta]
    total_slack = 0
    for variable in slack_variables:
        total_slack += sum([slack.x for _, slack in variable.items()])

    # logger.info(sum([slack.x for _, slack in delta.items()]))
    # logger.info(sum([slack.x for _, slack in socs_p.items()]))
    # logger.info(sum([slack.x for _, slack in socs_n.items()]))

    # logger.info("Charge")
    # logger.info(ys_charge[0, 0, 0].x)
    # logger.info(ys_charge[1, 0, 0].x)
    # logger.info(ys_charge[1, 1, 0].x)
    # logger.info(ys_charge[1, 2, 0].x)
    # logger.info("Discharge")
    # logger.info(ys_discharge[0, 0, 0].x)
    # logger.info(ys_discharge[1, 0, 0].x)
    # logger.info(ys_discharge[1, 1, 0].x)
    # logger.info(ys_discharge[1, 2, 0].x)
    # logger.info("SOC")
    # logger.info(soc[0, 0, 0].x)
    # logger.info(soc[1, 0, 0].x)
    # logger.info(soc[1, 1, 0].x)
    # logger.info(soc[1, 2, 0].x)

    logger.info("Solving finished.")
    logger.info("Status: %s", model.status)
    logger.info("Optimal value: %s", obj.getValue())
    logger.info("Total slack: %s", total_slack)
    logger.info("MIP gap: %s", model.MIPGap)

    runtime_logger.log_experiment_end()


if __name__ == "__main__":
    main()
