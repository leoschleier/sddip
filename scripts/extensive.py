import gurobipy as gp
from time import time
from sddip import parameters, tree, logger

test_case_name = "WB2"

log_manager = logger.LogManager()
log_dir = log_manager.create_log_dir(f"{test_case_name}_ext")
runtime_logger = logger.RuntimeLogger(log_dir)

runtime_logger.start()

print("Building the model...")

params = parameters.Parameters(test_case_name)

scenario_tree_construction_start_time = time()
scenario_tree = tree.ScenarioTree(params.n_realizations_per_stage)
runtime_logger.log_task_end(
    "scenario_tree_construction", scenario_tree_construction_start_time
)
print(scenario_tree)


########################################################################################################################
# Variables initialization
########################################################################################################################
model_building_start_time = time()

model = gp.Model("MSUC")

x = {}
y = {}
s_up = {}
s_down = {}
ys_p = {}
ys_n = {}

for t in range(params.n_stages):
    for node in scenario_tree.get_stage_nodes(t):
        n = node.index
        for g in range(params.n_gens):
            x[t, n, g] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{t+1}_{n+1}_{g+1}")
            y[t, n, g] = model.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=0, name=f"y_{t+1}_{n+1}_{g+1}"
            )
            s_up[t, n, g] = model.addVar(
                vtype=gp.GRB.BINARY, name=f"s_up_{t+1}_{n+1}_{g+1}"
            )
            s_down[t, n, g] = model.addVar(
                vtype=gp.GRB.BINARY, name=f"s_down_{t+1}_{n+1}_{g+1}"
            )
        ys_p[t, n] = model.addVar(
            vtype=gp.GRB.CONTINUOUS, lb=0, name=f"ys_p_{t+1}_{n+1}"
        )
        ys_n[t, n] = model.addVar(
            vtype=gp.GRB.CONTINUOUS, lb=0, name=f"ys_n_{t+1}_{n+1}"
        )

model.update()


########################################################################################################################
# Objective and constraints
########################################################################################################################

# Objective
obj = gp.quicksum(
    1
    / scenario_tree.n_nodes_per_stage[t]
    * (
        params.gc[g] * y[t, n, g]
        + params.suc[g] * s_up[t, n, g]
        + params.sdc[g] * s_down[t, n, g]
        + params.penalty * (ys_p[t, n] + ys_n[t, n])
    )
    for t in range(params.n_stages)
    for n in range(scenario_tree.n_nodes_per_stage[t])
    for g in range(params.n_gens)
)

model.setObjective(obj)


# Balance constraints
model.addConstrs(
    (
        gp.quicksum(y[t, n.index, g] for g in range(params.n_gens))
        + ys_p[t, n.index]
        - ys_n[t, n.index]
        == gp.quicksum(params.p_d[t][n.realization])
        for t in range(params.n_stages)
        for n in scenario_tree.get_stage_nodes(t)
    ),
    "balance",
)


# Generator constraints
model.addConstrs(
    (
        y[t, n, g] >= params.pg_min[g] * x[t, n, g]
        for g in range(params.n_gens)
        for t in range(params.n_stages)
        for n in range(scenario_tree.n_nodes_per_stage[t])
    ),
    "min-generation",
)

model.addConstrs(
    (
        y[t, n, g] <= params.pg_max[g] * x[t, n, g]
        for g in range(params.n_gens)
        for t in range(params.n_stages)
        for n in range(scenario_tree.n_nodes_per_stage[t])
    ),
    "max-generation",
)


# Power flow constraints
for t in range(params.n_stages):
    for node in scenario_tree.get_stage_nodes(t):
        n = node.index
        line_flows = [
            gp.quicksum(
                params.ptdf[l, b]
                * (
                    gp.quicksum(y[t, n, g] for g in params.gens_at_bus[b])
                    - params.p_d[t][node.realization][b]
                )
                for b in range(params.n_buses)
            )
            for l in range(params.n_lines)
        ]
        model.addConstrs(
            (line_flows[l] <= params.pl_max[l] for l in range(params.n_lines)),
            "power-flow(1)",
        )
        model.addConstrs(
            (-line_flows[l] <= params.pl_max[l] for l in range(params.n_lines)),
            "power-flow(2)",
        )


# Startup shutdown constraints
# t=0
x_init = [0] * params.n_gens
model.addConstrs(
    (x[0, 0, g] - x_init[g] <= s_up[0, 0, g] for g in range(params.n_gens)),
    "up-down(1)",
)
model.addConstrs(
    (
        x[0, 0, g] - x_init[g] == s_up[0, 0, g] - s_down[0, 0, g]
        for g in range(params.n_gens)
    ),
    "up-down(2)",
)
# t>0
for t in range(1, params.n_stages):
    for node in scenario_tree.get_stage_nodes(t):
        n = node.index
        a_n = node.parent.index
        model.addConstrs(
            (
                x[t, n, g] - x[t - 1, a_n, g] <= s_up[t, n, g]
                for g in range(params.n_gens)
            ),
            "up-down(1)",
        )
        model.addConstrs(
            (
                x[t, n, g] - x[t - 1, a_n, g] == s_up[t, n, g] - s_down[t, n, g]
                for g in range(params.n_gens)
            ),
            "up-down(2)",
        )

# Ramp rate constraints
# t=0
y_init = [0] * params.n_gens
model.addConstrs(
    (y[0, 0, g] - y_init[g] <= params.r_up[g] for g in range(params.n_gens)), "rate-up",
)
model.addConstrs(
    (y_init[g] - y[0, 0, g] <= params.r_down[g] for g in range(params.n_gens)),
    "rate-down(2)",
)
# t>0
for t in range(1, params.n_stages):
    for node in scenario_tree.get_stage_nodes(t):
        n = node.index
        a_n = node.parent.index
        model.addConstrs(
            (
                y[t, n, g] - y[t - 1, a_n, g] <= params.r_up[g]
                for g in range(params.n_gens)
            ),
            "rate-up",
        )
        model.addConstrs(
            (
                y[t - 1, a_n, g] - y[t, n, g] <= params.r_down[g]
                for g in range(params.n_gens)
            ),
            "rate-down",
        )

# Minimum up- and down-time constraints
for g in range(params.n_gens):
    for t in range(1, params.min_up_time[g]):
        for node in scenario_tree.get_stage_nodes(t):
            n = node.index
            ancestors = node.get_ancestors()
            model.addConstr(
                (
                    gp.quicksum(x[m.stage, m.index, g] for m in ancestors)
                    >= (t + 1) * s_down[t, n, g]
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
                ),
                "min-uptime",
            )

    for t in range(1, params.min_down_time[g]):
        for node in scenario_tree.get_stage_nodes(t):
            n = node.index
            ancestors = node.get_ancestors()
            model.addConstr(
                (
                    gp.quicksum((1 - x[m.stage, m.index, g]) for m in ancestors)
                    >= (t + 1) * s_up[t, n, g]
                ),
                "min-downtime",
            )

    for t in range(params.min_down_time[g], params.n_stages):
        for node in scenario_tree.get_stage_nodes(t):
            n = node.index
            ancestors = node.get_ancestors(params.min_down_time[g])
            model.addConstr(
                (
                    gp.quicksum((1 - x[m.stage, m.index, g]) for m in ancestors)
                    >= params.min_down_time[g] * s_up[t, n, g]
                ),
                "min-downtime",
            )

model.update()

runtime_logger.log_task_end(f"model_building", model_building_start_time)

########################################################################################################################
# Solving procedure
########################################################################################################################
model.setParam("OutputFlag", 0)

print("Solving process started...")
model_solving_start_time = time()
model.optimize()
runtime_logger.log_task_end("model_solving", model_solving_start_time)

print("Solving finished.")
print(f"Optimal value: {obj.getValue()}")

runtime_logger.log_experiment_end()
