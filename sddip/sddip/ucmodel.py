import gurobipy as gp
import numpy as np

class ModelBuilder:

    def __init__(self, n_buses:int, n_lines:int, n_generators:int, generators_at_bus:list) -> None:
        self.n_buses = n_buses
        self.n_lines = n_lines
        self.n_generators = n_generators
        self.generators_at_bus = generators_at_bus
        
        self.model = gp.Model("MILP: Unit commitment")

        # Commitment decisions
        self.x = []
        # Dispatch decisions
        self.y = []
        # Copy variable
        self.z = []
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

        # Balance constraints
        self.balance_constraints = None
        # Copy constraints
        self.copy_constraints = None
        # Cut constraints
        self.cut_constraints = None

        self.initialize_variables()
    

    def initialize_variables(self):
        for g in range(self.n_generators):
            self.x.append(self.model.addVar(vtype = gp.GRB.BINARY, name = "x_%i"%(g+1)))
            self.y.append(self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, name = "y_%i"%(g+1)))
            self.z.append(self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = "z_%i"%(g+1)))
            self.s_up.append(self.model.addVar(vtype = gp.GRB.BINARY, name = "s_up_%i"%(g+1)))
            self.s_down.append(self.model.addVar(vtype = gp.GRB.BINARY, name = "s_down_%i"%(g+1))) 
        self.theta = self.model.addVar(vtype = gp.GRB.CONTINUOUS, name = "theta")
        self.ys_p = self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, name = "ys_p")
        self.ys_n = self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, name = "ys_n")
        self.model.update()


    def add_objective(self, coefficients:list):
        variables = self.y + self.s_up + self.s_down + [self.ys_p, self.ys_n]
        objective = gp.LinExpr(coefficients, variables)
        self.model.setObjective(objective)
        self.update_model()


    def add_balance_constraints(self, demand:float):
        self.balance_constraints = self.model.addConstr(
            gp.quicksum(self.y) + self.ys_p - self.ys_n == demand, "balance")
        self.update_model()
    

    def add_generator_constraints(self, min_generation:list, max_generation:list):
        self.model.addConstrs((self.y[g] >= min_generation[g]*self.x[g] for g in range(self.n_generators)), 
            "min-generation")
        self.model.addConstrs((self.y[g] <= max_generation[g]*self.x[g] for g in range(self.n_generators)), 
            "max-generation")
        self.update_model()


    def add_power_flow_constraints(self, ptdf, max_line_capacities:list):
        line_flows = [gp.quicksum(ptdf[l,b] * gp.quicksum(self.y[g] for g in self.generators_at_bus[b]) 
            for b in range(self.n_buses)) for l in range(self.n_lines)]
        self.model.addConstrs((line_flows[l] <= max_line_capacities[l] for l in range(self.n_lines)), "power-flow(1)")
        self.model.addConstrs((-line_flows[l] <= max_line_capacities[l] for l in range(self.n_lines)), "power-flow(2)")
        self.update_model()


    def add_startup_shutdown_constraints(self):
        self.model.addConstrs((self.x[g] - self.z[g] <= self.s_up[g]  
            for g in range(self.n_generators)), "up-down(1)")
        self.model.addConstrs((self.x[g] - self.z[g] <= self.s_up[g] - self.s_down[g]  
            for g in range(self.n_generators)), "up-down(2)")
        self.update_model()


    def add_copy_constraints(self, trial_point:list):
        self.copy_constraints = self.model.addConstrs((self.z[g] == trial_point[g] 
            for g in range(self.n_generators)), "copy")
        self.update_model()

    
    def add_cut_constraints(self, cut_intercepts:list, cut_gradients:list):
        n_cuts = len(cut_intercepts)
        if not n_cuts == len(cut_gradients):
            raise ValueError("Number of cut intercepts and number of cuts gradients need to be equal.")

        x = np.array(self.x)
        self.cut_constraints = self.model.addConstrs(
            (self.theta >= cut_intercepts[i] + np.asscalar(np.array(cut_gradients[i]).T.dot(x)) for i in range(n_cuts)), "cut")
        self.update_model()


    def update_model(self):
        self.model.update()

    
    def update_balance_constraints(self, demand:float):
        self.model.remove(self.balance_constraints)
        self.update_model()
        self.add_balance_constraints(demand)


    def update_copy_constraints(self, trial_point:list):
        self.model.remove(self.copy_constraints)
        self.update_model()
        self.add_copy_constraints(trial_point)

    
    def update_cut_constraints(self, cut_intercepts:list, cut_gradients:list):
        self.model.remove(self.cut_constraints)
        self.update_model()
        self.add_cut_constraints(cut_intercepts, cut_gradients)