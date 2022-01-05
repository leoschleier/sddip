from abc import ABC, abstractmethod
import gurobipy as gp
import numpy as np
from scipy import linalg

class ModelBuilder(ABC):

    def __init__(self, n_buses:int, n_lines:int, n_generators:int, generators_at_bus:list) -> None:
        self.n_buses = n_buses
        self.n_lines = n_lines
        self.n_generators = n_generators
        self.generators_at_bus = generators_at_bus
        
        self.model = gp.Model("MILP: Unit commitment")

        # Commitment decision
        self.x = []
        # Dispatch decision
        self.y = []
        # Copy variables
        self.z_x = []
        self.z_y = []
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
    

    def initialize_variables(self):
        #TODO Default lower bounds
        for g in range(self.n_generators):
            self.x.append(self.model.addVar(vtype = gp.GRB.BINARY, name = "x_%i"%(g+1)))
            self.y.append(self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, name = "y_%i"%(g+1)))
            self.s_up.append(self.model.addVar(vtype = gp.GRB.BINARY, name = "s_up_%i"%(g+1)))
            self.s_down.append(self.model.addVar(vtype = gp.GRB.BINARY, name = "s_down_%i"%(g+1))) 
        self.theta = self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = "theta")
        self.ys_p = self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, name = "ys_p")
        self.ys_n = self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, name = "ys_n")
        self.model.update()

    @abstractmethod
    def initialize_copy_variables(self):
        pass


    def add_objective(self, coefficients:list):
        coefficients = coefficients + [1]
        variables = self.y + self.s_up + self.s_down + [self.ys_p, self.ys_n, self.theta]
        self.objective_terms = gp.LinExpr(coefficients, variables)
        self.model.setObjective(self.objective_terms)
        self.update_model()


    def add_balance_constraints(self, total_demand:float):
        self.balance_constraints = self.model.addConstr(
            gp.quicksum(self.y) + self.ys_p - self.ys_n == total_demand, "balance")
        self.update_model()
    

    def add_generator_constraints(self, min_generation:list, max_generation:list):
        self.model.addConstrs((self.y[g] >= min_generation[g]*self.x[g] for g in range(self.n_generators)), 
            "min-generation")
        self.model.addConstrs((self.y[g] <= max_generation[g]*self.x[g] for g in range(self.n_generators)), 
            "max-generation")
        self.update_model()


    def add_power_flow_constraints(self, ptdf, max_line_capacities:list, demand:list):
        line_flows = [gp.quicksum(ptdf[l,b] * (gp.quicksum(self.y[g] for g in self.generators_at_bus[b]) - demand[b])
            for b in range(self.n_buses)) for l in range(self.n_lines)]
        self.model.addConstrs((line_flows[l] <= max_line_capacities[l] for l in range(self.n_lines)), "power-flow(1)")
        self.model.addConstrs((-line_flows[l] <= max_line_capacities[l] for l in range(self.n_lines)), "power-flow(2)")
        self.update_model()


    def add_startup_shutdown_constraints(self):
        self.model.addConstrs((self.x[g] - self.z_x[g] <= self.s_up[g]  
            for g in range(self.n_generators)), "up-down(1)")
        self.model.addConstrs((self.x[g] - self.z_x[g] <= self.s_up[g] - self.s_down[g]  
            for g in range(self.n_generators)), "up-down(2)")
        self.update_model()


    def add_ramp_rate_constraints(self, max_rate_up:list, max_rate_down:list):
        self.model.addConstrs((self.y[g] - self.z_y[g] <= max_rate_up[g] for g in range(self.n_generators)), 
        "rate-up")
        self.model.addConstrs((self.z_y[g] - self.y[g] <= max_rate_down[g] for g in range(self.n_generators)), 
        "rate-down")


    @abstractmethod
    def add_copy_constraints(self, x_trial_point:list, y_trial_point):
        pass

    def add_cut_lower_bound(self, lower_bound:float):
        self.cut_lower_bound = self.model.addConstr((self.theta >= lower_bound), "cut-lb")


    def add_cut_constraints(self, cut_intercepts: list, cut_gradients: list, binary_multipliers:list):
        for intercept, gradient, multipliers in zip(cut_intercepts, cut_gradients, binary_multipliers):
            self.add_cut(intercept, gradient, multipliers)
        self.update_model()
        
    
    def add_cut(self, cut_intercept:float, cut_gradient:list, y_binary_multipliers:np.array):
        x_binary_multipliers = linalg.block_diag(*[1]*len(self.x))

        binary_multipliers = linalg.block_diag(x_binary_multipliers, y_binary_multipliers)
        
        n_var_approximations, n_binaries = y_binary_multipliers.shape

        ny = self.model.addVars(n_binaries, vtype = gp.GRB.CONTINUOUS, lb = 0, name = "ny")
        my = self.model.addVars(n_binaries, vtype = gp.GRB.CONTINUOUS, lb = 0, name = "my")
        eta = self.model.addVars(n_var_approximations, vtype = gp.GRB.CONTINUOUS, name = "eta")
        lmda = self.model.addVars(n_binaries, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = "lambda")

        state_vars = self.x + self.y

        w = self.model.addVars(n_binaries, vtype = gp.GRB.BINARY, name = "w")
        u = self.model.addVars(n_binaries, vtype = gp.GRB.BINARY, name = "u")

        # TODO Define Big-Ms
        m1 = 1
        m2 = [100000000]*n_binaries
        m3 = -1
        m4 = [100000000]*n_binaries       

        # Cut constraint
        self.model.addConstr((self.theta >= cut_intercept + lmda.prod(cut_gradient)), "cut")

        # KKT conditions
        self.model.addConstrs((0 == -cut_gradient[j] - ny[j] + my[j] + gp.quicksum(binary_multipliers[i,j]*eta[i] 
            for i in range(n_var_approximations)) 
            for j in range(n_binaries)), "KKT(1)")

        self.model.addConstrs((0 == gp.quicksum(binary_multipliers[i,j]*lmda[j] 
            for j in range(n_binaries)) - state_vars[i] for i in range(n_var_approximations)), "KKT(2)")

        self.model.addConstrs((lmda[i] <= m1*w[i] for i in range(n_binaries)), "KKT(3)")

        self.model.addConstrs((ny[i] <= m2[i]*(1-w[i]) for i in range(n_binaries)),"KKT(4)")

        self.model.addConstrs((lmda[i]-1 >= m3*u[i] for i in range(n_binaries)), "KKT(5)")

        self.model.addConstrs((my[i] <= m4[i]*(1-u[i]) for i in range(n_binaries)), "KKT(6)")

    
    def remove(self, gurobi_objects):
        # Remove if gurobi_objects not None or not empty
        if gurobi_objects: 
            self.model.remove(gurobi_objects)
            self.update_model


    def update_model(self):
        self.model.update()

    
    def update_balance_constraints(self, demand:float):
        self.remove(self.balance_constraints)
        self.add_balance_constraints(demand)

    @abstractmethod
    def update_copy_constraints(self, x_trial_point:list, y_trial_point:list):
        pass

    
    def update_cut_lower_bound(self, cut_lower_bound:float):
        self.remove(self.cut_lower_bound)
        self.add_cut_lower_bound(cut_lower_bound)


    def update_cut_constraints(self, cut_intercepts:list, cut_gradients:list):
        self.remove(self.cut_constraints)
        self.add_cut_constraints(cut_intercepts, cut_gradients)

    
    def disable_output(self):
        self.model.setParam("OutputFlag", 0)

    def enable_output(self):
        self.model.setParam("OutputFlag", 1)


class ForwardModelBuilder(ModelBuilder):

    def __init__(self, n_buses:int, n_lines:int, n_generators:int, generators_at_bus:list) -> None:
        super().__init__(n_buses, n_lines, n_generators, generators_at_bus)
        self.initialize_copy_variables()

    def initialize_copy_variables(self):
        for g in range(self.n_generators):
            self.z_x.append(self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = "z_x_%i"%(g+1)))
            self.z_y.append(self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = "z_y_%i"%(g+1)))

    def add_copy_constraints(self, x_trial_point:list, y_trial_point:list):
        self.copy_constraints_y = self.model.addConstrs((self.z_x[g] == x_trial_point[g] 
            for g in range(self.n_generators)), "copy-x")
        self.copy_constraints_x = self.model.addConstrs((self.z_y[g] == y_trial_point[g] 
            for g in range(self.n_generators)), "copy-y")
        self.update_model()
    
    def update_copy_constraints(self, x_trial_point:list, y_trial_point:list):
        self.remove(self.copy_constraints_x)
        self.remove(self.copy_constraints_y)
        self.add_copy_constraints(x_trial_point, y_trial_point)

class BackwardModelBuilder(ModelBuilder):

    def __init__(self, n_buses:int, n_lines:int, n_generators:int, generators_at_bus:list) -> None:
        super().__init__(n_buses, n_lines, n_generators, generators_at_bus)
        
        self.n_x_trial_binaries = None
        self.n_y_trial_binaries = None
        
        self.relaxed_terms = []


        # Copy variable for binary variables
        self.x_bin_copy_vars = []
        self.y_bin_copy_vars = []

        # Copy constraints
        self.copy_constraints_x = None
        self.copy_constraints_y = None

        self.initialize_copy_variables()

    def initialize_copy_variables(self):
        for g in range(self.n_generators):
            self.z_x.append(self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = "z_x_%i"%(g+1)))
            self.z_y.append(self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = "z_y_%i"%(g+1)))

    def add_relaxation(self, x_binary_trial_point:list, y_binary_trial_point:list):
        self.bin_copy_vars = []
        self.n_x_trial_binaries = len(x_binary_trial_point)
        self.n_y_trial_binaries = len(y_binary_trial_point)
        
        for j in range(self.n_x_trial_binaries):
            self.x_bin_copy_vars.append(self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, 
                name = "x_bin_copy_var_%i"%(j+1)))

        for j in range(self.n_y_trial_binaries):
            self.y_bin_copy_vars.append(self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, 
                name = "y_bin_copy_var_%i"%(j+1)))
        
        self.relax(x_binary_trial_point, y_binary_trial_point)


    def relax(self, x_binary_trial_point:list, y_binary_trial_point:list):      
        self.check_bin_copy_vars_not_empty()

        self.relaxed_terms += [x_binary_trial_point[j] - self.x_bin_copy_vars[j] 
            for j in range(self.n_x_trial_binaries)]

        self.relaxed_terms += [y_binary_trial_point[j] - self.y_bin_copy_vars[j] 
            for j in range(self.n_y_trial_binaries)]

    
    def add_copy_constraints(self, y_binary_trial_multipliers:np.array):      
        self.check_bin_copy_vars_not_empty()

        n_y_var_approximations, n_y_binaries = y_binary_trial_multipliers.shape

        self.copy_constraints_y = self.model.addConstrs((
            self.z_y[i] == gp.quicksum(y_binary_trial_multipliers[i,j]*self.y_bin_copy_vars[j] 
                for j in range(n_y_binaries)) 
                for i in range(n_y_var_approximations)), "copy-y")

        self.copy_constraints_x = self.model.addConstrs((
            self.z_x[i] == self.x_bin_copy_vars[i] for i in range(self.n_x_trial_binaries)), "copy-x")
        
        self.update_model()

    
    def update_copy_constraints(self, y_binary_trial_multipliers:np.array):
        self.remove(self.copy_constraints_x)
        self.remove(self.copy_constraints_y)
        self.add_copy_constraints(y_binary_trial_multipliers)

    
    def check_bin_copy_vars_not_empty(self):
        if not (self.x_bin_copy_vars and self.y_bin_copy_vars):
            raise ValueError("Copy variable does not exist. Call add_relaxation first.")
