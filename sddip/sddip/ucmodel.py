from abc import ABC, abstractmethod
import gurobipy as gp
import numpy as np

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

        # Objective
        self.objective_terms = None
        # Balance constraints
        self.balance_constraints = None
        # Copy constraints
        self.copy_constraints = None
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
            self.z.append(self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = "z_%i"%(g+1)))
            self.s_up.append(self.model.addVar(vtype = gp.GRB.BINARY, name = "s_up_%i"%(g+1)))
            self.s_down.append(self.model.addVar(vtype = gp.GRB.BINARY, name = "s_down_%i"%(g+1))) 
        self.theta = self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = "theta")
        self.ys_p = self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, name = "ys_p")
        self.ys_n = self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, name = "ys_n")
        self.model.update()


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
        self.model.addConstrs((self.x[g] - self.z[g] <= self.s_up[g]  
            for g in range(self.n_generators)), "up-down(1)")
        self.model.addConstrs((self.x[g] - self.z[g] <= self.s_up[g] - self.s_down[g]  
            for g in range(self.n_generators)), "up-down(2)")
        self.update_model()


    def add_ramp_rate_constraints(self, max_rate_up:list, max_rate_down:list):
        self.model.addConstrs((self.y[g] - self.z[g] <= max_rate_up[g] for g in range(self.n_generators)), 
        "rate-up")
        self.model.addConstrs((self.z[g] - self.y[g] <= max_rate_down[g] for g in range(self.n_generators)), 
        "rate-down")


    @abstractmethod
    def add_copy_constraints(self, trial_point:list):
        pass

    def add_cut_lower_bound(self, lower_bound:float):
        self.cut_lower_bound = self.model.addConstr((self.theta >= lower_bound), "cut-lb")


    def add_cut_constraints(self, cut_intercepts: list, cut_gradients: list, binary_multipliers:list):
        for intercept, gradient, multipliers in zip(cut_intercepts, cut_gradients, binary_multipliers):
            self.add_cut(intercept, gradient, multipliers)
        self.update_model()
        
    
    def add_cut(self, cut_intercept:float, cut_gradient:list, binary_multipliers:np.array):
        n_var_approximations, n_binaries = binary_multipliers.shape
        ny = self.model.addVars(n_binaries, vtype = gp.GRB.CONTINUOUS, lb = 0)
        my = self.model.addVars(n_binaries, vtype = gp.GRB.CONTINUOUS, lb = 0)
        eta = self.model.addVars(n_var_approximations, vtype = gp.GRB.CONTINUOUS)
        lmda = self.model.addVars(n_binaries, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1)

        non_binary_state_vars = self.y

        w = self.model.addVars(n_binaries, vtype = gp.GRB.BINARY)
        u = self.model.addVars(n_binaries, vtype = gp.GRB.BINARY)

        # TODO Define Big-Ms
        m1 = 1
        m2 = [10000]*n_binaries
        m3 = -1
        m4 = [10000]*n_binaries       

        # Cut constraint
        self.model.addConstr((self.theta >= cut_intercept + lmda.prod(cut_gradient)), "cut")

        # KKT conditions
        self.model.addConstrs(0 == -cut_gradient[j] - ny[j] + my[j] + gp.quicksum(binary_multipliers[i,j]*eta[i] 
            for i in range(n_var_approximations)) 
            for j in range(n_binaries))

        self.model.addConstrs( 0 == gp.quicksum(binary_multipliers[i,j]*lmda[j] 
            for j in range(n_binaries)) - non_binary_state_vars[i] for i in range(n_var_approximations))

        self.model.addConstrs(lmda[i] <= m1*w[i] for i in range(n_binaries))

        self.model.addConstrs(ny[i] <= m2[i]*(1-w[i]) for i in range(n_binaries))

        self.model.addConstrs(lmda[i]-1 >= m3*u[i] for i in range(n_binaries))

        self.model.addConstrs(my[i] <= m4[i]*(1-u[i]) for i in range(n_binaries))

    
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
    def update_copy_constraints(self, trial_point:list):
        pass

    
    def update_cut_lower_bound(self, cut_lower_bound:float):
        self.remove(self.cut_lower_bound)
        self.add_cut_lower_bound(cut_lower_bound)


    def update_cut_constraints(self, cut_intercepts:list, cut_gradients:list):
        self.remove(self.cut_constraints)
        self.add_cut_constraints(cut_intercepts, cut_gradients)

    
    def suppress_output(self):
        self.model.setParam("OutputFlag", 0)


class ForwardModelBuilder(ModelBuilder):

    def __init__(self, n_buses:int, n_lines:int, n_generators:int, generators_at_bus:list) -> None:
        super().__init__(n_buses, n_lines, n_generators, generators_at_bus)

    def add_copy_constraints(self, trial_point:list):
        self.copy_constraints = self.model.addConstrs((self.z[g] == trial_point[g] 
            for g in range(self.n_generators)), "copy")
        self.update_model()
    
    def update_copy_constraints(self, trial_point:list):
        self.remove(self.copy_constraints)
        self.add_copy_constraints(trial_point)

class BackwardModelBuilder(ModelBuilder):

    def __init__(self, n_buses:int, n_lines:int, n_generators:int, generators_at_bus:list) -> None:
        super().__init__(n_buses, n_lines, n_generators, generators_at_bus)
        
        self.n_trial_binaries = None        
        
        self.relaxed_terms = []


        # Copy variable for binary variables
        self.bin_copy_vars = []

        # Binary approximation constraints
        self.binary_approximation_constraints = None


    def add_relaxation(self, binary_trial_point:list):
        self.bin_copy_vars = []
        self.n_trial_binaries = len(binary_trial_point)
        
        for j in range(self.n_trial_binaries):
            self.bin_copy_vars.append(self.model.addVar(vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, 
                name = "bin_copy_vars_%i"%(j+1)))
        
        self.relax(binary_trial_point)


    def relax(self, binary_trial_point:list):      
        self.check_bin_copy_vars_not_empty()
        self.relaxed_terms = [binary_trial_point[j] - self.bin_copy_vars[j] for j in range(self.n_trial_binaries)]

    
    def add_copy_constraints(self, binary_trial_multipliers:np.array):      
        n_var_approximations, n_binaries = binary_trial_multipliers.shape
        
        self.check_bin_copy_vars_not_empty()

        self.copy_constraints = self.model.addConstrs((
            self.z[i] == gp.quicksum(binary_trial_multipliers[i,j]*self.bin_copy_vars[j] for j in range(n_binaries)) 
            for i in range(n_var_approximations)), 
            "copy")
        
        self.update_model()

    
    def update_copy_constraints(self, binary_trial_multipliers:np.array):
        self.remove(self.copy_constraints)
        self.add_copy_constraints(binary_trial_multipliers)

    
    def check_bin_copy_vars_not_empty(self):
        if not self.bin_copy_vars:
            raise ValueError("Copy variable does not exist. Call add_relaxation first.")
