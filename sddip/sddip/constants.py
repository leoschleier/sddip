class ResultKeys:
    # Result keys
    v_key = "v"
    x_key = "x"
    y_key = "y"
    x_bs_key = "x_bs"
    soc_key = "soc"
    primal_solution_keys = [x_key, y_key, x_bs_key, soc_key]

    dv_key = "dual_value"
    dm_key = "dual_multiplier"
    dual_solution_keys = [dv_key, dm_key]

    ci_key = "intercepts"
    cg_key = "gradients"
    y_bm_key = "y_multipliers"
    soc_bm_key = "soc_multipliers"
    cut_coefficient_keys = [ci_key, cg_key, y_bm_key, soc_bm_key]

    lb_key = "lb"
    ub_l_key = "ub_l"
    ub_r_key = "ub_r"
    bound_keys = [lb_key, ub_l_key, ub_r_key]

    bc_intercept_key = "intercept"
    bc_gradient_key = "gradient"
    bc_trial_point_key = "trial_point"
    benders_cut_keys = [bc_intercept_key, bc_gradient_key, bc_trial_point_key]

    ds_iterations = "iterations"
    ds_solver_time = "solver_time"
    dual_solver_keys = [ds_iterations, ds_solver_time]
