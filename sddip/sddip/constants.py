class ResultKeys:
    # Result keys
    x_key = "x"
    y_key = "y"
    x_bs_key = "x_bs"
    primal_solution_keys = [x_key, y_key, x_bs_key]

    dv_key = "dual_value"
    dm_key = "dual_multiplier"
    dual_solution_keys = [dv_key, dm_key]

    ci_key = "intercepts"
    cg_key = "gradients"
    bm_key = "multipliers"
    cut_coefficient_keys = [ci_key, cg_key, bm_key]
