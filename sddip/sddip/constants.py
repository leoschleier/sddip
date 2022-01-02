

class ResultKeys:
    # Result keys
    x_key = "x"
    y_key = "y"
    z_x_key = "zx"
    z_y_key = "zy"
    primal_solution_keys = [x_key, y_key, z_x_key, z_y_key]

    dv_key = "dual_value"
    dm_key = "dual_multiplier"
    dual_solution_keys = [dv_key, dm_key]

    ci_key = "intercepts"
    cg_key = "gradients"
    bm_key = "multipliers"
    cut_coefficient_keys = [ci_key, cg_key, bm_key]