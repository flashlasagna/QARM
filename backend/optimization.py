"""
optimization.py — Portfolio optimization and Solvency II frontier solver.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds
from backend.solvency_calc import scr_interest_rate, scr_eq, aggregate_market_scr


# =============================
#  OBJECTIVE FUNCTION
# =============================

def objective(x, r, gamma, T, asset_dur, liab_value, liab_duration,
              int_down_param, int_up_param, corr_downward, corr_upward):
    """
    Objective: maximize (expected return - γ * SCR_market)
    using dynamic interest-rate direction.
    """
    w = x[:6]    # asset weights
    s = x[6:]    # risk module sensitivities

    # Portfolio return
    port_ret = T * (w @ r)

    # --- Compute interest-rate SCR and detect direction
    scr_int = scr_interest_rate(
        w,
        asset_dur,
        liab_value,
        liab_duration,
        int_down_param,
        int_up_param
    )

    # Determine correlation matrix based on shock direction
    direction = scr_int.get("direction", "down")
    if direction.lower() == "up":
        R = corr_upward.values
    else:
        R = corr_downward.values

    # --- Approximate total market SCR
    scr_market = (s @ R @ s)

    # --- Objective: maximize (return - γ * SCR)
    return - (port_ret - gamma * scr_market)


# =============================
#  CONSTRAINT DEFINITIONS
# =============================

def define_constraints(T, asset_dur, lib_delta, allocation_limits, params):
    """
    Builds nonlinear & linear constraints used in the optimizer.
    """
    int_up_param   = params["interest_up"]
    int_down_param = params["interest_down"]
    equity_1_param = params["equity_type1"]
    equity_2_param = params["equity_type2"]
    prop_params    = params["property"]
    spread_params  = params["spread"]

    # === Nonlinear constraints ===

    def int_up_con(x):
        w = x[:6]; s_int = x[6]
        return s_int - int_up_param * (T * (asset_dur @ w) - lib_delta)

    def int_down_con(x):
        w = x[:6]; s_int = x[6]
        return s_int - int_down_param * (lib_delta - T * (asset_dur @ w))

    def eq_con(x):
        w = x[:6]; s_eq = x[7]
        _, _, w_eq1, w_eq2, _, _ = w
        return s_eq - T * (equity_1_param * w_eq1 + equity_2_param * w_eq2)

    def prop_con(x):
        w = x[:6]; s_prop = x[8]
        _, _, _, _, w_prop, _ = w
        return s_prop - prop_params * T * w_prop

    def spread_con(x):
        w = x[:6]; s_spread = x[9]
        _, w_corp, _, _, _, _ = w
        return s_spread - spread_params * T * w_corp

    int_up_constraint   = NonlinearConstraint(int_up_con, 0, np.inf)
    int_down_constraint = NonlinearConstraint(int_down_con, 0, np.inf)
    eq_constraint       = NonlinearConstraint(eq_con, 0, np.inf)
    prop_constraint     = NonlinearConstraint(prop_con, 0, np.inf)
    spread_constraint   = NonlinearConstraint(spread_con, 0, np.inf)

    # === Linear constraints ===
    # 1. Budget: sum of weights = 1
    A_budget = np.zeros((1, 10))
    A_budget[0, :6] = 1
    budget_constraint = LinearConstraint(A_budget, [1.0], [1.0])

    # 2. Allocation limits
    A_alloc = np.zeros((4, 10))
    A_alloc[0, 0] = 1              # gov
    A_alloc[1, [2, 3, 4]] = 1      # illiquid = eq1 + eq2 + prop
    A_alloc[2, 5] = 1              # t-bills
    A_alloc[3, 1] = 1              # corp

    alloc_constraint = LinearConstraint(
        A_alloc,
        lb=allocation_limits["min_weight"].values,
        ub=allocation_limits["max_weight"].values,
    )

    return [
        int_up_constraint,
        int_down_constraint,
        eq_constraint,
        prop_constraint,
        spread_constraint,
        budget_constraint,
        alloc_constraint,
    ]


# =============================
#  FRONTIER SOLVER
# =============================

def solve_frontier_combined(initial_asset, liab_value, liab_duration,
                            corr_downward, corr_upward, allocation_limits, params):
    """
    Solve the Solvency II efficient frontier for combined up/down interest rate shocks.

    Parameters
    ----------
    initial_asset : pd.DataFrame
        Asset data (val, dur, ret).
    liab_value : float
        Liabilities value.
    liab_duration : float
        Liabilities duration.
    corr_downward, corr_upward : np.ndarray
        Correlation matrices.
    allocation_limits : pd.DataFrame
        Min/max weight constraints.
    params : dict
        Model parameters from config (int shocks, equity shocks, etc.)

    Returns
    -------
    pd.DataFrame
        Optimization results with solvency ratios and expected returns.
    """
    T = initial_asset["asset_val"].sum()
    asset_dur = np.array(initial_asset["asset_dur"], dtype=float)
    lib_delta = liab_duration * liab_value

    constraints = define_constraints(T, asset_dur, lib_delta, allocation_limits, params)

    w0 = np.ones(6) / 6
    s0 = np.ones(4) * 0.1
    x0 = np.concatenate([w0, s0])
    r = initial_asset["asset_ret"].values

    bounds = Bounds(lb=np.zeros(10), ub=np.full(10, np.inf))

    gammas = np.logspace(-10, 3, 200)

    results = []

    for gamma in gammas:
        res = minimize(
            objective, x0,
            args=(r, gamma, T,
                initial_asset["asset_dur"].values,
                liab_value, liab_duration,
                params["interest_down"], params["interest_up"],
                corr_downward, corr_upward),
            constraints=constraints,
            method="SLSQP",
            bounds=bounds,
        )

        w_opt = res.x[:6]
        s_opt = res.x[6:]
        A_opt = w_opt * T
        port_return = w_opt @ r

        # Compute SCRs using solvency_model functions
        scr_interest = scr_interest_rate(
            A_opt, initial_asset["asset_dur"], liab_value, liab_duration,
            params["interest_down"], params["interest_up"]
        )

        A_eq1, A_eq2 = A_opt[2], A_opt[3]
        A_corp, A_prop = A_opt[1], A_opt[4]

        scr_equity   = scr_eq(A_eq1, A_eq2, params["equity_type1"], params["equity_type2"], params["rho"])
        scr_property = A_prop * params["property"]
        scr_spread   = A_corp * params["spread"]

        scr_total = aggregate_market_scr(
            scr_interest, scr_equity, scr_property, scr_spread, corr_downward, corr_upward
        )["SCR_market_final"]

        solvency_ratio = (T - liab_value) / scr_total

        results.append({
            "gamma": gamma,
            "return": port_return,
            "w_opt": w_opt,
            "SCR_market": scr_total,
            "solvency": solvency_ratio,
            "objective": -res.fun
        })

        x0 = res.x  # warm start for next iteration

    return pd.DataFrame(results)
