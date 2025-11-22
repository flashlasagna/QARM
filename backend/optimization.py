"""
optimization.py — Portfolio optimization and Solvency II frontier solver.
Fixed version with proper weight/amount handling.
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
    Objective: Maximize E[return on BOF] - γ * SCR_market

    x = [w1, w2, ..., w6, s_int, s_eq, s_prop, s_spread]
    where w are WEIGHTS (sum to 1), s are SCR slack variables
    """
    w = x[:6]          # weights (dimensionless, sum to 1)
    s = x[6:]          # SCR slacks [s_int, s_eq, s_prop, s_spread]

    # Convert weights → amounts (€M) for SCR calculations
    A = w * T

    # Portfolio return (as a RATE, not absolute €M)
    port_ret = w @ r

    # Compute interest-rate SCR using AMOUNTS
    scr_int_result = scr_interest_rate(
        A,
        asset_dur,
        liab_value,
        liab_duration,
        int_down_param,
        int_up_param
    )

    scr_int_value = scr_int_result["SCR_interest"]
    direction = scr_int_result.get("direction", "down").lower()

    # Choose correlation matrix based on interest rate direction
    R = corr_upward.values if direction == "up" else corr_downward.values

    # Market SCR approximation: sqrt(s^T R s)
    scr_market = np.sqrt(s @ R @ s)

    # Objective: maximize return - gamma * SCR => minimize negative
    return -(port_ret - gamma * scr_market / T)


# =============================
#  CONSTRAINT DEFINITIONS
# =============================

def define_constraints(T, asset_dur, liab_value, liab_duration, allocation_limits, params, corr_downward, corr_upward):
    """
    Builds nonlinear & linear constraints used in the optimizer.
    All constraints work in WEIGHT space, then convert to amounts internally.
    """
    int_up_param   = params["interest_up"]
    int_down_param = params["interest_down"]
    equity_1_param = params["equity_type1"]
    equity_2_param = params["equity_type2"]
    prop_params    = params["property"]
    spread_params  = params["spread"]
    rho = params["rho"]

    lib_delta = liab_duration * liab_value

    # === Nonlinear constraints ===

    def int_up_con(x):
        w = x[:6]
        s_int = x[6]
        asset_dur_contrib = T * (asset_dur @ w)
        return s_int - int_up_param * (asset_dur_contrib - lib_delta)

    def int_down_con(x):
        w = x[:6]
        s_int = x[6]
        asset_dur_contrib = T * (asset_dur @ w)
        return s_int - int_down_param * (lib_delta - asset_dur_contrib)

    def eq_con(x):
        w = x[:6]
        s_eq = x[7]
        w_eq1, w_eq2 = w[2], w[3]
        A_eq1, A_eq2 = w_eq1 * T, w_eq2 * T

        SCR_eq1 = equity_1_param * A_eq1
        SCR_eq2 = equity_2_param * A_eq2
        SCR_eq_total = np.sqrt(SCR_eq1**2 + 2*rho*SCR_eq1*SCR_eq2 + SCR_eq2**2)

        return s_eq - SCR_eq_total

    def prop_con(x):
        w = x[:6]
        s_prop = x[8]
        w_prop = w[4]
        return s_prop - prop_params * T * w_prop

    def spread_con(x):
        w = x[:6]
        s_spread = x[9]
        w_corp = w[1]
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

    # 2. Allocation limits (on weights)
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

    def min_solvency_con(x):
        """
        Constraint: Solvency Ratio >= 1.0 (or BOF - SCR >= 0)

        This constraint requires the calculation of SCR_market_final.
        It MUST be non-linear as SCR is calculated using complex rules.
        """
        w = x[:6]  # Portfolio weights
        s = x[6:]  # SCR slacks

        # 1. Compute SCR_market (using the approximation from the objective function)
        # Note: We must use the logic from objective() or call aggregate_market_scr
        # Here we re-use the slack variables logic for efficiency:
        # s = [s_int, s_eq, s_prop, s_spread]

        # Determine direction for correlation matrix (based on s_int relative to shocks)
        # This is tricky inside the solver. A robust approximation is to check the slacks directly
        # or conservatively use the WORST case correlation.
        # However, since we already optimize slacks, we can use the slack vector directly.

        # We need to fetch correlation matrices from params (passed into define_constraints)
        # Note: You need to update the function signature of define_constraints to accept
        # corr_downward and corr_upward

        # SIMPLIFIED APPROACH for Constraint:
        # Instead of full re-calculation, we use the slacks which represent the SCR components
        # The optimizer ensures slacks >= actual risk (via other constraints)
        # So we can just check: BOF >= sqrt(s' R s)

        # Since 'direction' flips the R matrix, we can just constrain against BOTH matrices
        # to be safe, or implement the switching logic if possible.
        # Safest and easiest: Constrain against the MAX of both directions.

        scr_down = np.sqrt(s @ corr_downward.values @ s)
        scr_up = np.sqrt(s @ corr_upward.values @ s)
        scr_est = np.maximum(scr_down, scr_up)  # Conservative estimate

        BOF = T - liab_value

        return BOF - scr_est

    min_solvency_constraint = NonlinearConstraint(min_solvency_con, 0, np.inf)

    return [
        int_up_constraint,
        int_down_constraint,
        eq_constraint,
        prop_constraint,
        spread_constraint,
        min_solvency_constraint,
        budget_constraint,
        alloc_constraint,
    ]



# =============================
#  FRONTIER SOLVER
# =============================

def solve_frontier_combined(initial_asset, liab_value, liab_duration,
                            corr_downward, corr_upward, allocation_limits, params):
    """
    Solve the Solvency II efficient frontier.

    User inputs AMOUNTS (€M), we convert to WEIGHTS for optimization,
    then convert back to AMOUNTS for results.
    """
    # Total assets (€M)
    T = initial_asset["asset_val"].sum()

    # Asset characteristics
    asset_dur = np.array(initial_asset["asset_dur"], dtype=float)
    r = initial_asset["asset_ret"].values

    # Build constraints
    constraints = define_constraints(
        T, asset_dur, liab_value, liab_duration, allocation_limits, params,
        corr_downward, corr_upward
    )

    # Initial guess: equal weights
    w0 = np.ones(6) / 6
    s0 = np.ones(4) * 10.0
    x0 = np.concatenate([w0, s0])

    # Bounds: weights in [0,1], SCRs >= 0
    bounds = Bounds(
        lb=np.concatenate([np.zeros(6), np.zeros(4)]),
        ub=np.concatenate([np.ones(6), np.full(4, np.inf)])
    )

    # Penalty parameter sweep (log-spaced)
    gammas = np.logspace(-3, 2, 150)

    results = []

    for gamma in gammas:
        res = minimize(
            objective, x0,
            args=(r, gamma, T,
                  asset_dur,
                  liab_value, liab_duration,
                  params["interest_down"], params["interest_up"],
                  corr_downward, corr_upward),
            constraints=constraints,
            method="SLSQP",
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9}
        )

        if not res.success:
            continue

        # Extract optimal weights and SCRs
        w_opt = res.x[:6]
        s_opt = res.x[6:]

        # Convert weights → amounts for reporting
        A_opt = w_opt * T

        # Portfolio return (rate)
        port_return = w_opt @ r

        # Recompute exact SCRs
        scr_interest = scr_interest_rate(
            A_opt, asset_dur, liab_value, liab_duration,
            params["interest_down"], params["interest_up"]
        )

        A_eq1, A_eq2 = A_opt[2], A_opt[3]
        A_corp, A_prop = A_opt[1], A_opt[4]

        scr_equity_result = scr_eq(
            A_eq1, A_eq2,
            params["equity_type1"], params["equity_type2"],
            params["rho"]
        )
        scr_equity = scr_equity_result["SCR_eq_total"]

        scr_property = A_prop * params["property"]
        scr_spread   = A_corp * params["spread"]

        # Aggregate market SCR
        scr_market_result = aggregate_market_scr(
            scr_interest,
            scr_equity_result,
            scr_property,
            scr_spread,
            corr_downward,
            corr_upward
        )
        scr_total = scr_market_result["SCR_market_final"]

        # Solvency ratio = BOF / SCR
        BOF = T - liab_value
        solvency_ratio = BOF / scr_total if scr_total > 0 else np.inf

        results.append({
            "gamma": gamma,
            "return": port_return,
            "w_opt": w_opt,
            "A_opt": A_opt,
            "SCR_market": scr_total,
            "solvency": solvency_ratio,
            "objective": -res.fun,
            "BOF": BOF
        })

        # Warm start for next iteration
        x0 = res.x

    return pd.DataFrame(results)