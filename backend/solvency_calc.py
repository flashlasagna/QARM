"""
solvency_calc.py — Solvency II SCR calculations
Fixed version with proper dimension handling
"""

import numpy as np
import pandas as pd
import math


def scr_interest_rate(A, D_A, L, D_L, delta_y_down, delta_y_up):
    """
    Calculate SCR for interest rate risk.
    """
    A = np.asarray(A, dtype=float)
    D_A = np.asarray(D_A, dtype=float)

    # Asset sensitivity (€M * years)
    asset_sens = np.sum(D_A * A)

    # Liability sensitivity (€M * years) - L and D_L are SCALARS
    liab_sens = D_L * L

    # Losses from up/down shocks
    loss_down = delta_y_down * (liab_sens - asset_sens)
    loss_up = delta_y_up * (asset_sens - liab_sens)

    # SCR is the maximum loss
    scr_interest = max(loss_down, loss_up, 0)
    direction = "down" if loss_down >= loss_up else "up"

    return {
        "SCR_interest": scr_interest,
        "direction": direction,
        "loss_down": loss_down,
        "loss_up": loss_up
    }


def scr_eq(A_eq1, A_eq2, shock_eq1, shock_eq2, rho):
    """
    Calculate SCR for equity risk (aggregated Type 1 and Type 2).
    """
    SCR1 = shock_eq1 * A_eq1
    SCR2 = shock_eq2 * A_eq2

    # Aggregated equity SCR using correlation
    SCR_eq = math.sqrt(SCR1 ** 2 + 2 * rho * SCR1 * SCR2 + SCR2 ** 2)

    return {
        "SCR_eq_type1": SCR1,
        "SCR_eq_type2": SCR2,
        "SCR_eq_total": SCR_eq
    }


def scr_prop(prop, shock):
    """Calculate SCR for property risk."""
    return prop * shock


def scr_sprd(corp_bond, shock):
    """Calculate SCR for spread risk."""
    return corp_bond * shock


def aggregate_market_scr(scr_interest, scr_equity, scr_property, scr_spread,
                         corr_downward, corr_upward):
    """
    Aggregate all market risk SCRs using correlation matrix.
    """
    # Extract direction from interest rate result
    direction = scr_interest.get("direction", "down").lower()

    # Build SCR vector [interest, equity, property, spread]
    scr_int = scr_interest['SCR_interest']
    scr_eq = scr_equity['SCR_eq_total'] if isinstance(scr_equity, dict) else scr_equity

    vec = np.array([scr_int, scr_eq, scr_property, scr_spread])

    # Choose correlation matrix based on interest rate direction
    if direction.startswith("down"):
        corr = np.asarray(corr_downward)
    else:
        corr = np.asarray(corr_upward)

    # Aggregate SCR: sqrt(v^T * R * v)
    scr_total = np.sqrt(vec @ corr @ vec)

    # Diversification benefit
    scr_sum = np.sum(vec)
    diversification = scr_sum - scr_total
    diversification_pct = (diversification / scr_sum * 100) if scr_sum > 0 else 0

    # Summary table
    summary = pd.DataFrame({
        "risk": ["interest", "equity", "property", "spread", "total", "diversification"],
        "SCR": np.append(vec, [scr_total, diversification])
    }).set_index("risk")

    return {
        "summary_table": summary,
        "chosen_panel": "Interest_DOWN" if direction == "down" else "Interest_UP",
        "SCR_market_final": scr_total,
        "Diversification_pct": diversification_pct,
        "diversification_amount": diversification
    }


def marginal_scr(v: np.ndarray, direction: str,
                 corr_downward: np.ndarray, corr_upward: np.ndarray) -> pd.DataFrame:
    """
    Compute marginal and allocated SCRs (Euler allocation).
    """
    v = np.asarray(v, dtype=float)

    # Choose correlation matrix
    if direction.upper().startswith("DOWN") or direction.lower().startswith("down"):
        corr = np.asarray(corr_downward, dtype=float)
    else:
        corr = np.asarray(corr_upward, dtype=float)

    # Validate dimensions
    if corr.shape[0] != corr.shape[1] or corr.shape[0] != len(v):
        raise ValueError(
            f"Correlation matrix shape {corr.shape} must match SCR vector length {len(v)}"
        )

    # Total SCR
    scr_total = np.sqrt(v @ corr @ v)

    # Marginal SCR factors
    marginals = (corr @ v) / scr_total if scr_total > 0 else np.zeros_like(v)

    # Allocated SCR (Euler allocation)
    allocated = v * marginals

    # Percentage contribution
    share_pct = (allocated / scr_total * 100) if scr_total > 0 else np.zeros_like(v)

    # Build output table
    df = pd.DataFrame({
        "risk": ["interest", "equity", "property", "spread"],
        "SCR": v,
        "marginal_factor": marginals,
        "allocated_SCR": allocated,
        "share_%": share_pct
    })

    return df


def allocate_marginal_scr(
        marginal_per_risk: pd.DataFrame,
        direction: str,
        initial_asset: pd.DataFrame,
        params: dict
) -> pd.DataFrame:
    """
    Break down marginal SCR by individual asset class.
    """
    m = marginal_per_risk.set_index("risk")["marginal_factor"]

    # Extract parameters
    int_down_param = params["interest_down"]
    int_up_param = params["interest_up"]
    spread_param = params["spread"]
    equity_1_param = params["equity_type1"]
    equity_2_param = params["equity_type2"]
    rho = params["rho"]
    prop_param = params["property"]

    # Choose interest shock based on direction
    int_param = int_down_param if "DOWN" in direction.upper() else int_up_param

    # Sign for interest rate sensitivity
    ir_sign = -1 if "DOWN" in direction.upper() else 1

    # Government bonds
    mSCR_gov = ir_sign * m["interest"] * int_param * initial_asset.loc["gov_bond", "asset_dur"]

    # Corporate bonds (interest + spread)
    mSCR_corp = (
            ir_sign * m["interest"] * int_param * initial_asset.loc["corp_bond", "asset_dur"] +
            m["spread"] * spread_param
    )

    # T-bills
    mSCR_tbills = ir_sign * m["interest"] * int_param * initial_asset.loc["t_bills", "asset_dur"]

    # Equity marginal SCRs
    A_eq1 = initial_asset.loc["equity_1", "asset_val"]
    A_eq2 = initial_asset.loc["equity_2", "asset_val"]

    scr_eq1 = equity_1_param * A_eq1
    scr_eq2 = equity_2_param * A_eq2
    scr_eq_total = np.sqrt(scr_eq1 ** 2 + 2 * rho * scr_eq1 * scr_eq2 + scr_eq2 ** 2)

    if scr_eq_total > 0:
        w1 = (scr_eq1 + rho * scr_eq2) / scr_eq_total
        w2 = (scr_eq2 + rho * scr_eq1) / scr_eq_total
    else:
        w1 = w2 = 0.5

    mSCR_eq1 = m["equity"] * w1 * equity_1_param
    mSCR_eq2 = m["equity"] * w2 * equity_2_param

    # Property
    mSCR_prop = m["property"] * prop_param

    # Build results table
    df_assets = pd.DataFrame({
        "asset": ["gov_bond", "corp_bond", "equity_1", "equity_2", "property", "t_bills"],
        "mSCR": [mSCR_gov, mSCR_corp, mSCR_eq1, mSCR_eq2, mSCR_prop, mSCR_tbills]
    })

    # Add E[r]/mSCR ratio (capital efficiency metric)
    asset_returns = initial_asset["asset_ret"]
    df_assets["expected_return"] = [
        asset_returns.loc["gov_bond"],
        asset_returns.loc["corp_bond"],
        asset_returns.loc["equity_1"],
        asset_returns.loc["equity_2"],
        asset_returns.loc["property"],
        asset_returns.loc["t_bills"]
    ]

    # Capital efficiency
    df_assets["return_per_mSCR"] = df_assets["expected_return"] / df_assets["mSCR"].abs()
    df_assets.loc[df_assets["mSCR"].abs() < 1e-6, "return_per_mSCR"] = np.inf

    return df_assets