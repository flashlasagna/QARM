import numpy as np
import pandas as pd
import math

def scr_interest_rate(A, D_A, L, D_L, delta_y_down, delta_y_up):
    A, D_A, L, D_L = map(np.array, (A, D_A, L, D_L))
    asset_sens, liab_sens = np.sum(D_A * A), np.sum(D_L * L)

    loss_down = delta_y_down * (liab_sens - asset_sens)
    loss_up   = delta_y_up   * (asset_sens - liab_sens)

    scr_interest = max(loss_down, loss_up)
    return {"SCR_interest": scr_interest, "direction": "down" if loss_down >= loss_up else "up"}


def scr_eq(A_eq1, A_eq2, shock_eq1, shock_eq2, rho):
    SCR1, SCR2 = shock_eq1 * A_eq1, shock_eq2 * A_eq2
    SCR_eq = math.sqrt(SCR1**2 + 2 * rho * SCR1 * SCR2 + SCR2**2)
    return {"SCR_eq_type1": SCR1, "SCR_eq_type2": SCR2, "SCR_eq_total": SCR_eq}

def scr_prop(prop, shock):
    return prop * shock

def scr_sprd(corp_bond, shock):
    return corp_bond * shock

def aggregate_market_scr(scr_interest, scr_equity, scr_property, scr_spread, corr_downward, corr_upward):
    direction = scr_interest["direction"].lower()
    vec = np.array([
        scr_interest['SCR_interest'],
        scr_equity['SCR_eq_total'],
        scr_property,
        scr_spread
    ])

    corr = np.array(corr_downward if direction.startswith("down") else corr_upward)
    scr_total = np.sqrt(vec @ corr @ vec)
    scr_sum = np.sum(vec)
    diversification = scr_sum - scr_total
    diversification_pct = 1 - (scr_total / scr_sum) if scr_sum > 0 else 0

    summary = pd.DataFrame({
        "risk": ["interest", "equity", "property", "spread", "total", "diversification"],
        "SCR": np.append(vec, [scr_total, diversification])
    }).set_index("risk")

    return {
        "summary_table": summary,
        "chosen_panel": "Interest_DOWN" if direction == "down" else "Interest_UP",
        "SCR_market_final": scr_total,
        "Diversification_pct": diversification_pct,
    }


import numpy as np
import pandas as pd

def marginal_scr(v: np.ndarray, direction: str, corr_downward: np.ndarray, corr_upward: np.ndarray) -> pd.DataFrame:
    """
    Compute marginal and allocated SCRs (Solvency II-style).

    Parameters
    ----------
    v : np.ndarray
        Vector of stand-alone SCRs [interest, equity, property, spread].
    direction : str
        Either 'Interest_DOWN' or 'Interest_UP' (controls which correlation matrix is used).
    corr_downward : np.ndarray
        Correlation matrix for downward scenario.
    corr_upward : np.ndarray
        Correlation matrix for upward scenario.

    Returns
    -------
    pd.DataFrame
        Table containing:
        - risk name
        - Stand-alone SCRs
        - Marginal SCR factors 
        - Absolute SCR contribution 
        - Relative SCR contribution (%)
    """

    # Ensure input is numpy array
    v = np.asarray(v, dtype=float)

    # Choose correlation matrix based on scenario direction
    if direction.upper().startswith("DOWN"):
        corr = np.asarray(corr_downward, dtype=float)
    else:
        corr = np.asarray(corr_upward, dtype=float)

    # Sanity check: dimension consistency
    if corr.shape[0] != corr.shape[1] or corr.shape[0] != len(v):
        raise ValueError("Correlation matrix must be square and match the size of the SCR vector")

    # === Core Solvency II marginal SCR formula ===
    scr_total = np.sqrt(v @ corr @ v)
    marginals = (corr @ v) / scr_total
    allocated = v * marginals
    share_pct = 100 * allocated / scr_total

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
    Break down total marginal SCR by asset type.

    Parameters
    ----------
    marginal_per_risk : pd.DataFrame
        Output of marginal_scr()
    direction : str
        'Interest_DOWN' or 'Interest_UP'
    initial_asset : pd.DataFrame
        Asset table (must include 'asset_val', 'asset_dur')
    params : dict
        Model parameters containing interest, spread, equity, and property shocks

    Returns
    -------
    pd.DataFrame
        Marginal SCR per asset
    """
    import numpy as np

    m = marginal_per_risk.set_index("risk")["marginal_factor"]

    int_down_param = params["interest_down"]
    int_up_param = params["interest_up"]
    spread_param = params["spread"]
    equity_1_param = params["equity_type1"]
    equity_2_param = params["equity_type2"]
    rho = params["rho"]
    prop_param = params["property"]

    # Choose interest shock direction
    int_param = int_down_param if direction == "Interest_DOWN" else int_up_param

    # Bond and bills sensitivity
    mSCR_gov_bond = -m["interest"] * int_param * initial_asset.loc["gov_bond", "asset_dur"]
    mSCR_corp_bond = (
        (-m["interest"] * int_param * initial_asset.loc["corp_bond", "asset_dur"]) +
        (m["spread"] * spread_param)
    )
    mSCR_t_bills = -m["interest"] * int_param * initial_asset.loc["t_bills", "asset_dur"]

    # Equity risk
    scr_eq_type1 = equity_1_param * initial_asset.loc["equity_1", "asset_val"]
    scr_eq_type2 = equity_2_param * initial_asset.loc["equity_2", "asset_val"]
    scr_eq_total = np.sqrt(scr_eq_type1**2 + 2 * rho * scr_eq_type1 * scr_eq_type2 + scr_eq_type2**2)

    w1 = (scr_eq_type1 + rho * scr_eq_type2) / scr_eq_total
    w2 = (scr_eq_type2 + rho * scr_eq_type1) / scr_eq_total

    mSCR_equity_1 = m["equity"] * w1 * equity_1_param
    mSCR_equity_2 = m["equity"] * w2 * equity_2_param

    # Property risk
    mSCR_property = m["property"] * prop_param

    df_assets = pd.DataFrame({
        "asset": [
            "gov_bond",
            "corp_bond",
            "equity_1",
            "equity_2",
            "property",
            "t_bills"
        ],
        "mSCR": [
            mSCR_gov_bond,
            mSCR_corp_bond,
            mSCR_equity_1,
            mSCR_equity_2,
            mSCR_property,
            mSCR_t_bills
        ]
    })
    return df_assets
