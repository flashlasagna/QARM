"""
frontend.py — Streamlit UI for Solvency II Asset Allocation Optimizer
Cleaner version with essential inputs only
"""

import streamlit as st
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# --- Make project root importable ---
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Backend imports ---
from backend.config_loader import load_config, get_corr_matrices, get_solvency_params
from backend.optimization import solve_frontier_combined
from backend.helpers import plot_frontier, summarize_portfolio

# ✅ Updated imports from data_calculator
from backend.data_calculator import (
    compute_expected_returns_from_etfs,
    get_gov_bond_return,
    compute_ir_shocks_duration_approx,
    compute_ir_shocks_from_eiopa,  # ✅ New: uses data_handler
    compute_spread_shock_eiopa      # ✅ New: uses data_handler
)


st.set_page_config(page_title="Solvency II Asset Allocation Optimizer", layout="wide")

# --------------------------
# Navigation
# --------------------------
if "nav" not in st.session_state:
    st.session_state["nav"] = "Inputs"

st.sidebar.title("Navigation")
st.session_state["nav"] = st.sidebar.radio(
    "Go to:",
    ["Inputs", "Results"],
    index=["Inputs", "Results"].index(st.session_state["nav"])
)


# --------------------------
# Helper Functions
# --------------------------

def fetch_ticker_prices(ticker, start_date, end_date):
    """
    Robust ticker price fetching with fallback logic.

    Returns:
        pd.Series: Price series, or None if failed
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            return None

        # Handle different data structures
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-index columns (multi-ticker download)
            if 'Close' in data.columns.get_level_values(0):
                prices = data['Close'].iloc[:, 0]
            elif 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close'].iloc[:, 0]
            else:
                return None
        else:
            # Single-level columns (normal case)
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif 'Close' in data.columns:
                prices = data['Close']
            else:
                return None

        # Clean and return
        prices = prices.dropna()
        return prices if len(prices) > 100 else None

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def validate_inputs(A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb, total_A,
                    gov_min, gov_max, corp_max, illiq_max, tb_min, tb_max):
    """Validate user inputs and return errors/warnings."""
    errs, warns = [], []

    tot = A_gov + A_corp + A_eq1 + A_eq2 + A_prop + A_tb

    if abs(tot - total_A) > 1e-6:
        errs.append(f"Amounts must sum to Total A (€{total_A:.1f}m). Current = €{tot:.1f}m.")

    if any(x < -1e-12 for x in (A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb)):
        errs.append("Allocations must be non-negative.")

    # Check limits as weights
    denom = max(total_A, 1e-9)
    w_gov = A_gov / denom
    w_corp = A_corp / denom
    w_eq1 = A_eq1 / denom
    w_eq2 = A_eq2 / denom
    w_prop = A_prop / denom
    w_tb = A_tb / denom

    if not (gov_min - 1e-9 <= w_gov <= gov_max + 1e-9):
        errs.append(f"Government weight {w_gov:.3f} violates [{gov_min:.2f}, {gov_max:.2f}].")
    if w_corp > corp_max + 1e-9:
        errs.append(f"Corporate weight {w_corp:.3f} exceeds {corp_max:.2f}.")
    if (w_eq1 + w_eq2 + w_prop) > illiq_max + 1e-9:
        errs.append(f"Illiquid (eq1+eq2+prop) {w_eq1 + w_eq2 + w_prop:.3f} exceeds {illiq_max:.2f}.")
    if not (tb_min - 1e-9 <= w_tb <= tb_max + 1e-9):
        warns.append(f"T-bills {w_tb:.3f} outside [{tb_min:.2f}, {tb_max:.2f}].")

    return errs, warns


def build_backend_inputs(A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb, total_A,
                         BE_value, BE_dur, dur_gov, dur_corp, dur_tb,
                         r_gov, r_corp, r_eq1, r_eq2, r_prop, r_tb,
                         ir_up, ir_down, corp_sp,
                         gov_min, gov_max, corp_max, illiq_max, tb_min, tb_max,
                         use_custom_shocks=False,
                         eq1_sh=None, eq2_sh=None, prop_sh=None):
    """Build inputs for backend optimization."""

    cfg = load_config()
    corr_down, corr_up = get_corr_matrices(cfg)
    solv = get_solvency_params(cfg)

    # Use config values unless custom shocks enabled
    if use_custom_shocks and eq1_sh is not None:
        equity_1_shock = eq1_sh
        equity_2_shock = eq2_sh
        property_shock = prop_sh
    else:
        equity_1_shock = solv["equity_1_param"]
        equity_2_shock = solv["equity_2_param"]
        property_shock = solv["prop_params"]

    # Asset table
    initial_asset = pd.DataFrame({
        "asset_val": [A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb],
        "asset_dur": [dur_gov, dur_corp, 0.0, 0.0, 0.0, dur_tb],
        "asset_ret": [r_gov, r_corp, r_eq1, r_eq2, r_prop, r_tb]
    }, index=["gov_bond", "corp_bond", "equity_1", "equity_2", "property", "t_bills"])

    initial_asset["asset_weight"] = initial_asset["asset_val"] / max(initial_asset["asset_val"].sum(), 1e-12)

    # Allocation limits
    allocation_limits = pd.DataFrame({
        "asset": ["gov_bond", "illiquid_assets", "t_bills", "corp_bond"],
        "min_weight": [gov_min, 0.0, tb_min, 0.0],
        "max_weight": [gov_max, illiq_max, tb_max, corp_max],
    }).set_index("asset")

    # Parameters
    params = {
        "interest_down": ir_down,
        "interest_up": ir_up,
        "spread": corp_sp,
        "equity_type1": equity_1_shock,
        "equity_type2": equity_2_shock,
        "property": property_shock,
        "rho": solv["rho"],
    }

    return initial_asset, BE_value, BE_dur, corr_down, corr_up, allocation_limits, params


# --------------------------
# INPUTS PAGE
# --------------------------
if st.session_state["nav"] == "Inputs":
    st.title("🏦 Solvency II Asset Allocation Optimizer")
    st.markdown("---")

    # Auto-calculate toggle (BEFORE columns)
    use_auto_params = st.checkbox(
        "🤖 Auto-calculate Parameters from Market Data",
        value=True,  # ✅ Default to TRUE now
        help="Automatically compute expected returns, durations, and shocks from selected ETF tickers"
    )

    if use_auto_params:
        st.success(
            "✨ **Automated Mode**: Select your ETF tickers below, and all parameters will be computed automatically!")
    else:
        st.info("📝 **Manual Mode**: You'll need to enter all parameters manually in Advanced Settings.")

    st.markdown("---")

    # ==========================================
    # TICKER SELECTION (Only shown if auto mode)
    # ==========================================
    if use_auto_params:
        st.subheader("🎯 Select Your ETF Tickers")

        st.markdown("""
        Choose the ETFs/tickers that represent each asset class in your portfolio. 
        The app will automatically compute expected returns and other parameters from historical data.
        """)

        # Create columns for ticker input
        ticker_col1, ticker_col2, ticker_col3 = st.columns(3)

        with ticker_col1:
            st.markdown("**Fixed Income**")
            ticker_gov = st.text_input(
                "Government Bonds",
                value="IBGM.L",
                help="Example: IBGM.L (iShares Global Govt Bond), IEAG.L (Euro Govt Bond)"
            )
            ticker_corp = st.text_input(
                "Corporate Bonds",
                value="IE15.L",
                help="Example: IE15.L (iShares Euro Corp 1-5y), SLXX.L (Euro Corp Bond)"
            )
            ticker_tbills = st.text_input(
                "Treasury Bills / Cash",
                value="CSH2.L",
                help="Example: CSH2.L (Euro Cash), or leave for manual input"
            )

        with ticker_col2:
            st.markdown("**Equity**")
            ticker_eq1 = st.text_input(
                "Equity Type 1 (Developed)",
                value="EUNL.DE",
                help="Example: EUNL.DE (iShares MSCI World EUR), IWDA.L (MSCI World USD)"
            )
            ticker_eq2 = st.text_input(
                "Equity Type 2 (Emerging)",
                value="IQQE.DE",
                help="Example: IQQE.DE (iShares MSCI EM EUR), EIMI.L (MSCI EM USD)"
            )

        with ticker_col3:
            st.markdown("**Real Assets**")
            ticker_prop = st.text_input(
                "Property / Real Estate",
                value="EUNK.DE",
                help="Example: EUNK.DE (iShares EU Property), IUKP.L (UK Property)"
            )

            st.markdown("**Data Period**")
            lookback_years = st.slider(
                "Historical data (years)",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of years of historical data for return calculation"
            )
            # Store tickers in session state for later use
            if 'ticker_gov' not in st.session_state or st.session_state.get('ticker_gov') != ticker_gov:
                st.session_state['ticker_gov'] = ticker_gov
                st.session_state['ticker_corp'] = ticker_corp
                st.session_state['ticker_tbills'] = ticker_tbills
                st.session_state['ticker_eq1'] = ticker_eq1
                st.session_state['ticker_eq2'] = ticker_eq2
                st.session_state['ticker_prop'] = ticker_prop
                st.session_state['lookback_years'] = lookback_years
        # Validate tickers button
        if st.button("🔍 Validate Tickers", type="secondary"):
            with st.spinner("Validating tickers and fetching data..."):
                from datetime import datetime, timedelta

                all_tickers = {
                    "Government Bonds": ticker_gov,
                    "Corporate Bonds": ticker_corp,
                    "Treasury Bills": ticker_tbills,
                    "Equity Type 1": ticker_eq1,
                    "Equity Type 2": ticker_eq2,
                    "Property": ticker_prop
                }

                validation_results = []

                for asset_class, ticker in all_tickers.items():
                    if not ticker or ticker.strip() == "":
                        validation_results.append({
                            "Asset Class": asset_class,
                            "Ticker": ticker,
                            "Status": f"✅ Valid ({ann_return:.2%})",
                            "Data Points": len(prices)
                        })
                        continue

                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=lookback_years * 365 + 100)

                    prices = fetch_ticker_prices(ticker, start_date, end_date)

                    if prices is None or len(prices) < 100:
                        validation_results.append({
                            "Asset Class": asset_class,
                            "Ticker": ticker,
                            "Status": "❌ Insufficient data",
                            "Data Points": len(prices) if prices is not None else 0
                        })
                    else:
                        # Calculate return to show in validation
                        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
                        years = len(prices) / 252
                        ann_return = (1 + total_return) ** (1 / years) - 1

                        validation_results.append({
                            "Asset Class": asset_class,
                            "Ticker": ticker,
                            "Status": f"✅ Valid ({ann_return:.2%})",
                            "Data Points": len(prices)
                        })
                # Display results
                validation_df = pd.DataFrame(validation_results)
                st.dataframe(validation_df, use_container_width=True, hide_index=True)

                # Summary
                valid_count = sum(1 for r in validation_results if "✅" in r["Status"])
                total_count = len([r for r in validation_results if r["Ticker"] != "Not provided"])

                if valid_count == total_count:
                    st.success(f"✅ All {valid_count} tickers validated successfully!")
                elif valid_count > 0:
                    st.warning(f"⚠️ {valid_count}/{total_count} tickers validated. Check failed tickers above.")
                else:
                    st.error("❌ No valid tickers found. Please check your inputs.")

        st.markdown("---")

    # ==========================================
    # PORTFOLIO ALLOCATION
    # ==========================================
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("📊 Portfolio Allocation")

        total_A = st.number_input(
            "Total Assets (€ millions)",
            min_value=0.0,
            value=1652.7,
            step=10.0,
            help="Total value of assets in millions of euros"
        )

        st.markdown("**Asset Allocation (€ millions)**")

        A_gov = st.number_input("Government Bonds", min_value=0.0, value=782.6, step=1.0)
        A_corp = st.number_input("Corporate Bonds", min_value=0.0, value=586.0, step=1.0)
        A_eq1 = st.number_input("Equity Type 1 (Developed Markets)", min_value=0.0, value=0.0, step=1.0)
        A_eq2 = st.number_input("Equity Type 2 (Emerging Markets)", min_value=0.0, value=102.5, step=1.0)
        A_prop = st.number_input("Property", min_value=0.0, value=42.0, step=1.0)
        A_tb = st.number_input("Treasury Bills", min_value=0.0, value=139.6, step=1.0)

        # Show total and progress
        total_allocated = A_gov + A_corp + A_eq1 + A_eq2 + A_prop + A_tb
        st.progress(min(1.0, total_allocated / max(total_A, 1)))

        if abs(total_allocated - total_A) < 1e-6:
            st.success(f"✓ Total allocated: €{total_allocated:.1f}m")
        else:
            st.warning(f"⚠ Total allocated: €{total_allocated:.1f}m (Target: €{total_A:.1f}m)")

    with col2:
        st.subheader("📋 Liabilities & Durations")

        BE_value = st.number_input(
            "Best Estimate Liabilities (€m)",
            min_value=0.0,
            value=1424.2,
            step=10.0,
            help="Best estimate of technical provisions"
        )
        BE_dur = st.number_input(
            "Liabilities Duration (years)",
            min_value=0.0,
            value=6.6,
            step=0.1,
            help="Modified duration of liabilities"
        )

        if not use_auto_params:
            st.markdown("**Asset Durations (years)**")
            dur_gov = st.number_input("Gov Bonds Duration", 0.0, 50.0, 5.2, 0.1)
            dur_corp = st.number_input("Corp Bonds Duration", 0.0, 50.0, 5.0, 0.1)
            dur_tb = st.number_input("T-Bills Duration", 0.0, 10.0, 0.1, 0.1)
        else:
            st.info("📊 Durations will be estimated from market data")
            # Set defaults for auto mode (will be overridden)
            dur_gov = 5.2
            dur_corp = 5.0
            dur_tb = 0.1

    st.markdown("---")

    # Advanced settings in expander
    with st.expander("⚙️ Advanced Settings", expanded=not use_auto_params):

        if use_auto_params:
            st.warning("⚠️ Auto-calculation is enabled. Manual inputs below will be **overridden** by computed values.")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Solvency II Shocks**")

            # ✅ EIOPA CURVES OPTION (only show in auto mode)
            if use_auto_params:
                use_eiopa_curves = st.checkbox(
                    "📊 Use EIOPA Risk-Free Curves",
                    value=True,  # ✅ Default to TRUE in auto mode
                    help="Load official EIOPA curves (Oct 2025) for regulatory-compliant interest rate shocks"
                )
                st.session_state['use_eiopa_curves'] = use_eiopa_curves  # Store in session

                if use_eiopa_curves:
                    st.info("✓ Will use EIOPA Oct 2025 curves (Portugal, NO VA)")
                else:
                    st.info("✓ Will use duration-based shock approximation")

                st.markdown("---")

            # Interest rate and spread shocks display/input
            if use_auto_params:
                st.info("✓ Shocks will be computed automatically")
                # Placeholders - will be overridden in optimize button
                ir_up = 0.011
                ir_down = 0.009
                corp_sp = 0.103
            else:
                # Manual mode - show input fields
                ir_up = st.number_input("IR Up", 0.0, 1.0, 0.011, 0.001, format="%.3f")
                ir_down = st.number_input("IR Down", 0.0, 1.0, 0.009, 0.001, format="%.3f")
                corp_sp = st.number_input(
                    "Spread (Corp)",
                    0.0, 1.0, 0.103, 0.001,
                    format="%.3f",
                    help="Corporate bond spread shock"
                )

            st.markdown("---")

            # Option to override equity/property shocks
            use_custom_shocks = st.checkbox(
                "Override Standard Formula",
                value=False,
                help="Enable to modify equity and property shocks. Default values are Solvency II regulatory standards (39%, 49%, 25%)."
            )

            if use_custom_shocks:
                eq1_sh = st.number_input("Equity Type 1 (custom)", 0.0, 1.0, 0.39, 0.01, format="%.2f")
                eq2_sh = st.number_input("Equity Type 2 (custom)", 0.0, 1.0, 0.49, 0.01, format="%.2f")
                prop_sh = st.number_input("Property (custom)", 0.0, 1.0, 0.25, 0.01, format="%.2f")
            else:
                eq1_sh = None
                eq2_sh = None
                prop_sh = None
                st.info(
                    "Using standard Solvency II shocks:\n- Equity Type 1: 39%\n- Equity Type 2: 49%\n- Property: 25%")

        with col_b:
            st.markdown("**Expected Returns (annual)**")

            if use_auto_params:
                st.info("✓ Returns will be computed from ETF historical data")
                # Show placeholders - will be overridden in optimize button
                r_gov = 0.029
                r_corp = 0.041
                r_eq1 = 0.064
                r_eq2 = 0.064
                r_prop = 0.056
                r_tb = 0.006
            else:
                r_gov = st.number_input("Gov Bonds", 0.0, 1.0, 0.029, 0.001, format="%.3f")
                r_corp = st.number_input("Corp Bonds", 0.0, 1.0, 0.041, 0.001, format="%.3f")
                r_eq1 = st.number_input("Equity Type 1", 0.0, 1.0, 0.064, 0.001, format="%.3f")
                r_eq2 = st.number_input("Equity Type 2", 0.0, 1.0, 0.064, 0.001, format="%.3f")
                r_prop = st.number_input("Property", 0.0, 1.0, 0.056, 0.001, format="%.3f")
                r_tb = st.number_input("T-Bills", 0.0, 1.0, 0.006, 0.001, format="%.3f")

        with col_c:
            st.markdown("**Allocation Limits (weights)**")
            gov_min = st.number_input("Gov Min", 0.0, 1.0, 0.25, 0.01, format="%.2f")
            gov_max = st.number_input("Gov Max", 0.0, 1.0, 0.75, 0.01, format="%.2f")
            corp_max = st.number_input("Corp Max", 0.0, 1.0, 0.50, 0.01, format="%.2f")
            illiq_max = st.number_input("Illiquid Max", 0.0, 1.0, 0.20, 0.01, format="%.2f")
            tb_min = st.number_input("T-Bills Min", 0.0, 1.0, 0.01, 0.01, format="%.2f")
            tb_max = st.number_input("T-Bills Max", 0.0, 1.0, 0.05, 0.01, format="%.2f")

    st.markdown("---")

    # Validation
    errs, warns = validate_inputs(
        A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb, total_A,
        gov_min, gov_max, corp_max, illiq_max, tb_min, tb_max
    )

    for e in errs:
        st.error(e)
    for w in warns:
        st.warning(w)

    # Optimize button
    can_optimize = (len(errs) == 0)

    if st.button("🚀 Optimize Portfolio", disabled=not can_optimize, type="primary", use_container_width=True):
        with st.spinner("Running optimization... This may take a minute."):
            try:
                # Auto-calculate parameters if enabled
                if use_auto_params:
                    st.info("🤖 Computing all parameters from selected tickers and EIOPA data...")

                    # Get tickers from session state or use defaults
                    ticker_mapping = {
                        'gov_bond': st.session_state.get('ticker_gov', "VGOV"),
                        'corp_bond': st.session_state.get('ticker_corp', "VCIT"),
                        'equity_1': st.session_state.get('ticker_eq1', "EUNL.DE"),
                        'equity_2': st.session_state.get('ticker_eq2', "IQQE.DE"),
                        'property': st.session_state.get('ticker_prop', "EUNK.DE"),
                        't_bills': st.session_state.get('ticker_tbills', "BIL")
                    }

                    lookback = st.session_state.get('lookback_years', 5)

                    # ==========================================
                    # 1. AUTO-CALCULATE EXPECTED RETURNS
                    # ==========================================
                    try:
                        with st.spinner("📊 Fetching ETF data and computing returns..."):
                            auto_returns = {}

                            for asset_class, ticker in ticker_mapping.items():
                                if not ticker or ticker.strip() == "":
                                    continue

                                end_date = datetime.now()
                                start_date = end_date - timedelta(days=lookback * 365 + 100)

                                prices = fetch_ticker_prices(ticker, start_date, end_date)

                                if prices is not None and len(prices) > lookback * 252 * 0.8:
                                    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
                                    actual_years = len(prices) / 252
                                    annualized_return = (1 + total_return) ** (1 / actual_years) - 1
                                    auto_returns[asset_class] = float(annualized_return)

                            # Apply computed returns
                            r_gov = auto_returns.get('gov_bond', 0.029)
                            r_corp = auto_returns.get('corp_bond', 0.041)
                            r_eq1 = auto_returns.get('equity_1', 0.064)
                            r_eq2 = auto_returns.get('equity_2', 0.064)
                            r_prop = auto_returns.get('property', 0.056)
                            r_tb = auto_returns.get('t_bills', 0.006)

                            # ✅ VALIDATE RETURNS - use paper defaults if unrealistic
                            returns_adjusted = False

                            if r_gov < 0:
                                st.warning(f"⚠️ Gov bond return ({r_gov:.2%}) is negative. Using paper default (2.9%)")
                                r_gov = 0.029
                                returns_adjusted = True

                            if r_corp < 0:
                                st.warning(
                                    f"⚠️ Corp bond return ({r_corp:.2%}) is negative. Using paper default (4.1%)")
                                r_corp = 0.041
                                returns_adjusted = True

                            if r_tb < 0:
                                st.warning(f"⚠️ T-bill return ({r_tb:.2%}) is negative. Using paper default (0.6%)")
                                r_tb = 0.006
                                returns_adjusted = True

                            if r_eq1 < -0.10 or r_eq1 > 0.20:
                                st.warning(
                                    f"⚠️ Equity 1 return ({r_eq1:.2%}) seems unrealistic. Using paper default (6.4%)")
                                r_eq1 = 0.064
                                returns_adjusted = True

                            if r_eq2 < -0.10 or r_eq2 > 0.25:
                                st.warning(
                                    f"⚠️ Equity 2 return ({r_eq2:.2%}) seems unrealistic. Using paper default (6.4%)")
                                r_eq2 = 0.064
                                returns_adjusted = True

                            if r_prop < -0.05 or r_prop > 0.15:
                                st.warning(
                                    f"⚠️ Property return ({r_prop:.2%}) seems unrealistic. Using paper default (5.6%)")
                                r_prop = 0.056
                                returns_adjusted = True

                            if returns_adjusted:
                                st.info("💡 Some returns adjusted to paper defaults due to unusual market conditions.")

                            st.success(
                                f"✓ Returns computed from {len(auto_returns)} tickers: "
                                f"Gov={r_gov:.2%}, Corp={r_corp:.2%}, Eq1={r_eq1:.2%}, "
                                f"Eq2={r_eq2:.2%}, Prop={r_prop:.2%}, TB={r_tb:.2%}"
                            )

                    except Exception as e:
                        st.warning(f"⚠️ Could not auto-calculate returns: {e}. Using defaults.")
                        r_gov, r_corp, r_eq1, r_eq2, r_prop, r_tb = 0.029, 0.041, 0.064, 0.064, 0.056, 0.006

                    # ==========================================
                    # 2. ESTIMATE DURATIONS
                    # ==========================================
                    try:
                        with st.spinner("⏱️ Estimating asset durations..."):
                            # Government bonds duration estimation
                            ticker_gov_str = ticker_mapping.get('gov_bond', '')
                            if 'short' in ticker_gov_str.lower() or '1-3' in ticker_gov_str.lower():
                                dur_gov = 2.0
                            elif 'long' in ticker_gov_str.lower() or '10+' in ticker_gov_str.lower():
                                dur_gov = 10.0
                            else:
                                dur_gov = 5.2  # Default mid-term

                            # Corporate bonds duration estimation
                            ticker_corp_str = ticker_mapping.get('corp_bond', '')
                            if 'short' in ticker_corp_str.lower() or '1-5' in ticker_corp_str.lower():
                                dur_corp = 3.0
                            elif 'long' in ticker_corp_str.lower():
                                dur_corp = 8.0
                            else:
                                dur_corp = 5.0  # Default

                            dur_tb = 0.1  # Always short for t-bills

                            st.success(
                                f"✓ Durations estimated: Gov={dur_gov:.1f}y, Corp={dur_corp:.1f}y, TB={dur_tb:.1f}y"
                            )

                    except Exception as e:
                        st.warning(f"⚠️ Could not estimate durations: {e}. Using defaults.")
                        dur_gov, dur_corp, dur_tb = 5.2, 5.0, 0.1

                    # ==========================================
                    # 3. AUTO-CALCULATE SHOCKS
                    # ==========================================

                    # Check if EIOPA mode is enabled
                    use_eiopa = st.session_state.get('use_eiopa_curves', False)

                    if use_eiopa:
                        # ✅ USE EIOPA CURVES
                        try:
                            with st.spinner("📊 Loading EIOPA curves and computing shocks..."):
                                # Compute IR shocks based on liability duration
                                ir_up, ir_down = compute_ir_shocks_from_eiopa(
                                    liab_duration=BE_dur,
                                    verbose=True  # Will print to console
                                )

                                # Compute spread shock using EIOPA formula
                                corp_sp = compute_spread_shock_eiopa(
                                    duration=dur_corp,
                                    verbose=True  # Will print to console
                                )

                                st.success(
                                    f"✓ EIOPA shocks computed:\n\n"
                                    f"**Interest Rate Shocks:**\n"
                                    f"- Up: {ir_up:.3%}\n"
                                    f"- Down: {ir_down:.3%}\n\n"
                                    f"**Spread Shock:** {corp_sp:.3%}"
                                )

                        except Exception as e:
                            st.error(f"❌ Could not load EIOPA curves: {e}")
                            st.info("Falling back to duration-based approximation")

                            # Fallback to duration approximation
                            try:
                                weights = np.array([A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb]) / total_A
                                durations = np.array([dur_gov, dur_corp, 0, 0, 0, dur_tb])
                                portfolio_dur = np.sum(weights * durations)

                                ir_up, ir_down = compute_ir_shocks_duration_approx(
                                    asset_dur=portfolio_dur,
                                    liab_dur=BE_dur,
                                    rate_change=0.01
                                )

                                corp_sp = 0.103  # Default

                                st.success(
                                    f"✓ Shocks (duration approx): IR Up={ir_up:.3%}, Down={ir_down:.3%}, Spread={corp_sp:.3%}")

                            except Exception as e2:
                                st.warning(f"⚠️ Fallback also failed: {e2}. Using defaults.")
                                ir_up, ir_down, corp_sp = 0.011, 0.009, 0.103

                    else:
                        # ✅ USE DURATION-BASED APPROXIMATION (No EIOPA)
                        try:
                            weights = np.array([A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb]) / total_A
                            durations = np.array([dur_gov, dur_corp, 0, 0, 0, dur_tb])
                            portfolio_dur = np.sum(weights * durations)

                            ir_up, ir_down = compute_ir_shocks_duration_approx(
                                asset_dur=portfolio_dur,
                                liab_dur=BE_dur,
                                rate_change=0.01
                            )

                            # Can still use EIOPA spread shock formula even without full curves
                            corp_sp = compute_spread_shock_eiopa(
                                duration=dur_corp,
                                verbose=True
                            )

                            st.success(
                                f"✓ Shocks computed (duration-based):\n\n"
                                f"**Interest Rate:**\n"
                                f"- Up: {ir_up:.3%}\n"
                                f"- Down: {ir_down:.3%}\n\n"
                                f"**Spread (EIOPA formula):** {corp_sp:.3%}"
                            )

                        except Exception as e:
                            st.warning(f"⚠️ Could not auto-calculate shocks: {e}. Using defaults.")
                            ir_up, ir_down, corp_sp = 0.011, 0.009, 0.103

                else:
                    # ==========================================
                    # MANUAL MODE - use input values from Advanced Settings
                    # ==========================================
                    pass  # Variables already defined from Advanced Settings inputs

                # ==========================================
                # BUILD BACKEND INPUTS & RUN OPTIMIZATION
                # ==========================================
                initial_asset, liab_value, liab_duration, corr_down, corr_up, allocation_limits, params = \
                    build_backend_inputs(
                        A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb, total_A,
                        BE_value, BE_dur, dur_gov, dur_corp, dur_tb,
                        r_gov, r_corp, r_eq1, r_eq2, r_prop, r_tb,
                        ir_up, ir_down, corp_sp,
                        gov_min, gov_max, corp_max, illiq_max, tb_min, tb_max,
                        use_custom_shocks=use_custom_shocks if 'use_custom_shocks' in locals() else False,
                        eq1_sh=eq1_sh if 'eq1_sh' in locals() else None,
                        eq2_sh=eq2_sh if 'eq2_sh' in locals() else None,
                        prop_sh=prop_sh if 'prop_sh' in locals() else None
                    )

                opt_df = solve_frontier_combined(
                    initial_asset=initial_asset,
                    liab_value=liab_value,
                    liab_duration=liab_duration,
                    corr_downward=corr_down,
                    corr_upward=corr_up,
                    allocation_limits=allocation_limits,
                    params=params
                )

                # Store results
                st.session_state["opt_df"] = opt_df
                st.session_state["initial_asset"] = initial_asset
                st.session_state["liab_value"] = liab_value
                st.session_state["liab_duration"] = liab_duration
                st.session_state["auto_calculated"] = use_auto_params
                st.session_state["nav"] = "Results"

                st.success("✓ Optimization completed successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                st.exception(e)


# --------------------------
# RESULTS PAGE
# --------------------------
else:
    st.title("📈 Optimization Results")

    # Show if auto-calculation was used
    if st.session_state.get("auto_calculated", False):
        st.info("🤖 **Auto-calculation was enabled** - Returns and IR shocks were computed from market data")

    if "opt_df" not in st.session_state:
        st.info("ℹ️ No results yet. Go to **Inputs** and click **Optimize Portfolio**.")
        st.stop()

    if "opt_df" not in st.session_state:
        st.info("ℹ️ No results yet. Go to **Inputs** and click **Optimize Portfolio**.")
        st.stop()

    opt_df = st.session_state["opt_df"]
    initial_asset = st.session_state["initial_asset"]  # ✅ Already here
    liab_value = st.session_state["liab_value"]  # ✅ Already here
    liab_duration = st.session_state.get("liab_duration", 6.6)  # ← Add this line

    if opt_df.empty:
        st.error("No feasible solutions found. Try adjusting your constraints.")
        st.stop()

    # Now add the portfolio summary
    st.subheader("📋 Portfolio Summary")
    summary_table = summarize_portfolio(initial_asset, liab_value, liab_duration)
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    if "opt_df" not in st.session_state:
        st.info("ℹ️ No results yet. Go to **Inputs** and click **Optimize Portfolio**.")
        st.stop()

    opt_df = st.session_state["opt_df"]
    initial_asset = st.session_state["initial_asset"]
    liab_value = st.session_state["liab_value"]

    if opt_df.empty:
        st.error("No feasible solutions found. Try adjusting your constraints.")
        st.stop()

    # Find best solution (highest objective value)
    best_idx = opt_df["objective"].idxmax()
    best = opt_df.loc[best_idx]

    # Current portfolio metrics
    current_ret = float((initial_asset["asset_ret"] * initial_asset["asset_weight"]).sum())
    current_SCR = float(best["SCR_market"])  # Approximation
    current_sol = (initial_asset['asset_val'].sum() - liab_value) / current_SCR

    # Best portfolio metrics
    best_ret = float(best["return"])
    best_sol = float(best["solvency"])
    best_SCR = float(best["SCR_market"])
    best_BOF = float(best["BOF"])

    st.markdown("---")

    # Key metrics
    st.subheader("📊 Key Metrics Comparison")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Expected Return",
            f"{best_ret:.2%}",
            delta=f"{(best_ret - current_ret):.2%}",
            help="Portfolio expected return (annualized)"
        )

    with col2:
        st.metric(
            "Solvency Ratio",
            f"{best_sol * 100:.1f}%",
            delta=f"{(best_sol - current_sol) * 100:.1f}%",
            help="BOF / SCR Market"
        )

    with col3:
        st.metric(
            "SCR Market",
            f"€{best_SCR:.1f}m",
            delta=f"€{(best_SCR - current_SCR):.1f}m",
            delta_color="inverse",
            help="Market risk capital requirement"
        )

    with col4:
        st.metric(
            "Basic Own Funds",
            f"€{best_BOF:.1f}m",
            help="Assets - Liabilities"
        )
    st.markdown("---")

    # Duration Analysis
    st.subheader("⚖️ Duration Analysis")

    # Calculate current portfolio duration
    current_dur = np.average(
        initial_asset["asset_dur"].values,
        weights=initial_asset["asset_weight"].values
    )

    # Calculate optimal portfolio duration
    optimal_dur = np.average(
        initial_asset["asset_dur"].values,
        weights=best["w_opt"]
    )

    # Duration gap (portfolio duration - liability duration)
    current_gap = current_dur - liab_duration
    optimal_gap = optimal_dur - liab_duration

    # Create columns for duration metrics
    dur_col1, dur_col2, dur_col3, dur_col4 = st.columns(4)

    with dur_col1:
        st.metric(
            "Liability Duration",
            f"{liab_duration:.2f} years",
            help="Modified duration of liabilities"
        )

    with dur_col2:
        st.metric(
            "Current Portfolio Duration",
            f"{current_dur:.2f} years",
            delta=f"{current_gap:+.2f} years gap",
            delta_color="inverse",
            help="Weighted average duration of current portfolio"
        )

    with dur_col3:
        st.metric(
            "Optimal Portfolio Duration",
            f"{optimal_dur:.2f} years",
            delta=f"{optimal_gap:+.2f} years gap",
            delta_color="inverse",
            help="Weighted average duration of optimal portfolio"
        )

    with dur_col4:
        improvement = abs(current_gap) - abs(optimal_gap)
        st.metric(
            "Gap Improvement",
            f"{improvement:.2f} years",
            delta=f"{(improvement / abs(current_gap) * 100) if current_gap != 0 else 0:.1f}%",
            help="Reduction in absolute duration gap"
        )

    # Visual representation of duration matching
    st.markdown("**Duration Matching Visualization**")

    fig_dur, ax_dur = plt.subplots(figsize=(10, 4))

    categories = ['Current\nPortfolio', 'Optimal\nPortfolio', 'Liabilities']
    durations = [current_dur, optimal_dur, liab_duration]
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

    bars = ax_dur.bar(categories, durations, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, dur in zip(bars, durations):
        height = bar.get_height()
        ax_dur.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{dur:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add horizontal line for liability duration
    ax_dur.axhline(y=liab_duration, color='gray', linestyle='--',
                   linewidth=2, alpha=0.5, label='Target (Liability Duration)')

    ax_dur.set_ylabel('Duration (years)', fontsize=11)
    ax_dur.set_title('Portfolio Duration vs Liability Duration', fontsize=13, fontweight='bold')
    ax_dur.legend(fontsize=9)
    ax_dur.grid(axis='y', alpha=0.3)

    st.pyplot(fig_dur, use_container_width=True)

    # Interpretation text
    if abs(optimal_gap) < abs(current_gap):
        st.success(
            f"✅ **Improved Duration Matching**: The optimal portfolio reduces the duration gap "
            f"from {abs(current_gap):.2f} years to {abs(optimal_gap):.2f} years, "
            f"improving interest rate risk management."
        )
    elif abs(optimal_gap) > abs(current_gap):
        st.warning(
            f"⚠️ **Duration Gap Increased**: The optimal portfolio prioritizes return generation "
            f"over duration matching, increasing the gap from {abs(current_gap):.2f} to "
            f"{abs(optimal_gap):.2f} years. This may increase interest rate risk."
        )
    else:
        st.info(
            f"ℹ️ **Duration Gap Unchanged**: The optimal portfolio maintains a similar duration "
            f"gap of {abs(optimal_gap):.2f} years."
        )

    # Detailed breakdown by asset class
    with st.expander("📊 Duration Contribution by Asset Class"):
        dur_contrib_df = pd.DataFrame({
            "Asset Class": ["Government Bonds", "Corporate Bonds", "Equity Type 1",
                            "Equity Type 2", "Property", "Treasury Bills"],
            "Duration (years)": initial_asset["asset_dur"].values,
            "Current Weight (%)": initial_asset["asset_weight"].values * 100,
            "Current Contribution": initial_asset["asset_dur"].values * initial_asset["asset_weight"].values,
            "Optimal Weight (%)": best["w_opt"] * 100,
            "Optimal Contribution": initial_asset["asset_dur"].values * best["w_opt"]
        })

        st.dataframe(
            dur_contrib_df.style.format({
                "Duration (years)": "{:.2f}",
                "Current Weight (%)": "{:.1f}",
                "Current Contribution": "{:.3f}",
                "Optimal Weight (%)": "{:.1f}",
                "Optimal Contribution": "{:.3f}"
            }),
            use_container_width=True
        )

        st.caption("Note: Duration contribution = Duration × Weight. Sum of contributions = Portfolio Duration.")
    st.markdown("---")
    st.markdown("---")

    # Sensitivity Analysis
    st.subheader("🔬 Sensitivity Analysis")

    st.markdown("""
        Explore how the optimal portfolio allocation changes under different scenarios.
        Adjust the parameters below to see the impact on asset allocation and risk metrics.
        """)

    # Create tabs for different sensitivity analyses
    sens_tab1, sens_tab2, sens_tab3 = st.tabs([
        "📊 Return Scenarios",
        "⚡ Shock Scenarios",
        "🎯 Custom Scenario"
    ])

    # ==========================================
    # TAB 1: Return Scenarios
    # ==========================================
    with sens_tab1:
        st.markdown("**Test different return assumptions for asset classes**")

        scenario_choice = st.selectbox(
            "Select Scenario",
            [
                "Base Case (Current)",
                "Optimistic (+0.5% all assets)",
                "Pessimistic (-0.5% all assets)",
                "Higher Equity Returns (+2.0%)",  # ✅ Changed label
                "Lower Bond Returns (-1.0%)",  # ✅ Changed label
                "Custom Returns"
            ],
            key="return_scenario"
        )

        # Get current returns
        base_returns = initial_asset["asset_ret"].values.copy()

        # Apply scenario
        if scenario_choice == "Base Case (Current)":
            scenario_returns = base_returns
        elif scenario_choice == "Optimistic (+0.5% all assets)":
            scenario_returns = base_returns + 0.005
        elif scenario_choice == "Pessimistic (-0.5% all assets)":
            scenario_returns = base_returns - 0.005
        elif scenario_choice == "Higher Equity Returns (+1.0%)":
            scenario_returns = base_returns.copy()
            scenario_returns[2] += 0.02  # ✅ Changed from 0.01 to 0.02 (Equity Type 1)
            scenario_returns[3] += 0.02  # ✅ Changed from 0.01 to 0.02 (Equity Type 2)
        elif scenario_choice == "Lower Bond Returns (-0.5%)":
            scenario_returns = base_returns.copy()
            scenario_returns[0] -= 0.01  # ✅ Changed from 0.005 to 0.01 (Gov bonds)
            scenario_returns[1] -= 0.01  # ✅ Changed from 0.005 to 0.01 (Corp bonds)
        else:  # Custom Returns
            col1, col2, col3 = st.columns(3)
            with col1:
                r_gov_sens = st.number_input("Gov Bonds", 0.0, 0.20, float(base_returns[0]), 0.001, format="%.3f",
                                             key="sens_r_gov")
                r_corp_sens = st.number_input("Corp Bonds", 0.0, 0.20, float(base_returns[1]), 0.001, format="%.3f",
                                              key="sens_r_corp")
            with col2:
                r_eq1_sens = st.number_input("Equity Type 1", 0.0, 0.20, float(base_returns[2]), 0.001, format="%.3f",
                                             key="sens_r_eq1")
                r_eq2_sens = st.number_input("Equity Type 2", 0.0, 0.20, float(base_returns[3]), 0.001, format="%.3f",
                                             key="sens_r_eq2")
            with col3:
                r_prop_sens = st.number_input("Property", 0.0, 0.20, float(base_returns[4]), 0.001, format="%.3f",
                                              key="sens_r_prop")
                r_tb_sens = st.number_input("T-Bills", 0.0, 0.20, float(base_returns[5]), 0.001, format="%.3f",
                                            key="sens_r_tb")
            scenario_returns = np.array([r_gov_sens, r_corp_sens, r_eq1_sens, r_eq2_sens, r_prop_sens, r_tb_sens])

        if st.button("🔄 Run Return Sensitivity", key="run_return_sens"):
            with st.spinner("Running sensitivity analysis..."):
                try:
                    # Create modified asset DataFrame
                    sens_asset = initial_asset.copy()
                    sens_asset["asset_ret"] = scenario_returns

                    # Get parameters from session state (use same allocation limits and shocks)
                    cfg = load_config()
                    corr_down, corr_up = get_corr_matrices(cfg)
                    solv = get_solvency_params(cfg)

                    # Build allocation limits from initial session
                    allocation_limits = pd.DataFrame({
                        "asset": ["gov_bond", "illiquid_assets", "t_bills", "corp_bond"],
                        "min_weight": [0.25, 0.0, 0.01, 0.0],
                        "max_weight": [0.75, 0.20, 0.05, 0.50],
                    }).set_index("asset")

                    params = {
                        "interest_down": 0.009,
                        "interest_up": 0.011,
                        "spread": 0.103,
                        "equity_type1": solv["equity_1_param"],
                        "equity_type2": solv["equity_2_param"],
                        "property": solv["prop_params"],
                        "rho": solv["rho"],
                    }

                    # Run optimization
                    sens_opt_df = solve_frontier_combined(
                        initial_asset=sens_asset,
                        liab_value=liab_value,
                        liab_duration=liab_duration,
                        corr_downward=corr_down,
                        corr_upward=corr_up,
                        allocation_limits=allocation_limits,
                        params=params
                    )

                    # Get best result
                    sens_best_idx = sens_opt_df["objective"].idxmax()
                    sens_best = sens_opt_df.loc[sens_best_idx]

                    # Compare with base case
                    st.success("✓ Sensitivity analysis completed!")

                    st.markdown("**Comparison: Base Case vs. Scenario**")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        base_ret = best["return"]
                        sens_ret = sens_best["return"]
                        st.metric(
                            "Expected Return",
                            f"{sens_ret:.2%}",
                            delta=f"{(sens_ret - base_ret):.2%}"
                        )
                    with col2:
                        base_scr = best["SCR_market"]
                        sens_scr = sens_best["SCR_market"]
                        st.metric(
                            "SCR Market",
                            f"€{sens_scr:.1f}m",
                            delta=f"€{(sens_scr - base_scr):.1f}m",
                            delta_color="inverse"
                        )
                    with col3:
                        base_solv = best["solvency"]
                        sens_solv = sens_best["solvency"]
                        st.metric(
                            "Solvency Ratio",
                            f"{sens_solv * 100:.1f}%",
                            delta=f"{(sens_solv - base_solv) * 100:.1f}%"
                        )

                    # Allocation comparison
                    st.markdown("**Asset Allocation Changes**")

                    comparison_df = pd.DataFrame({
                        "Asset Class": ["Gov Bonds", "Corp Bonds", "Equity Type 1",
                                        "Equity Type 2", "Property", "T-Bills"],
                        "Base Case (%)": best["w_opt"] * 100,
                        "Scenario (%)": sens_best["w_opt"] * 100,
                        "Change (pp)": (sens_best["w_opt"] - best["w_opt"]) * 100
                    })

                    st.dataframe(
                        comparison_df.style.format({
                            "Base Case (%)": "{:.1f}",
                            "Scenario (%)": "{:.1f}",
                            "Change (pp)": "{:+.1f}"
                        }).background_gradient(subset=["Change (pp)"], cmap="RdYlGn", vmin=-10, vmax=10),
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"Error in sensitivity analysis: {str(e)}")
                    st.exception(e)

    # ==========================================
    # TAB 2: Shock Scenarios
    # ==========================================
    with sens_tab2:
        st.markdown("**Test different Solvency II shock assumptions**")

        shock_scenario = st.selectbox(
            "Select Shock Scenario",
            [
                "Base Case (Current)",
                "Stressed Shocks (+50%)",  # ✅ Changed label
                "Relaxed Shocks (-30%)",  # ✅ Changed label
                "Higher Equity Shocks (+15pp)",  # ✅ Changed label
                "Lower Interest Rate Shocks (-20%)",
                "Custom Shocks"
            ],
            key="shock_scenario"
        )

        # Base shocks
        base_shocks = {
            "ir_up": 0.011,
            "ir_down": 0.009,
            "eq1": 0.39,
            "eq2": 0.49,
            "prop": 0.25,
            "spread": 0.103
        }

        # Apply scenario
        if shock_scenario == "Base Case (Current)":
            scenario_shocks = base_shocks.copy()
        elif shock_scenario == "Stressed Shocks (+20%)":
            scenario_shocks = {k: v * 1.5 for k, v in base_shocks.items()}  # ✅ Changed to 1.5 (50% increase)
        elif shock_scenario == "Relaxed Shocks (-20%)":
            scenario_shocks = {k: v * 0.7 for k, v in base_shocks.items()}  # ✅ Changed to 0.7 (30% decrease)
        elif shock_scenario == "Higher Equity Shocks (+10pp)":
            scenario_shocks = base_shocks.copy()
            scenario_shocks["eq1"] = min(0.60, base_shocks["eq1"] + 0.15)  # ✅ Changed to +15pp
            scenario_shocks["eq2"] = min(0.70, base_shocks["eq2"] + 0.15)  # ✅ Changed to +15pp
        elif shock_scenario == "Lower Interest Rate Shocks (-20%)":
            scenario_shocks = base_shocks.copy()
            scenario_shocks["ir_up"] *= 0.8
            scenario_shocks["ir_down"] *= 0.8
        else:  # Custom Shocks
            col1, col2 = st.columns(2)
            with col1:
                ir_up_shock = st.number_input("IR Up", 0.0, 0.05, base_shocks["ir_up"], 0.001, format="%.3f",
                                              key="sens_ir_up")
                ir_down_shock = st.number_input("IR Down", 0.0, 0.05, base_shocks["ir_down"], 0.001, format="%.3f",
                                                key="sens_ir_down")
                spread_shock = st.number_input("Spread", 0.0, 0.30, base_shocks["spread"], 0.001, format="%.3f",
                                               key="sens_spread")
            with col2:
                eq1_shock = st.number_input("Equity Type 1", 0.0, 1.0, base_shocks["eq1"], 0.01, format="%.2f",
                                            key="sens_eq1")
                eq2_shock = st.number_input("Equity Type 2", 0.0, 1.0, base_shocks["eq2"], 0.01, format="%.2f",
                                            key="sens_eq2")
                prop_shock = st.number_input("Property", 0.0, 1.0, base_shocks["prop"], 0.01, format="%.2f",
                                             key="sens_prop")
            scenario_shocks = {
                "ir_up": ir_up_shock,
                "ir_down": ir_down_shock,
                "eq1": eq1_shock,
                "eq2": eq2_shock,
                "prop": prop_shock,
                "spread": spread_shock
            }

        if st.button("🔄 Run Shock Sensitivity", key="run_shock_sens"):
            with st.spinner("Running shock sensitivity analysis..."):
                try:
                    cfg = load_config()
                    corr_down, corr_up = get_corr_matrices(cfg)
                    solv = get_solvency_params(cfg)

                    allocation_limits = pd.DataFrame({
                        "asset": ["gov_bond", "illiquid_assets", "t_bills", "corp_bond"],
                        "min_weight": [0.25, 0.0, 0.01, 0.0],
                        "max_weight": [0.75, 0.20, 0.05, 0.50],
                    }).set_index("asset")

                    params = {
                        "interest_down": scenario_shocks["ir_down"],
                        "interest_up": scenario_shocks["ir_up"],
                        "spread": scenario_shocks["spread"],
                        "equity_type1": scenario_shocks["eq1"],
                        "equity_type2": scenario_shocks["eq2"],
                        "property": scenario_shocks["prop"],
                        "rho": solv["rho"],
                    }

                    sens_opt_df = solve_frontier_combined(
                        initial_asset=initial_asset,
                        liab_value=liab_value,
                        liab_duration=liab_duration,
                        corr_downward=corr_down,
                        corr_upward=corr_up,
                        allocation_limits=allocation_limits,
                        params=params
                    )

                    sens_best_idx = sens_opt_df["objective"].idxmax()
                    sens_best = sens_opt_df.loc[sens_best_idx]

                    st.success("✓ Shock sensitivity analysis completed!")

                    st.markdown("**Impact of Changed Shocks**")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        base_ret = best["return"]
                        sens_ret = sens_best["return"]
                        st.metric(
                            "Expected Return",
                            f"{sens_ret:.2%}",
                            delta=f"{(sens_ret - base_ret):.2%}"
                        )
                    with col2:
                        base_scr = best["SCR_market"]
                        sens_scr = sens_best["SCR_market"]
                        st.metric(
                            "SCR Market",
                            f"€{sens_scr:.1f}m",
                            delta=f"€{(sens_scr - base_scr):.1f}m",
                            delta_color="inverse"
                        )
                    with col3:
                        base_solv = best["solvency"]
                        sens_solv = sens_best["solvency"]
                        st.metric(
                            "Solvency Ratio",
                            f"{sens_solv * 100:.1f}%",
                            delta=f"{(sens_solv - base_solv) * 100:.1f}%"
                        )

                    st.markdown("**Asset Allocation Response**")

                    comparison_df = pd.DataFrame({
                        "Asset Class": ["Gov Bonds", "Corp Bonds", "Equity Type 1",
                                        "Equity Type 2", "Property", "T-Bills"],
                        "Base Shocks (%)": best["w_opt"] * 100,
                        "Scenario Shocks (%)": sens_best["w_opt"] * 100,
                        "Change (pp)": (sens_best["w_opt"] - best["w_opt"]) * 100
                    })

                    st.dataframe(
                        comparison_df.style.format({
                            "Base Shocks (%)": "{:.1f}",
                            "Scenario Shocks (%)": "{:.1f}",
                            "Change (pp)": "{:+.1f}"
                        }).background_gradient(subset=["Change (pp)"], cmap="RdYlGn", vmin=-10, vmax=10),
                        use_container_width=True
                    )

                    st.info(
                        "💡 **Interpretation**: Higher shocks increase capital requirements for affected "
                        "asset classes, making them less attractive. The optimizer responds by shifting "
                        "allocation toward assets with relatively lower shocks."
                    )

                except Exception as e:
                    st.error(f"Error in shock sensitivity: {str(e)}")
                    st.exception(e)

    # ==========================================
    # TAB 3: Custom Scenario
    # ==========================================
    with sens_tab3:
        st.markdown("**Create your own custom scenario**")
        st.markdown("Combine different return and shock assumptions to test specific hypotheses.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📊 Custom Returns**")
            custom_r_gov = st.number_input("Gov Bonds Return", 0.0, 0.20, 0.029, 0.001, format="%.3f",
                                           key="custom_r_gov")
            custom_r_corp = st.number_input("Corp Bonds Return", 0.0, 0.20, 0.041, 0.001, format="%.3f",
                                            key="custom_r_corp")
            custom_r_eq1 = st.number_input("Equity 1 Return", 0.0, 0.20, 0.064, 0.001, format="%.3f",
                                           key="custom_r_eq1")
            custom_r_eq2 = st.number_input("Equity 2 Return", 0.0, 0.20, 0.064, 0.001, format="%.3f",
                                           key="custom_r_eq2")
            custom_r_prop = st.number_input("Property Return", 0.0, 0.20, 0.056, 0.001, format="%.3f",
                                            key="custom_r_prop")
            custom_r_tb = st.number_input("T-Bills Return", 0.0, 0.20, 0.006, 0.001, format="%.3f", key="custom_r_tb")

        with col2:
            st.markdown("**⚡ Custom Shocks**")
            custom_ir_up = st.number_input("IR Up Shock", 0.0, 0.05, 0.011, 0.001, format="%.3f", key="custom_ir_up")
            custom_ir_down = st.number_input("IR Down Shock", 0.0, 0.05, 0.009, 0.001, format="%.3f",
                                             key="custom_ir_down")
            custom_eq1 = st.number_input("Equity 1 Shock", 0.0, 1.0, 0.39, 0.01, format="%.2f", key="custom_eq1")
            custom_eq2 = st.number_input("Equity 2 Shock", 0.0, 1.0, 0.49, 0.01, format="%.2f", key="custom_eq2")
            custom_prop = st.number_input("Property Shock", 0.0, 1.0, 0.25, 0.01, format="%.2f", key="custom_prop")
            custom_spread = st.number_input("Spread Shock", 0.0, 0.30, 0.103, 0.001, format="%.3f", key="custom_spread")

        if st.button("🚀 Run Custom Scenario", key="run_custom_sens", type="primary"):
            with st.spinner("Running custom scenario analysis..."):
                try:
                    # Build custom asset and params
                    custom_asset = initial_asset.copy()
                    custom_asset["asset_ret"] = [custom_r_gov, custom_r_corp, custom_r_eq1,
                                                 custom_r_eq2, custom_r_prop, custom_r_tb]

                    cfg = load_config()
                    corr_down, corr_up = get_corr_matrices(cfg)
                    solv = get_solvency_params(cfg)

                    allocation_limits = pd.DataFrame({
                        "asset": ["gov_bond", "illiquid_assets", "t_bills", "corp_bond"],
                        "min_weight": [0.25, 0.0, 0.01, 0.0],
                        "max_weight": [0.75, 0.20, 0.05, 0.50],
                    }).set_index("asset")

                    custom_params = {
                        "interest_down": custom_ir_down,
                        "interest_up": custom_ir_up,
                        "spread": custom_spread,
                        "equity_type1": custom_eq1,
                        "equity_type2": custom_eq2,
                        "property": custom_prop,
                        "rho": solv["rho"],
                    }

                    custom_opt_df = solve_frontier_combined(
                        initial_asset=custom_asset,
                        liab_value=liab_value,
                        liab_duration=liab_duration,
                        corr_downward=corr_down,
                        corr_upward=corr_up,
                        allocation_limits=allocation_limits,
                        params=custom_params
                    )

                    custom_best_idx = custom_opt_df["objective"].idxmax()
                    custom_best = custom_opt_df.loc[custom_best_idx]

                    st.success("✓ Custom scenario completed!")

                    # Full comparison table
                    st.markdown("**Comprehensive Comparison**")

                    metrics_comparison = pd.DataFrame({
                        "Metric": ["Expected Return", "SCR Market (€m)", "Solvency Ratio (%)", "BOF (€m)"],
                        "Base Case": [
                            f"{best['return']:.2%}",
                            f"{best['SCR_market']:.1f}",
                            f"{best['solvency'] * 100:.1f}",
                            f"{best['BOF']:.1f}"
                        ],
                        "Custom Scenario": [
                            f"{custom_best['return']:.2%}",
                            f"{custom_best['SCR_market']:.1f}",
                            f"{custom_best['solvency'] * 100:.1f}",
                            f"{custom_best['BOF']:.1f}"
                        ]
                    })

                    st.dataframe(metrics_comparison, use_container_width=True, hide_index=True)

                    st.markdown("**Allocation Comparison**")

                    alloc_comparison = pd.DataFrame({
                        "Asset": ["Gov Bonds", "Corp Bonds", "Equity Type 1",
                                  "Equity Type 2", "Property", "T-Bills"],
                        "Base (%)": best["w_opt"] * 100,
                        "Custom (%)": custom_best["w_opt"] * 100,
                        "Δ (pp)": (custom_best["w_opt"] - best["w_opt"]) * 100
                    })

                    st.dataframe(
                        alloc_comparison.style.format({
                            "Base (%)": "{:.1f}",
                            "Custom (%)": "{:.1f}",
                            "Δ (pp)": "{:+.1f}"
                        }).background_gradient(subset=["Δ (pp)"], cmap="RdYlGn", vmin=-20, vmax=20),
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"Error in custom scenario: {str(e)}")
                    st.exception(e)
    # Efficient frontier plot
    st.subheader("📉 Efficient Frontier")

    fig, ax = plot_frontier(
        opt_df,
        current_sol=current_sol,
        current_ret=current_ret,
        min_sol_pct=90,
        title="Efficient Frontier: Return vs Solvency",
        show=False
    )
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # Optimal allocation
    st.subheader("💼 Optimal Portfolio Allocation")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Amounts (€ millions)**")
        A_opt = best["A_opt"]
        allocation_df = pd.DataFrame({
            "Asset Class": ["Government Bonds", "Corporate Bonds", "Equity Type 1",
                            "Equity Type 2", "Property", "Treasury Bills"],
            "Current (€m)": initial_asset["asset_val"].values,
            "Optimal (€m)": A_opt,
            "Change (€m)": A_opt - initial_asset["asset_val"].values
        })
        st.dataframe(allocation_df.style.format({
            "Current (€m)": "{:.1f}",
            "Optimal (€m)": "{:.1f}",
            "Change (€m)": "{:+.1f}"
        }), use_container_width=True)

    with col_right:
        st.markdown("**Weights (%)**")
        w_opt = best["w_opt"]
        weights_df = pd.DataFrame({
            "Asset Class": ["Government Bonds", "Corporate Bonds", "Equity Type 1",
                            "Equity Type 2", "Property", "Treasury Bills"],
            "Current (%)": initial_asset["asset_weight"].values * 100,
            "Optimal (%)": w_opt * 100,
            "Change (%)": (w_opt - initial_asset["asset_weight"].values) * 100
        })
        st.dataframe(weights_df.style.format({
            "Current (%)": "{:.1f}",
            "Optimal (%)": "{:.1f}",
            "Change (%)": "{:+.1f}"
        }), use_container_width=True)
    st.markdown("---")

    # Pie charts comparison
    st.markdown("**Visual Allocation Comparison**")

    fig_pie, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    asset_labels = ["Gov Bonds", "Corp Bonds", "Equity Type 1",
                    "Equity Type 2", "Property", "T-Bills"]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9']


    # Helper function to format labels
    def make_autopct(values):
        def my_autopct(pct):
            return f'{pct:.1f}%' if pct > 3 else ''  # Only show % if slice > 3%

        return my_autopct


    # Current allocation pie chart
    current_weights = initial_asset["asset_weight"].values * 100

    # Keep all data but control label display
    wedges1, texts1, autotexts1 = ax1.pie(
        current_weights,
        labels=None,  # Remove labels from pie
        colors=colors,
        autopct=make_autopct(current_weights),
        startangle=90,
        textprops={'fontsize': 10, 'weight': 'bold'},
        pctdistance=0.85
    )

    # Make percentage text more visible
    for autotext in autotexts1:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_weight('bold')

    ax1.set_title('Current Portfolio', fontsize=13, fontweight='bold', pad=20)

    # Optimal allocation pie chart
    optimal_weights = best["w_opt"] * 100

    wedges2, texts2, autotexts2 = ax2.pie(
        optimal_weights,
        labels=None,  # Remove labels from pie
        colors=colors,
        autopct=make_autopct(optimal_weights),
        startangle=90,
        textprops={'fontsize': 10, 'weight': 'bold'},
        pctdistance=0.85
    )

    for autotext in autotexts2:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_weight('bold')

    ax2.set_title('Optimal Portfolio', fontsize=13, fontweight='bold', pad=20)

    # Add a shared legend below the charts
    legend_labels = [f"{label}: {weight:.1f}%" for label, weight in zip(asset_labels, current_weights)]
    fig_pie.legend(
        wedges1,
        legend_labels,
        title="Asset Classes (Current)",
        loc="lower left",
        bbox_to_anchor=(0.05, -0.05),
        ncol=3,
        fontsize=9
    )

    legend_labels_opt = [f"{label}: {weight:.1f}%" for label, weight in zip(asset_labels, optimal_weights)]
    fig_pie.legend(
        wedges2,
        legend_labels_opt,
        title="Asset Classes (Optimal)",
        loc="lower right",
        bbox_to_anchor=(0.95, -0.05),
        ncol=3,
        fontsize=9
    )

    plt.tight_layout()
    st.pyplot(fig_pie, use_container_width=True)

    # Key changes summary
    st.markdown("**Key Allocation Changes:**")

    changes = []
    for i, label in enumerate(asset_labels):
        current_w = initial_asset["asset_weight"].values[i] * 100
        optimal_w = best["w_opt"][i] * 100
        change = optimal_w - current_w

        if abs(change) > 1.0:  # Only show significant changes
            direction = "↑" if change > 0 else "↓"
            color = "🟢" if change > 0 else "🔴"
            changes.append(
                f"{color} **{label}**: {direction} {abs(change):.1f}pp ({current_w:.1f}% → {optimal_w:.1f}%)")

    if changes:
        for change in changes:
            st.markdown(change)
    else:
        st.info("No significant allocation changes (< 1 percentage point)")
    st.markdown("---")

    # Frontier table
    with st.expander("📋 View Full Frontier Data"):
        view_df = opt_df[["gamma", "return", "SCR_market", "solvency", "objective"]].copy()
        view_df["return"] = view_df["return"] * 100
        view_df["solvency"] = view_df["solvency"] * 100

        st.dataframe(
            view_df.style.format({
                "gamma": "{:.4f}",
                "return": "{:.2f}%",
                "SCR_market": "€{:.1f}m",
                "solvency": "{:.1f}%",
                "objective": "{:.4f}"
            }),
            use_container_width=True
        )