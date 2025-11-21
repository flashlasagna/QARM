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
import json
import openpyxl
from io import BytesIO

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
    ["Inputs", "Results", "Interactive Portfolio Selector"],
    index=["Inputs", "Results", "Interactive Portfolio Selector"].index(st.session_state["nav"])
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
elif st.session_state["nav"] == "Results":
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
    # --- MARGINAL SCR ANALYSIS SECTION ---

st.subheader("🧮 Marginal SCR by Asset")

try:
    # Prepare SCR inputs using current portfolio
    # 1. Run "component SCR" for current portfolio
    from backend.config_loader import load_config, get_corr_matrices, get_solvency_params

    cfg = load_config()
    corr_down, corr_up = get_corr_matrices(cfg)
    solv = get_solvency_params(cfg)

    # Prepare SCR for each risk type (interest, equity, property, spread)
    from backend.solvency_calc import (
        scr_interest_rate, scr_eq, scr_prop, scr_sprd,
        aggregate_market_scr, marginal_scr, allocate_marginal_scr
    )

    # Get asset values and durations
    asset_values = initial_asset['asset_val']
    asset_durs = initial_asset['asset_dur']
    liab_value = st.session_state["liab_value"]
    liab_duration = st.session_state.get("liab_duration", 6.6)
    # Use params consistent with backend build process
    params = {
        "interest_down": best.get("interest_down", 0.009),
        "interest_up": best.get("interest_up", 0.011),
        "spread": best.get("spread", 0.103),
        "equity_type1": solv["equity_1_param"],
        "equity_type2": solv["equity_2_param"],
        "property": solv["prop_params"],
        "rho": solv["rho"],
    }
    
    # Component SCRs (same as in notebook)
    scr_interest = scr_interest_rate(
        asset_values,
        asset_durs,
        liab_value,
        liab_duration,
        params["interest_up"],
        params["interest_down"])
    scr_equity = scr_eq(
        asset_values.iloc[2],
        asset_values.iloc[3],
        params["equity_type1"],
        params["equity_type2"],
        params["rho"])
    scr_property = scr_prop(asset_values.iloc[4], params["property"])
    scr_spread = scr_sprd(asset_values.iloc[1], params["spread"])

    # SCR vector
    scr_vec = np.array([
        scr_interest["SCR_interest"],
        scr_equity["SCR_eq_total"],
        scr_property,
        scr_spread
    ])

    # Portfolio covariance used here for marginal SCR by risk type
    marginal_df = marginal_scr(
        v=scr_vec,
        direction=best.get("chosen_panel", "downward"),  # Or use pre-configured
        corr_downward=corr_down,
        corr_upward=corr_up
    )

    # Asset-level marginal SCR allocation
    asset_mSCR = allocate_marginal_scr(
        marginal_df,
        best.get("chosen_panel", "downward"),
        initial_asset,
        params
    )

    # For better UX, map keys to user-friendly names
    ASSET_LABELS = ["Government Bonds", "Corporate Bonds",
                    "Equity Type 1", "Equity Type 2",
                    "Property", "Treasury Bills"]
    asset_mSCR["Asset Class"] = ASSET_LABELS
    asset_mSCR = asset_mSCR[["Asset Class", "mSCR"]]

    # Show as table
    st.dataframe(
        asset_mSCR.style.format({"mSCR": "{:+.4f}"}),
        use_container_width=True,
        hide_index=True
    )

    # Optional: horizontal bar chart
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9']
    ax.barh(asset_mSCR["Asset Class"], asset_mSCR["mSCR"], color=colors)
    ax.set_xlabel("Marginal SCR Contribution")
    ax.set_title("Marginal SCR by Asset Class")
    ax.axvline(0, color="grey", linewidth=1)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.caption("Note: Marginal SCR represents the additional Solvency Capital Requirement from increasing exposure to each asset class, taking diversification into account. Units: proportional to portfolio size.")

except Exception as ex:
    st.error(f"Marginal SCR computation failed: {ex}")
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
    st.markdown("---")

    # Efficient frontier plot
    st.subheader("📉 Efficient Frontier")

    fig_frontier, ax_frontier = plt.subplots(figsize=(12, 7))

    # Modern styling
    ax_frontier.set_facecolor('#f8f9fa')
    fig_frontier.patch.set_facecolor('white')

    # Plot the efficient frontier line with simple styling
    ax_frontier.scatter(
        opt_df["solvency"] * 100,
        opt_df["return"] * 100,
        s=80,
        color='#4ECDC4',
        alpha=0.6,
        edgecolors='white',
        linewidth=1.5,
        zorder=2
    )

    # Connect with line
    ax_frontier.plot(
        opt_df["solvency"] * 100,
        opt_df["return"] * 100,
        '-',
        color='#4ECDC4',
        linewidth=2.5,
        alpha=0.4,
        label='Efficient Frontier',
        zorder=1
    )

    # Highlight the OPTIMAL portfolio with glow
    optimal_solvency = best["solvency"] * 100
    optimal_return = best["return"] * 100

    # Outer glow
    ax_frontier.scatter(
        optimal_solvency, optimal_return,
        s=1000, c='#FFD700', marker='*',
        alpha=0.2, edgecolors='none', zorder=4
    )
    # Middle glow
    ax_frontier.scatter(
        optimal_solvency, optimal_return,
        s=700, c='#FFD700', marker='*',
        alpha=0.4, edgecolors='none', zorder=4
    )
    # Main star
    ax_frontier.scatter(
        optimal_solvency, optimal_return,
        s=600, c='#FFD700', marker='*',
        edgecolors='#FF8C00', linewidth=3,
        label='Optimal Portfolio',
        zorder=5
    )

    # Add annotation with modern styling
    ax_frontier.annotate(
        f'OPTIMAL\n{optimal_return:.2f}% | {optimal_solvency:.1f}%',
        xy=(optimal_solvency, optimal_return),
        xytext=(25, 25),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.8',
            facecolor='#FFD700',
            edgecolor='#FF8C00',
            linewidth=2.5,
            alpha=0.9
        ),
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle='arc3,rad=0.3',
            color='#FF8C00',
            lw=2.5
        ),
        color='#2c3e50',
        zorder=6
    )

    # Plot the CURRENT portfolio with better styling
    ax_frontier.scatter(
        current_sol * 100, current_ret * 100,
        s=400, c='#E74C3C', marker='D',
        edgecolors='#C0392B', linewidth=3,
        label='Current Portfolio', zorder=4,
        alpha=0.9
    )

    # Annotation for current
    ax_frontier.annotate(
        f'Current\n{current_ret * 100:.2f}% | {current_sol * 100:.1f}%',
        xy=(current_sol * 100, current_ret * 100),
        xytext=(-70, -35),
        textcoords='offset points',
        fontsize=9,
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.6',
            facecolor='#E74C3C',
            edgecolor='#C0392B',
            linewidth=2,
            alpha=0.8
        ),
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle='arc3,rad=-0.3',
            color='#C0392B',
            lw=2
        ),
        color='white',
        zorder=6
    )

    # Add 100% solvency line
    ax_frontier.axvline(
        x=100,
        color='#95a5a6',
        linestyle='--',
        linewidth=2.5,
        alpha=0.6,
        label='100% Solvency',
        zorder=0
    )

    # Modern styling
    ax_frontier.set_xlabel('Solvency Ratio (%)', fontsize=12, fontweight='bold', color='#2c3e50')
    ax_frontier.set_ylabel('Expected Return (%)', fontsize=12, fontweight='bold', color='#2c3e50')
    ax_frontier.set_title('Efficient Frontier: Return vs Solvency', fontsize=15, fontweight='bold', pad=20,
                          color='#2c3e50')
    ax_frontier.grid(True, alpha=0.25, linestyle='--', linewidth=1, color='gray')

    # ✅ FIXED LEGEND - top right, smaller markers
    # ✅ LARGER LEGEND BOX with better spacing
    legend = ax_frontier.legend(
        loc='upper right',
        fontsize=10,  # ✅ Increased from 9 to 10
        frameon=True,
        shadow=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor='gray',
        ncol=1,
        borderpad=1.2,  # ✅ Increased from 0.8 to 1.2 (more internal padding)
        labelspacing=1.0,  # ✅ Increased from 0.6 to 1.0 (more space between items)
        handlelength=2.5,  # ✅ Longer marker lines
        handleheight=1.5,  # ✅ Taller marker area
        markerscale=0.7,  # ✅ Slightly larger markers (was 0.6)
        handletextpad=1.0,  # ✅ More space between marker and text
        borderaxespad=0.8  # ✅ Distance from plot edge
    )


    # Remove top and right spines
    ax_frontier.spines['top'].set_visible(False)
    ax_frontier.spines['right'].set_visible(False)
    ax_frontier.spines['left'].set_linewidth(1.5)
    ax_frontier.spines['bottom'].set_linewidth(1.5)

    # Set axis limits with padding
    x_min, x_max = opt_df["solvency"].min() * 100, opt_df["solvency"].max() * 100
    y_min, y_max = opt_df["return"].min() * 100, opt_df["return"].max() * 100

    x_padding = (x_max - x_min) * 0.12
    y_padding = (y_max - y_min) * 0.15

    ax_frontier.set_xlim(x_min - x_padding, x_max + x_padding)
    ax_frontier.set_ylim(y_min - y_padding, y_max + y_padding)

    plt.tight_layout()
    st.pyplot(fig_frontier, use_container_width=True)

    # Add interpretation text
    improvement_return = (optimal_return - current_ret * 100)
    improvement_solvency = (optimal_solvency - current_sol * 100)

    if improvement_return > 0 and improvement_solvency > 0:
        st.success(
            f"✅ **Win-Win Optimization**: The optimal portfolio improves both metrics:\n"
            f"- Return: +{improvement_return:.2f}pp ({current_ret * 100:.2f}% → {optimal_return:.2f}%)\n"
            f"- Solvency: +{improvement_solvency:.1f}pp ({current_sol * 100:.1f}% → {optimal_solvency:.1f}%)"
        )
    elif improvement_return > 0:
        st.info(
            f"📊 **Return Focus**: The optimal portfolio trades some solvency for higher returns:\n"
            f"- Return: +{improvement_return:.2f}pp\n"
            f"- Solvency: {improvement_solvency:.1f}pp"
        )
    elif improvement_solvency > 0:
        st.info(
            f"🛡️ **Safety Focus**: The optimal portfolio prioritizes solvency:\n"
            f"- Return: {improvement_return:.2f}pp\n"
            f"- Solvency: +{improvement_solvency:.1f}pp"
        )
    else:
        st.warning(
            f"⚠️ The current portfolio is already near-optimal on the efficient frontier."
        )

    st.markdown("---")

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

    fig_pie, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Modern styling
    fig_pie.patch.set_facecolor('white')

    asset_labels = ["Gov Bonds", "Corp Bonds", "Equity Type 1",
                    "Equity Type 2", "Property", "T-Bills"]

    # Modern color palette (more vibrant)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9']


    def make_autopct(values):
        def my_autopct(pct):
            return f'{pct:.1f}%' if pct > 3 else ''

        return my_autopct


    # Current allocation pie chart
    current_weights = initial_asset["asset_weight"].values * 100

    wedges1, texts1, autotexts1 = ax1.pie(
        current_weights,
        labels=None,
        colors=colors,
        autopct=make_autopct(current_weights),
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'},
        pctdistance=0.80,
        explode=[0.03 if w > 10 else 0 for w in current_weights],  # Slight explode for large slices
        shadow=True,  # Add shadow
        wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'antialiased': True}
    )

    # Style percentage labels
    for autotext in autotexts1:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
        autotext.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3))

    ax1.set_title('Current Portfolio', fontsize=14, fontweight='bold', pad=25, color='#2c3e50')

    # Optimal allocation pie chart
    optimal_weights = best["w_opt"] * 100

    wedges2, texts2, autotexts2 = ax2.pie(
        optimal_weights,
        labels=None,
        colors=colors,
        autopct=make_autopct(optimal_weights),
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'},
        pctdistance=0.80,
        explode=[0.03 if w > 10 else 0 for w in optimal_weights],
        shadow=True,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'antialiased': True}
    )

    for autotext in autotexts2:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
        autotext.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3))

    ax2.set_title('Optimal Portfolio', fontsize=14, fontweight='bold', pad=25, color='#2c3e50')

    # Modern legend
    fig_pie.legend(
        wedges1,
        [f"{label}" for label in asset_labels],
        title="Asset Classes",
        title_fontsize=11,
        fontsize=10,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=6,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        edgecolor='gray'
    )

    plt.tight_layout()
    st.pyplot(fig_pie, use_container_width=True)

    st.markdown("---")

    # Key changes with better formatting
    st.markdown("**Key Allocation Changes:**")

    changes = []
    for i, label in enumerate(asset_labels):
        current_w = initial_asset["asset_weight"].values[i] * 100
        optimal_w = best["w_opt"][i] * 100
        change = optimal_w - current_w

        if abs(change) > 1.0:
            if change > 0:
                changes.append(f"🟢 **{label}**: ↑ {abs(change):.1f}pp ({current_w:.1f}% → {optimal_w:.1f}%)")
            else:
                changes.append(f"🔴 **{label}**: ↓ {abs(change):.1f}pp ({current_w:.1f}% → {optimal_w:.1f}%)")

    if changes:
        cols = st.columns(2)
        mid = len(changes) // 2
        with cols[0]:
            for change in changes[:mid]:
                st.markdown(change)
        with cols[1]:
            for change in changes[mid:]:
                st.markdown(change)
    else:
        st.info("No significant allocation changes (< 1 percentage point)")
    st.markdown("---")

    # Export/Download Section
    st.subheader("💾 Export Results")

    st.markdown("Download your optimization results in various formats for reporting and further analysis.")

    col_export1, col_export2, col_export3 = st.columns(3)

    with col_export1:
        st.markdown("**📊 Allocation Data**")

        # Create allocation export dataframe
        allocation_export = pd.DataFrame({
            "Asset Class": ["Government Bonds", "Corporate Bonds", "Equity Type 1",
                            "Equity Type 2", "Property", "Treasury Bills"],
            "Current Amount (€m)": initial_asset["asset_val"].values,
            "Optimal Amount (€m)": best["A_opt"],
            "Change Amount (€m)": best["A_opt"] - initial_asset["asset_val"].values,
            "Current Weight (%)": initial_asset["asset_weight"].values * 100,
            "Optimal Weight (%)": best["w_opt"] * 100,
            "Change Weight (pp)": (best["w_opt"] - initial_asset["asset_weight"].values) * 100
        })

        csv_allocation = allocation_export.to_csv(index=False)
        st.download_button(
            label="📥 Download Allocation",
            data=csv_allocation,
            file_name=f"optimal_allocation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download optimal vs current allocation as CSV"
        )

    with col_export2:
        st.markdown("**📈 Frontier Data**")

        # Create frontier export dataframe
        frontier_export = opt_df[["gamma", "return", "SCR_market", "solvency", "objective"]].copy()
        frontier_export["return_pct"] = frontier_export["return"] * 100
        frontier_export["solvency_pct"] = frontier_export["solvency"] * 100

        # Add allocations to frontier data
        for i, asset in enumerate(["Gov", "Corp", "Eq1", "Eq2", "Prop", "TB"]):
            frontier_export[f"{asset}_weight_%"] = opt_df["w_opt"].apply(lambda x: x[i] * 100)

        frontier_export = frontier_export[[
            "gamma", "return_pct", "solvency_pct", "SCR_market", "objective",
            "Gov_weight_%", "Corp_weight_%", "Eq1_weight_%", "Eq2_weight_%", "Prop_weight_%", "TB_weight_%"
        ]]

        csv_frontier = frontier_export.to_csv(index=False)
        st.download_button(
            label="📥 Download Frontier",
            data=csv_frontier,
            file_name=f"efficient_frontier_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download complete efficient frontier data"
        )

    with col_export3:
        st.markdown("**📄 Summary Report**")

        # Create comprehensive text report
        summary_text = f"""SOLVENCY II ASSET ALLOCATION OPTIMIZATION REPORT
    {'=' * 60}
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    {'=' * 60}
    PORTFOLIO SUMMARY
    {'=' * 60}
    Total Assets:              €{initial_asset['asset_val'].sum():>12.1f}m
    Best Estimate Liabilities: €{liab_value:>12.1f}m
    Liability Duration:        {liab_duration:>12.2f} years

    {'=' * 60}
    CURRENT PORTFOLIO
    {'=' * 60}
    Expected Return:           {current_ret:>12.2%}
    Solvency Ratio:            {current_sol * 100:>12.1f}%
    SCR Market:                €{current_SCR:>12.1f}m
    Portfolio Duration:        {current_dur:>12.2f} years
    Duration Gap:              {current_gap:>12.2f} years

    {'=' * 60}
    OPTIMAL PORTFOLIO
    {'=' * 60}
    Expected Return:           {best_ret:>12.2%}
    Solvency Ratio:            {best_sol * 100:>12.1f}%
    SCR Market:                €{best_SCR:>12.1f}m
    Basic Own Funds:           €{best_BOF:>12.1f}m
    Portfolio Duration:        {optimal_dur:>12.2f} years
    Duration Gap:              {optimal_gap:>12.2f} years
    Optimization Objective:    {best['objective']:>12.4f}

    {'=' * 60}
    IMPROVEMENTS
    {'=' * 60}
    Return Change:             {(best_ret - current_ret):>12.2%} ({(best_ret - current_ret) * 100:+.2f}pp)
    Solvency Change:           {(best_sol - current_sol) * 100:>12.1f}% ({(best_sol - current_sol) * 100:+.1f}pp)
    SCR Change:                €{(best_SCR - current_SCR):>12.1f}m
    Duration Gap Change:       {(optimal_gap - current_gap):>12.2f} years

    {'=' * 60}
    OPTIMAL ALLOCATION
    {'=' * 60}
    Asset Class              Amount (€m)    Weight (%)    Change (€m)    Change (pp)
    {'-' * 80}
    """

        for i, asset in enumerate(["Government Bonds", "Corporate Bonds", "Equity Type 1",
                                   "Equity Type 2", "Property", "Treasury Bills"]):
            curr_amt = initial_asset["asset_val"].values[i]
            opt_amt = best["A_opt"][i]
            curr_wgt = initial_asset["asset_weight"].values[i] * 100
            opt_wgt = best["w_opt"][i] * 100

            summary_text += f"{asset:<24} {opt_amt:>10.1f}    {opt_wgt:>9.1f}    {(opt_amt - curr_amt):>10.1f}    {(opt_wgt - curr_wgt):>+10.1f}\n"

        summary_text += f"""
    {'=' * 60}
    RISK METRICS
    {'=' * 60}
    SCR Market Risk:           €{best_SCR:>12.1f}m
    Risk Aversion Parameter:   {best['gamma']:>12.4f}
    """

        # Add duration analysis
        summary_text += f"""
    {'=' * 60}
    DURATION ANALYSIS
    {'=' * 60}
    Target (Liability):        {liab_duration:>12.2f} years
    Current Portfolio:         {current_dur:>12.2f} years
    Optimal Portfolio:         {optimal_dur:>12.2f} years
    Gap Improvement:           {abs(current_gap) - abs(optimal_gap):>12.2f} years

    """

        if abs(optimal_gap) < abs(current_gap):
            summary_text += f"✓ Duration matching IMPROVED ({abs(current_gap):.2f}y → {abs(optimal_gap):.2f}y)\n"
        elif abs(optimal_gap) > abs(current_gap):
            summary_text += f"⚠ Duration gap INCREASED for higher returns ({abs(current_gap):.2f}y → {abs(optimal_gap):.2f}y)\n"
        else:
            summary_text += f"→ Duration gap UNCHANGED ({abs(optimal_gap):.2f}y)\n"

        summary_text += f"""
    {'=' * 60}
    NOTES
    {'=' * 60}
    - All amounts in millions of euros (€m)
    - Returns are annualized
    - Duration is modified duration
    - pp = percentage points
    - SCR = Solvency Capital Requirement

    {'=' * 60}
    End of Report
    {'=' * 60}
    """

        st.download_button(
            label="📥 Download Report",
            data=summary_text,
            file_name=f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
            help="Download comprehensive text report"
        )

        # Additional export options in expandable section
        with st.expander("🔧 Advanced Export Options"):

            col_xl1, col_xl2 = st.columns(2)

            with col_xl1:
                st.markdown("**📊 Export as Excel Workbook**")
                st.caption("Comprehensive workbook with multiple sheets")

                # Create Excel file in memory
                output = BytesIO()

                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Sheet 1: Summary
                    summary_df = pd.DataFrame({
                        "Metric": [
                            "Total Assets (€m)",
                            "Liabilities (€m)",
                            "Liability Duration (years)",
                            "",
                            "=== CURRENT PORTFOLIO ===",
                            "Expected Return (%)",
                            "Solvency Ratio (%)",
                            "SCR Market (€m)",
                            "Portfolio Duration (years)",
                            "Duration Gap (years)",
                            "",
                            "=== OPTIMAL PORTFOLIO ===",
                            "Expected Return (%)",
                            "Solvency Ratio (%)",
                            "SCR Market (€m)",
                            "Basic Own Funds (€m)",
                            "Portfolio Duration (years)",
                            "Duration Gap (years)",
                            "Objective Value",
                            "",
                            "=== IMPROVEMENTS ===",
                            "Return Change (pp)",
                            "Solvency Change (pp)",
                            "SCR Change (€m)",
                            "Duration Gap Change (years)"
                        ],
                        "Value": [
                            initial_asset['asset_val'].sum(),
                            liab_value,
                            liab_duration,
                            "",
                            "",
                            current_ret * 100,
                            current_sol * 100,
                            current_SCR,
                            current_dur,
                            current_gap,
                            "",
                            "",
                            best_ret * 100,
                            best_sol * 100,
                            best_SCR,
                            best_BOF,
                            optimal_dur,
                            optimal_gap,
                            best["objective"],
                            "",
                            "",
                            (best_ret - current_ret) * 100,
                            (best_sol - current_sol) * 100,
                            best_SCR - current_SCR,
                            optimal_gap - current_gap
                        ]
                    })
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)

                    # Sheet 2: Allocation Comparison
                    allocation_detail = pd.DataFrame({
                        "Asset Class": ["Government Bonds", "Corporate Bonds", "Equity Type 1",
                                        "Equity Type 2", "Property", "Treasury Bills"],
                        "Current Amount (€m)": initial_asset["asset_val"].values,
                        "Current Weight (%)": initial_asset["asset_weight"].values * 100,
                        "Optimal Amount (€m)": best["A_opt"],
                        "Optimal Weight (%)": best["w_opt"] * 100,
                        "Change Amount (€m)": best["A_opt"] - initial_asset["asset_val"].values,
                        "Change Weight (pp)": (best["w_opt"] - initial_asset["asset_weight"].values) * 100,
                        "Current Return (%)": initial_asset["asset_ret"].values * 100,
                        "Duration (years)": initial_asset["asset_dur"].values
                    })
                    allocation_detail.to_excel(writer, sheet_name='Allocation', index=False)

                    # Sheet 3: Efficient Frontier
                    frontier_detail = opt_df[["gamma", "return", "SCR_market", "solvency", "objective"]].copy()
                    frontier_detail["Return (%)"] = frontier_detail["return"] * 100
                    frontier_detail["Solvency (%)"] = frontier_detail["solvency"] * 100
                    frontier_detail = frontier_detail.drop(columns=["return", "solvency"])

                    # Add allocations
                    for i, asset in enumerate(
                            ["Gov Bonds", "Corp Bonds", "Equity 1", "Equity 2", "Property", "T-Bills"]):
                        frontier_detail[f"{asset} (%)"] = opt_df["w_opt"].apply(lambda x: x[i] * 100)

                    frontier_detail.to_excel(writer, sheet_name='Efficient Frontier', index=False)

                    # Sheet 4: Duration Analysis
                    duration_df = pd.DataFrame({
                        "Metric": [
                            "Liability Duration (years)",
                            "Current Portfolio Duration (years)",
                            "Optimal Portfolio Duration (years)",
                            "Current Duration Gap (years)",
                            "Optimal Duration Gap (years)",
                            "Gap Improvement (years)",
                            "",
                            "=== DURATION CONTRIBUTION ===",
                            "Asset Class",
                            "Government Bonds",
                            "Corporate Bonds",
                            "Equity Type 1",
                            "Equity Type 2",
                            "Property",
                            "Treasury Bills"
                        ],
                        "Value": [
                            liab_duration,
                            current_dur,
                            optimal_dur,
                            current_gap,
                            optimal_gap,
                            abs(current_gap) - abs(optimal_gap),
                            "",
                            "",
                            "Duration (years)",
                            initial_asset["asset_dur"].values[0],
                            initial_asset["asset_dur"].values[1],
                            initial_asset["asset_dur"].values[2],
                            initial_asset["asset_dur"].values[3],
                            initial_asset["asset_dur"].values[4],
                            initial_asset["asset_dur"].values[5]
                        ],
                        "Current Weight (%)": [
                            "", "", "", "", "", "", "", "", "",
                            initial_asset["asset_weight"].values[0] * 100,
                            initial_asset["asset_weight"].values[1] * 100,
                            initial_asset["asset_weight"].values[2] * 100,
                            initial_asset["asset_weight"].values[3] * 100,
                            initial_asset["asset_weight"].values[4] * 100,
                            initial_asset["asset_weight"].values[5] * 100
                        ],
                        "Optimal Weight (%)": [
                            "", "", "", "", "", "", "", "", "",
                            best["w_opt"][0] * 100,
                            best["w_opt"][1] * 100,
                            best["w_opt"][2] * 100,
                            best["w_opt"][3] * 100,
                            best["w_opt"][4] * 100,
                            best["w_opt"][5] * 100
                        ]
                    })
                    duration_df.to_excel(writer, sheet_name='Duration Analysis', index=False)

                    # Sheet 5: Risk Metrics
                    cfg = load_config()
                    solv = get_solvency_params(cfg)

                    risk_df = pd.DataFrame({
                        "Asset Class": ["Government Bonds", "Corporate Bonds", "Equity Type 1",
                                        "Equity Type 2", "Property", "Treasury Bills"],
                        "Optimal Weight (%)": best["w_opt"] * 100,
                        "Optimal Amount (€m)": best["A_opt"],
                        "Shock (%)": [
                            1.1,  # IR shock approx
                            10.3,  # Spread shock approx
                            solv["equity_1_param"] * 100,
                            solv["equity_2_param"] * 100,
                            solv["prop_params"] * 100,
                            1.1  # IR shock approx
                        ],
                        "Risk Contribution (€m)": [
                            best["A_opt"][0] * 0.011,
                            best["A_opt"][1] * 0.103,
                            best["A_opt"][2] * solv["equity_1_param"],
                            best["A_opt"][3] * solv["equity_2_param"],
                            best["A_opt"][4] * solv["prop_params"],
                            best["A_opt"][5] * 0.011
                        ]
                    })
                    risk_df.to_excel(writer, sheet_name='Risk Metrics', index=False)

                    # Sheet 6: Metadata
                    metadata_df = pd.DataFrame({
                        "Parameter": [
                            "Report Generated",
                            "Optimization Tool",
                            "Model",
                            "",
                            "Total Frontier Points",
                            "Optimal Gamma",
                            "Optimal Objective Value",
                            "",
                            "Auto-Calculation Used"
                        ],
                        "Value": [
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "Solvency II Asset Allocation Optimizer",
                            "Mean-Variance with Solvency Constraints",
                            "",
                            len(opt_df),
                            best["gamma"],
                            best["objective"],
                            "",
                            "Yes" if st.session_state.get("auto_calculated", False) else "No"
                        ]
                    })
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

                    # Format the workbook
                    workbook = writer.book

                    # Format Summary sheet
                    summary_sheet = writer.sheets['Summary']
                    summary_sheet.column_dimensions['A'].width = 35
                    summary_sheet.column_dimensions['B'].width = 20

                    # Format Allocation sheet
                    allocation_sheet = writer.sheets['Allocation']
                    for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
                        allocation_sheet.column_dimensions[col].width = 18

                    # Format Frontier sheet
                    frontier_sheet = writer.sheets['Efficient Frontier']
                    for col in range(1, 15):  # Adjust based on columns
                        frontier_sheet.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 15

                    # Format Duration sheet
                    duration_sheet = writer.sheets['Duration Analysis']
                    duration_sheet.column_dimensions['A'].width = 35
                    duration_sheet.column_dimensions['B'].width = 20
                    duration_sheet.column_dimensions['C'].width = 20
                    duration_sheet.column_dimensions['D'].width = 20

                    # Format Risk sheet
                    risk_sheet = writer.sheets['Risk Metrics']
                    for col in ['A', 'B', 'C', 'D', 'E']:
                        risk_sheet.column_dimensions[col].width = 20

                # Get the Excel file from memory
                excel_data = output.getvalue()

                st.download_button(
                    label="📥 Download Excel Workbook",
                    data=excel_data,
                    file_name=f"portfolio_optimization_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    help="Download comprehensive Excel workbook with 6 sheets"
                )

                st.caption(
                    "📋 **Includes 6 sheets**: Summary, Allocation, Efficient Frontier, Duration Analysis, Risk Metrics, Metadata")

            with col_xl2:
                # JSON export for programmatic use
                st.markdown("**📦 Export as JSON (for API/Python)**")
                st.caption("Structured data for programmatic access")

                json_export = {
                    "metadata": {
                        "generated": datetime.now().isoformat(),
                        "total_assets": float(initial_asset['asset_val'].sum()),
                        "liabilities": float(liab_value),
                        "liability_duration": float(liab_duration),
                        "auto_calculation_used": st.session_state.get("auto_calculated", False)
                    },
                    "current_portfolio": {
                        "return": float(current_ret),
                        "solvency_ratio": float(current_sol),
                        "SCR_market": float(current_SCR),
                        "duration": float(current_dur),
                        "duration_gap": float(current_gap),
                        "allocation": {
                            "Government_Bonds": float(initial_asset["asset_val"].values[0]),
                            "Corporate_Bonds": float(initial_asset["asset_val"].values[1]),
                            "Equity_Type_1": float(initial_asset["asset_val"].values[2]),
                            "Equity_Type_2": float(initial_asset["asset_val"].values[3]),
                            "Property": float(initial_asset["asset_val"].values[4]),
                            "Treasury_Bills": float(initial_asset["asset_val"].values[5])
                        },
                        "weights": {
                            "Government_Bonds": float(initial_asset["asset_weight"].values[0]),
                            "Corporate_Bonds": float(initial_asset["asset_weight"].values[1]),
                            "Equity_Type_1": float(initial_asset["asset_weight"].values[2]),
                            "Equity_Type_2": float(initial_asset["asset_weight"].values[3]),
                            "Property": float(initial_asset["asset_weight"].values[4]),
                            "Treasury_Bills": float(initial_asset["asset_weight"].values[5])
                        }
                    },
                    "optimal_portfolio": {
                        "return": float(best_ret),
                        "solvency_ratio": float(best_sol),
                        "SCR_market": float(best_SCR),
                        "BOF": float(best_BOF),
                        "duration": float(optimal_dur),
                        "duration_gap": float(optimal_gap),
                        "objective": float(best["objective"]),
                        "gamma": float(best["gamma"]),
                        "allocation": {
                            "Government_Bonds": float(best["A_opt"][0]),
                            "Corporate_Bonds": float(best["A_opt"][1]),
                            "Equity_Type_1": float(best["A_opt"][2]),
                            "Equity_Type_2": float(best["A_opt"][3]),
                            "Property": float(best["A_opt"][4]),
                            "Treasury_Bills": float(best["A_opt"][5])
                        },
                        "weights": {
                            "Government_Bonds": float(best["w_opt"][0]),
                            "Corporate_Bonds": float(best["w_opt"][1]),
                            "Equity_Type_1": float(best["w_opt"][2]),
                            "Equity_Type_2": float(best["w_opt"][3]),
                            "Property": float(best["w_opt"][4]),
                            "Treasury_Bills": float(best["w_opt"][5])
                        }
                    },
                    "improvements": {
                        "return_change_pp": float((best_ret - current_ret) * 100),
                        "solvency_change_pp": float((best_sol - current_sol) * 100),
                        "SCR_change": float(best_SCR - current_SCR),
                        "duration_gap_change": float(optimal_gap - current_gap)
                    }
                }

                json_string = json.dumps(json_export, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_string,
                    file_name=f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Structured JSON format for APIs and Python scripts"
                )

        st.markdown("---")

    
# --------------------------
# INTERACTIVE PORTFOLIO SELECTOR PAGE
# --------------------------
elif st.session_state["nav"] == "Interactive Portfolio Selector":
    st.title("🎚️ Interactive Portfolio Selector")


    st.markdown("---")
    st.markdown("Explore different risk-return tradeoffs along the efficient frontier by selecting alternative portfolios.")

    # Check if optimization has been run
    if "opt_df" not in st.session_state or st.session_state["opt_df"].empty:
        st.warning("⚠️ No optimization results available. Please run the optimization from the **Inputs** page first.")
        st.stop()

    # Load data from session
    opt_df = st.session_state["opt_df"]
    initial_asset = st.session_state["initial_asset"]
    liab_value = st.session_state["liab_value"]
    liab_duration = st.session_state.get("liab_duration", 6.6)

    # Find optimal portfolio
    best_idx = opt_df["objective"].idxmax()
    best = opt_df.loc[best_idx]

    # Current portfolio metrics
    current_ret = float((initial_asset["asset_ret"] * initial_asset["asset_weight"]).sum())
    current_SCR = float(best["SCR_market"])
    current_sol = (initial_asset['asset_val'].sum() - liab_value) / current_SCR

    st.markdown("---")

    # ==========================================
    # INTERACTIVE FRONTIER CHART
    # ==========================================
    st.subheader("📈 Interactive Efficient Frontier")

    # Initialize selected index (default to optimal)
    if "selected_frontier_idx" not in st.session_state:
        st.session_state["selected_frontier_idx"] = best_idx

    # Slider for selecting frontier point
    st.markdown("**Use the slider below to explore different portfolios along the efficient frontier:**")
    st.info("""
        💡 **Why does the slider sometimes feel jumpy?**

        Due to regulatory constraints and investment policy limits, many portfolios on the frontier 
        have similar allocations. This is normal for Solvency II-compliant portfolios where the 
        feasible investment space is intentionally restricted for safety.

        Even small differences in allocation can produce meaningful differences in return and solvency.
        """)

    selected_idx = st.slider(
        "Select Portfolio Point",
        min_value=0,
        max_value=len(opt_df) - 1,
        value=int(st.session_state["selected_frontier_idx"]),
        format="Point %d / " + str(len(opt_df) - 1),
        help="Slide to explore different risk-return profiles. Point 0 = most aggressive (highest return), Point N = most conservative (highest solvency)",
        key="frontier_slider"
    )

    # Update session state
    st.session_state["selected_frontier_idx"] = selected_idx

    # Get selected portfolio
    selected_portfolio = opt_df.iloc[selected_idx]

    # Create interactive frontier chart
    fig_interactive, ax_interactive = plt.subplots(figsize=(12, 7))

    # Styling
    ax_interactive.set_facecolor('#f8f9fa')
    fig_interactive.patch.set_facecolor('white')

    # Plot frontier line
    ax_interactive.plot(
        opt_df["solvency"] * 100,
        opt_df["return"] * 100,
        '-',
        color='#4ECDC4',
        linewidth=2.5,
        alpha=0.4,
        zorder=1
    )

    # Plot all frontier points (small dots)
    ax_interactive.scatter(
        opt_df["solvency"] * 100,
        opt_df["return"] * 100,
        s=60,
        color='#4ECDC4',
        alpha=0.4,
        edgecolors='white',
        linewidth=1,
        zorder=2
    )

    # Plot CURRENT portfolio (red diamond)
    ax_interactive.scatter(
        current_sol * 100,
        current_ret * 100,
        s=350,
        c='#E74C3C',
        marker='D',
        edgecolors='#C0392B',
        linewidth=2.5,
        label='Current Portfolio',
        zorder=4,
        alpha=0.9
    )

    # Plot OPTIMAL portfolio (gold star)
    optimal_solvency = best["solvency"] * 100
    optimal_return = best["return"] * 100

    ax_interactive.scatter(
        optimal_solvency,
        optimal_return,
        s=600,
        c='#FFD700',
        marker='*',
        edgecolors='#FF8C00',
        linewidth=3,
        label='Optimal Portfolio',
        zorder=5
    )

    # Plot SELECTED portfolio (purple circle) - THE MOVING ONE
    selected_solvency = selected_portfolio["solvency"] * 100
    selected_return = selected_portfolio["return"] * 100

    # Outer glow for selected
    ax_interactive.scatter(
        selected_solvency,
        selected_return,
        s=800,
        c='#9B59B6',
        marker='o',
        alpha=0.2,
        edgecolors='none',
        zorder=6
    )

    # Main selected point
    ax_interactive.scatter(
        selected_solvency,
        selected_return,
        s=500,
        c='#9B59B6',
        marker='o',
        edgecolors='#6C3483',
        linewidth=3,
        label='Selected Portfolio',
        zorder=7,
        alpha=0.9
    )

    # Annotation for selected point
    ax_interactive.annotate(
        f'SELECTED\n{selected_return:.2f}% | {selected_solvency:.1f}%',
        xy=(selected_solvency, selected_return),
        xytext=(25, 25),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.8',
            facecolor='#9B59B6',
            edgecolor='#6C3483',
            linewidth=2.5,
            alpha=0.9
        ),
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle='arc3,rad=0.3',
            color='#6C3483',
            lw=2.5
        ),
        color='white',
        zorder=8
    )

    # Add 100% solvency line
    ax_interactive.axvline(
        x=100,
        color='#95a5a6',
        linestyle='--',
        linewidth=2.5,
        alpha=0.6,
        label='100% Solvency',
        zorder=0
    )

    # Styling
    ax_interactive.set_xlabel('Solvency Ratio (%)', fontsize=12, fontweight='bold', color='#2c3e50')
    ax_interactive.set_ylabel('Expected Return (%)', fontsize=12, fontweight='bold', color='#2c3e50')
    ax_interactive.set_title('Interactive Efficient Frontier', fontsize=15, fontweight='bold', pad=20, color='#2c3e50')
    ax_interactive.grid(True, alpha=0.25, linestyle='--', linewidth=1, color='gray')

    # Legend
    legend = ax_interactive.legend(
        loc='upper right',
        fontsize=10,
        frameon=True,
        shadow=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor='gray',
        ncol=1,
        borderpad=1.2,
        labelspacing=1.0,
        handlelength=2.5,
        handleheight=1.5,
        markerscale=0.7,
        handletextpad=1.0,
        borderaxespad=0.8
    )
    legend.get_frame().set_linewidth(1.5)

    # Remove spines
    ax_interactive.spines['top'].set_visible(False)
    ax_interactive.spines['right'].set_visible(False)
    ax_interactive.spines['left'].set_linewidth(1.5)
    ax_interactive.spines['bottom'].set_linewidth(1.5)

    # Axis limits - IMPROVED FOCUS
    x_min, x_max = opt_df["solvency"].min() * 100, opt_df["solvency"].max() * 100
    y_min, y_max = opt_df["return"].min() * 100, opt_df["return"].max() * 100

    # Option 1: Show the full range (will show all 150 points clearly)
    x_padding = (x_max - x_min) * 0.05  # ✅ Reduced padding
    y_padding = (y_max - y_min) * 0.10  # ✅ Reduced padding

    ax_interactive.set_xlim(x_min - x_padding, x_max + x_padding)
    ax_interactive.set_ylim(y_min - y_padding, y_max + y_padding)

    plt.tight_layout()
    st.pyplot(fig_interactive, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # METRICS DASHBOARD
    # ==========================================
    st.subheader("📊 Selected Portfolio Metrics")

    col_met1, col_met2, col_met3, col_met4 = st.columns(4)

    with col_met1:
        st.metric(
            "Expected Return",
            f"{selected_portfolio['return'] * 100:.2f}%",
            delta=f"{(selected_portfolio['return'] - best['return']) * 100:+.2f}pp vs Optimal",
            help="Annual expected return of selected portfolio"
        )

    with col_met2:
        st.metric(
            "Solvency Ratio",
            f"{selected_portfolio['solvency'] * 100:.1f}%",
            delta=f"{(selected_portfolio['solvency'] - best['solvency']) * 100:+.1f}pp vs Optimal",
            help="BOF / SCR Market ratio"
        )

    with col_met3:
        st.metric(
            "SCR Market",
            f"€{selected_portfolio['SCR_market']:.1f}m",
            delta=f"€{(selected_portfolio['SCR_market'] - best['SCR_market']):+.1f}m vs Optimal",
            delta_color="inverse",
            help="Market risk capital requirement"
        )

    with col_met4:
        st.metric(
            "Risk Aversion (γ)",
            f"{selected_portfolio['gamma']:.4f}",
            help="Higher γ = more conservative (prioritizes solvency)"
        )

    # Additional metrics row
    col_met5, col_met6, col_met7, col_met8 = st.columns(4)

    with col_met5:
        st.metric(
            "Basic Own Funds",
            f"€{selected_portfolio['BOF']:.1f}m",
            help="Assets - Liabilities"
        )

    with col_met6:
        selected_dur = np.average(
            initial_asset["asset_dur"].values,
            weights=selected_portfolio["w_opt"]
        )
        st.metric(
            "Portfolio Duration",
            f"{selected_dur:.2f} years",
            delta=f"{(selected_dur - liab_duration):+.2f}y gap",
            delta_color="inverse",
            help="Modified duration of selected portfolio"
        )

    with col_met7:
        st.metric(
            "Objective Value",
            f"{selected_portfolio['objective']:.4f}",
            delta=f"{(selected_portfolio['objective'] - best['objective']):+.4f} vs Optimal",
            help="Optimization objective (higher = better balance)"
        )

    with col_met8:
        # Position indicator
        position_pct = (selected_idx / (len(opt_df) - 1)) * 100
        if position_pct < 33:
            position_label = "Aggressive"
            position_color = "🔴"
        elif position_pct < 67:
            position_label = "Balanced"
            position_color = "🟡"
        else:
            position_label = "Conservative"
            position_color = "🟢"

        st.metric(
            "Profile",
            f"{position_color} {position_label}",
            help="Portfolio risk profile based on frontier position"
        )

    st.markdown("---")

    # ==========================================
    # ALLOCATION DISPLAY
    # ==========================================
    st.subheader("💼 Selected Portfolio Allocation")

    col_alloc_left, col_alloc_right = st.columns([1.2, 1])

    with col_alloc_left:
        st.markdown("**Detailed Allocation**")

        asset_labels = ["Government Bonds", "Corporate Bonds", "Equity Type 1",
                        "Equity Type 2", "Property", "Treasury Bills"]

        allocation_detail = pd.DataFrame({
            "Asset Class": asset_labels,
            "Weight (%)": selected_portfolio["w_opt"] * 100,
            "Amount (€m)": selected_portfolio["A_opt"],
            "Return (%)": initial_asset["asset_ret"].values * 100,
            "Duration (y)": initial_asset["asset_dur"].values
        })

        st.dataframe(
            allocation_detail.style.format({
                "Weight (%)": "{:.1f}",
                "Amount (€m)": "{:.1f}",
                "Return (%)": "{:.2f}",
                "Duration (y)": "{:.2f}"
            }).background_gradient(subset=["Weight (%)"], cmap="Blues", vmin=0, vmax=100),
            use_container_width=True,
            hide_index=True
        )

    with col_alloc_right:
        st.markdown("**Visual Allocation**")

        # Pie chart for selected allocation
        fig_selected_pie, ax_selected_pie = plt.subplots(figsize=(7, 7))

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9']

        wedges, texts, autotexts = ax_selected_pie.pie(
            selected_portfolio["w_opt"] * 100,
            labels=["Gov", "Corp", "Eq1", "Eq2", "Prop", "TB"],
            colors=colors,
            autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
            startangle=90,
            textprops={'fontsize': 10, 'weight': 'bold'},
            pctdistance=0.80,
            explode=[0.03 if w > 10 else 0 for w in selected_portfolio["w_opt"] * 100],
            shadow=True,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax_selected_pie.set_title('Selected Portfolio', fontsize=12, fontweight='bold', pad=15)

        plt.tight_layout()
        st.pyplot(fig_selected_pie, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # COMPARISON SECTION
    # ==========================================
    st.subheader("📊 Comparison Analysis")

    # Tabs for different comparisons
    comp_tab1, comp_tab2 = st.tabs(["📈 vs Optimal Portfolio", "📉 vs Current Portfolio"])

    with comp_tab1:
        st.markdown("**Selected vs Optimal Portfolio**")

        comparison_optimal = pd.DataFrame({
            "Metric": [
                "Expected Return (%)",
                "Solvency Ratio (%)",
                "SCR Market (€m)",
                "Basic Own Funds (€m)",
                "Portfolio Duration (years)",
                "Objective Value"
            ],
            "Selected": [
                f"{selected_portfolio['return'] * 100:.2f}",
                f"{selected_portfolio['solvency'] * 100:.1f}",
                f"{selected_portfolio['SCR_market']:.1f}",
                f"{selected_portfolio['BOF']:.1f}",
                f"{selected_dur:.2f}",
                f"{selected_portfolio['objective']:.4f}"
            ],
            "Optimal": [
                f"{best['return'] * 100:.2f}",
                f"{best['solvency'] * 100:.1f}",
                f"{best['SCR_market']:.1f}",
                f"{best['BOF']:.1f}",
                f"{np.average(initial_asset['asset_dur'].values, weights=best['w_opt']):.2f}",
                f"{best['objective']:.4f}"
            ],
            "Difference": [
                f"{(selected_portfolio['return'] - best['return']) * 100:+.2f}pp",
                f"{(selected_portfolio['solvency'] - best['solvency']) * 100:+.1f}pp",
                f"{(selected_portfolio['SCR_market'] - best['SCR_market']):+.1f}",
                f"{(selected_portfolio['BOF'] - best['BOF']):+.1f}",
                f"{(selected_dur - np.average(initial_asset['asset_dur'].values, weights=best['w_opt'])):+.2f}",
                f"{(selected_portfolio['objective'] - best['objective']):+.4f}"
            ]
        })

        st.dataframe(comparison_optimal, use_container_width=True, hide_index=True)

        # Allocation comparison
        st.markdown("**Allocation Differences**")

        alloc_comp_optimal = pd.DataFrame({
            "Asset Class": asset_labels,
            "Selected (%)": selected_portfolio["w_opt"] * 100,
            "Optimal (%)": best["w_opt"] * 100,
            "Difference (pp)": (selected_portfolio["w_opt"] - best["w_opt"]) * 100
        })

        st.dataframe(
            alloc_comp_optimal.style.format({
                "Selected (%)": "{:.1f}",
                "Optimal (%)": "{:.1f}",
                "Difference (pp)": "{:+.1f}"
            }).background_gradient(subset=["Difference (pp)"], cmap="RdYlGn", vmin=-20, vmax=20),
            use_container_width=True,
            hide_index=True
        )

    with comp_tab2:
        st.markdown("**Selected vs Current Portfolio**")

        comparison_current = pd.DataFrame({
            "Metric": [
                "Expected Return (%)",
                "Solvency Ratio (%)",
                "SCR Market (€m)",
                "Portfolio Duration (years)"
            ],
            "Selected": [
                f"{selected_portfolio['return'] * 100:.2f}",
                f"{selected_portfolio['solvency'] * 100:.1f}",
                f"{selected_portfolio['SCR_market']:.1f}",
                f"{selected_dur:.2f}"
            ],
            "Current": [
                f"{current_ret * 100:.2f}",
                f"{current_sol * 100:.1f}",
                f"{current_SCR:.1f}",
                f"{np.average(initial_asset['asset_dur'].values, weights=initial_asset['asset_weight'].values):.2f}"
            ],
            "Difference": [
                f"{(selected_portfolio['return'] - current_ret) * 100:+.2f}pp",
                f"{(selected_portfolio['solvency'] - current_sol) * 100:+.1f}pp",
                f"{(selected_portfolio['SCR_market'] - current_SCR):+.1f}",
                f"{(selected_dur - np.average(initial_asset['asset_dur'].values, weights=initial_asset['asset_weight'].values)):+.2f}"
            ]
        })

        st.dataframe(comparison_current, use_container_width=True, hide_index=True)

        # Allocation comparison
        st.markdown("**Allocation Differences**")

        alloc_comp_current = pd.DataFrame({
            "Asset Class": asset_labels,
            "Selected (%)": selected_portfolio["w_opt"] * 100,
            "Current (%)": initial_asset["asset_weight"].values * 100,
            "Difference (pp)": (selected_portfolio["w_opt"] - initial_asset["asset_weight"].values) * 100
        })

        st.dataframe(
            alloc_comp_current.style.format({
                "Selected (%)": "{:.1f}",
                "Current (%)": "{:.1f}",
                "Difference (pp)": "{:+.1f}"
            }).background_gradient(subset=["Difference (pp)"], cmap="RdYlGn", vmin=-30, vmax=30),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # ==========================================
    # EXPORT SELECTED PORTFOLIO
    # ==========================================
    st.subheader("💾 Export Selected Portfolio")

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        st.markdown("**📊 Allocation CSV**")

        export_selected_alloc = pd.DataFrame({
            "Asset Class": asset_labels,
            "Weight (%)": selected_portfolio["w_opt"] * 100,
            "Amount (€m)": selected_portfolio["A_opt"],
            "Expected Return (%)": initial_asset["asset_ret"].values * 100,
            "Duration (years)": initial_asset["asset_dur"].values
        })

        csv_selected = export_selected_alloc.to_csv(index=False)
        st.download_button(
            label="📥 Download Allocation",
            data=csv_selected,
            file_name=f"selected_portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_exp2:
        st.markdown("**📄 Summary Report**")

        selected_report = f"""SELECTED PORTFOLIO REPORT
    {'=' * 60}
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Frontier Point: {selected_idx} of {len(opt_df) - 1}
    
    {'=' * 60}
    PORTFOLIO METRICS
    {'=' * 60}
    Expected Return:         {selected_portfolio['return']:>12.2%}
    Solvency Ratio:          {selected_portfolio['solvency'] * 100:>12.1f}%
    SCR Market:              €{selected_portfolio['SCR_market']:>12.1f}m
    Basic Own Funds:         €{selected_portfolio['BOF']:>12.1f}m
    Portfolio Duration:      {selected_dur:>12.2f} years
    Objective Value:         {selected_portfolio['objective']:>12.4f}
    Risk Aversion (gamma):   {selected_portfolio['gamma']:>12.4f}
    
    {'=' * 60}
    ALLOCATION
    {'=' * 60}
    Asset Class              Weight (%)    Amount (€m)
    {'-' * 60}
    """
        for i, asset in enumerate(asset_labels):
            selected_report += f"{asset:<24} {selected_portfolio['w_opt'][i] * 100:>9.1f}    {selected_portfolio['A_opt'][i]:>10.1f}\n"

        selected_report += f"""
    {'=' * 60}
    COMPARISON VS OPTIMAL
    {'=' * 60}
    Return Difference:       {(selected_portfolio['return'] - best['return']) * 100:>12.2f}pp
    Solvency Difference:     {(selected_portfolio['solvency'] - best['solvency']) * 100:>12.1f}pp
    SCR Difference:          €{(selected_portfolio['SCR_market'] - best['SCR_market']):>12.1f}m
    
    {'=' * 60}
    End of Report
    {'=' * 60}
    """

        st.download_button(
            label="📥 Download Report",
            data=selected_report,
            file_name=f"selected_portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col_exp3:
        st.markdown("**📦 JSON Export**")

        selected_json = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "frontier_point": int(selected_idx),
                "total_points": len(opt_df)
            },
            "metrics": {
                "return": float(selected_portfolio['return']),
                "solvency_ratio": float(selected_portfolio['solvency']),
                "SCR_market": float(selected_portfolio['SCR_market']),
                "BOF": float(selected_portfolio['BOF']),
                "duration": float(selected_dur),
                "objective": float(selected_portfolio['objective']),
                "gamma": float(selected_portfolio['gamma'])
            },
            "allocation": {
                "Government_Bonds": {
                    "weight": float(selected_portfolio["w_opt"][0]),
                    "amount": float(selected_portfolio["A_opt"][0])
                },
                "Corporate_Bonds": {
                    "weight": float(selected_portfolio["w_opt"][1]),
                    "amount": float(selected_portfolio["A_opt"][1])
                },
                "Equity_Type_1": {
                    "weight": float(selected_portfolio["w_opt"][2]),
                    "amount": float(selected_portfolio["A_opt"][2])
                },
                "Equity_Type_2": {
                    "weight": float(selected_portfolio["w_opt"][3]),
                    "amount": float(selected_portfolio["A_opt"][3])
                },
                "Property": {
                    "weight": float(selected_portfolio["w_opt"][4]),
                    "amount": float(selected_portfolio["A_opt"][4])
                },
                "Treasury_Bills": {
                    "weight": float(selected_portfolio["w_opt"][5]),
                    "amount": float(selected_portfolio["A_opt"][5])
                }
            }
        }

        json_selected = json.dumps(selected_json, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_selected,
            file_name=f"selected_portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")

    # Help section
    with st.expander("ℹ️ How to Use This Page"):
        st.markdown("""
            ### Interactive Portfolio Selector Guide
    
            **Purpose**: Explore alternative portfolios along the efficient frontier and compare them to the optimal solution.
    
            **How It Works**:
            1. **Use the slider** to select different points on the efficient frontier
            2. **Watch the chart update** - the purple circle shows your selected portfolio
            3. **Review metrics** - compare selected portfolio to optimal and current
            4. **Download results** - export any portfolio you want to analyze further
    
            **Understanding Risk Profiles**:
            - 🔴 **Aggressive** (Low points): Higher returns, lower solvency
            - 🟡 **Balanced** (Middle points): Trade-off between return and safety
            - 🟢 **Conservative** (High points): Lower returns, higher solvency
    
            **Why Explore?**:
            - The "optimal" portfolio maximizes a specific objective function
            - Your organization may have different priorities (e.g., minimum solvency requirement)
            - This tool lets you find portfolios that meet YOUR specific constraints
    
            **Tips**:
            - Gold star (⭐) = Mathematically optimal portfolio
            - Purple circle (🟣) = Your currently selected portfolio
            - Red diamond (♦️) = Your current portfolio for reference
            """)
