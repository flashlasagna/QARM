import streamlit as st
from backend.style_utils import apply_sidebar_style
st.set_page_config(page_title="Results", layout="wide")
apply_sidebar_style()
import os, sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta



# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- Backend Imports ---
from backend.config_loader import load_config, get_corr_matrices, get_solvency_params
from backend.optimization import solve_frontier_combined
from backend.data_handler import get_interpolated_ecb_yield
from backend.data_calculator import (
    compute_expected_returns_from_etfs,
    compute_ir_shocks_duration_approx,
    compute_ir_shocks_from_eiopa,
    compute_spread_shock_eiopa
)


# --------------------------
# Helper Functions
# --------------------------
def fetch_ticker_prices(ticker, start_date, end_date):
    """Robust ticker price fetching with fallback logic."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty: return None

        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                prices = data['Close'].iloc[:, 0]
            elif 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close'].iloc[:, 0]
            else:
                return None
        else:
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif 'Close' in data.columns:
                prices = data['Close']
            else:
                return None

        prices = prices.dropna()
        return prices if len(prices) > 100 else None
    except Exception as e:
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

    denom = max(total_A, 1e-9)
    w_gov = A_gov / denom
    w_corp = A_corp / denom
    w_eq1, w_eq2, w_prop = A_eq1 / denom, A_eq2 / denom, A_prop / denom
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
                         use_custom_shocks=False, eq1_sh=None, eq2_sh=None, prop_sh=None):
    cfg = load_config()
    corr_down, corr_up = get_corr_matrices(cfg)
    solv = get_solvency_params(cfg)

    if use_custom_shocks and eq1_sh is not None:
        equity_1_shock, equity_2_shock, property_shock = eq1_sh, eq2_sh, prop_sh
    else:
        equity_1_shock, equity_2_shock, property_shock = solv["equity_1_param"], solv["equity_2_param"], solv[
            "prop_params"]

    initial_asset = pd.DataFrame({
        "asset_val": [A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb],
        "asset_dur": [dur_gov, dur_corp, 0.0, 0.0, 0.0, dur_tb],
        "asset_ret": [r_gov, r_corp, r_eq1, r_eq2, r_prop, r_tb]
    }, index=["gov_bond", "corp_bond", "equity_1", "equity_2", "property", "t_bills"])

    initial_asset["asset_weight"] = initial_asset["asset_val"] / max(initial_asset["asset_val"].sum(), 1e-12)

    allocation_limits = pd.DataFrame({
        "asset": ["gov_bond", "illiquid_assets", "t_bills", "corp_bond"],
        "min_weight": [gov_min, 0.0, tb_min, 0.0],
        "max_weight": [gov_max, illiq_max, tb_max, corp_max],
    }).set_index("asset")

    params = {
        "interest_down": ir_down, "interest_up": ir_up, "spread": corp_sp,
        "equity_type1": equity_1_shock, "equity_type2": equity_2_shock, "property": property_shock,
        "rho": solv["rho"],
    }

    return initial_asset, BE_value, BE_dur, corr_down, corr_up, allocation_limits, params


# --------------------------
# UI Content
# --------------------------

st.title("🏦 Solvency II Asset Allocation Optimizer")

st.markdown("---")

# Auto-calculate toggle
# Updated help text to reflect that tickers are now for reference/input in both modes
use_auto_params = st.checkbox(
    "🤖 Auto-calculate Returns & Shocks from Market Data",
    value=st.session_state.get("auto_calculated", True),
    help="If checked, the app uses historical data from the tickers below to estimate returns and risks. If unchecked, it uses the manual values in Advanced Settings."
)

if use_auto_params:
    st.success("✨ **Automated Mode**: Parameters will be computed from the tickers below.")
else:
    st.info("📝 **Manual Mode**: Tickers are recorded for reference, but parameters must be set in Advanced Settings.")

st.markdown("---")

# ==========================================
# TICKER SELECTION (ALWAYS VISIBLE)
# ==========================================
st.subheader("🎯 Select Your ETF Tickers")
st.markdown("Choose the ETFs/tickers that represent each asset class in your portfolio.")

ticker_col1, ticker_col2, ticker_col3 = st.columns(3)

with ticker_col1:
    st.markdown("**Fixed Income**")
    ticker_gov = st.text_input("Government Bonds", value=st.session_state.get('ticker_gov', "IBGM.L"),
                               help="e.g., IBGM.L")
    ticker_corp = st.text_input("Corporate Bonds", value=st.session_state.get('ticker_corp', "IE15.L"),
                                help="e.g., IE15.L")
    ticker_tbills = st.text_input("Treasury Bills / Cash", value=st.session_state.get('ticker_tbills', "CSH2.L"))

with ticker_col2:
    st.markdown("**Equity**")
    ticker_eq1 = st.text_input("Equity Type 1 (Developed)", value=st.session_state.get('ticker_eq1', "EUNL.DE"))
    ticker_eq2 = st.text_input("Equity Type 2 (Emerging)", value=st.session_state.get('ticker_eq2', "IQQE.DE"))

with ticker_col3:
    st.markdown("**Real Assets**")
    ticker_prop = st.text_input("Property / Real Estate", value=st.session_state.get('ticker_prop', "EUNK.DE"))
    st.markdown("**Data Period**")
    lookback_years = st.slider("Historical data (years)", 1, 10, st.session_state.get('lookback_years', 5))

# Save tickers to session state
st.session_state.update({
    'ticker_gov': ticker_gov, 'ticker_corp': ticker_corp, 'ticker_tbills': ticker_tbills,
    'ticker_eq1': ticker_eq1, 'ticker_eq2': ticker_eq2, 'ticker_prop': ticker_prop,
    'lookback_years': lookback_years
})

if st.button("🔍 Validate Tickers", type="secondary"):
    with st.spinner("Validating tickers and fetching data..."):
        all_tickers = {
            "Government Bonds": ticker_gov, "Corporate Bonds": ticker_corp, "Treasury Bills": ticker_tbills,
            "Equity Type 1": ticker_eq1, "Equity Type 2": ticker_eq2, "Property": ticker_prop
        }
        validation_results = []
        for asset_class, ticker in all_tickers.items():
            if not ticker.strip():
                continue

            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_years * 365 + 100)
            prices = fetch_ticker_prices(ticker, start_date, end_date)

            if prices is None or len(prices) < 100:
                status = "❌ Insufficient data"
            else:
                total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
                years = len(prices) / 252
                ann_return = (1 + total_return) ** (1 / years) - 1
                status = f"✅ Valid ({ann_return:.2%})"

            validation_results.append({
                "Asset Class": asset_class, "Ticker": ticker, "Status": status,
                "Data Points": len(prices) if prices is not None else 0
            })

        st.dataframe(pd.DataFrame(validation_results), use_container_width=True, hide_index=True)

st.markdown("---")

# ==========================================
# PORTFOLIO ALLOCATION
# ==========================================
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📊 Portfolio Allocation")
    total_A = st.number_input("Total Assets (€ millions)", min_value=0.0, value=1652.7, step=10.0)

    st.markdown("**Asset Allocation (€ millions)**")
    A_gov = st.number_input("Government Bonds", min_value=0.0, value=782.6, step=1.0)
    A_corp = st.number_input("Corporate Bonds", min_value=0.0, value=586.0, step=1.0)
    A_eq1 = st.number_input("Equity Type 1 (Developed Markets)", min_value=0.0, value=0.0, step=1.0)
    A_eq2 = st.number_input("Equity Type 2 (Emerging Markets)", min_value=0.0, value=102.5, step=1.0)
    A_prop = st.number_input("Property", min_value=0.0, value=42.0, step=1.0)
    A_tb = st.number_input("Treasury Bills", min_value=0.0, value=139.6, step=1.0)

    total_allocated = A_gov + A_corp + A_eq1 + A_eq2 + A_prop + A_tb
    st.progress(min(1.0, total_allocated / max(total_A, 1)))

    if abs(total_allocated - total_A) < 1e-6:
        st.success(f"✓ Total allocated: €{total_allocated:.1f}m")
    else:
        st.warning(f"⚠ Total allocated: €{total_allocated:.1f}m (Target: €{total_A:.1f}m)")

with col2:
    st.subheader("📋 Liabilities & Durations")
    BE_value = st.number_input("Best Estimate Liabilities (€m)", min_value=0.0, value=1424.2, step=10.0)
    BE_dur = st.number_input("Liabilities Duration (years)", min_value=0.0, value=6.6, step=0.1)

    # ----------------------------------------------------
    # CHANGED: Durations are now ALWAYS visible inputs
    # ----------------------------------------------------
    st.markdown("**Asset Durations (years)**")
    st.info("ℹ️ Please input the modified duration from your fund factsheet.")
    dur_gov = st.number_input("Gov Bonds Duration", 0.0, 50.0, 5.2, 0.1)
    dur_corp = st.number_input("Corp Bonds Duration", 0.0, 50.0, 5.0, 0.1)
    dur_tb = st.number_input("T-Bills Duration", 0.0, 10.0, 0.1, 0.1)

st.markdown("---")

# ==========================================
# ADVANCED SETTINGS
# ==========================================
with st.expander("⚙️ Advanced Settings", expanded=not use_auto_params):
    if use_auto_params:
        st.warning(
            "⚠️ Auto-calculation is enabled. Returns and Shocks below will be **overridden** by computed values.")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Solvency II Shocks**")
        if use_auto_params:
            use_eiopa_curves = st.checkbox("📊 Use EIOPA Risk-Free Curves", value=True)
            st.session_state['use_eiopa_curves'] = use_eiopa_curves
            if use_eiopa_curves:
                st.info("✓ Will use EIOPA Oct 2025 curves")
            else:
                st.info("✓ Will use duration-based approx")
            ir_up, ir_down, corp_sp = 0.011, 0.009, 0.103  # Placeholders
        else:
            ir_up = st.number_input("IR Up", 0.0, 1.0, 0.011, format="%.3f")
            ir_down = st.number_input("IR Down", 0.0, 1.0, 0.009, format="%.3f")
            corp_sp = st.number_input("Spread (Corp)", 0.0, 1.0, 0.103, format="%.3f")

        st.markdown("---")
        use_custom_shocks = st.checkbox("Override Standard Formula", value=False)
        if use_custom_shocks:
            eq1_sh = st.number_input("Equity Type 1 (custom)", 0.0, 1.0, 0.39)
            eq2_sh = st.number_input("Equity Type 2 (custom)", 0.0, 1.0, 0.49)
            prop_sh = st.number_input("Property (custom)", 0.0, 1.0, 0.25)
        else:
            eq1_sh, eq2_sh, prop_sh = None, None, None
            st.info("Using standard Solvency II shocks (39%, 49%, 25%)")

    with col_b:
        st.markdown("**Risk-Free/Floors Fallback**")
        # --- NEW INPUT HERE ---
        # 1. Risk Free Rate (Base)
        risk_free_rate = st.number_input(
            "Risk Free Rate (Base)",
            min_value=0.0, max_value=0.15, value=0.030, step=0.001, format="%.3f",
            help="Used as the baseline for floors and proxies."
        )
        st.info("ℹ️ This rate is used as a fallback if the automated ECB interpolation fails.")

        # 2. Credit Spread (for Corp Bonds proxy)
        credit_spread = st.number_input(
            "Credit Spread Proxy",
            min_value=0.0, max_value=0.10, value=0.015, step=0.001, format="%.3f",
            help="Added to Risk Free Rate if historical corporate bond returns are negative."
        )

        # 3. Min Equity Premium (for Equity floor)
        equity_risk_premium = st.number_input(
            "Min Equity Risk Premium",
            min_value=0.0, max_value=0.10, value=0.020, step=0.001, format="%.3f",
            help="Added to Risk Free Rate to set the minimum floor for equity returns."
        )
        if use_auto_params:
            st.info("✓ Returns will be computed from ETF historical data")
            r_gov, r_corp, r_eq1, r_eq2, r_prop, r_tb = 0.029, 0.041, 0.064, 0.064, 0.056, 0.006
        else:
            r_gov = st.number_input("Gov Bonds", 0.0, 1.0, 0.029, format="%.3f")
            r_corp = st.number_input("Corp Bonds", 0.0, 1.0, 0.041, format="%.3f")
            r_eq1 = st.number_input("Equity Type 1", 0.0, 1.0, 0.064, format="%.3f")
            r_eq2 = st.number_input("Equity Type 2", 0.0, 1.0, 0.064, format="%.3f")
            r_prop = st.number_input("Property", 0.0, 1.0, 0.056, format="%.3f")
            r_tb = st.number_input("T-Bills", 0.0, 1.0, 0.006, format="%.3f")

    with col_c:
        st.markdown("**Allocation Limits (weights)**")
        gov_min = st.number_input("Gov Min", 0.0, 1.0, 0.25)
        gov_max = st.number_input("Gov Max", 0.0, 1.0, 0.75)
        corp_max = st.number_input("Corp Max", 0.0, 1.0, 0.50)
        illiq_max = st.number_input("Illiquid Max", 0.0, 1.0, 0.20)
        tb_min = st.number_input("T-Bills Min", 0.0, 1.0, 0.01)
        tb_max = st.number_input("T-Bills Max", 0.0, 1.0, 0.05)

st.markdown("---")

# Validation
errs, warns = validate_inputs(A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb, total_A,
                              gov_min, gov_max, corp_max, illiq_max, tb_min, tb_max)

for e in errs: st.error(e)
for w in warns: st.warning(w)

can_optimize = (len(errs) == 0)

# ==========================================
# OPTIMIZE BUTTON & LOGIC
# ==========================================
if st.button("🚀 Optimize Portfolio", disabled=not can_optimize, type="primary", use_container_width=True):
    with st.spinner("Running optimization... This may take a minute."):
        try:
            # 1. AUTO-CALCULATE LOGIC
            if use_auto_params:
                st.info("🤖 Computing parameters from market data...")

                # Fetch Returns
                ticker_mapping = {
                    'gov_bond': st.session_state.get('ticker_gov', "IBGM.L"),
                    'corp_bond': st.session_state.get('ticker_corp', "IE15.L"),
                    'equity_1': st.session_state.get('ticker_eq1', "EUNL.DE"),
                    'equity_2': st.session_state.get('ticker_eq2', "IQQE.DE"),
                    'property': st.session_state.get('ticker_prop', "EUNK.DE"),
                    't_bills': st.session_state.get('ticker_tbills', "CSH2.L")
                }

                auto_returns = {}
                lookback = st.session_state.get('lookback_years', 5)

                for asset, ticker in ticker_mapping.items():
                    if not ticker.strip(): continue
                    end_d = datetime.now()
                    start_d = end_d - timedelta(days=lookback * 365 + 100)
                    prices = fetch_ticker_prices(ticker, start_d, end_d)

                    if prices is not None and len(prices) > lookback * 252 * 0.8:
                        tot_ret = (prices.iloc[-1] / prices.iloc[0]) - 1
                        yrs = len(prices) / 252
                        ann_ret = (1 + tot_ret) ** (1 / yrs) - 1
                        auto_returns[asset] = float(ann_ret)


                # --- FETCH LIVE ECB RATES ---
                try:
                    live_rfr = get_interpolated_ecb_yield(10.0, verbose=False)
                    st.success(f"✓ Fetched live ECB Risk Free Rate (10Y): {live_rfr:.2%}")
                except:
                    live_rfr = risk_free_rate  # Fallback to user input
                    st.warning(f"⚠️ ECB interpolation failed. Using manual fallback: {live_rfr:.2%}")

                    # 2. Calculate Returns using this live rate
                    # Gov Bonds: Use exact duration from input (dur_gov)
                try:
                    r_gov = get_interpolated_ecb_yield(dur_gov, verbose=False)
                except:
                    r_gov = live_rfr  # Fallback

                    # T-Bills: Use exact duration from input (dur_tb)
                try:
                    r_tb = get_interpolated_ecb_yield(dur_tb, verbose=False)
                except:
                    r_tb = live_rfr  # Fallback


                # Corp Bonds: Risk Free Rate + Credit Spread (approx 1.5%)
                hist_corp = auto_returns.get('corp_bond', -1)
                if hist_corp < 0:
                    r_corp = live_rfr + credit_spread # <--- FIX: Uses User Input now
                    st.warning(f"⚠️ Historical Corp Bond return was negative. Using Proxy (RFR {live_rfr:.1%} + Spread {credit_spread:.1%} = {r_corp:.1%}).")
                else:
                    r_corp = hist_corp
                # Equities: Floor at Risk Free Rate + Min Premium (e.g., 2%)
                min_equity_threshold = live_rfr + equity_risk_premium

                val_eq1 = auto_returns.get('equity_1', -1)
                if val_eq1 < min_equity_threshold:
                    r_eq1 = min_equity_threshold
                    st.warning(
                        f"⚠️ Hist. Equity 1 return ({val_eq1:.1%}) < RFR. Using long-term assumption ({r_eq1:.1%}).")
                else:
                    r_eq1 = val_eq1

                val_eq2 = auto_returns.get('equity_2', -1)
                if val_eq2 < min_equity_threshold:
                    r_eq2 = min_equity_threshold
                    st.warning(
                        f"⚠️ Hist. Equity 2 return ({val_eq2:.1%}) < RFR. Using long-term assumption ({r_eq2:.1%}).")
                else:
                    r_eq2 = val_eq2


                st.success(
                    f"✓ Returns: Gov={r_gov:.1%}, Corp={r_corp:.1%}, Eq1={r_eq1:.1%}, Eq2={r_eq2:.1%}, Prop={r_prop:.1%}, TB={r_tb:.1%}")


                # Compute Shocks (Using user-defined duration)
                if st.session_state.get('use_eiopa_curves', True):
                    try:
                        ir_up, ir_down = compute_ir_shocks_from_eiopa(liab_duration=BE_dur)
                        # We use the USER INPUT duration here
                        corp_sp = compute_spread_shock_eiopa(duration=dur_corp)
                        st.success(f"✓ EIOPA Shocks: IR Up={ir_up:.2%}, IR Down={ir_down:.2%}, Spread={corp_sp:.2%}")
                    except:
                        st.warning("⚠️ EIOPA failed. Using duration approx.")
                        # Fallback logic
                        weights = np.array([A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb]) / total_A
                        durs = np.array([dur_gov, dur_corp, 0, 0, 0, dur_tb])
                        port_dur = np.sum(weights * durs)
                        ir_up, ir_down = compute_ir_shocks_duration_approx(port_dur, BE_dur, 0.01)
                        corp_sp = 0.103
                else:
                    # Duration Approx Logic
                    weights = np.array([A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb]) / total_A
                    durs = np.array([dur_gov, dur_corp, 0, 0, 0, dur_tb])
                    port_dur = np.sum(weights * durs)
                    ir_up, ir_down = compute_ir_shocks_duration_approx(port_dur, BE_dur, 0.01)
                    corp_sp = compute_spread_shock_eiopa(duration=dur_corp)

            # 2. BUILD BACKEND INPUTS
            initial_asset, liab_value, liab_duration, corr_down, corr_up, allocation_limits, params = \
                build_backend_inputs(
                    A_gov, A_corp, A_eq1, A_eq2, A_prop, A_tb, total_A,
                    BE_value, BE_dur, dur_gov, dur_corp, dur_tb,
                    r_gov, r_corp, r_eq1, r_eq2, r_prop, r_tb,
                    ir_up, ir_down, corp_sp,
                    gov_min, gov_max, corp_max, illiq_max, tb_min, tb_max,
                    use_custom_shocks, eq1_sh, eq2_sh, prop_sh
                )

            # 3. RUN SOLVER
            opt_df = solve_frontier_combined(
                initial_asset, liab_value, liab_duration, corr_down, corr_up, allocation_limits, params
            )
            # === NEW ERROR CHECK ===
            if opt_df.empty:
                st.error("❌ Optimization failed: No feasible portfolios found.")
                st.warning("""
                            **Possible causes:**
                            1. **Solvency Constraint:** It might be mathematically impossible to achieve a 100% Solvency Ratio with the current Assets & Liabilities.
                            2. **Constraints:** Your Min/Max allocation limits might be too restrictive.
                            3. **Shocks:** The computed shocks might be too severe for the available capital.
                            """)
                # Prevent navigating to results
                st.session_state["optimization_run"] = False
            else:
                # 4. STORE RESULTS (Only if successful)
                st.session_state["opt_df"] = opt_df
                st.session_state["initial_asset"] = initial_asset
                st.session_state["liab_value"] = liab_value
                st.session_state["liab_duration"] = liab_duration
                st.session_state["auto_calculated"] = use_auto_params
                st.session_state["optimization_run"] = True
                st.session_state["params"] = params  # Save the exact shocks used

                st.success("✅ Optimization completed successfully! Navigate to 'Results' page.")

        except Exception as e:
            st.error(f"❌ Optimization failed: {str(e)}")
            st.exception(e)
