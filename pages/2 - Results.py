import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
import json
import os
import sys


# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- Backend imports ---
from backend.config_loader import load_config, get_corr_matrices, get_solvency_params
from backend.optimization import solve_frontier_combined
from backend.helpers import summarize_portfolio
from backend.helpers import plot_scenario_comparison
from backend.data_calculator import (
    compute_ir_shocks_from_eiopa,
    compute_spread_shock_eiopa

st.title("📈 Optimization Results")

# --- 1. Session State Checks ---
if not st.session_state.get("optimization_run"):
    st.warning("Please run the optimization on the **Inputs** page first.")
    st.stop()

# Show if auto-calculation was used
if st.session_state.get("auto_calculated", False):
    st.info("🤖 **Auto-calculation was enabled** - Returns and IR shocks were computed from market data")

# --- Load Data ---
# Ensure liab_duration is explicitly loaded into the script's main scope
if "opt_df" in st.session_state:
    opt_df = st.session_state["opt_df"]
    best = opt_df.loc[opt_df["objective"].idxmax()]
    initial_asset = st.session_state["initial_asset"]
    liab_value = st.session_state["liab_value"]

    # FIX: Explicitly define liab_duration here
    liab_duration = st.session_state["liab_duration"]
else:
    # Handle missing data gracefully if needed
    st.info("Optimization data not fully loaded.")
    st.stop()

# Find Best Solution
best_idx = opt_df["objective"].idxmax()
best = opt_df.loc[best_idx]

# Calculate Current Metrics (ignoring duration)
current_ret = float((initial_asset["asset_ret"] * initial_asset["asset_weight"]).sum())
current_SCR = float(best["SCR_market"])  # Approximation using optimal's SCR logic for comparison
current_sol = (initial_asset['asset_val'].sum() - liab_value) / current_SCR

# Best Metrics
best_ret = float(best["return"])
best_sol = float(best["solvency"])
best_SCR = float(best["SCR_market"])
best_BOF = float(best["BOF"])

# === NEW: CRITICAL SOLVENCY CHECK ===
if best_sol < 1.0:
    st.error(f"🚨 **CRITICAL WARNING: INSOLVENT PORTFOLIO**")
    st.warning(f"""
        **The optimizer could not find a portfolio compliant with Solvency II (100% Ratio).**
        
        The displayed portfolio is the **"Least Bad"** option available under your current constraints.
        
        - **Best Achievable Ratio:** {best_sol:.1%} (Target: 100%)
        - **Capital Shortfall:** €{(best['SCR_market'] - best['BOF']):,.1f}m
        
        **Recommended Actions:**
        1. **Inject Capital:** You cannot solve this by asset allocation alone.
        2. **Relax Constraints:** Allow more allocation to high-return/low-capital assets (if any).
        3. **Reduce Liabilities:** Revisit liability duration matching or reinsurance.
    """)

# --- 3. Key Metrics Comparison ---
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


# --- 5. Efficient Frontier Plot (High Fidelity) ---
st.subheader("📉 Efficient Frontier")

# 1. SORTING: Order by Solvency Ratio to ensure the line draws sequentially
# We sort descending (High Solvency -> Low Solvency) to help the filter logic
opt_sorted = opt_df.sort_values(by="solvency", ascending=False).copy()

# 2. PARETO FILTER: Keep only the "Upper Envelope"
# Logic: Scanning from the safest (Right) to riskiest (Left), 
# we only keep a point if it offers a HIGHER return than everything safer than it.
# If a risky portfolio has a lower return than a safer one, it's inefficient -> Trash it.

opt_sorted["max_return_seen"] = opt_sorted["return"].cummax()
# Keep points where the return is the new highest we've seen so far
pareto_frontier = opt_sorted[opt_sorted["return"] >= opt_sorted["max_return_seen"]]

# Sort back to ascending for proper line plotting (Left to Right)
pareto_frontier = pareto_frontier.sort_values(by="solvency")

fig_frontier, ax_frontier = plt.subplots(figsize=(12, 7))
ax_frontier.set_facecolor('#f8f9fa')
fig_frontier.patch.set_facecolor('white')

# A. Plot the "Feasible Set" (All points) as faint dots
# This shows the user the full search space, including the inefficient 'hooks'
ax_frontier.scatter(
    opt_df["solvency"] * 100, 
    opt_df["return"] * 100, 
    s=30, 
    color='gray', 
    alpha=0.2, 
    label='Feasible Portfolios'
)

# B. Plot the "Efficient Frontier" (Filtered Line)
# This will now be a smooth curve without zig-zags
ax_frontier.plot(
    pareto_frontier["solvency"] * 100, 
    pareto_frontier["return"] * 100, 
    '-', 
    color='#4ECDC4', 
    linewidth=3, 
    label='Efficient Frontier',
    zorder=2
)

# Optimal Point (Gold Star)
# We stick to the mathematically optimal point found by the solver
optimal_solvency = best["solvency"] * 100
optimal_return = best["return"] * 100

ax_frontier.scatter(
    optimal_solvency, optimal_return, 
    s=600, c='#FFD700', marker='*',
    edgecolors='#FF8C00', linewidth=3, 
    label='Optimal Portfolio', 
    zorder=5
)

# Annotation for Optimal
ax_frontier.annotate(
    f'OPTIMAL\n{optimal_return:.2f}% | {optimal_solvency:.1f}%',
    xy=(optimal_solvency, optimal_return), xytext=(25, 25),
    textcoords='offset points', fontsize=10, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFD700', edgecolor='#FF8C00', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='#FF8C00', lw=2.5), zorder=6
)

# Current Point (Red Diamond)
ax_frontier.scatter(
    current_sol * 100, current_ret * 100, 
    s=400, c='#E74C3C', marker='D',
    edgecolors='#C0392B', linewidth=3, 
    label='Current Portfolio', 
    zorder=4, alpha=0.9
)

# Styling
ax_frontier.axvline(x=100, color='#95a5a6', linestyle='--', linewidth=2.5, alpha=0.6, label='100% Solvency')
ax_frontier.set_xlabel('Solvency Ratio (%)', fontsize=12, fontweight='bold', color='#2c3e50')
ax_frontier.set_ylabel('Expected Return (%)', fontsize=12, fontweight='bold', color='#2c3e50')
ax_frontier.grid(True, alpha=0.25, linestyle='--', linewidth=1)
ax_frontier.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

st.pyplot(fig_frontier, use_container_width=True)
st.markdown("---")

# --- 6. Optimal Allocation Tables ---
st.subheader("💼 Optimal Portfolio Allocation")

col_left, col_right = st.columns([1, 1])
A_opt = best["A_opt"]
w_opt = best["w_opt"]

with col_left:
    st.markdown("**Amounts (€ millions)**")
    allocation_df = pd.DataFrame({
        "Asset Class": ["Gov Bonds", "Corp Bonds", "Equity 1", "Equity 2", "Property", "T-Bills"],
        "Current (€m)": initial_asset["asset_val"].values,
        "Optimal (€m)": A_opt,
        "Change (€m)": A_opt - initial_asset["asset_val"].values
    })

    # FIX: Apply formatting only to specific numeric columns using a dictionary
    st.dataframe(
        allocation_df.style.format({
            "Current (€m)": "{:.1f}",
            "Optimal (€m)": "{:.1f}",
            "Change (€m)": "{:+.1f}"  # Adds a +/- sign for changes
        }),
        use_container_width=True,
        hide_index=True
    )

with col_right:
    st.markdown("**Weights (%)**")
    weights_df = pd.DataFrame({
        "Asset Class": ["Gov Bonds", "Corp Bonds", "Equity 1", "Equity 2", "Property", "T-Bills"],
        "Current (%)": initial_asset["asset_weight"].values * 100,
        "Optimal (%)": w_opt * 100,
        "Change (%)": (w_opt - initial_asset["asset_weight"].values) * 100
    })

    # FIX: Apply formatting only to specific numeric columns
    st.dataframe(
        weights_df.style.format({
            "Current (%)": "{:.1f}",
            "Optimal (%)": "{:.1f}",
            "Change (%)": "{:+.1f}"
        }),
        use_container_width=True,
        hide_index=True
    )
st.markdown("---")

# --- 7. Pie Charts (Restored Visuals) ---
st.markdown("**Visual Allocation Comparison**")
fig_pie, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig_pie.patch.set_facecolor('white')

asset_labels = ["Gov Bonds", "Corp Bonds", "Equity 1", "Equity 2", "Property", "T-Bills"]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9']


def make_autopct(values):
    def my_autopct(pct):
        return f'{pct:.1f}%' if pct > 3 else ''

    return my_autopct


# Current Pie
current_weights = initial_asset["asset_weight"].values * 100
wedges1, _, autotexts1 = ax1.pie(
    current_weights, colors=colors, autopct=make_autopct(current_weights),
    startangle=90, textprops={'fontsize': 11, 'weight': 'bold'}, pctdistance=0.80,
    explode=[0.03 if w > 10 else 0 for w in current_weights], shadow=True,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
ax1.set_title('Current Portfolio', fontsize=14, fontweight='bold', pad=25)

# Optimal Pie
optimal_weights = w_opt * 100
wedges2, _, autotexts2 = ax2.pie(
    optimal_weights, colors=colors, autopct=make_autopct(optimal_weights),
    startangle=90, textprops={'fontsize': 11, 'weight': 'bold'}, pctdistance=0.80,
    explode=[0.03 if w > 10 else 0 for w in optimal_weights], shadow=True,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
ax2.set_title('Optimal Portfolio', fontsize=14, fontweight='bold', pad=25)

# Legend
fig_pie.legend(wedges1, asset_labels, title="Asset Classes", loc="lower center",
               bbox_to_anchor=(0.5, -0.08), ncol=6, frameon=True, shadow=True)
st.pyplot(fig_pie, use_container_width=True)

st.markdown("---")

# --- 8. Key Allocation Changes (Text) ---
st.markdown("**Key Allocation Changes:**")
changes = []
for i, label in enumerate(asset_labels):
    change = (w_opt[i] * 100) - (initial_asset["asset_weight"].values[i] * 100)
    if abs(change) > 1.0:
        icon = "🟢" if change > 0 else "🔴"
        direction = "↑" if change > 0 else "↓"
        changes.append(f"{icon} **{label}**: {direction} {abs(change):.1f}pp")

if changes:
    cols = st.columns(2)
    mid = len(changes) // 2
    with cols[0]:
        for c in changes[:mid]: st.markdown(c)
    with cols[1]:
        for c in changes[mid:]: st.markdown(c)
else:
    st.info("No significant allocation changes (< 1 percentage point)")

st.markdown("---")

# =========================================================
# 🧩 MARGINAL SCR ANALYSIS (NEW SECTION)
# =========================================================
st.subheader("🧩 Marginal SCR Contribution")
st.markdown("Breakdown of how each asset class contributes to the total Market SCR.")

# --- 1. Calculate Marginal SCR (Logic adapted from notebook) ---
# We need to reconstruct the SCR components for the OPTIMAL portfolio
# (The 'best' object contains aggregated results, but we need component vectors for marginal calc)

# Re-calculate component SCRs for the optimal allocation
# Note: This logic repeats some backend work but is necessary if 'best' doesn't store raw components
from backend.solvency_calc import scr_interest_rate, scr_eq, scr_prop, scr_sprd, marginal_scr, allocate_marginal_scr

# Get parameters from config (or session state if stored, otherwise re-load)
cfg = load_config()
corr_down, corr_up = get_corr_matrices(cfg)
solv = get_solvency_params(cfg)

# Recalculate SCR components for the BEST portfolio
# 1. Interest Rate Risk
# We need shocks from session state or re-compute
if st.session_state.get('use_eiopa_curves', True):
    ir_up, ir_down = compute_ir_shocks_from_eiopa(liab_duration=liab_value,
                                                  verbose=False)  # Note: passed liab_value as placeholder in previous turn, ideally use liab_duration
else:
    # Fallback or pull from somewhere else. For now re-compute or assume safe defaults if missing
    # Better: Store shocks in session state in Inputs.py
    ir_up, ir_down = 0.011, 0.009

# Compute Interest SCR
scr_int_res = scr_interest_rate(
    best["A_opt"],
    initial_asset["asset_dur"],
    liab_value,
    liab_duration,
    ir_up,
    ir_down
)

# 2. Equity Risk
A_eq1_opt = best["A_opt"][2]
A_eq2_opt = best["A_opt"][3]
scr_eq_res = scr_eq(A_eq1_opt, A_eq2_opt, solv["equity_1_param"], solv["equity_2_param"], solv["rho"])

# 3. Property Risk
scr_prop_val = scr_prop(best["A_opt"][4], solv["prop_params"])

# 4. Spread Risk
# Need spread shock (re-compute or store)
corp_dur = initial_asset.loc["corp_bond", "asset_dur"]
spread_shock_val = compute_spread_shock_eiopa(corp_dur, verbose=False)
scr_sprd_val = scr_sprd(best["A_opt"][1], spread_shock_val)

# 5. Prepare Vector for Marginal Calculation
scr_vec = np.array([
    scr_int_res["SCR_interest"],
    scr_eq_res["SCR_eq_total"],
    scr_prop_val,
    scr_sprd_val
])

# Determine direction (Up/Down) from Interest result
direction = scr_int_res["direction"]

# Run Marginal Calculation (Risk Level)
marg_risk_df = marginal_scr(scr_vec, direction, corr_down.values, corr_up.values)

# Run Allocation Calculation (Asset Level)
# Need to reconstruct params dict
alloc_params = {
    "interest_down": ir_down,
    "interest_up": ir_up,
    "spread": spread_shock_val,
    "equity_type1": solv["equity_1_param"],
    "equity_type2": solv["equity_2_param"],
    "property": solv["prop_params"],
    "rho": solv["rho"]
}

# We need a DataFrame for the Optimal Portfolio state to pass to allocate_marginal_scr
# The function expects 'initial_asset' structure but with optimal values?
# Actually, looking at the function 'allocate_marginal_scr' in notebook/file:
# It uses 'initial_asset' for durations and 'asset_val' for Equities.
# We should pass a dataframe representing the OPTIMAL portfolio.
opt_asset_df = initial_asset.copy()
opt_asset_df["asset_val"] = best["A_opt"]

mscr_assets = allocate_marginal_scr(marg_risk_df, direction, opt_asset_df, alloc_params)

# --- 2. Display Results ---

col_m1, col_m2 = st.columns([1, 1])

with col_m1:
    st.markdown("**Asset Class Contribution Table**")
    # Clean up table for display
    disp_df = mscr_assets[["asset", "mSCR"]].copy()
    disp_df["Contribution (%)"] = (disp_df["mSCR"] / disp_df["mSCR"].sum()) * 100

    # Rename assets for cleaner display
    asset_map = {
        "gov_bond": "Gov Bonds", "corp_bond": "Corp Bonds",
        "equity_1": "Equity 1", "equity_2": "Equity 2",
        "property": "Property", "t_bills": "T-Bills"
    }
    disp_df["asset"] = disp_df["asset"].map(asset_map)

    st.dataframe(
        disp_df.style.format({
            "mSCR": "{:.1f}",
            "Contribution (%)": "{:.1f}%"
        }).background_gradient(subset=["mSCR"], cmap="Reds"),
        use_container_width=True,
        hide_index=True
    )

with col_m2:
    st.markdown("**SCR Contribution by Asset**")

    # Prepare data for Pie Chart (handle negative mSCR if any, usually 0 or small)
    # mSCR can be negative for hedging assets (Gov bonds), so a bar chart might be better technically,
    # but user asked for Pie. We will clip negatives to 0 for Pie or separate them.
    # Let's use a waterfall or bar chart for better accuracy, OR a Donut chart for positive contributors.

    plot_data = disp_df[disp_df["mSCR"] > 0].copy()

    fig_mscr, ax_mscr = plt.subplots(figsize=(6, 6))
    ax_mscr.pie(
        plot_data["mSCR"],
        labels=plot_data["asset"],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    ax_mscr.set_title("Risk Contribution (Positive Only)")
    st.pyplot(fig_mscr, use_container_width=True)

    if (disp_df["mSCR"] < 0).any():
        st.caption("Note: Assets with negative marginal SCR (risk reducers/hedges) are excluded from the pie chart.")

st.markdown("---")

# =========================================================
# 💾 EXPORT SECTION (Updated to include Marginal SCR)
# =========================================================
st.subheader("💾 Export Results")
col_export1, col_export2, col_export3 = st.columns(3)

# Prepare DataFrames for Export
# 1. Allocation
allocation_df = pd.DataFrame({
    "Asset Class": ["Gov Bonds", "Corp Bonds", "Equity 1", "Equity 2", "Property", "T-Bills"],
    "Current (€m)": initial_asset["asset_val"].values,
    "Optimal (€m)": best["A_opt"],
    "Change (€m)": best["A_opt"] - initial_asset["asset_val"].values,
    "Weight (%)": best["w_opt"] * 100
})

# 2. Frontier
frontier_export = opt_df[["gamma", "return", "SCR_market", "solvency", "objective"]].copy()

# 3. Risk Contribution (The new data)
# We use 'disp_df' calculated in the Marginal SCR section above
# Check if it exists to avoid errors if that section failed
if 'disp_df' not in locals():
    risk_export_df = pd.DataFrame()  # Empty if calculation failed
else:
    risk_export_df = disp_df.copy()

with col_export1:
    st.markdown("**📊 Allocation Data**")
    csv_allocation = allocation_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Allocation CSV", csv_allocation, "optimal_allocation.csv", "text/csv")

with col_export2:
    st.markdown("**📈 Frontier Data**")
    csv_frontier = frontier_export.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Frontier CSV", csv_frontier, "efficient_frontier.csv", "text/csv")

with col_export3:
    st.markdown("**📄 Summary Report**")
    summary_text = f"""SOLVENCY II OPTIMIZATION REPORT
{'=' * 60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Total Assets: €{initial_asset['asset_val'].sum():,.1f}m
Liabilities:  €{liab_value:,.1f}m

KEY METRICS (Optimal)
{'=' * 60}
Return:   {best_ret:.2%} (Current: {current_ret:.2%})
Solvency: {best_sol:.1%} (Current: {current_sol:.1%})
SCR:      €{best_SCR:,.1f}m
BOF:      €{best_BOF:,.1f}m

RISK CONTRIBUTION
{'=' * 60}
"""
    if not risk_export_df.empty:
        for _, row in risk_export_df.iterrows():
            summary_text += f"{row['asset']:<15} SCR: {row['mSCR']:>8.1f} ({row['Contribution (%)']:>5.1f}%)\n"

    st.download_button("📥 Download Text Report", summary_text, "report.txt", "text/plain")

# --- Advanced Exports (Excel & JSON) ---
with st.expander("🔧 Advanced Export Options"):
    col_xl1, col_xl2 = st.columns(2)

    with col_xl1:
        st.markdown("**📊 Export as Excel Workbook**")
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Summary
            pd.DataFrame({
                "Metric": ["Total Assets", "Liabilities", "Optimal Return", "Optimal Solvency", "Optimal SCR"],
                "Value": [initial_asset['asset_val'].sum(), liab_value, best_ret, best_sol, best_SCR]
            }).to_excel(writer, sheet_name='Summary', index=False)

            # Sheet 2: Allocation
            allocation_df.to_excel(writer, sheet_name='Allocation', index=False)

            # Sheet 3: Risk Contribution (NEW)
            if not risk_export_df.empty:
                risk_export_df.to_excel(writer, sheet_name='Risk Contribution', index=False)

            # Sheet 4: Frontier
            frontier_export.to_excel(writer, sheet_name='Efficient Frontier', index=False)

        st.download_button(
            "📥 Download Excel Workbook", output.getvalue(),
            "optimization_results.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with col_xl2:
        st.markdown("**📦 Export as JSON**")

        # Build JSON structure
        json_export = {
            "metadata": {"generated": str(datetime.now())},
            "optimal_portfolio": {
                "metrics": {
                    "return": best_ret,
                    "solvency": best_sol,
                    "SCR": best_SCR,
                    "BOF": best_BOF
                },
                "allocation": {
                    "amounts": best["A_opt"].tolist(),
                    "weights": best["w_opt"].tolist()
                },
                "risk_decomposition": risk_export_df.to_dict(orient="records") if not risk_export_df.empty else []
            }
        }

        st.download_button("📥 Download JSON", json.dumps(json_export, indent=2), "results.json", "application/json")
# Sensitivity Analysis
st.subheader("🔬 Sensitivity Analysis")

st.markdown("""
    Explore how the optimal portfolio allocation changes under different scenarios.
    Adjust the parameters below to see the impact on asset allocation and risk metrics.
    """)
sens_tab1, sens_tab2, sens_tab3 = st.tabs([
    "📊 Return Scenarios",
    "⚡ Shock Scenarios",
    "🎯 Custom Scenario"
])

# --- TAB 1: Return Scenarios ---
with sens_tab1:
    scenario_choice = st.selectbox(
        "Select Scenario",
        [
            "Base Case (Current)",
            "Optimistic (+0.5% all assets)",
            "Pessimistic (-0.5% all assets)",
            "Higher Equity Returns (+2.0%)",
            "Lower Bond Returns (-1.0%)"
        ],
        key="return_scenario"
    )

    if st.button("🔄 Run Return Sensitivity", key="run_return_sens"):
        with st.spinner("Running sensitivity analysis..."):
            try:
                base_returns = initial_asset["asset_ret"].values.copy()

                if scenario_choice == "Base Case (Current)":
                    scenario_returns = base_returns
                elif "Optimistic" in scenario_choice:
                    scenario_returns = base_returns + 0.005
                elif "Pessimistic" in scenario_choice:
                    scenario_returns = base_returns - 0.005
                elif "Higher Equity" in scenario_choice:
                    scenario_returns = base_returns.copy()
                    scenario_returns[2] += 0.02
                    scenario_returns[3] += 0.02
                elif "Lower Bond" in scenario_choice:
                    scenario_returns = base_returns.copy()
                    scenario_returns[0] -= 0.01
                    scenario_returns[1] -= 0.01

                # Setup Inputs
                sens_asset = initial_asset.copy()
                sens_asset["asset_ret"] = scenario_returns

                # Re-run optimization
                cfg = load_config()
                corr_down, corr_up = get_corr_matrices(cfg)
                solv = get_solvency_params(cfg)

                # Reconstruct limits (simplified for sensitivity)
                allocation_limits = pd.DataFrame({
                    "asset": ["gov_bond", "illiquid_assets", "t_bills", "corp_bond"],
                    "min_weight": [0.25, 0.0, 0.01, 0.0],
                    "max_weight": [0.75, 0.20, 0.05, 0.50],
                }).set_index("asset")

                # Default params (using session state shocks would be better, but defaulting for safety)
                params = {
                    "interest_down": 0.009, "interest_up": 0.011, "spread": 0.103,
                    "equity_type1": solv["equity_1_param"], "equity_type2": solv["equity_2_param"],
                    "property": solv["prop_params"], "rho": solv["rho"],
                }

                sens_opt_df = solve_frontier_combined(
                    initial_asset=sens_asset, liab_value=liab_value, liab_duration=liab_value,
                    # Using liab_value as placeholder for actual liab_duration
                    corr_downward=corr_down, corr_upward=corr_up, allocation_limits=allocation_limits, params=params
                )

                if not sens_opt_df.empty:
                    sens_best = sens_opt_df.loc[sens_opt_df["objective"].idxmax()]
                    st.success("Sensitivity analysis completed!")

                    # --- NEW PLOT GENERATION ---
                    st.subheader("Scenario Optimal Portfolio Comparison")
                    fig = plot_scenario_comparison(opt_df, best, sens_best, current_ret, current_sol, sens_df=sens_opt_df)
                    st.pyplot(fig, use_container_width=True)

                    # Comparison Metrics
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("New Return", f"{sens_best['return']:.2%}", f"{(sens_best['return'] - best_ret):.2%}")
                    sc2.metric("New Solvency", f"{sens_best['solvency'] * 100:.1f}%",
                               f"{(sens_best['solvency'] - best_sol) * 100:.1f}pp")

                    # Allocation Change
                    st.markdown("**Asset Allocation Change (pp)**")
                    diff = (sens_best["w_opt"] - best["w_opt"]) * 100
                    df_diff = pd.DataFrame([diff], columns=["Gov", "Corp", "Eq1", "Eq2", "Prop", "TB"])
                    st.dataframe(df_diff.style.format("{:+.2f}"), hide_index=True)
                else:
                    st.error("Optimization failed for this scenario.")

            except Exception as e:
                st.error(f"Error: {e}")

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

                    # --- NEW PLOT GENERATION ---
                    st.subheader("Scenario Optimal Portfolio Comparison")
                    fig = plot_scenario_comparison(opt_df, best, sens_best, current_ret, current_sol, sens_df=sens_opt_df)
                    st.pyplot(fig, use_container_width=True)

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
            # Initialize with current asset returns as defaults
            defaults = initial_asset["asset_ret"].values
            custom_r_gov = st.number_input("Gov Bonds Return", -0.10, 0.20, float(defaults[0]), 0.001, format="%.3f", key="c_r_gov")
            custom_r_corp = st.number_input("Corp Bonds Return", -0.10, 0.20, float(defaults[1]), 0.001, format="%.3f", key="c_r_corp")
            custom_r_eq1 = st.number_input("Equity 1 Return", -0.10, 0.20, float(defaults[2]), 0.001, format="%.3f", key="c_r_eq1")
            custom_r_eq2 = st.number_input("Equity 2 Return", -0.10, 0.20, float(defaults[3]), 0.001, format="%.3f", key="c_r_eq2")
            custom_r_prop = st.number_input("Property Return", -0.10, 0.20, float(defaults[4]), 0.001, format="%.3f", key="c_r_prop")
            custom_r_tb = st.number_input("T-Bills Return", -0.10, 0.20, float(defaults[5]), 0.001, format="%.3f", key="c_r_tb")

        with col2:
            st.markdown("**⚡ Custom Shocks**")
            # Load default parameters from config/session for defaults
            cfg = load_config()
            solv = get_solvency_params(cfg)
            
            # Use session state if available (from auto-calc), else defaults
            # Ideally these should come from st.session_state if stored in Inputs.py
            # Here we use safe defaults or previously computed values if possible
            
            custom_ir_up = st.number_input("IR Up Shock", 0.0, 0.20, 0.011, 0.001, format="%.3f", key="c_ir_up")
            custom_ir_down = st.number_input("IR Down Shock", 0.0, 0.20, 0.009, 0.001, format="%.3f", key="c_ir_down")
            custom_spread = st.number_input("Spread Shock", 0.0, 0.30, 0.103, 0.001, format="%.3f", key="c_spread")
            
            custom_eq1 = st.number_input("Equity 1 Shock", 0.0, 1.0, solv["equity_1_param"], 0.01, format="%.2f", key="c_eq1")
            custom_eq2 = st.number_input("Equity 2 Shock", 0.0, 1.0, solv["equity_2_param"], 0.01, format="%.2f", key="c_eq2")
            custom_prop = st.number_input("Property Shock", 0.0, 1.0, solv["prop_params"], 0.01, format="%.2f", key="c_prop")

        if st.button("🚀 Run Custom Scenario", key="run_custom_sens", type="primary"):
            with st.spinner("Running custom scenario analysis..."):
                try:
                    # 1. Build Custom Inputs
                    custom_asset = initial_asset.copy()
                    custom_asset["asset_ret"] = [custom_r_gov, custom_r_corp, custom_r_eq1,
                                                 custom_r_eq2, custom_r_prop, custom_r_tb]

                    # Re-load matrices
                    corr_down, corr_up = get_corr_matrices(cfg)

                    # Build Allocation Limits (Same as base case)
                    allocation_limits = pd.DataFrame({
                        "asset": ["gov_bond", "illiquid_assets", "t_bills", "corp_bond"],
                        "min_weight": [0.25, 0.0, 0.01, 0.0],
                        "max_weight": [0.75, 0.20, 0.05, 0.50],
                    }).set_index("asset")

                    # Build Params Dictionary
                    custom_params = {
                        "interest_down": custom_ir_down,
                        "interest_up": custom_ir_up,
                        "spread": custom_spread,
                        "equity_type1": custom_eq1,
                        "equity_type2": custom_eq2,
                        "property": custom_prop,
                        "rho": solv["rho"],
                    }

                    # 2. Run Optimization
                    custom_opt_df = solve_frontier_combined(
                        initial_asset=custom_asset,
                        liab_value=liab_value,
                        liab_duration=liab_duration,
                        corr_downward=corr_down,
                        corr_upward=corr_up,
                        allocation_limits=allocation_limits,
                        params=custom_params
                    )

                    if not custom_opt_df.empty:
                        custom_best_idx = custom_opt_df["objective"].idxmax()
                        custom_best = custom_opt_df.loc[custom_best_idx]

                        st.success("✓ Custom scenario completed!")

                        # --- A. PLOT SCENARIO COMPARISON ---
                        st.subheader("Scenario Optimal Portfolio Comparison")
                        # Pass 'sens_df=custom_opt_df' to see the new frontier line
                        fig = plot_scenario_comparison(opt_df, best, custom_best, current_ret, current_sol, sens_df=custom_opt_df)
                        st.pyplot(fig, use_container_width=True)

                        # --- B. KEY METRICS ---
                        st.markdown("**Impact of Custom Assumptions**")
                        col_m1, col_m2, col_m3 = st.columns(3)
                        
                        with col_m1:
                            st.metric(
                                "Expected Return", 
                                f"{custom_best['return']:.2%}", 
                                f"{(custom_best['return'] - best_ret):.2%}"
                            )
                        with col_m2:
                            st.metric(
                                "Solvency Ratio", 
                                f"{custom_best['solvency']*100:.1f}%", 
                                f"{(custom_best['solvency'] - best_sol)*100:.1f}pp"
                            )
                        with col_m3:
                            st.metric(
                                "SCR Market", 
                                f"€{custom_best['SCR_market']:.1f}m", 
                                f"€{(custom_best['SCR_market'] - best_SCR):.1f}m",
                                delta_color="inverse"
                            )

                        # --- C. ALLOCATION CHANGE ---
                        st.markdown("**Asset Allocation Response**")
                        
                        alloc_comparison = pd.DataFrame({
                            "Asset Class": ["Gov Bonds", "Corp Bonds", "Equity Type 1",
                                            "Equity Type 2", "Property", "T-Bills"],
                            "Base Case (%)": best["w_opt"] * 100,
                            "Custom Case (%)": custom_best["w_opt"] * 100,
                            "Change (pp)": (custom_best["w_opt"] - best["w_opt"]) * 100
                        })

                        st.dataframe(
                            alloc_comparison.style.format({
                                "Base Case (%)": "{:.1f}",
                                "Custom Case (%)": "{:.1f}",
                                "Change (pp)": "{:+.1f}"
                            }).background_gradient(subset=["Change (pp)"], cmap="RdYlGn", vmin=-10, vmax=10),
                            use_container_width=True
                        )
                        
                    else:
                        st.error("❌ Optimization failed for this custom scenario. Try relaxing constraints or improving returns.")

                except Exception as e:
                    st.error(f"Error in custom scenario: {str(e)}")
                    # st.exception(e) # Uncomment for debugging
    st.markdown("---")
