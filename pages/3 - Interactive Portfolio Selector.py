import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from datetime import datetime
import json
from backend.style_utils import apply_sidebar_style

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

st.set_page_config(page_title="Interactive Selector", layout="wide")

st.title("🎚️ Interactive Portfolio Selector")

apply_sidebar_style()

st.markdown("Explore different risk-return tradeoffs along the efficient frontier by selecting alternative portfolios.")

# --- 1. Session State Checks ---
if not st.session_state.get("optimization_run"):
    st.warning("⚠️ Please run the optimization on the **Inputs** page first.")
    st.stop()

# --- 2. Load Data ---
opt_df = st.session_state["opt_df"]
initial_asset = st.session_state["initial_asset"]
liab_value = st.session_state["liab_value"]

# Find Optimal (Best)
best_idx = opt_df["objective"].idxmax()
best = opt_df.loc[best_idx]

# Calculate Current Metrics (Approx)
current_ret = float((initial_asset["asset_ret"] * initial_asset["asset_weight"]).sum())
current_SCR = float(best["SCR_market"])
current_sol = (initial_asset['asset_val'].sum() - liab_value) / current_SCR

# --- 3. Interactive Slider ---
if "selected_frontier_idx" not in st.session_state:
    st.session_state["selected_frontier_idx"] = int(best_idx)

selected_idx = st.slider(
    "Select Portfolio Point",
    min_value=0,
    max_value=len(opt_df) - 1,
    value=st.session_state["selected_frontier_idx"],
    key="frontier_slider",
    help="Slide to explore profiles. 0 = Aggressive, N = Conservative"
)
st.session_state["selected_frontier_idx"] = selected_idx
selected_port = opt_df.iloc[selected_idx]

# --- 4. Interactive Chart ---
fig_interactive, ax = plt.subplots(figsize=(12, 7))
ax.set_facecolor('#f8f9fa')
fig_interactive.patch.set_facecolor('white')

# Plot Frontier
ax.plot(opt_df["solvency"] * 100, opt_df["return"] * 100, '-', color='#4ECDC4', linewidth=2.5, alpha=0.4, zorder=1)
ax.scatter(opt_df["solvency"] * 100, opt_df["return"] * 100, s=60, color='#4ECDC4', alpha=0.4, edgecolors='white',
           zorder=2)

# Current Portfolio (Red Diamond)
ax.scatter(current_sol * 100, current_ret * 100, s=350, c='#E74C3C', marker='D', edgecolors='#C0392B', linewidth=2.5,
           label='Current Portfolio', zorder=4)

# Optimal Portfolio (Gold Star)
ax.scatter(best["solvency"] * 100, best["return"] * 100, s=600, c='#FFD700', marker='*', edgecolors='#FF8C00',
           linewidth=3, label='Optimal Portfolio', zorder=5)

# Selected Portfolio (Purple Circle - Moving)
sel_sol = selected_port["solvency"] * 100
sel_ret = selected_port["return"] * 100
ax.scatter(sel_sol, sel_ret, s=500, c='#9B59B6', marker='o', edgecolors='#6C3483', linewidth=3,
           label='Selected Portfolio', zorder=7)

# Annotations
ax.annotate(f'SELECTED\n{sel_ret:.2f}% | {sel_sol:.1f}%', xy=(sel_sol, sel_ret), xytext=(25, 25),
            textcoords='offset points', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#9B59B6', edgecolor='#6C3483', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='#6C3483', lw=2.5), color='white', zorder=8)

ax.axvline(x=100, color='#95a5a6', linestyle='--', linewidth=2.5, alpha=0.6, label='100% Solvency')

ax.set_xlabel('Solvency Ratio (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Expected Return (%)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fancybox=True, framealpha=0.9, shadow=True)
ax.grid(True, alpha=0.25, linestyle='--')

st.pyplot(fig_interactive, use_container_width=True)

st.markdown("---")

# --- 5. Metrics Dashboard ---
st.subheader("📊 Selected Portfolio Metrics")

# Row 1
c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected Return", f"{selected_port['return'] * 100:.2f}%",
          f"{(selected_port['return'] - best['return']) * 100:+.2f}pp vs Opt")
c2.metric("Solvency Ratio", f"{selected_port['solvency'] * 100:.1f}%",
          f"{(selected_port['solvency'] - best['solvency']) * 100:+.1f}pp vs Opt")
c3.metric("SCR Market", f"€{selected_port['SCR_market']:.1f}m",
          f"€{(selected_port['SCR_market'] - best['SCR_market']):+.1f}m", delta_color="inverse")
c4.metric("Risk Aversion (γ)", f"{selected_port['gamma']:.4f}")

# Row 2 (No Duration)
c5, c6, c7 = st.columns(3)
c5.metric("Basic Own Funds", f"€{selected_port['BOF']:.1f}m")
c6.metric("Objective Value", f"{selected_port['objective']:.4f}",
          f"{(selected_port['objective'] - best['objective']):+.4f}")

# Profile Indicator
pos_pct = (selected_idx / (len(opt_df) - 1)) * 100
if pos_pct < 33:
    prof_lbl, prof_col = "Aggressive", "🔴"
elif pos_pct < 67:
    prof_lbl, prof_col = "Balanced", "🟡"
else:
    prof_lbl, prof_col = "Conservative", "🟢"
c7.metric("Profile", f"{prof_col} {prof_lbl}")

st.markdown("---")

# --- 6. Allocation (Table + Pie) ---
st.subheader("💼 Selected Portfolio Allocation")
col_alloc_l, col_alloc_r = st.columns([1.2, 1])

labels = ["Gov Bonds", "Corp Bonds", "Equity 1", "Equity 2", "Property", "T-Bills"]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9']

with col_alloc_l:
    st.markdown("**Detailed Allocation**")
    df_detail = pd.DataFrame({
        "Asset Class": labels,
        "Weight (%)": selected_port["w_opt"] * 100,
        "Amount (€m)": selected_port["A_opt"],
        "Return (%)": initial_asset["asset_ret"].values * 100
    })
    st.dataframe(
        df_detail.style.format({"Weight (%)": "{:.1f}", "Amount (€m)": "{:.1f}", "Return (%)": "{:.2f}"})
        .background_gradient(subset=["Weight (%)"], cmap="Blues"),
        use_container_width=True, hide_index=True
    )

with col_alloc_r:
    st.markdown("**Visual Allocation**")
    fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
    weights = selected_port["w_opt"] * 100


    def make_autopct(values):
        def my_autopct(pct):
            return f'{pct:.1f}%' if pct > 3 else ''

        return my_autopct


    ax_pie.pie(weights, labels=None, colors=colors, autopct=make_autopct(weights),
               startangle=90, textprops={'fontsize': 10, 'weight': 'bold'}, pctdistance=0.85,
               explode=[0.03 if w > 10 else 0 for w in weights], shadow=True,
               wedgeprops={'edgecolor': 'white', 'linewidth': 2})

    ax_pie.legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    st.pyplot(fig_pie, use_container_width=True)

st.markdown("---")

# --- 7. Comparison Tabs ---
st.subheader("📊 Comparison Analysis")
tab1, tab2 = st.tabs(["📈 vs Optimal Portfolio", "📉 vs Current Portfolio"])

with tab1:
    comp_opt = pd.DataFrame({
        "Metric": ["Expected Return (%)", "Solvency Ratio (%)", "SCR Market (€m)", "Basic Own Funds (€m)", "Objective"],
        "Selected": [
            f"{selected_port['return'] * 100:.2f}", f"{selected_port['solvency'] * 100:.1f}",
            f"{selected_port['SCR_market']:.1f}", f"{selected_port['BOF']:.1f}", f"{selected_port['objective']:.4f}"
        ],
        "Optimal": [
            f"{best['return'] * 100:.2f}", f"{best['solvency'] * 100:.1f}",
            f"{best['SCR_market']:.1f}", f"{best['BOF']:.1f}", f"{best['objective']:.4f}"
        ],
        "Diff": [
            f"{(selected_port['return'] - best['return']) * 100:+.2f}pp",
            f"{(selected_port['solvency'] - best['solvency']) * 100:+.1f}pp",
            f"{(selected_port['SCR_market'] - best['SCR_market']):+.1f}",
            f"{(selected_port['BOF'] - best['BOF']):+.1f}",
            f"{(selected_port['objective'] - best['objective']):+.4f}"
        ]
    })
    st.dataframe(comp_opt, use_container_width=True, hide_index=True)

with tab2:
    comp_curr = pd.DataFrame({
        "Metric": ["Expected Return (%)", "Solvency Ratio (%)", "SCR Market (€m)"],
        "Selected": [
            f"{selected_port['return'] * 100:.2f}", f"{selected_port['solvency'] * 100:.1f}",
            f"{selected_port['SCR_market']:.1f}"
        ],
        "Current": [
            f"{current_ret * 100:.2f}", f"{current_sol * 100:.1f}", f"{current_SCR:.1f}"
        ],
        "Diff": [
            f"{(selected_port['return'] - current_ret) * 100:+.2f}pp",
            f"{(selected_port['solvency'] - current_sol) * 100:+.1f}pp",
            f"{(selected_port['SCR_market'] - current_SCR):+.1f}"
        ]
    })
    st.dataframe(comp_curr, use_container_width=True, hide_index=True)

st.markdown("---")

# --- 8. Export Section ---
st.subheader("💾 Export Selected Portfolio")
c_ex1, c_ex2, c_ex3 = st.columns(3)

with c_ex1:
    st.markdown("**📊 Allocation CSV**")
    csv_data = pd.DataFrame({
        "Asset": labels, "Weight (%)": selected_port["w_opt"] * 100, "Amount (€m)": selected_port["A_opt"]
    }).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv_data, "selected_portfolio.csv", "text/csv")

with c_ex2:
    st.markdown("**📄 Summary Report**")
    report_txt = f"""SELECTED PORTFOLIO REPORT
Date: {datetime.now()}
Return: {selected_port['return']:.2%}
Solvency: {selected_port['solvency']:.1%}
SCR: €{selected_port['SCR_market']:.1f}m

ALLOCATION:
"""
    for i, lab in enumerate(labels):
        report_txt += f"{lab}: {selected_port['w_opt'][i] * 100:.1f}% (€{selected_port['A_opt'][i]:.1f}m)\n"

    st.download_button("📥 Download Text Report", report_txt, "report.txt", "text/plain")

with c_ex3:
    st.markdown("**📦 JSON Export**")
    json_data = {
        "metrics": {"return": float(selected_port['return']), "solvency": float(selected_port['solvency'])},
        "allocation": list(selected_port["A_opt"])
    }
    st.download_button("📥 Download JSON", json.dumps(json_data, indent=2), "selected.json", "application/json")
