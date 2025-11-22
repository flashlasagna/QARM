import streamlit as st
import os
import sys

# --- Page Config ---
st.set_page_config(
    page_title="Solvency II Asset Allocation Optimizer",
    layout="wide"
)

# --- Session State Initialization ---
# We initialize these to ensure they exist across all pages
keys_to_init = [
    "opt_df", "initial_asset", "liab_value",
    "liab_duration", "auto_calculated", "optimization_run"
]

for key in keys_to_init:
    if key not in st.session_state:
        st.session_state[key] = None

if "optimization_run" not in st.session_state:
    st.session_state["optimization_run"] = False

# --- Main Landing Content ---
st.title("🏦 Solvency II Asset Allocation Optimizer")

st.markdown("""
### Welcome
This application optimizes asset allocation under Solvency II constraints.

**How to use:**
1. Navigate to the **Inputs** page (sidebar) to define your assets, liabilities, and constraints.
2. Click **Optimize Portfolio**.
3. View the **Results** page for the optimal solution and reports.
4. Use the **Interactive Portfolio Selector** to explore the efficient frontier.
""")

st.info("👈 Please start by selecting **Inputs** from the sidebar.")