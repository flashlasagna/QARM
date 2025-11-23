import streamlit as st
import os
import sys
from backend.style_utils import apply_sidebar_style


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

# Page configuration
st.set_page_config(
    page_title="Solvency II Portfolio Optimiser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance that adapts to Light/Dark mode automatically
st.markdown("""
    <style>
    /* Global font styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* REMOVED: Fixed background colors. Now inherits from Streamlit theme. */

    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: inherit; /* Adapts to dark/light theme automatically */
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .subtitle {
        font-size: 1.4rem;
        color: inherit;
        opacity: 0.8; /* Uses opacity instead of fixed grey for contrast */
        margin-bottom: 1.5rem;
        font-weight: 400;
        line-height: 1.6;
    }

    .intro-text {
        font-size: 1.05rem;
        color: inherit;
        line-height: 1.8;
        margin-bottom: 2.5rem;
        max-width: 95%;
    }

    /* Navigation box styling */
    .nav-box {
        /* Transparent grey background - looks good on white AND black */
        background-color: rgba(128, 128, 128, 0.1); 
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    .nav-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: inherit;
        margin-bottom: 0.75rem;
    }

    .nav-box ul {
        margin-left: 0;
        padding-left: 1.2rem;
        line-height: 1.8;
    }

    .nav-box li {
        color: inherit;
        margin-bottom: 0.5rem;
    }

    /* Section headers */
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: inherit;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 0.75rem;
        letter-spacing: -0.01em;
    }

    /* Team section styling */
    .team-category {
        font-size: 1.4rem;
        font-weight: 700;
        /* Brighter blue that is legible on both dark and light backgrounds */
        color: #60a5fa; 
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        letter-spacing: -0.01em;
    }

    .profile-container {
        margin-bottom: 1.2rem;
    }

    .profile-name {
        font-weight: 500;
        font-size: 1.05rem;
        color: inherit;
        margin-bottom: 0.2rem;
    }

    .profile-role {
        color: inherit;
        opacity: 0.7;
        font-size: 0.9rem;
        font-style: italic;
    }

    /* Key metrics styling */
    .metric-box {
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #60a5fa; /* Bright blue */
        margin-bottom: 0.25rem;
    }

    .metric-label {
        font-size: 0.9rem;
        color: inherit;
        opacity: 0.8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Tab content styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: inherit;
        opacity: 0.7;
    }

    .stTabs [data-baseweb="tab"]:hover {
        opacity: 1;
    }

    /* Info box */
    .info-box {
        /* Semi-transparent blue background */
        background-color: rgba(59, 130, 246, 0.1); 
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin: 1.5rem 0;
    }

    .info-box p {
        color: inherit;
        margin: 0;
        line-height: 1.6;
    }

    /* General Text overrides to ensure nothing is hidden */
    p, span, div, h1, h2, h3, h4, h5, h6, li {
        color: inherit;
    }

    /* Footer styling */
    .footer {
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 2px solid rgba(128, 128, 128, 0.2);
        text-align: center;
        color: inherit;
        opacity: 0.6;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

apply_sidebar_style()
# ============================================================================
# HEADER SECTION
# ============================================================================

col_left, col_right = st.columns([3, 1])

with col_left:
    st.markdown('<div class="main-title">Solvency II Portfolio Optimiser</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Exploring the trade-off between asset allocation, profitability, and Solvency II Market Risk capital.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="intro-text">
    This web application enables life insurers to optimise their investment portfolios under the Solvency II regulatory framework. 
    By taking the insurer's balance sheet as input, the tool applies the <strong>Solvency II Standard Formula</strong> for the 
    Market Risk module to compute capital-efficient portfolios along an efficient frontier. The optimiser balances expected 
    returns on assets against the Solvency Capital Requirement (SCR), helping insurers make informed decisions that enhance 
    profitability while maintaining strong solvency positions. The methodology is based on the academic paper by 
    <em>Machado (2024)</em>, adapted for practical portfolio management.
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("### Key Benefits")
    st.markdown(
        '<div class="metric-box"><div class="metric-value">99.5%</div><div class="metric-label">Confidence Level</div></div>',
        unsafe_allow_html=True)
    st.markdown("")
    st.markdown(
        '<div class="metric-box"><div class="metric-value">6</div><div class="metric-label">Asset Classes</div></div>',
        unsafe_allow_html=True)
    st.markdown("")
    st.markdown(
        '<div class="metric-box"><div class="metric-value">Real-time</div><div class="metric-label">Optimization</div></div>',
        unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# QUICK NAVIGATION - NO BUTTONS, NO CARDS
# ============================================================================

st.markdown("### Quick Navigation")
st.markdown("")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="nav-box">', unsafe_allow_html=True)
    st.markdown('<div class="nav-title">1️⃣ Input Data</div>', unsafe_allow_html=True)
    st.markdown("""
    - Upload or enter current asset allocation
    - Define liability profile and balance sheet
    - Set investment constraints and regulatory limits
    - Specify duration parameters for interest rate sensitivity
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="nav-box">', unsafe_allow_html=True)
    st.markdown('<div class="nav-title">2️⃣ Results & Efficient Frontier</div>', unsafe_allow_html=True)
    st.markdown("""
    - View the efficient frontier curve
    - Analyse expected return vs SCR trade-off
    - Review marginal SCR contributions by asset class
    - Compare initial vs optimized allocations
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="nav-box">', unsafe_allow_html=True)
    st.markdown('<div class="nav-title">3️⃣ Interactive Portfolio Selector</div>', unsafe_allow_html=True)
    st.markdown("""
    - Navigate along the efficient frontier
    - Inspect optimal allocations at different risk levels
    - Explore SCR composition and diversification benefits
    - Evaluate capital efficiency metrics (ER/mSCR ratios)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# METHODOLOGY & SOLVENCY II OVERVIEW
# ============================================================================

st.markdown('<div class="section-header">Methodology & Solvency II Background</div>', unsafe_allow_html=True)

tabs = st.tabs(["📖 Concept", "⚠️ Solvency II Market Risk", "🔢 Formulas", "📊 Capital Efficiency"])

with tabs[0]:
    st.markdown("""
    **Life insurers operating under Solvency II must hold sufficient capital to meet regulatory requirements.**

    Specifically, they must maintain a Solvency Capital Requirement (SCR) calibrated to a **99.5% confidence level** 
    over a one-year horizon. The SCR ensures the insurer can withstand extreme market events without jeopardising 
    policyholder protection or financial stability.

    **Asset allocation plays a critical role in determining both profitability and the SCR.**

    Different asset classes carry different risk profiles under the Market Risk module of Solvency II. 
    Riskier assets such as equities may offer higher expected returns but also result in higher capital charges, 
    increasing the SCR. Conversely, safer assets like government bonds contribute less to the SCR but may limit 
    overall profitability. 

    **This optimiser helps insurers search for portfolios that maximise expected return on assets while controlling 
    the Market Risk component of the SCR.** By embedding regulatory formulas within a convex quadratic programming 
    framework, the tool generates an efficient frontier that balances profitability with regulatory compliance.

    The methodology is inspired by academic research (Kouwenberg, 2017, 2018; Braun et al., 2015) and applied to 
    real-world Portuguese life insurer data, incorporating investment limits to ensure practical applicability.
    """)

    st.markdown(
        '<div class="info-box"><p><strong>Key Insight:</strong> The optimiser penalises high capital requirements '
        'while maximising expected returns, enabling insurers to identify portfolios that achieve the best balance '
        'between profitability and capital efficiency.</p></div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown("**Market Risk Sub-Modules Modelled:**")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        - **Interest Rate Risk**: Sensitivity to upward and downward shifts in the interest rate term structure, 
          affecting both assets (bonds, T-bills) and liabilities (Best Estimate of technical provisions)
        - **Equity Risk**: Risk from changes in equity prices
          - *Type 1*: EEA/OECD equities (~39% shock)
          - *Type 2*: Non-EEA, unlisted, funds (~49% shock)
        - **Property Risk**: Risk from fluctuations in real estate values (~25% shock)
        """)

    with col_b:
        st.markdown("""
        - **Spread Risk**: Risk from changes in credit spreads of corporate bonds (~10.3% risk factor for bonds)
        - **Currency Risk**: Assumed negligible (EUR-denominated portfolio with hedging)
        - **Concentration Risk**: Assumed negligible (diversified benchmarks ensure exposure below regulatory thresholds)
        """)

    st.markdown("")
    st.markdown("**Asset Classes Considered:**")

    asset_data = {
        "Asset Class": [
            "Government Bonds",
            "Corporate Bonds",
            "Equity Type 1",
            "Equity Type 2",
            "Property",
            "Treasury Bills"
        ],
        "Benchmark/Index": [
            "Bloomberg Euro Aggregate Treasury",
            "Bloomberg Euro Aggregate Corporate",
            "MSCI World",
            "Non-EEA equities, unlisted shares, investment funds",
            "FTSE EPRA/NAREIT Developed Europe",
            "German 3-month Treasury Bills"
        ],
        "Primary Solvency II Risk": [
            "Interest Rate",
            "Interest Rate + Spread",
            "Equity (39% shock)",
            "Equity (49% shock)",
            "Property (25% shock)",
            "Interest Rate (minimal)"
        ]
    }

    st.table(asset_data)

    st.markdown("""
    Under the Solvency II Standard Formula, each asset class is subject to prescribed shocks. These shocks are used 
    to compute **stand-alone capital requirements** for each risk type. The stand-alone SCRs are then aggregated 
    using a **correlation matrix** to capture diversification benefits and produce the total Market Risk SCR.
    """)

with tabs[2]:
    st.markdown("**Stand-alone SCR for a risk type i:**")
    st.markdown(
        "This measures the capital required to absorb losses from a single risk factor, calculated as the absolute change in Basic Own Funds (BOF) after applying the regulatory shock.")
    st.latex(r"SCR_i = |V_0 - V_{\text{shock}}|")
    st.markdown(
        "where $V_0$ is the market value before the shock and $V_{\\text{shock}}$ is the market value after applying the regulatory stress.")

    st.markdown("")
    st.markdown("**Spread Risk:**")
    st.markdown(
        "The capital requirement for spread risk aggregates the product of market value and spread risk factor for each bond $j$.")
    st.latex(r"SCR_{\text{spread}} = \sum_{j} MV_j \cdot s_j")
    st.markdown(
        "where $MV_j$ is the market value of bond $j$ and $s_j$ is the spread risk factor (approximately 10.3% for corporate bonds).")

    st.markdown("")
    st.markdown("**Aggregation of Market Risk Sub-Modules:**")
    st.markdown(
        "The total Market Risk SCR accounts for diversification benefits using a correlation matrix $\\rho_{ij}$:")
    st.latex(r"SCR_{\text{total}} = \sqrt{\sum_{i} \sum_{j} SCR_i \cdot SCR_j \cdot \rho_{ij}}")
    st.markdown(
        "This formula ensures that risks are not simply added together; instead, **correlation effects** reduce the total capital requirement when asset classes do not move perfectly together.")

    st.markdown("")
    st.markdown("**Objective Function:**")
    st.markdown(
        "The optimiser maximises expected return on Basic Own Funds while penalising high SCR through a penalty parameter $\\lambda$:")
    st.latex(r"\max \; \mathbb{E}[r^T x] - \lambda \cdot \sqrt{SCR^T \cdot \rho \cdot SCR}")
    st.markdown("""
    By adjusting $\\lambda$, the model generates an **efficient frontier** showing the trade-off between expected 
    return and the Market Risk solvency ratio. Higher $\\lambda$ values prioritise capital efficiency (lower SCR), 
    while lower values prioritise return maximisation.
    """)

with tabs[3]:
    st.markdown("**Capital Efficiency Metrics**")

    st.markdown("""
    A critical insight from the Machado (2024) paper is the concept of **capital efficiency**, measured by the ratio 
    of expected return to marginal SCR contribution (ER/mSCR ratio). This metric helps identify which asset classes 
    provide the best "bang for the buck" in terms of return per unit of capital consumed.
    """)

    st.markdown("**Key Findings from the Portuguese Life Insurer Case Study:**")

    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown("""
        **Most Capital-Efficient Assets:**
        - **Corporate Bonds** (ER/mSCR ≈ 0.60-0.61): Highest capital efficiency despite spread risk charges
        - **Property** (ER/mSCR ≈ 0.29-0.31): Second-best efficiency, attractive for long-term investors
        - **Government Bonds & T-Bills**: No spread risk charge, negative marginal SCR (reduce overall capital needs)
        """)

    with col_y:
        st.markdown("""
        **Less Capital-Efficient Assets:**
        - **Equity Type 1** (MSCI World): High capital charge (~39%) limits attractiveness
        - **Equity Type 2** (ER/mSCR ≈ 0.14-0.15): Lowest efficiency due to 49% capital charge, often excluded from optimised portfolios
        """)

    st.markdown("""
    **Practical Implications:**

    The case study demonstrated that by reallocating from equities to corporate bonds and property, the Portuguese 
    life insurer increased expected return on assets from **3.40% to 3.74%** while maintaining the same Market Risk 
    SCR. This improvement was achieved by:

    1. **Maximising allocation to corporate bonds** (up to investment limits)
    2. **Increasing property exposure** from 2.5% to 10.4% of assets
    3. **Eliminating Equity Type 2** holdings entirely due to poor capital efficiency
    4. **Reducing low-return treasury bills** to minimum liquidity requirements

    The optimised portfolio achieved a **Market Risk Solvency Ratio of 186%**, well above regulatory minimums, 
    while significantly improving profitability.
    """)

# ============================================================================
# HOW TO USE THIS WEBAPP
# ============================================================================

st.markdown('<div class="section-header">How to Use This Webapp</div>', unsafe_allow_html=True)

st.markdown("""
**Follow these steps to optimise your portfolio:**

1. **Define Your Current Position**: Navigate to the **Inputs** page and select your ETF tickers. 
   You can choose to **"Auto-calculate Returns & Shocks"** to fetch market data automatically, or uncheck it to enter manual assumptions.
   Enter your current asset allocation across the six asset classes and define your liability profile (Best Estimate and Duration).

2. **Set Investment Constraints**: Specify investment limits for each asset class to align the optimisation with your 
   investment strategy. Common constraints include:
   - Maximum corporate bonds allocation (e.g., 50%)
   - Maximum illiquid assets (equity + property) allocation (e.g., 30%)
   - Minimum treasury bills for liquidity requirements (e.g., 1-5%)
   - Upper and lower bounds for government bonds

3. **Configure Solvency II Parameters**: 
   - **Durations**: You must manually input the modified duration for your bond portfolios and liabilities.
   - **Auto-Mode**: If enabled, the app fetches live ECB yield curves for the risk-free rate. You can customize the *Base Risk Free Rate*, *Credit Spread Proxy*, and *Equity Risk Premium*.
   - **Manual Mode**: You can fully override all expected returns and Solvency II shock parameters (Interest Rate Up/Down, Spread, Equity, Property).

4. **Run the Optimisation**: Click **"Optimize Portfolio"**. The model calculates the entire efficient frontier by solving 
   multiple convex quadratic programs (sweeping the penalty parameter $\\lambda$) to maximize return while ensuring 
   the **Solvency Ratio remains above 100%**.

5. **Review Key Performance Indicators**: On the **Results** page, examine:
   - **Efficient Frontier Chart**: A visual plot of Expected Return vs Solvency Ratio, highlighting the current vs. optimal portfolio.
   - **Allocation Tables**: Detailed comparison of weights and amounts (in €m) between your initial and the optimal portfolio.
   - **Risk Decomposition**: A breakdown of the Marginal SCR contribution by asset class.
   - **Capital Efficiency**: Key metrics including the Solvency Ratio, Market SCR, and Basic Own Funds (BOF).

6. **Conduct Sensitivity Analysis**: Use the tabs in the Results section to test robustness:
   - **Return Scenarios**: Test how the optimal portfolio changes under "Pessimistic" or "Optimistic" market views.
   - **Shock Scenarios**: See how the portfolio reacts if regulatory shocks (e.g., Equity shock) increase.
   - **Custom Scenarios**: Define your own stress tests to see the impact on the optimal allocation.

7. **Select and Analyse Target Portfolio**: Use the **Interactive Portfolio Selector** page to:
   - Slide along the efficient frontier to pick a portfolio that matches your specific risk appetite (e.g., "Aggressive" vs "Conservative").
   - Inspect the specific allocation and risk metrics for that chosen point.
   - Export the final portfolio weights and summary report to CSV, Excel, or JSON.
""")

# ============================================================================
# ACADEMIC FOUNDATION
# ============================================================================

st.markdown('<div class="section-header">Academic Foundation</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("""
    **This application is based on the Master's thesis:**

    *"Trade-Off Between Asset Allocation and Solvency II Requirements: The Case of a Portuguese Life Insurer"*  
    **Daniel Alexandre da Silva Machado (2024)**  
    Master in Actuarial Science, ISEG Lisbon School of Economics & Management

    **Key References:**
    - Kouwenberg, R. (2017, 2018): Strategic Asset Allocation and Risk Budgeting for Insurers under Solvency II
    - Braun et al. (2015): Portfolio Optimization under Solvency II Constraints
    - Höring, D. (2013): Will Solvency II Market Risk Requirements Bite?
    - Escobar et al. (2018): Implications of Solvency II on Investment Strategies

    The optimisation framework uses **Convex Quadratic Programming (CQP)** to solve the trade-off between profitability 
    and capital efficiency, incorporating real balance sheet data, regulatory shocks from the Solvency II Standard Formula, 
    and investment limits to ensure practical applicability.
    """)

with col_right:
    st.markdown("**Regulatory Framework**")
    st.markdown("""
    - **Directive 2009/138/EC** (Solvency II)
    - **Delegated Regulation (EU) 2015/35** (Standard Formula specifications)
    - **EIOPA Guidelines** on Market Risk calibration

    **Data Sources:**
    - Bloomberg Euro Aggregate Indices
    - MSCI World Index
    - FTSE EPRA/NAREIT Index
    - EIOPA Risk-Free Rate Term Structures
    """)

# ============================================================================
# ABOUT THE PROJECT TEAM
# ============================================================================

st.markdown('<div class="section-header">About the Project Team</div>', unsafe_allow_html=True)

st.markdown('<div class="team-category">Back-end Development (Portfolio Optimisation & Modelling)</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    st.markdown('<div class="profile-name">Hoai Thuong Phan</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">MSc Quantitative Asset & Risk Management</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    st.markdown('<div class="profile-name">Jacopo Sinigaglia</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">MSc Quantitative Asset & Risk Management</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    st.markdown('<div class="profile-name">Angélique Nhât-Ngân Trinh</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">MSc Quantitative Asset & Risk Management</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")
st.markdown('<div class="team-category">Front-end Development (Web Application & Visualisation)</div>',
            unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    st.markdown('<div class="profile-name">Ruben Mimouni</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">MSc Quantitative Asset & Risk Management</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    st.markdown('<div class="profile-name">Maxime Bezier</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">MSc Quantitative Asset & Risk Management</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")

col_disclaimer, col_contact = st.columns([3, 1])

with col_disclaimer:
    st.markdown("""
    This web application was developed as part of a **Quantitative Asset & Risk Management** course project, 
    inspired by the paper *"Trade-Off Between Asset Allocation and Solvency II Requirements – The Case of a 
    Portuguese Life Insurer"* (Machado, 2024). The tool is designed for **educational and prototype purposes only** 
    and should not be considered as regulatory or investment advice. 

    **Important Disclaimers:**
    - Results depend on input assumptions (expected returns, durations, shocks) which are subject to uncertainty
    - The model focuses on Market Risk only; other Solvency II modules (e.g., Life Underwriting Risk) are excluded
    - Investment limits and constraints should be reviewed by qualified professionals
    - Regulatory approval may be required before implementing optimised portfolios

    Users are encouraged to consult with qualified actuaries, risk managers, and regulatory experts before making 
    investment decisions based on this tool.
    """)


with col_contact:
    st.markdown("**Supervision**")
    st.markdown("""
    **Prof. Divernois Marc-Aurèle**  
    *Academic Supervisor*

    HEC Lausanne  
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "*© 2024 Solvency II Portfolio Optimiser | Master of Science in Finance | HEC Lausanne*")
st.markdown("*For educational purposes only – Not for regulatory or investment advice*")
st.markdown("</div>", unsafe_allow_html=True)
