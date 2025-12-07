import streamlit as st
from backend.style_utils import apply_sidebar_style
st.set_page_config(
    page_title="Solvency II Portfolio Optimiser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_sidebar_style()
import os
import sys



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

st.markdown("""
    <style>
    /* --- HOME PAGE CONTENT STYLING --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: inherit;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .subtitle {
        font-size: 1.4rem;
        color: inherit;
        opacity: 0.8;
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

    /* Navigation box */
    .nav-box {
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

    /* Team section */
    .team-category {
        font-size: 1.4rem;
        font-weight: 700;
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

    /* Metrics */
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
        color: #60a5fa;
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

    /* Tab Styling */
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

    /* Overrides */
    p, span, div, h1, h2, h3, h4, h5, h6, li {
        color: inherit;
    }

    /* Footer */
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


# ============================================================================
# HEADER SECTION
# ============================================================================

col_left, col_right = st.columns([3, 1])

with col_left:
    st.markdown('<div class="main-title">Solvency II Portfolio Optimiser</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Exploring the trade-off between investment returns and Solvency II Market Risk capital for insurers.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="intro-text">
    This web application enables insurers to optimise their investment portfolios under the Solvency II regulatory framework. 
    By taking the insurer's balance sheet as input, the tool applies the <strong>Solvency II Standard Formula</strong> for the 
    Market Risk module to compute capital-efficient portfolios along an efficient frontier. The optimiser balances expected 
    returns on assets against the Solvency Capital Requirement (SCR), helping insurers make informed decisions that enhance 
    profitability while maintaining strong solvency positions. The methodology is based on the academic paper by 
    <em>Kouwenberg, R. (2017, 2018)</em>.
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
    - Define client's balance sheet profile 
    - Set investment constraints and regulatory limits
    - Set features for measuring asset's expected return
    - Specify solvency II shocks parameters 
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="nav-box">', unsafe_allow_html=True)
    st.markdown('<div class="nav-title">2️⃣ Results & Efficient Frontier</div>', unsafe_allow_html=True)
    st.markdown("""
    - View the efficient frontier curve
    - Compare initial vs optimized allocations
    - Compare initial vs optimized SCR and marginal SCR
    - Perform sensitivity analysis under different scenarios 
    - Export results to csv or json for further analysis or reporting purposes 
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="nav-box">', unsafe_allow_html=True)
    st.markdown('<div class="nav-title">3️⃣ Interactive Portfolio Selector</div>', unsafe_allow_html=True)
    st.markdown("""
    - Navigate along the efficient frontier
    - Explore optimal allocations across different solvency-ratio levels 
    - Review key metrics for all frontier portfolios and export any you choose
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# METHODOLOGY & SOLVENCY II OVERVIEW
# ============================================================================

st.markdown('<div class="section-header">Methodology & Solvency II Background</div>', unsafe_allow_html=True)

tabs = st.tabs(["Concept", "Solvency II Market Risk", "Asset Class", "SCR Formulas", "Optimization"])

with tabs[0]:
    st.markdown("""
    
    Solvency II is the European Union’s risk-based regulatory framework that sets capital, governance, and reporting requirements to ensure insurers remain sufficiently resilient to financial and actuarial shocks.           
                
    **Insurers operating under Solvency II must hold sufficient capital to meet regulatory requirements.**

    Specifically, they must maintain a Solvency Capital Requirement (SCR) calibrated to a **99.5% confidence level** 
    over a one-year horizon. The SCR ensures the insurer can withstand extreme market events without jeopardising 
    policyholder protection or financial stability.

    **Asset allocation plays a critical role in determining both profitability and the SCR.**

    Different asset classes carry different risk profiles under the Market Risk module of Solvency II. 
    Riskier assets such as equities may offer higher expected returns but also result in higher capital charges, 
    increasing the SCR. Conversely, safer assets like government bonds contribute less to the SCR but may limit 
    overall profitability. 

    **This optimiser helps insurers search for portfolios that maximise expected return on assets while controlling 
    the Market Risk component of the SCR.** 
    
    By embedding regulatory formulas within a convex quadratic programming framework, the tool generates an efficient frontier that balances profitability with regulatory compliance.

    The methodology is inspired by academic research (Kouwenberg, 2017, 2018; Braun et al., 2015). To ensure practical applicability, we propose
    to incorporate investment limits on each asset class. 
    """)

    st.markdown(
        '<div class="info-box"><p><strong>Key Insight:</strong> The optimiser penalises high capital requirements '
        'while maximising expected returns, enabling insurers to identify portfolios that achieve the best balance '
        'between profitability and capital efficiency.</p></div>', unsafe_allow_html=True)

with tabs[1]:

    col_a, col_b = st.columns(2)

    # LEFT COLUMN — Market Risk Sub-Modules
    with col_a:
        st.markdown("""
        **Market Risk Sub-Modules Modelled**

        - **Interest Rate Risk**  
          Sensitivity to upward and downward shifts in the interest rate term structure 

        - **Equity Risk**  
          Risk from changes in equity prices  
            - *Type 1*: EEA/OECD equities (~39% shock)  
            - *Type 2*: Non-EEA, unlisted, funds (~49% shock)  
                    

        - **Spread Risk**  
          Risk from changes in credit spreads of corporate bonds, defined via a duration-based piecewise stress  

        - **Property Risk**  
          Risk from fluctuations in real estate values (~25% shock)  

        - **Currency Risk**  
          Assumed negligible in our work for EUR-denominated mandates with hedging

        - **Concentration Risk**  
          Assumed negligible in our work given diversified benchmarks staying below regulatory limits
        """)

    # RIGHT COLUMN — Shock Parameter Details
    with col_b:
        st.markdown("""
        **Solvency II Shock Parameters**

        **Interest Rate Shock**  
        The latest EIOPA risk-free rate term structure (Oct 2025), including the base curve and shocked up/down curves, is used. 
        Shocks are linearly interpolated to match the client’s liability duration.  
        (Solvency II Delegated Regulation (EU) 2015/35, Articles 164–167)

        **Equity Shock**  
        - Type 1: **39%**  
        - Type 2: **49%**  
        (Solvency II Delegated Regulation (EU) 2015/35, Articles 168–169)

        **Spread Shock**  
        The spread shock is applied using the Solvency II duration-based formula, where d is modified duration of corporate bond portfolio:
        """)

        st.latex(r"""
        {\scriptsize
        \text{shock}(d)=
        \begin{cases}
        0.03\,d, & d \le 5 \\[3pt]
        0.15 + 0.017(d-5), & 5 < d \le 10 \\[3pt]
        0.235 + 0.012(d-10), & 10 < d \le 20 \\[3pt]
        \min\{0.355 + 0.005(d-20),\, 1.0\}, & d > 20
        \end{cases}
        }
        """)

        st.markdown("*Solvency II Delegated Regulation (EU) 2015/35, Article 176*")


        st.markdown("""
        **Property Shock:** 25%  
        (Solvency II Delegated Regulation (EU) 2015/35, Article 170)
        """)

# ASSET CLASSES
with tabs[2]:

    st.markdown("**Asset Classes Considered**")

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
    Under the Solvency II Standard Formula, each asset class is assigned prescribed shocks. These shocks determine the 
    **stand-alone capital requirements (SCRs)** for each risk type. Stand-alone SCRs are then aggregated using the 
    regulatory **correlation matrix**, capturing diversification benefits to produce the total Market Risk SCR.
    """)

with tabs[3]:

    st.markdown("**Stand-Alone SCR — Interest Rate Risk**")
    st.markdown("""
    The SCR for interest rate risk is the maximum loss between the upward and downward shocks, 
    based on duration-weighted sensitivities of assets and liabilities:
    """)
    st.latex(r"""
    SCR_{\text{IR}} =
    \max\left\{
    s_{\text{down}} \cdot (D_L L - \sum_i D_{A,i}A_i),\;
    s_{\text{up}} \cdot (\sum_i D_{A,i}A_i - D_L L),\;
    0
    \right\}
    """)

    st.markdown("**Stand-Alone SCR — Equity Risk**")
    st.markdown("""
    Equity Type 1 and Type 2 are shocked separately and then aggregated using the Solvency II correlation.
    The correlation parameter ρ = 0.75 is prescribed in the Solvency II Delegated Regulation (EU) 2015/35,
    Annex IV – Market Risk Correlation Matrix.
    """)

    st.latex(r"""
    SCR_1 = A_{\text{eq1}} \cdot s_1 (0.39),\quad 
    SCR_2 = A_{\text{eq2}} \cdot s_2 (0.49)
    """)

    st.latex(r"""
    SCR_{\text{eq}} =
    \sqrt{
    SCR_1^2 + 2\rho\,SCR_1 SCR_2 + SCR_2^2
    }
    """)

    st.markdown("**Stand-Alone SCR — Spread Risk**")
    st.markdown("""
    The SCR for spread risk is calculated by applying the duration-based Solvency II spread shock 
    to corporate bond exposure:
    """)
    st.latex(r"""
    SCR_{\text{spr}} = A_{\text{corp}} \cdot s_{\text{spread}}(d)
    """)


    st.markdown("**Stand-Alone SCR — Property Risk**")
    st.latex(r"""
    SCR_{\text{prop}} = A_{\text{prop}} \cdot \text{s}_{property} (0.25)
    """)

    st.markdown("")
    st.markdown("**Aggregation of Market Risk Sub-Modules:**")
    st.markdown(
        "The total Market Risk SCR accounts for diversification benefits using a correlation matrix $\\rho_{ij}$:")
    st.latex(r"SCR_{\text{total}} = \sqrt{\sum_{i} \sum_{j} SCR_i \cdot SCR_j \cdot \rho_{ij}}")

    st.markdown("""
               EIOPA provides two versions of correlation matrices (Annex IV), depending on whether the decisive interest rate shock is
                a downward or upward movement.
    """)

with tabs[4]:

    st.markdown("**Objective Function:**")
    st.markdown(
        "The optimiser maximises expected return on asset portfolio while penalising high SCR through a penalty parameter $\\gamma$:")
    st.latex(r"""
    \max_{w,s} \; E[f(w,s)] = E[r_A^{T} w] - \gamma\, s^{T} R s
    """)
    st.markdown("""
                Where $w$ is the vector of weights invested in each asset class, and $s$ is the vector of stand-alone SCR. 
                $R$ is the Solvency II correlation matrix to aggregate stand-alone SCR.""")
    st.markdown("""
    By adjusting $\\gamma$, the model generates an efficient frontier showing the trade-off between expected 
    return and the Market Risk solvency ratio (=BOF/SCR). Higher $\\gamma$ values prioritise capital efficiency (lower SCR), 
    while lower values prioritise return maximisation.
    """)

    st.markdown("**Optimization Constraints:**")

    st.markdown("All portfolios in the optimization must satisfy the following regulatory and model-imposed constraints:")

    col1, col2 = st.columns(2)

    # -------------------------
    # LEFT COLUMN
    # -------------------------
    with col1:

        # ---- 1. Interest Rate ----
        with st.expander("### 1. Interest Rate Risk Constraints (Nonlinear)", expanded=False):
            st.markdown("The stand-alone SCR for interest rate risk must cover both upward and downward shocks:")
            st.latex(r"""
            s_{\text{IR}} \ge 
            \Delta y_{\text{up}}\,(A_{\text{dur}} - L_{\text{dur}})
            """)
            st.latex(r"""
            s_{\text{IR}} \ge 
            \Delta y_{\text{down}}\,(L_{\text{dur}} - A_{\text{dur}})
            """)
            st.markdown("where:")
            st.latex(r"A_{\text{dur}} = T \sum_i w_i D_{A,i}")
            st.latex(r"L_{\text{dur}} = D_L\, L")

        # ---- 2. Equity ----
        with st.expander("### 2. Equity Risk Constraint (Nonlinear)"):
            st.markdown("The Solvency II aggregation must hold between Equity Type 1 and Type 2 exposures:")
            st.latex(r"""
            s_{\text{eq}} \ge
            \sqrt{
            SCR_1^2 + 2\rho\, SCR_1 SCR_2 + SCR_2^2
            }
            """)
            st.markdown("with:")
            st.latex(r"SCR_1 = s_1\, A_{\text{eq1}}")
            st.latex(r"SCR_2 = s_2\, A_{\text{eq2}}")
            st.markdown("and the Solvency II correlation:")
            st.latex(r"\rho = 0.75")

        # ---- 3. Property ----
        with st.expander("### 3. Property Risk Constraint (Nonlinear)"):
            st.markdown("The property SCR must cover the 25% market-value shock:")
            st.latex(r"""
            s_{\text{prop}} \ge 0.25\, A_{\text{prop}}
            """)

    # -------------------------
    # RIGHT COLUMN
    # -------------------------
    with col2:

        # ---- 4. Spread ----
        with st.expander("### 4. Spread Risk Constraint (Nonlinear)"):
            st.markdown("Spread SCR must cover the Solvency II duration-based spread shock:")
            st.latex(r"""
            s_{\text{spr}} \ge s_{\text{spread}}(d)\, A_{\text{corp}}
            """)
            st.markdown("where $s_{\\text{spread}}(d)$ is the duration-based Solvency II spread stress.")

        # ---- 5. Budget ----
        with st.expander("### 5. Budget Constraint (Linear)"):
            st.markdown("Total portfolio weights must sum to 1:")
            st.latex(r"""
            \sum_{i=1}^{6} w_i = 1
            """)

        # ---- 6. Allocation Limits ----
        with st.expander("### 6. Allocation Limits (Linear)"):
            st.markdown("Each asset class must lie within its allowed minimum and maximum weights:")
            st.latex(r"""
            w_i^{\min} \le w_i \le w_i^{\max}
            """)

        # ---- 7. Solvency Ratio ----
        with st.expander("### 7. Solvency Ratio Constraint (Nonlinear)"):
            st.markdown("The portfolio must satisfy a minimum solvency ratio defined by user:")
            st.latex(r"""
            \frac{BOF}{SCR_{\text{market}}} \ge \text{Solvency}_{\min}
            """)
            st.markdown("which is equivalent to:")
            st.latex(r"""
            BOF - \text{Solvency}_{\min} \cdot SCR_{\text{market}} \ge 0
            """)

        # ---- 8. Non-Negativity ----
        with st.expander("### 8. Non-Negativity of Weights and SCR Components"):
            st.markdown("All portfolio weights and SCR components must be non-negative:")
            st.latex(r"w_i \ge 0 \quad \text{for all asset classes } i")
            st.latex(r"""
            s_{\text{IR}},\; s_{\text{eq}},\; s_{\text{prop}},\; s_{\text{spr}} \ge 0
            """)


# ============================================================================
# HOW TO USE THIS WEBAPP
# ============================================================================

st.markdown('<div class="section-header">How to Use This Webapp</div>', unsafe_allow_html=True)

st.markdown("""
**Follow these steps to optimise your portfolio:**

1. **Define Your Current Position**: Navigate to the **Inputs** page. 
    - Enter your current asset allocation and durations across the six asset classes and define your liability profile (Best Estimate and Duration).
    - For expected returns and shock, you can choose to **"Auto-calculate Returns & Shocks"** to fetch market data automatically, or uncheck it to enter manual assumptions.
    - You may select the historical window used to estimate expected returns. 

2. **Set Investment Constraints**: Specify investment limits for each asset class to align the optimisation with your 
   investment strategy. Common constraints include:
   - Maximum corporate bonds allocation (e.g., 50%)
   - Maximum illiquid assets (equity + property) allocation (e.g., 30%)
   - Minimum treasury bills for liquidity requirements (e.g., 1-5%)
   - Upper and lower bounds for government bonds

3. **Configure Solvency II Parameters**: 
   - **Auto-Mode**: If enabled, the app fetches live ECB yield curves for the risk-free rate. You can customize the *Base Risk Free Rate*, *Credit Spread Proxy*, and *Equity Risk Premium*.
   - **Manual Mode**: You can fully override all expected returns and Solvency II shock parameters (Interest Rate Up/Down, Spread, Equity, Property).
   - You can set minimum solvency ratio you want to use in the optimization algorithm (default is 150%).

4. **Run the Optimisation**: Click **"Optimize Portfolio"**. The model calculates the entire efficient frontier by solving 
   multiple convex quadratic programs (sweeping the penalty parameter $\\lambda$) to maximize return while ensuring 
   the **Solvency Ratio remains above the defined level.

5. **Review Key Performance Indicators**: On the **Results** page, examine:
   - **Efficient Frontier Chart**: A visual plot of Expected Return vs Solvency Ratio, highlighting the current vs. optimal portfolio.
   - **Allocation Tables**: Detailed comparison of weights and amounts (in €m) between your initial and the optimal portfolio.
   - **Risk Decomposition**: A breakdown of the SCR contribution by risk type and marginal contribution by asset class.
   - **Capital Efficiency**: Key metrics including the Solvency Ratio, Market SCR, and Basic Own Funds (BOF).
""")

st.markdown("""
6. **Conduct Sensitivity Analysis**: Use the scenario tabs in the Results section to test the robustness of the optimal portfolio under different market and regulatory conditions.""")

with st.expander("Return Scenarios (Profitability Stress)", expanded=False):

    st.markdown("*Optimistic (+0.5%)*")
    st.markdown("""
    Simulates a mild bull market where expected returns across all asset classes 
    increase by 50 bps. Tests whether the portfolio captures upside potential 
    without breaching solvency limits.
    """)

    st.markdown("*Pessimistic (-0.5%)*")
    st.markdown("""
    Represents a weak or stagnant market where expected returns fall by 50 bps. 
    Evaluates whether the portfolio remains profitable and solvent under poor performance.
    """)

    st.markdown("*Higher Equity Returns (+2.0%)*")
    st.markdown("""
    Assumes a strong conviction in equity markets: expected equity returns increase 
    by 200 bps. Tests whether the optimizer shifts toward equities despite higher risk.
    """)

    st.markdown("*Lower Bond Returns (-1.0%)*")
    st.markdown("""
    Models a fixed-income downturn, with a 100 bps drop in bond returns. 
    Examines whether the optimizer reallocates toward Property or Equities 
    to maintain profitability.
    """)

# =======================
# SHOCK SCENARIOS
# =======================
with st.expander("Shock Scenarios (Capital Requirement Stress)", expanded=False):

    st.markdown("*Stressed Shocks (+50%)*")
    st.markdown("""
    A severe systemic stress test where all Solvency II capital charges increase by 50%. Tests the portfolio’s resilience under extreme regulatory tightening.
    """)

    st.markdown("*Relaxed Shocks (-30%)*")
    st.markdown("""
    A benign environment with capital charges reduced by 30%. Shows how much additional risk capacity appears in calm markets.
    """)

    st.markdown("*Higher Equity Shocks (+15pp)*")
    st.markdown("""
    Capital charges for equities rise by 15 percentage points (e.g., 39% → 54%, 49% → 64%). Tests whether equities remain attractive when their capital cost increases.
    """)

    st.markdown("*Lower Interest Rate Shocks (-20%)*")
    st.markdown("""
    Reduces interest rate shocks by 20%, reflecting a period of rate stability. Lowers the penalty for duration mismatch and may free up capital for other assets.
    """)

st.markdown("""
7. **Select and Analyse Target Portfolio**: Use the **Interactive Portfolio Selector** page to:
   - Slide along the efficient frontier to pick a portfolio that matches your specific risk appetite (e.g., "Aggressive" vs "Conservative").
   - Inspect the specific allocation and risk metrics for that chosen point.
   - Export the final portfolio weights and summary report to CSV, TXT, or JSON.
""")

# ============================================================================
# ACADEMIC FOUNDATION
# ============================================================================

st.markdown('<div class="section-header">Academic Foundation</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("""
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
    st.markdown('<div class="profile-role">MSc Asset & Risk Management</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    st.markdown('<div class="profile-name">Jacopo Sinigaglia</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">MSc Asset & Risk Management</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    st.markdown('<div class="profile-name">Angélique Nhât-Ngân Trinh</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">MSc Asset & Risk Management</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")
st.markdown('<div class="team-category">Front-end Development (Web Application & Visualisation)</div>',
            unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    st.markdown('<div class="profile-name">Ruben Mimouni</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">MSc Asset & Risk Management</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="profile-container">', unsafe_allow_html=True)
    st.markdown('<div class="profile-name">Maxime Bezier</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">MSc Asset & Risk Management</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")

col_disclaimer, col_contact = st.columns([3, 1])

with col_disclaimer:
    st.markdown("""
    This web application was developed as part of a **Quantitative Asset & Risk Management** course project.
    The tool is designed for **educational and prototype purposes only** 
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
