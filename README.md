# Solvency II Portfolio Optimizer

A comprehensive web application for life insurers to optimize their investment portfolios under the Solvency II regulatory framework. This tool enables insurers to balance expected returns against Solvency Capital Requirements (SCR) while maintaining regulatory compliance.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Academic Foundation](#academic-foundation)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Team](#team)
- [Disclaimer](#disclaimer)

## 🎯 Overview

This web application applies the **Solvency II Standard Formula** for the Market Risk module to compute capital-efficient portfolios along an efficient frontier. The optimizer helps insurers make informed decisions that enhance profitability while maintaining strong solvency positions.

### Key Benefits

- **99.5% Confidence Level**: Calibrated to regulatory standards
- **6 Asset Classes**: Comprehensive portfolio coverage
- **Real-time Optimization**: Instant results with interactive visualizations
- **Auto-calculation Mode**: Fetches live market data from ECB and ETF providers

## ✨ Features

### 1. Portfolio Optimization
- Multi-objective optimization balancing expected returns vs. SCR
- Efficient frontier generation using convex quadratic programming
- Support for custom investment constraints and allocation limits

### 2. Market Data Integration
- **Auto-calculation Mode**: 
  - Live ECB risk-free rate curves (EIOPA October 2025)
  - Historical ETF returns via Yahoo Finance
  - Automatic interest rate and spread shock calculations
- **Manual Mode**: Full control over all parameters and assumptions

### 3. Asset Classes Supported
| Asset Class | Benchmark/Index | Primary Risk |
|-------------|-----------------|--------------|
| Government Bonds | Bloomberg Euro Aggregate Treasury | Interest Rate |
| Corporate Bonds | Bloomberg Euro Aggregate Corporate | Interest Rate + Spread |
| Equity Type 1 | MSCI World | Equity (39% shock) |
| Equity Type 2 | Non-EEA equities, unlisted shares | Equity (49% shock) |
| Property | FTSE EPRA/NAREIT Developed Europe | Property (25% shock) |
| Treasury Bills | German 3-month T-Bills | Interest Rate (minimal) |

### 4. Risk Analysis
- **SCR Decomposition**: Interest Rate, Equity, Property, and Spread risks
- **Marginal SCR Contribution**: Asset-level risk attribution
- **Capital Efficiency Metrics**: ER/mSCR ratios for each asset class
- **Diversification Benefits**: Correlation-based aggregation

### 5. Sensitivity Analysis
- **Return Scenarios**: Test optimistic/pessimistic market assumptions
- **Shock Scenarios**: Stress-test under increased regulatory shocks
- **Custom Scenarios**: Define your own parameter combinations

### 6. Interactive Visualizations
- Efficient frontier plots with current vs. optimal portfolios
- Portfolio allocation pie charts and comparison tables
- Risk decomposition waterfall charts
- Interactive portfolio selector along the efficient frontier

### 7. Export Capabilities
- CSV, Excel, and JSON export formats
- Comprehensive summary reports
- Downloadable allocation tables and risk metrics

## 📚 Academic Foundation

This application is based on the Master's thesis:

**"Trade-Off Between Asset Allocation and Solvency II Requirements: The Case of a Portuguese Life Insurer"**  
*Daniel Alexandre da Silva Machado (2024)*  
Master in Actuarial Science, ISEG Lisbon School of Economics & Management

### Key References
- Kouwenberg, R. (2017, 2018): Strategic Asset Allocation and Risk Budgeting for Insurers under Solvency II
- Braun et al. (2015): Portfolio Optimization under Solvency II Constraints
- Höring, D. (2013): Will Solvency II Market Risk Requirements Bite?
- Escobar et al. (2018): Implications of Solvency II on Investment Strategies

### Regulatory Framework
- **Directive 2009/138/EC** (Solvency II)
- **Delegated Regulation (EU) 2015/35** (Standard Formula specifications)
- **EIOPA Guidelines** on Market Risk calibration

## 🚀 Installation

### Prerequisites
- Python 3.11+
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/solvency-ii-optimizer.git
cd solvency-ii-optimizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run Home.py
```

The application will open in your default browser at `http://localhost:8501`

### Docker Setup (Optional)

The project includes a `.devcontainer` configuration for VS Code:

```bash
# Open in VS Code Dev Container
code .
# Select "Reopen in Container" when prompted
```

## 📖 Usage

### Step-by-Step Guide

#### 1. Define Your Current Position
Navigate to the **Inputs** page:
- Select ETF tickers for each asset class
- Choose **Auto-calculate** mode or enter manual parameters
- Input current asset allocation (€ millions)
- Define liability profile (Best Estimate and Duration)

#### 2. Set Investment Constraints
Specify allocation limits:
- Maximum corporate bonds allocation (e.g., 50%)
- Maximum illiquid assets (equity + property, e.g., 30%)
- Minimum treasury bills for liquidity (e.g., 1-5%)
- Government bonds bounds

#### 3. Configure Solvency II Parameters
- **Durations**: Manually input modified durations for bonds and liabilities
- **Auto-Mode**: Fetches live ECB yield curves for risk-free rates
- **Manual Mode**: Override all expected returns and shock parameters

#### 4. Run Optimization
Click **"Optimize Portfolio"** to:
- Calculate the entire efficient frontier
- Maximize return while maintaining Solvency Ratio ≥ 100%
- Generate optimal allocation recommendations

#### 5. Review Results
Examine on the **Results** page:
- Efficient Frontier chart (Current vs. Optimal)
- Allocation comparison tables (weights and amounts)
- Marginal SCR contribution by asset class
- Capital efficiency metrics and solvency ratios

#### 6. Sensitivity Analysis
Test robustness using:
- **Return Scenarios**: Pessimistic/Optimistic market views
- **Shock Scenarios**: Increased regulatory stress
- **Custom Scenarios**: Your own stress tests

#### 7. Interactive Selection
Use the **Portfolio Selector** to:
- Navigate along the efficient frontier
- Select risk/return profiles matching your appetite
- Export final recommendations in multiple formats

## 📂 Project Structure

```
solvency-ii-optimizer/
├── Home.py                      # Main landing page
├── pages/
│   ├── 1 - Inputs.py           # Portfolio input and configuration
│   ├── 2 - Results.py          # Optimization results and analysis
│   └── 3 - Interactive Portfolio Selector.py  # Frontier navigation
├── backend/
│   ├── config_loader.py        # YAML configuration management
│   ├── data_handler.py         # EIOPA data and yield curve fetching
│   ├── data_calculator.py      # Auto-calculation of returns/shocks
│   ├── optimization.py         # Convex quadratic programming solver
│   ├── solvency_calc.py        # SCR calculations and aggregation
│   ├── helpers.py              # Plotting and utility functions
│   └── style_utils.py          # UI styling
├── config/
│   └── config.yaml             # ETF universe and correlation matrices
├── data/
│   └── EIOPA_RFR_20251031_Term_Structures.xlsx  # EIOPA risk-free rates
├── requirements.txt            # Python dependencies
├── test_notebook.ipynb         # Development testing notebook
└── README.md                   # This file
```

## 🔬 Methodology

### Optimization Framework

The optimizer uses **Convex Quadratic Programming (CQP)** to solve:

$$\max \; \mathbb{E}[r^T x] - \lambda \cdot \sqrt{SCR^T \cdot \rho \cdot SCR}$$

Where:
- $r$ = expected returns vector
- $x$ = portfolio weights
- $\lambda$ = penalty parameter (swept to generate frontier)
- $SCR$ = stand-alone capital requirements vector
- $\rho$ = correlation matrix

### SCR Calculation

**Stand-alone SCR for risk type $i$:**

$$SCR_i = |V_0 - V_{\text{shock}}|$$

**Spread Risk:**

$$SCR_{\text{spread}} = \sum_{j} MV_j \cdot s_j$$

**Aggregated Market Risk:**

$$SCR_{\text{total}} = \sqrt{\sum_{i} \sum_{j} SCR_i \cdot SCR_j \cdot \rho_{ij}}$$

### Capital Efficiency

The tool calculates the **ER/mSCR ratio** (Expected Return / Marginal SCR) to identify which assets provide the best return per unit of capital consumed.

**Key Findings from Case Study:**
- **Most Efficient**: Corporate Bonds (ER/mSCR ≈ 0.60-0.61)
- **Second Best**: Property (ER/mSCR ≈ 0.29-0.31)
- **Least Efficient**: Equity Type 2 (ER/mSCR ≈ 0.14-0.15)

## 👥 Team

### Back-end Development (Portfolio Optimization & Modelling)
- **Hoai Thuong Phan** - MSc Quantitative Asset & Risk Management
- **Jacopo Sinigaglia** - MSc Quantitative Asset & Risk Management
- **Angélique Nhât-Ngân Trinh** - MSc Quantitative Asset & Risk Management

### Front-end Development (Web Application & Visualization)
- **Ruben Mimouni** - MSc Quantitative Asset & Risk Management
- **Maxime Bezier** - MSc Quantitative Asset & Risk Management

### Academic Supervision
- **Prof. Divernois Marc-Aurèle** - HEC Lausanne

## ⚠️ Disclaimer

This web application was developed for **educational and prototype purposes only**. It should not be considered as regulatory or investment advice.

**Important Notes:**
- Results depend on input assumptions which are subject to uncertainty
- The model focuses on Market Risk only; other Solvency II modules (e.g., Life Underwriting Risk) are excluded
- Investment limits and constraints should be reviewed by qualified professionals
- Regulatory approval may be required before implementing optimized portfolios
- Users should consult with qualified actuaries, risk managers, and regulatory experts before making investment decisions

## 📄 License

This project is intended for educational purposes only. For commercial use, please contact the development team.

## 🙏 Acknowledgments

- EIOPA for providing risk-free rate term structures
- Bloomberg and MSCI for benchmark indices
- HEC Lausanne for academic support
- Daniel Machado for the foundational research

---

*© 2024 Solvency II Portfolio Optimizer | Master of Science in Finance | HEC Lausanne*
