"""
data_calculator.py — Automated parameter calculations for Solvency II optimization
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ✅ Import from data_handler
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backend.data_handler import (
    load_eiopa_excel,
    interpolate_shocks,
    spread_shock,
    get_interpolated_ecb_yield,  # Bonus: real yield curve data
    expected_return_yf  # Bonus: alternative return calculation
)


# ==========================================
#  EIOPA CURVES (Wrapper for data_handler)
# ==========================================

def load_eiopa_curves(verbose=True):
    """
    Load EIOPA risk-free curves using data_handler.

    Returns
    -------
    dict
        'df': DataFrame with maturity, int_base, int_up, int_down
        'base': List of base rates
        'up': List of up shock rates
        'down': List of down shock rates
        'maturities': List of maturities
    """
    try:
        if verbose:
            print("📊 Loading EIOPA data...")

        df = load_eiopa_excel()  # Uses default path

        if verbose:
            print(f"✓ Loaded EIOPA curves:")
            print(f"  Maturities: {df['maturity'].min():.1f} to {df['maturity'].max():.1f} years")
            print(f"  Sample rates (1y, 5y, 10y, 20y):")
            for mat in [1, 5, 10, 20]:
                idx = (df['maturity'] - mat).abs().argmin()
                row = df.iloc[idx]
                print(f"    {mat}y: base={row['int_base']:.4f}, "
                      f"up={row['int_up']:.4f}, down={row['int_down']:.4f}")

        return {
            'df': df,
            'base': df['int_base'].tolist(),
            'up': df['int_up'].tolist(),
            'down': df['int_down'].tolist(),
            'maturities': df['maturity'].tolist()
        }

    except Exception as e:
        print(f"❌ Error loading EIOPA data: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_ir_shocks_from_eiopa(liab_duration, verbose=True):
    """
    Compute IR shocks for a given liability duration using EIOPA curves.

    Parameters
    ----------
    liab_duration : float
        Liability duration in years
    verbose : bool
        Print detailed output

    Returns
    -------
    tuple
        (shock_up, shock_down) as decimals
    """
    try:
        # Load EIOPA data
        eiopa = load_eiopa_curves(verbose=False)
        if eiopa is None:
            if verbose:
                print("⚠️ EIOPA data unavailable, using defaults")
            return 0.011, 0.009

        # Interpolate shocks for liability duration
        shock_up, shock_down = interpolate_shocks(eiopa['df'], liab_duration)

        if verbose:
            print(f"  EIOPA IR shocks for {liab_duration:.1f}y duration:")
            print(f"    Up shock:   {shock_up:.4f} ({shock_up*100:.2f}%)")
            print(f"    Down shock: {shock_down:.4f} ({shock_down*100:.2f}%)")

        return float(shock_up), float(shock_down)

    except Exception as e:
        print(f"❌ Error computing EIOPA IR shocks: {e}")
        return 0.011, 0.009


def compute_spread_shock_eiopa(duration, verbose=True):
    """
    Compute spread shock using official EIOPA formula.

    Parameters
    ----------
    duration : float
        Modified duration of corporate bonds
    verbose : bool
        Print detailed output

    Returns
    -------
    float
        Spread shock as decimal
    """
    try:
        shock = spread_shock(duration)

        if verbose:
            print(f"  EIOPA spread shock for {duration:.1f}y duration: "
                  f"{shock:.4f} ({shock*100:.2f}%)")

        return float(shock)

    except Exception as e:
        print(f"❌ Error computing spread shock: {e}")
        return 0.103  # Fallback


# ==========================================
#  GOVERNMENT BOND RETURNS (ECB Data)
# ==========================================

def get_gov_bond_return(duration=5.0, use_ecb=True, verbose=True):
    """
    Estimate government bond return.

    Parameters
    ----------
    duration : float
        Target duration in years
    use_ecb : bool
        If True, fetch real ECB yield curve data
        If False, use historical average
    verbose : bool
        Print detailed output

    Returns
    -------
    float
        Expected return as decimal
    """
    if use_ecb:
        try:
            if verbose:
                print(f"📊 Fetching ECB yield for {duration:.1f}y maturity...")

            yield_rate = get_interpolated_ecb_yield(duration, verbose=verbose)

            if verbose:
                print(f"✓ ECB yield: {yield_rate:.3%}")

            return yield_rate

        except Exception as e:
            print(f"⚠️ Could not fetch ECB data: {e}")
            print("   Using fallback: 2.9%")
            return 0.029
    else:
        # Fallback to paper's value
        return 0.029


# ==========================================
#  EXPECTED RETURNS (ETF Historical Data)
# ==========================================

def compute_expected_returns_from_etfs(etf_config, lookback_years=5, method='geometric'):
    """
    Compute expected returns from ETF historical data.

    Parameters
    ----------
    etf_config : dict
        ETF tickers from config.yaml
    lookback_years : int
        Number of years of historical data
    method : str
        'geometric' (default) or 'arithmetic'

    Returns
    -------
    dict
        Expected annualized returns for each asset class
    """
    trading_days = 252
    lookback_days = trading_days * lookback_years

    returns = {}

    asset_mapping = {
        'corp_bond': 'corp_bond',
        'property': 'property',
        'equity_2': 'equity_2',
        'equity_1': 'equity_1'
    }

    for config_key, asset_class in asset_mapping.items():
        if config_key not in etf_config:
            continue

        ticker = etf_config[config_key]['ticker']

        try:
            if method == 'arithmetic':
                # Use data_handler's function
                period = f"{lookback_years}y"
                ret = expected_return_yf(ticker, trading_days=trading_days, period=period)
                if not np.isnan(ret):
                    returns[asset_class] = ret
                    continue

            # Geometric method (default)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 100)

            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if data.empty or len(data) < lookback_days * 0.8:
                print(f"Warning: Insufficient data for {ticker} ({asset_class})")
                returns[asset_class] = None
                continue

            # Handle Adj Close vs Close
            if 'Adj Close' in data.columns:
                prices = data['Adj Close'].dropna()
            elif 'Close' in data.columns:
                prices = data['Close'].dropna()
            else:
                continue

            total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
            actual_years = len(prices) / trading_days
            annualized_return = (1 + total_return) ** (1 / actual_years) - 1

            returns[asset_class] = float(annualized_return)

        except Exception as e:
            print(f"Error fetching {ticker} for {asset_class}: {e}")
            returns[asset_class] = None

    return returns


# ==========================================
#  DURATION-BASED IR SHOCK APPROXIMATION
# ==========================================

def compute_ir_shocks_duration_approx(asset_dur, liab_dur, rate_change=0.01):
    """
    Approximate IR shocks using duration-based approach (simplified).

    Use this when EIOPA data is unavailable.

    Parameters
    ----------
    asset_dur : float
        Portfolio weighted duration
    liab_dur : float
        Liability duration
    rate_change : float
        Assumed parallel shift (default 1%)

    Returns
    -------
    tuple
        (shock_up, shock_down) as decimals
    """
    duration_gap = liab_dur - asset_dur

    if duration_gap > 0:
        shock_down = rate_change * 1.1
        shock_up = rate_change * 0.9
    else:
        shock_up = rate_change * 1.1
        shock_down = rate_change * 0.9

    return float(shock_up), float(shock_down)


# ==========================================
#  BOND DURATION CALCULATION
# ==========================================

def compute_bond_duration(bonds_data):
    """
    Compute weighted average modified duration for bond portfolio.

    Parameters
    ----------
    bonds_data : pd.DataFrame
        Must contain: 'market_value', 'coupon', 'maturity', 'ytm'

    Returns
    -------
    float
        Weighted average modified duration
    """
    def modified_duration_single(coupon, maturity, ytm, face_value=100):
        """Compute modified duration for a single bond."""
        if ytm <= -1:
            return 0

        periods = int(maturity)
        if periods == 0:
            return 0

        pv_weighted_time = 0
        pv_total = 0

        for t in range(1, periods + 1):
            cashflow = coupon * face_value if t < periods else coupon * face_value + face_value
            pv = cashflow / ((1 + ytm) ** t)
            pv_weighted_time += pv * t
            pv_total += pv

        if pv_total == 0:
            return 0

        macaulay_dur = pv_weighted_time / pv_total
        modified_dur = macaulay_dur / (1 + ytm)

        return modified_dur

    bonds_data['duration'] = bonds_data.apply(
        lambda row: modified_duration_single(
            row['coupon'],
            row['maturity'],
            row['ytm']
        ),
        axis=1
    )

    total_value = bonds_data['market_value'].sum()
    if total_value == 0:
        return 0

    weighted_duration = (bonds_data['duration'] * bonds_data['market_value']).sum() / total_value

    return float(weighted_duration)