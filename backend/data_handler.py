import pandas as pd
import requests
import yfinance as yf
from io import StringIO
import numpy as np

def load_eiopa_excel(path: str = "data/EIOPA_RFR_20251031_Term_Structures.xlsx") -> pd.DataFrame:
    """
    Load EIOPA risk-free rate and shock data from the default Excel file,
    unless a custom path is specified.
    """
    xls = pd.ExcelFile(path)
    base  = pd.read_excel(xls, "RFR_spot_no_VA", usecols=[1, 2]).iloc[9:].reset_index(drop=True)
    up    = pd.read_excel(xls, "Spot_NO_VA_shock_UP", usecols=[2]).iloc[9:].reset_index(drop=True)
    down  = pd.read_excel(xls, "Spot_NO_VA_shock_DOWN", usecols=[2]).iloc[9:].reset_index(drop=True)

    base.columns = ["maturity", "int_base"]
    up.columns, down.columns = ["int_up"], ["int_down"]
    df = pd.concat([base, up, down], axis=1).apply(pd.to_numeric, errors="coerce")

    return df.dropna(subset=["maturity"]).reset_index(drop=True)


def interpolate_shocks(df: pd.DataFrame, maturity_input: float) -> tuple[float, float]:
    """
    Linearly interpolate upward and downward shocks for a given maturity.
    Returns both shocks as decimals (e.g. 0.011, 0.009).
    """
    df = df.sort_values("maturity")
    maturities = df["maturity"].astype(float).values
    up_shocks = df["int_up"].values
    down_shocks = df["int_down"].values

    up_interp = np.interp(maturity_input, maturities, up_shocks)
    down_interp = np.interp(maturity_input, maturities, down_shocks)

    return float(up_interp), float(down_interp)


def spread_shock(duration):
    """
    Compute the spread shock according to EIOPA formula.
    
    Parameters
    ----------
    duration : float or array-like
        Modified duration (dur_i).

    Returns
    -------
    shock : float or np.ndarray
        Spread shock in decimal (e.g., 0.15 = 15%).
    """
    d = np.asarray(duration, dtype=float)
    shock = np.zeros_like(d)

    shock[d <= 5] = 0.03 * d[d <= 5]
    mask = (d > 5) & (d <= 10)
    shock[mask] = 0.15 + 0.017 * (d[mask] - 5)
    mask = (d > 10) & (d <= 20)
    shock[mask] = 0.235 + 0.012 * (d[mask] - 10)
    mask = (d > 20)
    shock[mask] = np.minimum(0.355 + 0.005 * (d[mask] - 20), 1.0)

    return float(shock) if np.isscalar(duration) else shock



def get_ecb_yield(maturity_code: str) -> pd.DataFrame:
    """
    Fetch Euro area AAA government bond yield curve data from the ECB Data API.

    Parameters
    ----------
    maturity_code : str
        Maturity to fetch, such as "3M", "1Y", "5Y", "10Y", "30Y".

    Returns
    -------
    pd.DataFrame
        Indexed by date, with yields in **decimal form** (e.g. 0.028 = 2.8%).
    """
    base_url = "https://data-api.ecb.europa.eu/service/data/YC/"
    url = f"{base_url}B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{maturity_code}?format=csvdata"

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    df = df.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "yield"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[["date", "yield"]].dropna().sort_values("date").set_index("date")

    # ✅ Convert from percent to decimal
    df["yield"] = df["yield"] / 100.0

    return df


def get_interpolated_ecb_yield(maturity_input: float, verbose: bool = True) -> float:
    """
    Interpolate or fetch Euro area AAA government bond yield for any maturity.

    Handles edge cases where requested maturity < min or > max available tenor.
    Returns yields as decimals (e.g. 0.028 = 2.8%).
    """
    avail = np.array([0.25, 0.5, 0.75, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30])

    def fmt(x):
        return f"{int(x*12)}M" if x < 1 else f"{int(x)}Y"

    # Clip to valid range
    if maturity_input <= avail.min():
        if verbose:
            print(f"Maturity {maturity_input}Y below {avail.min()}Y — using {avail.min()}Y instead.")
        return float(get_ecb_yield(fmt(avail.min())).iloc[-1, 0])

    if maturity_input >= avail.max():
        if verbose:
            print(f"Maturity {maturity_input}Y above {avail.max()}Y — using {avail.max()}Y instead.")
        return float(get_ecb_yield(fmt(avail.max())).iloc[-1, 0])

    # Exact match
    if maturity_input in avail:
        code = fmt(maturity_input)
        if verbose:
            print(f"Fetching {code} directly from ECB (no interpolation)...")
        df = get_ecb_yield(code)
        return float(df.iloc[-1, 0])

    # Interpolate between nearest maturities
    lower = avail[avail <= maturity_input].max()
    upper = avail[avail >= maturity_input].min()
    lower_code, upper_code = fmt(lower), fmt(upper)

    if verbose:
        print(f"Interpolating between {lower_code} ({lower}Y) and {upper_code} ({upper}Y)...")

    lower_df = get_ecb_yield(lower_code)
    upper_df = get_ecb_yield(upper_code)
    merged = lower_df.rename(columns={"yield": f"yield_{lower_code}"}).join(
        upper_df.rename(columns={"yield": f"yield_{upper_code}"}), how="inner"
    )
    latest = merged.iloc[-1]
    y_lower, y_upper = latest[f"yield_{lower_code}"], latest[f"yield_{upper_code}"]
    y_interp = y_lower + (maturity_input - lower) * (y_upper - y_lower) / (upper - lower)

    if verbose:
        print(f"Latest date: {merged.index[-1].date()}, Interpolated {maturity_input:.2f}Y yield = {y_interp:.3%}")

    return float(y_interp)


def expected_return_yf(ticker: str, trading_days: int = 252, period: str = "1y"):
    """
    Calculate expected annualized arithmetic return from Yahoo Finance data.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. 'EUNL.DE').
    trading_days : int
        Number of trading days per year (default 252).
    period : str
        Lookback period for history(), e.g. '1y', '5y', '6mo'.

    Returns
    -------
    float
        Annualized arithmetic expected return (in decimal, e.g. 0.065 for 6.5%).
        Returns np.nan if data unavailable.
    """
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty or "Close" not in data.columns:
            return np.nan

        daily_ret = data["Close"].pct_change().dropna()
        return float(daily_ret.mean() * trading_days)

    except Exception as e:
        print(f"⚠️ Error fetching {ticker}: {e}")
        return np.nan
