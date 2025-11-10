import pandas as pd
import matplotlib.pyplot as plt


def summarize_portfolio(initial_asset: pd.DataFrame, liab_value: float, liab_duration: float) -> pd.DataFrame:
    """
    Summarize the key metrics of the asset portfolio and liability profile.

    Parameters
    ----------
    initial_asset : pd.DataFrame
        DataFrame containing at least 'asset_val', 'asset_dur', 'asset_ret', 'asset_weight'.
    liab_value : float
        Present value of liabilities.
    liab_duration : float
        Duration of liabilities.

    Returns
    -------
    pd.DataFrame
        Summary table with total asset value, weighted duration, expected return,
        liability value/duration, and duration gap.
    """

    # Compute portfolio metrics
    total_asset = initial_asset["asset_val"].sum()
    weighted_duration = (initial_asset["asset_dur"] * initial_asset["asset_weight"]).sum()
    portfolio_return = (initial_asset["asset_ret"] * initial_asset["asset_weight"]).sum()

    # Construct summary table
    summary = pd.DataFrame({
        "Metric": [
            "Total Asset Value",
            "Weighted Duration",
            "Expected Asset Return",
            "Liability Value",
            "Liability Duration"
        ],
        "Value": [
            f"{total_asset:,.1f}",
            f"{weighted_duration:.2f} years",
            f"{portfolio_return:.2%}",
            f"{liab_value:,.1f}",
            f"{liab_duration:.2f} years"
        ]
    })

    return summary


def plot_frontier(opt_df: pd.DataFrame,
                  current_sol: float = None,
                  current_ret: float = None,
                  min_sol_pct: float = 100.0,
                  title: str = "Efficient Frontier — Combined Interest UP & DOWN",
                  show: bool = True):
    """
    Plot the efficient frontier (Expected Return vs Solvency Ratio).

    Parameters
    ----------
    opt_df : pd.DataFrame
        DataFrame returned by `solve_frontier_combined()` containing 'solvency' and 'return'.
    current_sol : float, optional
        Current portfolio solvency ratio (as a multiple, e.g. 1.8 = 180%).
    current_ret : float, optional
        Current portfolio expected return (in decimal, e.g. 0.035 = 3.5%).
    min_sol_pct : float, optional
        Minimum solvency ratio (in %) to highlight the efficient frontier (default 150%).
    title : str
        Plot title.
    show : bool
        If True, displays the plot immediately. Set False to return the fig/ax for Streamlit.

    Returns
    -------
    (fig, ax) : tuple
        Matplotlib Figure and Axes objects (useful for Streamlit `st.pyplot`).
    """

    if opt_df is None or opt_df.empty:
        raise ValueError("opt_df is empty — run solve_frontier_combined() first")

    # Filter based on solvency threshold
    mask = opt_df["solvency"] * 100 >= min_sol_pct
    opt_filtered = opt_df[mask]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the efficient frontier
    ax.plot(opt_filtered["solvency"] * 100,
            opt_filtered["return"],
            marker="o", color="seagreen",
            label=f"Efficient Frontier")

    # Plot current portfolio (optional)
    if current_sol is not None and current_ret is not None:
        ax.scatter(current_sol * 100, current_ret,
                   color="red", s=100, label="Current Portfolio", zorder=5)
        ax.axvline(current_sol * 100, color="red", linestyle="--", alpha=0.5)
        ax.axhline(current_ret, color="red", linestyle="--", alpha=0.5)

    ax.set_xlabel("Solvency Ratio (%)")
    ax.set_ylabel("Expected Return on Asset")
    ax.set_title(title)
    ax.set_xlim(100, 500)
    ax.grid(True)
    ax.legend()

    if show:
        plt.show()

    return fig, ax

