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


# --------------------------
# Plotting Helper Function
# --------------------------

'''
def plot_scenario_comparison(opt_df, base_best, sens_best, current_ret, current_sol, sens_df=None):
    """
    Generates a comparison plot showing Base Optimal, Current, and New Scenario Optimal.
    Applies Pareto Filtering to ensure smooth curves.
    """
    # --- INTERNAL HELPER: PARETO FILTER ---
    def get_pareto_frontier(df):
        if df is None or df.empty:
            return None
        # 1. Sort by Solvency (High -> Low)
        sorted_df = df.sort_values(by="solvency", ascending=False).copy()
        # 2. Filter: Keep points only if they have higher return than any safer point seen so far
        sorted_df["max_return_seen"] = sorted_df["return"].cummax()
        clean_df = sorted_df[sorted_df["return"] >= sorted_df["max_return_seen"]]
        # 3. Sort back (Low -> High) for plotting
        return clean_df.sort_values(by="solvency")
    # ---------------------------------------

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # 1. Process & Plot BASE Frontier
    base_curve = get_pareto_frontier(opt_df)
    if base_curve is not None:
        ax.plot(base_curve["solvency"] * 100, base_curve["return"] * 100, '-',
                color='#4ECDC4', linewidth=2.5, alpha=0.4, label='Base Efficient Frontier', zorder=1)

    # 2. Process & Plot SCENARIO Frontier (if provided)
    if sens_df is not None and not sens_df.empty:
        sens_curve = get_pareto_frontier(sens_df)
        if sens_curve is not None:
            ax.plot(sens_curve["solvency"] * 100, sens_curve["return"] * 100, '--',
                    color='#9B59B6', linewidth=2.0, alpha=0.6, label='Scenario Frontier', zorder=2)

    # 3. Base Optimal Portfolio (Star)
    ax.scatter(base_best["solvency"] * 100, base_best["return"] * 100, s=400, c='#FFD700', marker='*',
               edgecolors='#FF8C00', linewidth=2, label='Base Optimal Portfolio', zorder=5)

    # 4. Current Portfolio (Diamond)
    ax.scatter(current_sol * 100, current_ret * 100, s=250, c='#E74C3C', marker='D',
               edgecolors='#C0392B', linewidth=2, label='Current Portfolio', zorder=4)

    # 5. New Scenario Optimal Portfolio (Triangle)
    sens_sol = sens_best["solvency"] * 100
    sens_ret = sens_best["return"] * 100
    ax.scatter(sens_sol, sens_ret, s=400, c='#9B59B6', marker='^',
               edgecolors='#6C3483', linewidth=3, label='Scenario Optimal Portfolio', zorder=6)
    
    # Annotation for New Scenario
    ax.annotate(f'NEW OPTIMAL\n{sens_ret:.2f}% | {sens_sol:.1f}%',
                xy=(sens_sol, sens_ret), xytext=(-50, 40),
                textcoords='offset points', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#9B59B6', edgecolor='#6C3483', alpha=0.9),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', color='#6C3483', lw=2), zorder=6)

    # Styling
    ax.set_xlabel('Solvency Ratio (%)', fontsize=12)
    ax.set_ylabel('Expected Return (%)', fontsize=12)
    ax.set_title('Scenario Comparison: Optimal Portfolios', fontsize=15)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc='lower right', frameon=True, shadow=True)
    
    return fig

'''

def plot_scenario_comparison(opt_df, base_best, sens_best, current_ret, current_sol, sens_df=None):
    """
    Scenario Comparison Plot — Styled exactly like the Efficient Frontier plot.
    Shows:
    - Base Efficient Frontier
    - Scenario Frontier
    - Base Optimal Portfolio (Gold Star)
    - Scenario Optimal Portfolio (Purple Triangle)
    - Current Portfolio (Red Dot)
    """

    # --- INTERNAL HELPER: PARETO CLEANER ---
    def get_pareto(df):
        if df is None or df.empty:
            return None
        df = df.sort_values("solvency", ascending=False).copy()
        df["max_return_seen"] = df["return"].cummax()
        df = df[df["return"] >= df["max_return_seen"]]
        return df.sort_values("solvency")

    # Initialize figure
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # === 1. FEASIBLE SET (grey points) ===
    ax.scatter(
        opt_df["solvency"] * 100,
        opt_df["return"] * 100,
        s=30,
        color="gray",
        alpha=0.25,
        label="Feasible Portfolios",
        zorder=1
    )

    # === 2. BASE FRONTIER (dark blue, consistent with main plot) ===
    base_front = get_pareto(opt_df)
    if base_front is not None:
        ax.plot(
            base_front["solvency"] * 100,
            base_front["return"] * 100,
            "-",
            color="#003366",
            linewidth=3,
            label="Base Efficient Frontier",
            zorder=2
        )

    # === 3. SCENARIO FRONTIER (purple dashed) ===
    if sens_df is not None and not sens_df.empty:
        scen_front = get_pareto(sens_df)
        if scen_front is not None:
            ax.plot(
                scen_front["solvency"] * 100,
                scen_front["return"] * 100,
                "--",
                color="#9B59B6",
                linewidth=3,
                alpha=0.8,
                label="Scenario Frontier",
                zorder=3
            )

    # === 4. BASE OPTIMAL PORTFOLIO (gold star) ===
    base_sol = base_best["solvency"] * 100
    base_ret = base_best["return"] * 100
    ax.scatter(
        base_sol, base_ret,
        s=200, c="#FFD700", marker="*",
        edgecolors="#FF8C00",
        linewidth=2.5,
        label="Base Optimal",
        zorder=5
    )
    
    ax.annotate(
        f'BASE OPTIMAL\n{base_ret:.2f}% | {base_sol:.1f}%',
        xy=(base_sol, base_ret),
        xytext=(25, 25),
        textcoords="offset points",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFD700",
                  edgecolor="#FF8C00", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#FF8C00", lw=2.2),
        zorder=6
    )

    # === 5. CURRENT PORTFOLIO (red circle) ===
    cur_sol = current_sol * 100
    cur_ret = current_ret * 100
    ax.scatter(
        cur_sol, cur_ret,
        s=80, c="#E74C3C", marker="o",
        edgecolors="#C0392B", linewidth=2,
        label="Current Portfolio",
        zorder=4
    )
    
    ax.annotate(
        f'CURRENT\n{cur_ret:.2f}% | {cur_sol:.1f}%',
        xy=(cur_sol, cur_ret),
        xytext=(-50, -35),
        textcoords="offset points",
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FADBD8",
                  edgecolor="#C0392B", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=2),
        zorder=6
    )

    # === 6. SCENARIO OPTIMAL (purple triangle) ===
    scen_sol = sens_best["solvency"] * 100
    scen_ret = sens_best["return"] * 100
    ax.scatter(
        scen_sol, scen_ret,
        s=220, c="#9B59B6", marker="^",
        edgecolors="#6C3483",
        linewidth=3,
        label="Scenario Optimal",
        zorder=6
    )

    ax.annotate(
        f'SCENARIO OPTIMAL\n{scen_ret:.2f}% | {scen_sol:.1f}%',
        xy=(scen_sol, scen_ret),
        xytext=(-60, 40),
        textcoords="offset points",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#D7BDE2",
                  edgecolor="#6C3483", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#6C3483", lw=2),
        zorder=7
    )

    # === 7. Styling (same as your frontier plot) ===
    ax.axvline(
        x=100, color="#95A5A6",
        linestyle="--", linewidth=2.2, alpha=0.6,
        label="100% Solvency"
    )

    ax.set_xlabel("Solvency Ratio (%)", fontsize=12, fontweight="bold", color="#2c3e50")
    ax.set_ylabel("Expected Return (%)", fontsize=12, fontweight="bold", color="#2c3e50")
    ax.set_title("Scenario Comparison: Optimal Portfolios", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")

    ax.tick_params(colors="#2c3e50")
    
    ax.legend(
    loc="upper right",          # ← move legend to upper-right
    bbox_to_anchor=(1, 1),      # ← ensure it sits inside the axes
    frameon=True, shadow=True, fancybox=True
    )

    # Force x-axis to start at 90%
    ax.set_xlim(90, None)

    return fig
