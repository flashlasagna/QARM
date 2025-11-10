import yaml
import os
import pandas as pd

def load_config(filename="config.yaml"):
    path = os.path.join(os.path.dirname(__file__), '..', 'config', filename)
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_corr_matrices(config):
    """Convert YAML correlation matrices to pandas DataFrames."""
    down_labels = config["correlations"]["downward"]["labels"]
    up_labels = config["correlations"]["upward"]["labels"]
    corr_down = pd.DataFrame(
        config["correlations"]["downward"]["matrix"],
        index=down_labels,
        columns=down_labels
    )
    corr_up = pd.DataFrame(
        config["correlations"]["upward"]["matrix"],
        index=up_labels,
        columns=up_labels
    )
    return corr_down, corr_up


def get_solvency_params(config):
    """Flatten solvency2 parameters into a dict for modeling."""
    solv = config["solvency2"]
    return {
        "equity_1_param": solv["equity"]["type1"],
        "equity_2_param": solv["equity"]["type2"],
        "prop_params": solv["property"],
        "rho": solv["correlation_base"],
    }


def get_etf_universe(config):
    """
    Convert the 'etfs' section of the YAML config into a clean DataFrame.

    Example output:
    ┌───────────────┬───────────────────────────────────────────────────────┬───────────┬────────────────────┐
    │ key           │ name                                                  │ ticker    │ category           │
    ├───────────────┼───────────────────────────────────────────────────────┼───────────┼────────────────────┤
    │ corporate_bond│ iShares Euro Corporate Bond 1–5yr UCITS ETF           │ IE15.L    │ Fixed Income       │
    │ property      │ iShares European Property Yield UCITS ETF             │ EUNK.DE   │ Real Estate        │
    │ equity_emerging│ iShares MSCI Emerging Markets UCITS ETF EUR (Dist)   │ IQQE.DE   │ Equity - Emerging  │
    │ equity_developed│ iShares MSCI World UCITS ETF EUR (Acc)              │ EUNL.DE   │ Equity - Developed │
    └───────────────┴───────────────────────────────────────────────────────┴───────────┴────────────────────┘
    """
    etfs = config.get("etfs", {})
    if not etfs:
        raise ValueError("No 'etfs' section found in config.yaml")

    df = pd.DataFrame(etfs).T.reset_index(names="key")
    return df[["key", "name", "ticker", "category"]]