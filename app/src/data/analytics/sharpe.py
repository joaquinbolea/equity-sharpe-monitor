import numpy as np
import pandas as pd


def compute_log_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    rets = np.log(prices_df / prices_df.shift(1))
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return rets


def compute_sharpe(rets: pd.DataFrame, horizons: dict, ann_factor=252) -> pd.DataFrame:
    rows = []

    for ric in rets.columns:
        r = rets[ric].dropna()
        row = {"ticker": ric}

        for label, window in horizons.items():
            if len(r) < window:
                row[f"sharpe_{label.lower()}"] = np.nan
            else:
                r_win = r.iloc[-window:]
                mu = r_win.mean()
                sig = r_win.std(ddof=1)
                row[f"sharpe_{label.lower()}"] = (
                    np.nan if sig == 0 else (mu / sig) * np.sqrt(ann_factor)
                )

        rows.append(row)

    return pd.DataFrame(rows)
