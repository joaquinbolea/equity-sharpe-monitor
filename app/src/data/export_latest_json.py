import os, json, datetime as dt
import numpy as np
import pandas as pd
import yaml

from data.lseg_loader import open_session, load_universe, load_prices, load_pe
from analytics.sharpe import compute_log_returns, compute_sharpe

OUT_PATH = os.path.join("docs", "data", "latest.json")

def winsorize(s: pd.Series, p_low=0.05, p_high=0.95):
    s = pd.to_numeric(s, errors="coerce")
    lo = s.quantile(p_low)
    hi = s.quantile(p_high)
    return s.clip(lower=lo, upper=hi)

def scale_sizes(s: pd.Series, min_sz=10, max_sz=55):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(min_sz, index=s.index)
    if s.max() == s.min():
        return pd.Series((min_sz + max_sz) / 2, index=s.index)
    return min_sz + (s - s.min()) / (s.max() - s.min()) * (max_sz - min_sz)

def main():
    open_session()

    # horizons.yaml
    with open("app/src/analytics/config/horizons.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    SHORT = cfg["short"]
    LONG  = cfg["long"]

    rics = load_universe()
    prices_df = load_prices(rics)
    pe = load_pe(rics)

    rets = compute_log_returns(prices_df)
    sh = compute_sharpe(rets, {**SHORT, **LONG})

    df = sh.merge(pe, on="ticker", how="left").replace([np.inf, -np.inf], np.nan)

    # bubble_size: P/E winsor (igual a tu streamlit)
    df["pe_w"] = winsorize(df["pe"], 0.05, 0.95)
    df["bubble_size"] = scale_sizes(df["pe_w"], 10, 55)

    payload = {
        "asof": dt.date.today().isoformat(),
        "index": ".OEX",
        "n": int(df.shape[0]),
        "rows": df.to_dict(orient="records")
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print("OK ->", OUT_PATH, "| n =", payload["n"])

if __name__ == "__main__":
    main()
