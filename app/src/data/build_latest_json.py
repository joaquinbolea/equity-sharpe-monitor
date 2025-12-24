import os
import json
import datetime as dt

import numpy as np
import pandas as pd
import refinitiv.data as rd


# =========================
# CONFIG
# =========================
INDEX_RIC = ".OEX"                 # S&P 100
PRICE_FIELD = "TR.PriceClose"
PE_FIELD = "TR.PE"                 # el que te funcionó
START = "2018-01-01"               # para tener 5A (1260 días aprox)
OUT_PATH = os.path.join("docs", "data", "latest.json")

SHORT = {"1m": 21, "3m": 63, "6m": 126, "1a": 252}
LONG  = {"1a": 252, "2a": 504, "3a": 756, "5a": 1260}
HORIZONS = {**SHORT, **LONG}


# =========================
# HELPERS
# =========================
def open_session():
    # En Workspace/Codebook suele bastar con esto
    try:
        rd.open_session()
    except Exception:
        # si ya hay sesión abierta, no pasa nada
        pass


def get_sp100_rics(index_ric: str) -> list:
    const = rd.get_data(
        universe=index_ric,
        fields=["Instrument", "Constituent RIC"]
    )[0]

    # tolerante a nombres de columnas
    col = None
    for c in ["Constituent RIC", "Constituent RIC "]:
        if c in const.columns:
            col = c
            break
    if col is None:
        # fallback: agarrar la segunda columna si existe
        col = const.columns[1]

    rics = pd.Series(const[col]).dropna().astype(str).unique().tolist()
    return rics


def download_prices(rics: list, start: str, end: str) -> pd.DataFrame:
    series = {}
    for i, ric in enumerate(rics, 1):
        df = rd.get_history(
            universe=ric,
            fields=[PRICE_FIELD],
            start=start,
            end=end,
            interval="daily",
        )
        if df is None or df.empty:
            continue

        s = df.iloc[:, 0].rename(ric)
        series[ric] = s

        if i % 20 == 0:
            print(f"prices OK {i}/{len(rics)} | {ric} | rows={len(s)}")

    prices_df = pd.concat(series.values(), axis=1).sort_index()
    prices_df = prices_df.apply(pd.to_numeric, errors="coerce")
    return prices_df


def compute_daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    rets = prices_df.pct_change()
    return rets


def sharpe_table(rets: pd.DataFrame, horizons: dict) -> pd.DataFrame:
    out = pd.DataFrame({"ticker": rets.columns.astype(str)})

    for label, window in horizons.items():
        # usamos últimos N días hábiles
        r_win = rets.tail(window)
        mu = r_win.mean(axis=0)
        sig = r_win.std(axis=0, ddof=1)

        sharpe = (mu / sig) * np.sqrt(252)
        out[f"sharpe_{label}"] = sharpe.values

    return out


def get_pe_snapshot(rics: list) -> pd.DataFrame:
    df = rd.get_data(
        universe=rics,
        fields=[PE_FIELD]
    )[0].copy()

    # Normalizar nombres
    inst_col = "Instrument" if "Instrument" in df.columns else df.columns[0]
    pe_col = None
    for c in df.columns:
        if c != inst_col:
            pe_col = c
            break

    df = df[[inst_col, pe_col]].rename(columns={inst_col: "ticker", pe_col: "pe"})
    df["pe"] = pd.to_numeric(df["pe"], errors="coerce")
    return df


def bubble_from_pe(pe: pd.Series) -> pd.Series:
    """
    Convertimos P/E a tamaño de burbuja estable.
    - cap a percentiles para evitar outliers (p.ej. 3000+)
    - escalamos a rango [12, 55]
    """
    x = pe.copy()
    x = x.replace([np.inf, -np.inf], np.nan)

    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    x = x.clip(lower=lo, upper=hi)

    # si todo NaN, devolvemos tamaño default
    if np.all(np.isnan(x)):
        return pd.Series(18.0, index=pe.index)

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if xmin == xmax:
        return pd.Series(22.0, index=pe.index)

    # normalizar 0..1
    z = (x - xmin) / (xmax - xmin)
    return 12 + z * (55 - 12)


# =========================
# MAIN
# =========================
def main():
    open_session()

    end = dt.date.today().isoformat()
    print("Downloading constituents...")
    rics = get_sp100_rics(INDEX_RIC)
    print("n rics:", len(rics), "| sample:", rics[:10])

    print("Downloading prices...")
    prices_df = download_prices(rics, START, end)
    print("prices_df shape:", prices_df.shape)

    print("Computing returns & sharpes...")
    rets = compute_daily_returns(prices_df)
    sh = sharpe_table(rets, HORIZONS)

    print("Downloading PE snapshot...")
    pe = get_pe_snapshot(rics)

    # merge
    out = sh.merge(pe, on="ticker", how="left")

    # bubble size desde pe
    out["bubble_size"] = bubble_from_pe(out["pe"])

    # limpiar: solo filas con al menos X e Y posibles (dejamos todo, front filtra por NaN)
    out = out.replace([np.inf, -np.inf], np.nan)

    payload = {
        "asof": end,
        "index": INDEX_RIC,
        "n": int(out.shape[0]),
        "rows": out.to_dict(orient="records"),
    }

    # escribir
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
