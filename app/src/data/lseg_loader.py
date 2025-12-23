# LSEG price loader
import datetime as dt
import pandas as pd
import refinitiv.data as rd

INDEX_RIC = ".OEX"
PRICE_FIELD = "TR.PriceClose"
PE_FIELD = "TR.PE"


def open_session():
    rd.open_session()


def load_universe():
    const = rd.get_data(
        universe=[INDEX_RIC],
        fields=["TR.IndexConstituentRIC"]
    )
    col = const.columns[-1]
    rics = list(pd.Series(const[col]).dropna().astype(str).unique())
    return rics


def load_prices(rics, start="2018-01-01", end=None):
    if end is None:
        end = dt.date.today().isoformat()

    prices = {}
    for ric in rics:
        df = rd.get_history(
            universe=ric,
            fields=[PRICE_FIELD],
            start=start,
            end=end,
            interval="daily",
        )
        prices[ric] = df.iloc[:, 0].rename(ric)

    prices_df = pd.concat(prices.values(), axis=1).sort_index()
    prices_df = prices_df.apply(pd.to_numeric, errors="coerce")
    return prices_df


def load_pe(rics):
    pe_raw = rd.get_data(
        universe=rics,
        fields=[PE_FIELD]
    )
    pe_col = pe_raw.columns[-1]
    pe = pe_raw.rename(columns={pe_col: "pe"}).copy()
    pe["ticker"] = pe["Instrument"].astype(str)
    return pe[["ticker", "pe"]]
