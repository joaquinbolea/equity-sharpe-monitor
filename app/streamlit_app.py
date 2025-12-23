# Streamlit app entry point
import streamlit as st
import yaml
import plotly.express as px

from data.lseg_loader import open_session, load_universe, load_prices, load_pe
from analytics.sharpe import compute_log_returns, compute_sharpe


@st.cache_data(show_spinner=True)
def load_all():
    open_session()
    rics = load_universe()
    prices = load_prices(rics)
    pe = load_pe(rics)
    return rics, prices, pe


st.set_page_config(layout="wide")
st.title("Equity Sharpe Monitor — S&P 100")

# Load config
with open("analytics/config/horizons.yaml", "r") as f:
    cfg = yaml.safe_load(f)

short = cfg["short"]
long = cfg["long"]

# Load data
rics, prices_df, pe = load_all()

rets = compute_log_returns(prices_df)
sh = compute_sharpe(rets, {**short, **long})

df = sh.merge(pe, on="ticker", how="left")

# Selectors
x_key = st.selectbox("Sharpe corto", list(short.keys()), index=1)
y_key = st.selectbox("Sharpe largo", list(long.keys()), index=1)

x_col = f"sharpe_{x_key.lower()}"
y_col = f"sharpe_{y_key.lower()}"

fig = px.scatter(
    df,
    x=x_col,
    y=y_col,
    size="pe",
    hover_name="ticker",
    template="plotly_dark",
)

st.plotly_chart(fig, use_container_width=True)
