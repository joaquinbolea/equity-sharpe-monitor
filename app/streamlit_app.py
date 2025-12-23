import streamlit as st
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data.lseg_loader import open_session, load_universe, load_prices, load_pe
from analytics.sharpe import compute_log_returns, compute_sharpe

# =========================
# Helpers
# =========================
def winsorize(s: pd.Series, p_low=0.05, p_high=0.95):
    s = pd.to_numeric(s, errors="coerce")
    lo = s.quantile(p_low)
    hi = s.quantile(p_high)
    return s.clip(lower=lo, upper=hi), float(lo), float(hi)

def scale_sizes(s: pd.Series, min_sz=10, max_sz=55):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(min_sz, index=s.index)
    if s.max() == s.min():
        return pd.Series((min_sz + max_sz) / 2, index=s.index)
    return min_sz + (s - s.min()) / (s.max() - s.min()) * (max_sz - min_sz)

def pct(s: pd.Series, q: float):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.quantile(q)) if len(s) else np.nan

def pick_labels(df, score_col, k=12):
    # etiqueta top/bottom por “score” y algunos outliers (x o y extremos)
    tmp = df.copy()
    tmp["score"] = tmp[score_col]
    tmp = tmp.dropna(subset=["score"])

    top = tmp.nlargest(k, "score")["ticker"].tolist()
    bot = tmp.nsmallest(k, "score")["ticker"].tolist()

    # extremos por ejes (para que siempre haya nombres en bordes)
    xext = df.dropna(subset=["x"]).copy()
    yext = df.dropna(subset=["y"]).copy()
    x_ext = xext.nlargest(6, "x")["ticker"].tolist() + xext.nsmallest(6, "x")["ticker"].tolist()
    y_ext = yext.nlargest(6, "y")["ticker"].tolist() + yext.nsmallest(6, "y")["ticker"].tolist()

    labels = set(top + bot + x_ext + y_ext)
    return labels

# =========================
# Streamlit config
# =========================
st.set_page_config(layout="wide")
st.title("Equity Sharpe Monitor — S&P 100")
st.caption("Sharpe short vs long. Bubble size: P/E (winsorizado 5%-95%).")

# =========================
# Load horizons
# =========================
with open("analytics/config/horizons.yaml", "r") as f:
    cfg = yaml.safe_load(f)

SHORT = cfg["short"]
LONG = cfg["long"]

# =========================
# Load data (cached)
# =========================
@st.cache_data(show_spinner=True)
def load_all():
    open_session()
    rics = load_universe()
    prices = load_prices(rics)
    pe = load_pe(rics)
    return rics, prices, pe

with st.spinner("Cargando datos desde LSEG..."):
    rics, prices_df, pe = load_all()

rets = compute_log_returns(prices_df)
sh = compute_sharpe(rets, {**SHORT, **LONG})
df = sh.merge(pe, on="ticker", how="left")

# =========================
# UI selectors
# =========================
c1, c2, c3, c4 = st.columns([1,1,1,1])

with c1:
    x_key = st.selectbox("Eje X (Sharpe corto)", list(SHORT.keys()), index=1)  # default 3M
with c2:
    y_key = st.selectbox("Eje Y (Sharpe largo)", list(LONG.keys()), index=1)   # default 2A
with c3:
    label_k = st.slider("Cantidad de labels (top/bottom)", 6, 25, 12, 1)
with c4:
    bubble_mode = st.selectbox("Tamaño burbuja", ["P/E winsor", "log(P/E) winsor"], index=0)

x_col = f"sharpe_{x_key.lower()}"
y_col = f"sharpe_{y_key.lower()}"

df = df.copy()
df["x"] = df[x_col]
df["y"] = df[y_col]

# =========================
# Bubble size (P/E) robusto
# =========================
df["pe_w"], pe_lo, pe_hi = winsorize(df["pe"], 0.05, 0.95)

if bubble_mode.startswith("log"):
    pe_for_size = np.log(df["pe_w"].where(df["pe_w"] > 0))
else:
    pe_for_size = df["pe_w"]

df["bubble_size"] = scale_sizes(pe_for_size, 10, 55)

# =========================
# Quadrants + colors (como la imagen)
# líneas rojas en percentil 75
# =========================
x_thr = pct(df["x"], 0.75)
y_thr = pct(df["y"], 0.75)

def quad(row):
    if pd.isna(row["x"]) or pd.isna(row["y"]):
        return "NA"
    if row["x"] >= x_thr and row["y"] >= y_thr:
        return "Q1 (High/High)"
    if row["x"] >= x_thr and row["y"] < y_thr:
        return "Q2 (High/Low)"
    if row["x"] < x_thr and row["y"] >= y_thr:
        return "Q3 (Low/High)"
    return "Q4 (Low/Low)"

df["quad"] = df.apply(quad, axis=1)

# paleta simple estilo “neón” sobre fondo oscuro
quad_color = {
    "Q1 (High/High)": "rgba(255, 80, 200, 0.85)",   # magenta
    "Q2 (High/Low)":  "rgba(200, 200, 255, 0.75)",  # lila
    "Q3 (Low/High)":  "rgba(130, 200, 255, 0.75)",  # celeste
    "Q4 (Low/Low)":   "rgba(140, 140, 140, 0.60)",  # gris
    "NA":             "rgba(120, 120, 120, 0.35)"
}

# =========================
# Labels selectivos
# =========================
df["score"] = df["x"].fillna(0) + df["y"].fillna(0)
labels = pick_labels(df, "score", k=label_k)

df["label"] = df["ticker"].where(df["ticker"].isin(labels), "")

# =========================
# Plotly Figure
# =========================
fig = go.Figure()

# una traza por cuadrante para colores limpios
for qname in ["Q1 (High/High)", "Q2 (High/Low)", "Q3 (Low/High)", "Q4 (Low/Low)"]:
    sub = df[df["quad"] == qname].copy()
    if sub.empty:
        continue

    fig.add_trace(go.Scatter(
        x=sub["x"],
        y=sub["y"],
        mode="markers+text",
        text=sub["label"],
        textposition="top center",
        textfont=dict(size=10),
        name=qname,
        marker=dict(
            size=sub["bubble_size"],
            sizemode="diameter",
            color=quad_color[qname],
            opacity=0.80,
            line=dict(width=0.8, color="rgba(255,255,255,0.35)")
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            f"Sharpe {x_key} (short): "+"%{x:.2f}<br>"
            f"Sharpe {y_key} (long): "+"%{y:.2f}<br>"
            "P/E raw: %{customdata[1]:.1f}<br>"
            "P/E winsor: %{customdata[2]:.1f}<extra></extra>"
        ),
        customdata=np.stack([
            sub["ticker"].to_numpy(),
            pd.to_numeric(sub["pe"], errors="coerce").to_numpy(),
            pd.to_numeric(sub["pe_w"], errors="coerce").to_numpy(),
        ], axis=1),
    ))

# líneas rojas punteadas (percentil 75) como en tu screenshot
fig.add_shape(
    type="line", x0=x_thr, x1=x_thr, y0=0, y1=1, xref="x", yref="paper",
    line=dict(color="rgba(255,0,0,0.65)", width=2, dash="dash")
)
fig.add_shape(
    type="line", x0=0, x1=1, y0=y_thr, y1=y_thr, xref="paper", yref="y",
    line=dict(color="rgba(255,0,0,0.65)", width=2, dash="dash")
)

# watermark tipo “Balanz” (opcional)
fig.add_annotation(
    text="Balanz",
    x=0.5, y=0.1, xref="paper", yref="paper",
    showarrow=False,
    font=dict(size=80, color="rgba(255,255,255,0.06)"),
)

fig.update_layout(
    template="plotly_dark",
    height=780,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=55, r=25, t=40, b=55),
    xaxis=dict(title=f"Sharpe corto ({x_key})", gridcolor="rgba(255,255,255,0.08)", zeroline=False),
    yaxis=dict(title=f"Sharpe largo ({y_key})", gridcolor="rgba(255,255,255,0.08)", zeroline=False),
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Debug / thresholds"):
    st.write({
        "x_threshold_p75": x_thr,
        "y_threshold_p75": y_thr,
        "pe_winsor_low": pe_lo,
        "pe_winsor_high": pe_hi,
        "missing_pe": int(df["pe"].isna().sum()),
    })
