async function loadData() {
  const res = await fetch("./data/latest.json");
  return await res.json();
}

function quantile(arr, q) {
  const a = arr.filter(x => x !== null && x !== undefined && !Number.isNaN(x)).slice().sort((x,y)=>x-y);
  if (!a.length) return null;
  const pos = (a.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return a[base+1] !== undefined ? a[base] + rest*(a[base+1]-a[base]) : a[base];
}

function draw(rows, xKey, yKey) {
  const x = rows.map(r => r[xKey]);
  const y = rows.map(r => r[yKey]);
  const t = rows.map(r => r.ticker);
  const s = rows.map(r => r.bubble_size ?? 12);

  const x75 = quantile(x, 0.75);
  const y75 = quantile(y, 0.75);

  const shapes = [];
  if (x75 !== null) shapes.push({type:"line", x0:x75, x1:x75, y0:0, y1:1, xref:"x", yref:"paper",
    line:{width:2, dash:"dash", color:"rgba(255,0,0,0.65)"}});
  if (y75 !== null) shapes.push({type:"line", x0:0, x1:1, y0:y75, y1:y75, xref:"paper", yref:"y",
    line:{width:2, dash:"dash", color:"rgba(255,0,0,0.65)"}});

  const trace = {
    type: "scatter",
    mode: "markers+text",
    x, y, text: t,
    marker: {
      size: s,
      opacity: 0.78,
      line: {width: 0.6, color: "rgba(255,255,255,0.35)"}
    },
    hovertemplate: "<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>"
  };

  const layout = {
    paper_bgcolor: "#0b0f14",
    plot_bgcolor: "#0b0f14",
    font: {color: "#e6edf3"},
    xaxis: {title: xKey, gridcolor: "rgba(255,255,255,0.08)", zeroline: false},
    yaxis: {title: yKey, gridcolor: "rgba(255,255,255,0.08)", zeroline: false},
    shapes,
    margin: {l:60, r:30, t:20, b:50},
    annotations: [{
      text: "Balanz",
      x: 0.5, y: 0.12, xref: "paper", yref: "paper",
      showarrow: false,
      font: {size: 80, color: "rgba(255,255,255,0.06)"}
    }]
  };

  Plotly.newPlot("chart", [trace], layout, {displayModeBar: true});
}

(async () => {
  const data = await loadData();
  const rows = data.rows ?? data; // soporta {rows:[...]} o [...]
  const xSel = document.getElementById("xSel");
  const ySel = document.getElementById("ySel");

  function rerender() {
    draw(rows, xSel.value, ySel.value);
  }

  xSel.addEventListener("change", rerender);
  ySel.addEventListener("change", rerender);

  rerender();
})();
