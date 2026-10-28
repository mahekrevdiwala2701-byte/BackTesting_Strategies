"""🏠 Main Dashboard: same-date, year-over-year returns of the TOTAL PORTFOLIO
under the 3 strategies of the Calculated Investment page, all on one screen (no tabs).

    🧺 Buy & Hold   ·   🔁 Rebalancing to the same weights   ·   📐 Different Weights (growth-based)

For every trading day from the first investment date, the whole portfolio (GoldBEES, Nifty 50, MidCap,
SmallCap by your weights) is bought and valued on the same date N years later.

Layout (top to bottom):
    1. Three strategy cards side by side: key numbers + bell curve
    2. Line chart: return for every investment date, all 3 strategies
    3. Year by year (left)  |  Portfolio vs each asset (right)
    4. Comparison table (left)  |  CSV download + preview (right)
"""
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import (
    panel, chart, hero, note, sidebar_nav, sidebar_footer, sidebar_label, weight_badge,
    inr, pct, hex_rgba, STRAT_COLORS,
    PLOT_LAYOUT, AMBER, GOOD, BAD, FAINT, MUTED, TEXT,
)
from src.asset_data import ASSETS, ASSET_LABELS, ASSET_SHORT, ASSET_ICONS, COLORS, style_change_cols, blank_missing
from src.period_picker import today_ist
from src.strategies import REBAL_FREQ, DIFF_REBAL_FREQ, LOOKBACK_OPTIONS
from src.yoy_returns import (STRATEGY_KEYS, STRATEGY_NAMES, strategy_yoy, summary_stats, yearly_summary,
                             normal_curve, to_csv)

DEFAULT_START = date(2010, 1, 1)
LOOKBACKS = {**LOOKBACK_OPTIONS, "2 Years": 24, "2.5 Years": 30}
ICONS = {"bh": "🧺", "periodic": "🔁", "diff": "📐"}
H = 330                                   # chart height: keeps every row on one screen


def _yrs(y):
    return f"{y} year" + ("s" if y > 1 else "")


# =========================================================
# Page
# =========================================================
def render_home_page(asset_dfs, fetch_errors):
    available = [a for a in ASSETS if a in asset_dfs]

    # ---------------- Sidebar ----------------
    sidebar_nav("home")

    sidebar_label("💼 Portfolio")
    picked = st.sidebar.multiselect("Assets in the portfolio", available, default=available,
                                    format_func=lambda n: ASSET_LABELS[n], key="yoy_assets")
    amount = st.sidebar.number_input("Amount invested on each date (₹)", min_value=1000.0, value=100000.0,
                                     step=5000.0, format="%.0f", key="yoy_amount")
    raw_w = {}
    if picked:
        sidebar_label("⚖️ Weights (%)")
        share = round(100 / len(picked), 2)
        defaults = [share] * (len(picked) - 1) + [round(100 - share * (len(picked) - 1), 2)]
        raw_w = {a: st.sidebar.number_input(f"{ASSET_ICONS[a]} {ASSET_SHORT[a]}", min_value=0.0, max_value=100.0,
                                            value=d, step=5.0, key=f"yoy_w_{a}_{len(picked)}")
                 for a, d in zip(picked, defaults)}
        with st.sidebar:
            weight_badge(sum(raw_w.values()))
    weights = {a: w for a, w in raw_w.items() if w > 0}

    sidebar_label("🎯 Year-over-year")
    if weights:
        data_min = max(asset_dfs[a]['Date'].min() for a in weights).date()
        data_max = min(min(asset_dfs[a]['Date'].max() for a in weights).date(), today_ist())
    else:
        data_min, data_max = DEFAULT_START, today_ist()
    from_date = st.sidebar.date_input("📍 First investment date", value=max(DEFAULT_START, data_min),
                                      min_value=data_min, max_value=data_max, format="DD-MM-YYYY", key="yoy_from")
    to_date = st.sidebar.date_input("🏁 Last comparison date", value=data_max, min_value=data_min,
                                    max_value=data_max, format="DD-MM-YYYY", key="yoy_to")
    if from_date > to_date:
        st.sidebar.error("First investment date must be before the last comparison date. Swapped them.")
        from_date, to_date = to_date, from_date
    years = st.sidebar.select_slider("⏳ Compare with the same date after", options=list(range(1, 11)),
                                     value=1, format_func=_yrs, key="yoy_years")
    st.sidebar.caption("📆 Trading days only: a day is used when every asset has a real recent price.")

    sidebar_label("⚙️ Strategy settings")
    p_freq = st.sidebar.selectbox("🔁 Rebalancing: every", list(REBAL_FREQ), index=1, key="yoy_p_freq")
    d_freq = st.sidebar.selectbox("📐 Different Weights: rebalance every", list(DIFF_REBAL_FREQ), index=0,
                                  key="yoy_d_freq")
    d_look = st.sidebar.selectbox("📐 Different Weights: compare growth over", list(LOOKBACKS), index=1,
                                  key="yoy_d_look")
    sidebar_footer(fetch_errors, key="home")

    # ---------------- Calculation ----------------
    if not weights:
        hero("Main Dashboard", "Pick at least one asset with a weight above 0% in the sidebar.",
             page="home", eyebrow="Portfolio year-over-year returns")
        return

    table = _calc(asset_dfs, tuple(sorted(weights.items())), amount, from_date, to_date, years,
                  REBAL_FREQ[p_freq], DIFF_REBAL_FREQ[d_freq], LOOKBACKS[d_look])
    w_norm = {a: w / sum(weights.values()) for a, w in weights.items()}
    mix = " · ".join(f"{ASSET_ICONS[a]} {ASSET_SHORT[a]} {w * 100:.0f}%" for a, w in w_norm.items())
    settings = {"bh": "bought once, never touched",
                "periodic": f"back to your weights {p_freq.lower()}",
                "diff": f"{d_freq.lower()} · growth over previous {d_look}"}

    hero("Main Dashboard",
         f"{inr(amount)} portfolio bought on every trading day, valued on the same date {_yrs(years)} later, "
         f"under all 3 Calculated Investment strategies.",
         page="home", eyebrow="Portfolio year-over-year returns",
         chips=[f"💼 {mix}", f"📅 {from_date:%d-%m-%Y} → {to_date:%d-%m-%Y}", f"⏳ {_yrs(years)}",
                f"🧮 {len(table):,} trading days"])

    if table.empty:
        note(f"Not enough data: the range must be longer than <b>{_yrs(years)}</b> and every asset needs prices "
             f"in it ({data_min:%d-%m-%Y} to {data_max:%d-%m-%Y}).", AMBER, "⚠️")
        return

    _data_check(table)
    stats = {k: summary_stats(table[f'{STRATEGY_NAMES[k]} Return (%)']) for k in STRATEGY_KEYS}

    # ---------------- Row 1: the 3 strategy dashboards ----------------
    lo = min(s['min'] for s in stats.values())
    hi = max(s['max'] for s in stats.values())
    for col, key in zip(st.columns(3), STRATEGY_KEYS):
        with col, panel():
            _strategy_card(table, key, stats[key], settings[key], amount, years, (lo, hi))

    # ---------------- Row 2: line chart ----------------
    with panel():
        chart(_line_figure(table, years), key="yoy_line")

    # ---------------- Row 3: year by year | portfolio vs assets ----------------
    c1, c2 = st.columns(2)
    with c1, panel():
        chart(_yearly_figure(table), key="yoy_yearly")
    with c2, panel():
        chart(_vs_assets_figure(table, w_norm, stats, years), key="yoy_vs")

    # ---------------- Row 4: comparison table | CSV ----------------
    c1, c2 = st.columns([1.05, 1.2])
    with c1, panel():
        _comparison_table(table, w_norm, stats)
    with c2, panel():
        _csv_block(table, years)


@st.cache_data(show_spinner="Calculating every trading day...", ttl=600)
def _calc(asset_dfs, weights_items, amount, from_date, to_date, years, p_months, d_months, lookback):
    return strategy_yoy(asset_dfs, dict(weights_items), amount, from_date, to_date, years,
                        p_months, d_months, lookback, short_names=ASSET_SHORT)


# =========================================================
# Pieces
# =========================================================
def _data_check(table):
    removed = table.attrs.get('removed_prices', [])
    excluded = table.attrs.get('excluded_days', 0)
    if not removed and not excluded:
        return
    parts = []
    if removed:
        per = pd.Series([r[0] for r in removed]).map(ASSET_SHORT).value_counts()
        parts.append(f"{len(removed)} wrong price rows ignored ({', '.join(f'{a}: {n}' for a, n in per.items())})")
    if excluded:
        parts.append(f"{excluded:,} trading days left out (an asset has a hole in its data)")
    with st.expander("🧹 Data check: " + "; ".join(parts), expanded=False):
        st.caption("Fix the source sheet to include these dates.")
        if removed:
            df = pd.DataFrame(removed, columns=['Asset', 'Date', 'Close'])
            df['Asset'] = df['Asset'].map(ASSET_SHORT)
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d-%m-%Y')
            st.dataframe(df, hide_index=True, width="stretch", height=200)


def _strategy_card(table, key, s, setting, amount, years, x_range):
    color = STRAT_COLORS[key]
    name = STRATEGY_NAMES[key]
    gain = table[f'{name} Value (₹)'].mean() - amount
    cell = ('<div style="flex:1;min-width:0"><div style="color:{m};font-size:.72rem;text-transform:uppercase;'
            'letter-spacing:.04em">{l}</div><div style="color:{c};font-weight:700;font-size:1.05rem">{v}</div></div>')
    cells = "".join(cell.format(m=MUTED, l=lbl, c=c, v=v) for lbl, v, c in [
        ("Positive", f"{s['pct_positive']:.1f}%", GOOD),
        ("Average", pct(s['mean']), GOOD if s['mean'] >= 0 else BAD),
        ("Worst", pct(s['min']), BAD),
        ("Best", pct(s['max']), GOOD),
    ])
    st.markdown(
        f'<div style="border-left:3px solid {color};padding:2px 0 2px 10px;margin-bottom:6px">'
        f'<div style="font-weight:800;color:{TEXT};font-size:1.02rem">{ICONS[key]} {name}</div>'
        f'<div style="color:{MUTED};font-size:.78rem;line-height:1.35;min-height:2.7em">{setting}<br>'
        f'avg gain {inr(gain, 0, sign=True)} on {inr(amount)}'
        f'</div></div><div style="display:flex;gap:8px;margin:4px 0 2px">{cells}</div>',
        unsafe_allow_html=True)
    chart(_bell_figure(table[f'{name} Return (%)'], s, color, x_range), key=f"yoy_bell_{key}")


def _bell_figure(r, s, color, x_range):
    lo, hi = x_range
    bins = np.linspace(lo, hi, 46)
    counts, edges = np.histogram(r, bins=bins)
    width = edges[1] - edges[0]
    centers = (edges[:-1] + edges[1:]) / 2
    density = counts / max(counts.sum() * width, 1e-9)
    fig = go.Figure()
    fig.add_vrect(x0=s['mean'] - s['std'], x1=s['mean'] + s['std'], fillcolor=hex_rgba(color, 0.08), line_width=0)
    fig.add_trace(go.Bar(
        x=centers, y=density, width=width * 0.92, marker_color=[GOOD if c > 0 else BAD for c in centers],
        marker_line_width=0, opacity=0.85, showlegend=False,
        customdata=np.stack([edges[:-1], edges[1:], counts, counts / max(counts.sum(), 1) * 100], axis=1),
        hovertemplate="%{customdata[0]:.1f}% to %{customdata[1]:.1f}%<br>%{customdata[2]:,} days "
                      "(%{customdata[3]:.1f}%)<extra></extra>"))
    x, y = normal_curve(s['mean'], s['std'], lo, hi)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=color, width=2.6), hoverinfo='skip',
                             showlegend=False))
    fig.add_vline(x=0, line_color=FAINT, line_dash="dot")
    fig.add_vline(x=s['mean'], line_color=color, line_width=1.6, annotation_text=f"mean {s['mean']:+.1f}%",
                  annotation_position="top right", annotation_font=dict(color=color, size=11))
    fig.update_layout(**PLOT_LAYOUT, height=H - 40, bargap=0, margin=dict(l=8, r=8, t=18, b=8),
                      xaxis=dict(ticksuffix="%", range=[lo, hi]), yaxis=dict(showticklabels=False))
    return fig


def _with_gaps(table):
    """Insert an empty row wherever the data has a hole, so lines break instead of bridging it."""
    t = table.set_index('Invest Date')
    gap = t.index.to_series().diff().dt.days > 10
    if not gap.any():
        return t
    blanks = pd.DataFrame(index=t.index[gap] - pd.Timedelta(days=1), columns=t.columns)
    return pd.concat([t, blanks]).sort_index()


def _line_figure(table, years):
    t = _with_gaps(table)
    fig = go.Figure()
    for key in STRATEGY_KEYS:
        name = STRATEGY_NAMES[key]
        fig.add_trace(go.Scatter(x=t.index, y=t[f'{name} Return (%)'].astype(float), mode='lines', name=name,
                                 connectgaps=False, line=dict(color=STRAT_COLORS[key], width=1.5),
                                 hovertemplate=f"{name}: %{{y:+.2f}}%<extra></extra>"))
    fig.add_hline(y=0, line_color=FAINT, line_dash="dot")
    fig.update_layout(**PLOT_LAYOUT, height=H, hovermode='x unified',
                      title=f"Portfolio return after {_yrs(years)}, for every investment date",
                      yaxis=dict(ticksuffix="%"),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig


def _yearly_figure(table):
    ys = yearly_summary(table)
    fig = go.Figure()
    for key in STRATEGY_KEYS:
        name = STRATEGY_NAMES[key]
        fig.add_trace(go.Bar(x=ys['Year'].astype(str), y=ys[f'{name} Avg (%)'], name=name,
                             marker_color=STRAT_COLORS[key], marker_line_width=0,
                             customdata=ys[f'{name} % Positive'],
                             hovertemplate=f"{name} · invested in %{{x}}<br>average %{{y:+.2f}}%"
                                           f" · %{{customdata:.0f}}% positive<extra></extra>"))
    fig.add_hline(y=0, line_color=FAINT, line_dash="dot")
    fig.update_layout(**PLOT_LAYOUT, height=H, barmode='group', bargap=0.25, title="Year by year (average return)",
                      yaxis=dict(ticksuffix="%"), xaxis=dict(type='category'),
                      legend=dict(orientation='h', yanchor='top', y=-0.1, x=0), margin=dict(l=12, r=12, t=50, b=40))
    return fig


def _vs_assets_figure(table, w_norm, stats, years):
    curves = [(STRATEGY_NAMES[k], stats[k]['mean'], stats[k]['std'], STRAT_COLORS[k], 3.2) for k in STRATEGY_KEYS]
    for a in w_norm:
        r = table[f'{ASSET_SHORT[a]} Return (%)']
        curves.append((ASSET_SHORT[a], r.mean(), r.std(), COLORS[a], 1.4))
    lo = min(m - 3 * sd for _, m, sd, _, _ in curves)
    hi = max(m + 3 * sd for _, m, sd, _, _ in curves)
    fig = go.Figure()
    for name, m, sd, color, wd in curves:
        x, y = normal_curve(m, sd, lo, hi)
        is_asset = wd < 2
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name,
                                 line=dict(color=color, width=wd, dash='dot' if is_asset else 'solid'),
                                 hovertemplate=f"{name}: mean {m:+.1f}%, σ {sd:.1f}%<extra></extra>"))
    fig.add_vline(x=0, line_color=FAINT, line_dash="dot")
    fig.update_layout(**PLOT_LAYOUT, height=H, title=f"Portfolio vs each asset ({_yrs(years)} bell curves)",
                      xaxis=dict(ticksuffix="%"), yaxis=dict(showticklabels=False),
                      legend=dict(orientation='h', yanchor='top', y=-0.1, x=0), margin=dict(l=12, r=12, t=50, b=40))
    return fig


def _comparison_table(table, w_norm, stats):
    rows = []
    for key in STRATEGY_KEYS:
        s = stats[key]
        rows.append({'': f"{ICONS[key]} {STRATEGY_NAMES[key]}", 'Positive': s['pct_positive'],
                     'Average': s['mean'], 'σ': s['std'], 'Worst': s['min'], 'Best': s['max']})
    for a, w in w_norm.items():
        s = summary_stats(table[f'{ASSET_SHORT[a]} Return (%)'])
        rows.append({'': f"{ASSET_ICONS[a]} {ASSET_SHORT[a]} ({w * 100:.0f}%)", 'Positive': s['pct_positive'],
                     'Average': s['mean'], 'σ': s['std'], 'Worst': s['min'], 'Best': s['max']})
    df = pd.DataFrame(rows)
    signed = ['Average', 'Worst', 'Best']
    fmt = {c: '{:+.1f}%' for c in signed}
    fmt.update({'Positive': '{:.1f}%', 'σ': '{:.1f}%'})
    st.markdown("**📋 Portfolio strategies vs each asset**")
    st.dataframe(style_change_cols(df.style.format(fmt), signed), width="stretch", hide_index=True,
                 height=38 + 35 * len(df))


def _csv_block(table, years):
    c1, c2 = st.columns([1.6, 1], vertical_alignment="center")
    with c1:
        st.markdown(f"**📥 CSV** · {len(table):,} trading days · each asset's prices + all 3 strategies")
    with c2:
        st.download_button("⬇️  Download CSV", to_csv(table), type="primary", width="stretch",
                           file_name=f"portfolio_strategies_{years}y_returns.csv", mime="text/csv",
                           key="yoy_dl_main")
    cols = ['Invest Date', 'Compare Date'] + [f'{STRATEGY_NAMES[k]} {c}' for k in STRATEGY_KEYS
                                              for c in ('Value (₹)', 'Return (%)')]
    view = table[cols].copy()
    for c in ('Invest Date', 'Compare Date'):
        view[c] = view[c].dt.strftime('%d-%m-%Y')
    fmt = {c: (lambda v: inr(v, 0)) if '₹' in c else '{:+.2f}%' for c in cols[2:]}
    view, fmt = blank_missing(view, fmt)
    st.dataframe(style_change_cols(view.style.format(fmt), [c for c in cols if '%' in c]), width="stretch",
                 hide_index=True, height=38 + 35 * (len(table.attrs.get('assets', [])) + 3) - 36)