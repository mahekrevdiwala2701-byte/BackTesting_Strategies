"""🏠 Main Dashboard: same-date, year-over-year returns of the TOTAL PORTFOLIO
under the 3 strategies of the Calculated Investment page, all on one screen (no tabs).

    🧺 Buy & Hold   ·   🔁 Rebalancing to the same weights   ·   📐 Different Weights (growth-based)

For every trading day from the first investment date, the whole portfolio (GoldBEES, Nifty 50, MidCap,
SmallCap by your weights) is bought and valued on the same date N years later.

Layout (top to bottom):
    0. Plain-language summary strip (which strategy won, chance of profit, bad case)
    1. Three strategy cards side by side: key numbers + histogram (% of start dates)
    2. Line chart: profit/loss area for one strategy (or all 3), picked with a toggle
    3. Year by year (left)  |  Range of outcomes: portfolio vs each asset (right)
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
                             to_csv)

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

    # ---------------- Row 0: plain-language summary ----------------
    _verdict(stats, amount, years)

    # ---------------- Row 1: the 3 strategy dashboards ----------------
    lo = min(s['min'] for s in stats.values())
    hi = max(s['max'] for s in stats.values())
    for col, key in zip(st.columns(3), STRATEGY_KEYS):
        with col, panel():
            _strategy_card(table, key, stats[key], settings[key], amount, years, (lo, hi))

    # ---------------- Row 2: line chart ----------------
    with panel():
        show = st.segmented_control(
            "Show on the chart", ["Buy & Hold", "Rebalancing", "Different Weights", "All 3 together"],
            default="Buy & Hold", key="yoy_line_show")
        chart(_line_figure(table, years, show or "Buy & Hold"), key="yoy_line")
        st.caption(f"**How to read:** each point is one start date. Its height is what you'd have earned if you "
                   f"invested that day and checked again {_yrs(years)} later. Green = profit, red = loss. "
                   f"Empty stretches = missing price data for at least one asset.")

    # ---------------- Row 3: year by year | portfolio vs assets ----------------
    c1, c2 = st.columns(2)
    with c1, panel():
        chart(_yearly_figure(table, years), key="yoy_yearly")
        st.caption("**How to read:** the average result for everyone who started investing in that year. "
                   "Hover a bar to see how many of those start dates were profitable.")
    with c2, panel():
        chart(_vs_assets_figure(table, w_norm, stats, years), key="yoy_vs")
        st.caption("**How to read:** the thick bar holds the middle 90% of outcomes, the thin line runs from "
                   "the worst to the best case, and the dot is the average. A shorter bar = steadier returns.")

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


def _verdict(stats, amount, years):
    """One-line answers to 'which strategy won?' and 'how risky was it?', in rupees."""
    best = max(STRATEGY_KEYS, key=lambda k: stats[k]['mean'])
    safest = max(STRATEGY_KEYS, key=lambda k: stats[k]['p5'])
    s = stats[best]
    cells = [
        ("🏆 Highest average", f"{ICONS[best]} {STRATEGY_NAMES[best]}",
         f"{pct(s['mean'])} → {inr(amount * (1 + s['mean'] / 100))} on {inr(amount)}"),
        ("🛡️ Smallest bad-case loss", f"{ICONS[safest]} {STRATEGY_NAMES[safest]}",
         f"1-in-20 worst case {pct(stats[safest]['p5'])}"),
        ("🎯 Chance of profit", f"{s['pct_positive']:.0f} in 100 start dates",
         f"made money after {_yrs(years)} ({STRATEGY_NAMES[best]})"),
        ("⚖️ Gap between strategies", f"{max(x['mean'] for x in stats.values()) - min(x['mean'] for x in stats.values()):.1f} pts",
         "difference in average return (small = they behave alike)"),
    ]
    html = "".join(
        f'<div style="flex:1;min-width:200px;background:{hex_rgba("#8A94B3", .07)};border-radius:12px;padding:10px 14px">'
        f'<div style="color:{MUTED};font-size:.74rem;text-transform:uppercase;letter-spacing:.04em">{l}</div>'
        f'<div style="color:{TEXT};font-weight:800;font-size:1.08rem;margin:2px 0">{v}</div>'
        f'<div style="color:{MUTED};font-size:.8rem">{d}</div></div>' for l, v, d in cells)
    st.markdown(f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:4px 0 14px">{html}</div>',
                unsafe_allow_html=True)


def _strategy_card(table, key, s, setting, amount, years, x_range):
    color = STRAT_COLORS[key]
    name = STRATEGY_NAMES[key]
    gain = table[f'{name} Value (₹)'].mean() - amount
    cell = ('<div style="flex:1;min-width:0"><div style="color:{m};font-size:.72rem;text-transform:uppercase;'
            'letter-spacing:.04em">{l}</div><div style="color:{c};font-weight:700;font-size:1.05rem">{v}</div></div>')
    cells = "".join(cell.format(m=MUTED, l=lbl, c=c, v=v) for lbl, v, c in [
        ("Profitable", f"{s['pct_positive']:.0f}%", GOOD),
        ("Average", pct(s['mean'], 1), GOOD if s['mean'] >= 0 else BAD),
        ("Worst", pct(s['min'], 1), BAD),
        ("Best", pct(s['max'], 1), GOOD),
    ])
    st.markdown(
        f'<div style="border-left:3px solid {color};padding:2px 0 2px 10px;margin-bottom:6px">'
        f'<div style="font-weight:800;color:{TEXT};font-size:1.02rem">{ICONS[key]} {name}</div>'
        f'<div style="color:{MUTED};font-size:.78rem;line-height:1.35;min-height:2.7em">{setting}<br>'
        f'average result {inr(amount + gain, 0)} from {inr(amount)} ({inr(gain, 0, sign=True)})'
        f'</div></div><div style="display:flex;gap:8px;margin:4px 0 2px">{cells}</div>',
        unsafe_allow_html=True)
    chart(_bell_figure(table[f'{name} Return (%)'], s, color, x_range, years), key=f"yoy_bell_{key}")


def _bell_figure(r, s, color, x_range, years=1):
    """Histogram: how many start dates ended with each return. Y = % of start dates (readable),
    no fitted normal curve (real returns are lopsided, the curve misled more than it helped)."""
    lo, hi = x_range
    step = max(1.0, round((hi - lo) / 35))                      # ~35 bars, edges on whole % steps
    edges = np.arange(np.floor(lo / step) * step, hi + step, step)
    counts, edges = np.histogram(r.dropna(), bins=edges)
    share = counts / max(counts.sum(), 1) * 100
    centers = (edges[:-1] + edges[1:]) / 2
    loss = share[centers < 0].sum()
    fig = go.Figure(go.Bar(
        x=centers, y=share, width=step * 0.9, marker_line_width=0,
        marker_color=[hex_rgba(BAD, .85) if c < 0 else hex_rgba(GOOD, .85) for c in centers],
        customdata=np.stack([edges[:-1], edges[1:], counts], axis=1), showlegend=False,
        hovertemplate="Return %{customdata[0]:.0f}% to %{customdata[1]:.0f}%<br>"
                      "%{customdata[2]:,} start dates (%{y:.1f}%)<extra></extra>"))
    fig.add_vline(x=0, line_color=TEXT, line_width=1, line_dash="dot")
    fig.add_vline(x=s['mean'], line_color=color, line_width=2,
                  annotation_text=f"Average {s['mean']:+.1f}%", annotation_position="top right",
                  annotation_font=dict(color=color, size=11))
    top = share.max() * 1.18 if share.size else 1
    fig.add_annotation(x=lo, y=top, xanchor="left", yanchor="top", showarrow=False,
                       text=f"<b>{loss:.0f}%</b> lost money", font=dict(color=BAD, size=11))
    fig.add_annotation(x=hi, y=top, xanchor="right", yanchor="top", showarrow=False,
                       text=f"<b>{100 - loss:.0f}%</b> made money", font=dict(color=GOOD, size=11))
    fig.update_layout(**PLOT_LAYOUT, height=H - 30, bargap=0, margin=dict(l=8, r=8, t=12, b=8),
                      xaxis=dict(ticksuffix="%", range=[lo, hi], title=dict(text=f"Return after {_yrs(years)}",
                                 font=dict(size=11, color=MUTED))),
                      yaxis=dict(ticksuffix="%", range=[0, top], title=dict(text="% of start dates",
                                 font=dict(size=11, color=MUTED))))
    return fig


def _with_gaps(table):
    """Insert an empty row wherever the data has a hole, so lines break instead of bridging it."""
    t = table.set_index('Invest Date')
    gap = t.index.to_series().diff().dt.days > 10
    if not gap.any():
        return t
    before = t.index.to_series().shift(1)[gap] + pd.Timedelta(days=1)     # day after the last real point
    after = t.index[gap] - pd.Timedelta(days=1)                            # day before data resumes
    blanks = pd.DataFrame(index=pd.DatetimeIndex(list(before) + list(after)), columns=t.columns)
    return pd.concat([t, blanks]).sort_index()


def _line_figure(table, years, show="Buy & Hold"):
    """One strategy: green area above 0 (profit), red below (loss).
    'All 3 together': the 3 lines (they usually overlap, which itself tells you they behave alike)."""
    t = _with_gaps(table)
    fig = go.Figure()
    if show in STRATEGY_NAMES.values():
        y = t[f'{show} Return (%)'].astype(float)
        # gap rows become 0 so the coloured areas close at the hole instead of bridging it
        for part, c, label in ((y.where(y >= 0, 0).fillna(0), GOOD, "Profit"),
                               (y.where(y < 0, 0).fillna(0), BAD, "Loss")):
            fig.add_trace(go.Scatter(x=t.index, y=part, mode='lines', name=label, connectgaps=False,
                                     line=dict(color=c, width=0.8), fill='tozeroy', fillcolor=hex_rgba(c, .30),
                                     hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=t.index, y=y, mode='lines', name=show, connectgaps=False,
                                 line=dict(color=TEXT, width=1.1),
                                 hovertemplate=f"Invested %{{x|%d %b %Y}}<br>{show}: <b>%{{y:+.1f}}%</b>"
                                               f" after {_yrs(years)}<extra></extra>"))
        title = f"{show}: return after {_yrs(years)}, by the date you invested"
    else:
        for key in STRATEGY_KEYS:
            name = STRATEGY_NAMES[key]
            fig.add_trace(go.Scatter(x=t.index, y=t[f'{name} Return (%)'].astype(float), mode='lines', name=name,
                                     connectgaps=False, line=dict(color=STRAT_COLORS[key], width=1.6),
                                     hovertemplate=f"{name}: %{{y:+.1f}}%<extra></extra>"))
        title = f"All 3 strategies: return after {_yrs(years)}, by the date you invested"
    fig.add_hline(y=0, line_color=TEXT, line_width=1, line_dash="dot")
    fig.update_layout(**PLOT_LAYOUT, height=H + 40, hovermode='x unified' if show not in STRATEGY_NAMES.values()
                      else 'closest', title=title,
                      xaxis=dict(title=dict(text="Date you invested", font=dict(size=11, color=MUTED))),
                      yaxis=dict(ticksuffix="%", title=dict(text=f"Return {_yrs(years)} later",
                                 font=dict(size=11, color=MUTED))),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig


def _yearly_figure(table, years=1):
    ys = yearly_summary(table)
    fig = go.Figure()
    for key in STRATEGY_KEYS:
        name = STRATEGY_NAMES[key]
        fig.add_trace(go.Bar(x=ys['Year'].astype(str), y=ys[f'{name} Avg (%)'], name=name,
                             marker_color=STRAT_COLORS[key], marker_line_width=0,
                             customdata=ys[f'{name} % Positive'],
                             hovertemplate=f"{name} · started in %{{x}}<br>average %{{y:+.1f}}%"
                                           f" · %{{customdata:.0f}}% of start dates profitable<extra></extra>"))
    fig.add_hline(y=0, line_color=TEXT, line_width=1, line_dash="dot")
    fig.update_layout(**PLOT_LAYOUT, height=H, barmode='group', bargap=0.25,
                      title=f"Average {_yrs(years)} return, by the year you started",
                      yaxis=dict(ticksuffix="%", title=dict(text="Average return", font=dict(size=11, color=MUTED))),
                      xaxis=dict(type='category', title=dict(text="Year you invested", font=dict(size=11, color=MUTED))),
                      legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1),
                      margin=dict(l=12, r=12, t=60, b=40))
    return fig


def _vs_assets_figure(table, w_norm, stats, years):
    """Range of outcomes for each strategy and each single asset (replaces the overlapping bell curves):
    thin line = worst to best, thick bar = middle 90% (5th to 95th percentile), dot = average."""
    rows = [(f"{ICONS[k]} {STRATEGY_NAMES[k]}", stats[k], STRAT_COLORS[k]) for k in STRATEGY_KEYS]
    rows += [(f"{ASSET_ICONS[a]} {ASSET_SHORT[a]} only", summary_stats(table[f'{ASSET_SHORT[a]} Return (%)']),
              COLORS[a]) for a in w_norm]
    rows = rows[::-1]                                             # strategies at the top
    names = [r[0] for r in rows]
    fig = go.Figure()
    for name, s, c in rows:
        fig.add_trace(go.Scatter(x=[s['min'], s['max']], y=[name, name], mode='lines',
                                 line=dict(color=hex_rgba(c, .55), width=2), hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=[s['p5'], s['p95']], y=[name, name], mode='lines',
                                 line=dict(color=c, width=14), showlegend=False,
                                 hovertemplate=f"<b>{name}</b><br>worst {s['min']:+.1f}% · best {s['max']:+.1f}%<br>"
                                               f"middle 90%: {s['p5']:+.1f}% to {s['p95']:+.1f}%<br>"
                                               f"average {s['mean']:+.1f}% · profitable {s['pct_positive']:.0f}%"
                                               f"<extra></extra>"))
        fig.add_trace(go.Scatter(x=[s['mean']], y=[name], mode='markers', showlegend=False, hoverinfo='skip',
                                 marker=dict(color=TEXT, size=9, line=dict(color="#0A0F1E", width=2))))
    fig.add_vline(x=0, line_color=TEXT, line_width=1, line_dash="dot")
    fig.update_layout(**PLOT_LAYOUT, height=H, title=f"Portfolio vs each asset alone: range of {_yrs(years)} results",
                      xaxis=dict(ticksuffix="%", title=dict(text=f"Return after {_yrs(years)}",
                                 font=dict(size=11, color=MUTED))),
                      yaxis=dict(categoryorder='array', categoryarray=names),
                      margin=dict(l=12, r=12, t=50, b=40))
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