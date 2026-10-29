"""🏠 Main Dashboard: same-date, year-over-year returns for the TOTAL PORTFOLIO (all on one page, no tabs).

Portfolio = GoldBEES + Nifty 50 + MidCap + SmallCap, always all 4 (weights changeable in the sidebar).

"If I had invested ₹1,00,000 in the portfolio on 01-01-2010, what was it worth on 01-01-2011?
 If I had invested on 02-01-2010, what was it worth on 02-01-2011?"  ... for every day of every year.

Shows: headline numbers, a bell curve (distribution of all portfolio returns), a line curve (return for
every investment date), a year-wise table, a single-date lookup, portfolio vs each asset, CSV downloads.
"""
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import (
    panel, hero, section, kpi_row, note, sidebar_nav, sidebar_footer, sidebar_label, weight_badge,
    tone_of, inr, pct, hex_rgba,
    PLOT_LAYOUT, VIOLET, CYAN, AMBER, PINK, GOOD, BAD, FAINT, MUTED, INK,
)
from src.asset_data import ASSETS, ASSET_SHORT, ASSET_ICONS, COLORS, style_change_cols, blank_missing
from src.period_picker import today_ist
from src.yoy_returns import (same_date_returns, portfolio_same_date_returns, summary_stats, yearly_summary,
                             normal_curve, to_csv)

DEFAULT_START = date(2010, 1, 1)
PORT_COLOR = VIOLET



# =========================================================
# Zoom-friendly charts
# =========================================================
ZOOM_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,                                    # mouse wheel zooms where the pointer is
    "doubleClick": "reset",                                # double-click = back to the full view
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "autoScale2d"],
}


def _chart(fig, key=None, height=None):
    """Show a chart that zooms properly:
    - drag a box (or use the mouse wheel) and it zooms exactly on that part, left-right only,
      so the curves keep their shape;
    - no slide animation while zooming;
    - double-click (or the house button) brings back the full view;
    - the zoom stays put when the page refreshes after a setting change in the sidebar."""
    if height:
        fig.update_layout(height=height)
    fig.update_layout(transition=dict(duration=0), dragmode="zoom", uirevision=key or True)
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=True)                      # zoom only left-right
    st.plotly_chart(fig, theme=None, width="stretch", key=key, config=ZOOM_CONFIG)

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
    picked = available                       # always all 4: GoldBEES, Nifty 50, MidCap, SmallCap
    amount = st.sidebar.number_input("Amount invested on each date (₹)", min_value=1000.0, value=100000.0,
                                     step=5000.0, format="%.0f", key="yoy_amount")
    raw_w = {}
    if picked:
        sidebar_label("⚖️ Weights (%)")
        share = round(100 / len(picked), 2)
        defaults = [share] * (len(picked) - 1) + [round(100 - share * (len(picked) - 1), 2)]   # adds up to 100
        raw_w = {a: st.sidebar.number_input(f"{ASSET_ICONS[a]} {ASSET_SHORT[a]}", min_value=0.0, max_value=100.0,
                                            value=d, step=5.0, key=f"yoy_w_{a}_{len(picked)}")
                 for a, d in zip(picked, defaults)}
        with st.sidebar:
            weight_badge(sum(raw_w.values()))
    weights = {a: w for a, w in raw_w.items() if w > 0}

    sidebar_label("🎯 Year-over-year settings")
    if weights:
        data_min = max(asset_dfs[a]['Date'].min() for a in weights).date()      # all assets must have data
        data_max = min(min(asset_dfs[a]['Date'].max() for a in weights).date(), today_ist())
    else:
        data_min, data_max = DEFAULT_START, today_ist()
    from_date = st.sidebar.date_input("📍 First investment date", value=max(DEFAULT_START, data_min),
                                      min_value=data_min, max_value=data_max, format="DD-MM-YYYY",
                                      key="yoy_from")
    to_date = st.sidebar.date_input("🏁 Last comparison date", value=data_max, min_value=data_min,
                                    max_value=data_max, format="DD-MM-YYYY", key="yoy_to")
    if from_date > to_date:
        st.sidebar.error("First investment date must be before the last comparison date. Swapped them.")
        from_date, to_date = to_date, from_date

    years = st.sidebar.select_slider("⏳ Compare with the same date after", options=list(range(1, 11)),
                                     value=1, format_func=_yrs, key="yoy_years")
    basis = st.sidebar.radio("📆 Investment dates", ["Every calendar day", "Trading days only"],
                             key="yoy_basis",
                             help="Every calendar day: 1 Jan, 2 Jan, 3 Jan ... even holidays (the previous "
                                  "close is used). Trading days only: only dates where every asset has a price.")
    trading_only = basis == "Trading days only"
    sidebar_footer(fetch_errors, key="home")

    # ---------------- Calculation ----------------
    if not weights:
        hero("Main Dashboard", "Pick at least one asset with a weight above 0% in the sidebar.",
             page="home", eyebrow="Portfolio year-over-year returns")
        note("Choose the portfolio assets and give at least one a weight above 0%.", AMBER, "👈")
        return

    table = portfolio_same_date_returns(asset_dfs, weights, amount, from_date, to_date, years, trading_only,
                                        short_names=ASSET_SHORT)
    w_norm = {a: w / sum(weights.values()) for a, w in weights.items()}
    mix = " · ".join(f"{ASSET_ICONS[a]} {ASSET_SHORT[a]} {w * 100:.0f}%" for a, w in w_norm.items())

    # ---------------- Hero ----------------
    hero("Main Dashboard",
         f"Invest {inr(amount)} in the portfolio on any date, then compare its value on the same date "
         f"{_yrs(years)} later. Every day of every year, as a bell curve, a line curve and a CSV.",
         page="home", eyebrow="Portfolio year-over-year returns",
         chips=[f"💼 {mix}",
                f"📅 {from_date:%d-%m-%Y} → {to_date:%d-%m-%Y}",
                f"⏳ {_yrs(years)} holding",
                f"🧮 {len(table):,} comparisons"],
         line=_portfolio_index(asset_dfs, w_norm))

    if table.empty:
        note(f"Not enough data. The range must be longer than <b>{_yrs(years)}</b> and every asset in the "
             f"portfolio needs prices in it. Shared data runs from <b>{data_min:%d-%m-%Y}</b> to "
             f"<b>{data_max:%d-%m-%Y}</b>.", AMBER, "⚠️")
        return

    if 'SmallCap' in weights and not trading_only:
        note("SmallCap data is <b>monthly</b>, so on most days its previous month's close is used. "
             "Switch <b>Investment dates</b> to <b>Trading days only</b> for cleaner results.", AMBER, "💡")

    _data_quality_note(table)

    stats = summary_stats(table)
    _headline(stats, table, amount, w_norm)

    # ---------------- Charts & tables ----------------
    # All results on one page, one after another (no tabs)
    section("Bell curve", "How often each return happened", "🔔", PINK)
    _bell_tab(table, stats, years)

    section("CSV", "Every investment date with each asset's prices and the portfolio value", "📥", CYAN)
    _data_tab(table, years)

    section("Line chart", "Return for every investment date", "📈", VIOLET)
    _line_tab(table, years)

    section("Year by year", "Average, best and worst return per investment year", "🗓️", AMBER)
    _yearly_tab(table)

    section("Portfolio vs assets", "The portfolio next to each asset on its own", "🧩", GOOD)
    _vs_assets_tab(table, asset_dfs, w_norm, from_date, to_date, years, trading_only)


def _portfolio_index(asset_dfs, w_norm):
    """Decorative line for the hero: weighted growth of the portfolio over the last ~3 years."""
    series = []
    for a, w in w_norm.items():
        s = asset_dfs[a].set_index('Date')['Close'].iloc[-800:]
        series.append(w * s / s.iloc[0])
    return pd.concat(series, axis=1).sort_index().ffill().dropna().sum(axis=1).to_numpy()


def _data_quality_note(table):
    removed = pd.DataFrame(table.attrs.get('removed_prices', []), columns=['Asset', 'Date', 'Close'])
    removed['Date'] = pd.to_datetime(removed['Date'])
    skipped = table.attrs.get('skipped_dates', 0)
    not_common = table.attrs.get('dropped_not_common', 0)
    if removed.empty and not skipped and not not_common:
        return
    parts = []
    if not removed.empty:
        per = ", ".join(f"{a}: {n}" for a, n in removed['Asset'].value_counts().items())
        parts.append(f"<b>{len(removed)}</b> price rows look wrong (far from nearby prices; {per}) and were ignored")
    if skipped or not_common:
        parts.append(f"<b>{not_common:,}</b> investment dates were left out because at least one asset has no "
                     f"usable price there (a hole in its data)")
    with st.expander("🧹 Data check: " + "; ".join(parts).replace("<b>", "").replace("</b>", ""), expanded=False):
        st.caption("Fix the source sheet to include these dates.")
        if not removed.empty:
            shown = removed.assign(Date=removed['Date'].dt.strftime('%d-%m-%Y'))
            st.dataframe(shown, hide_index=True, width="stretch", height=220)


# =========================================================
# Headline cards
# =========================================================
def _headline(s, table, amount, w_norm):
    section("Headline", f"Total portfolio · {s['count']:,} investment dates compared", "⚡", CYAN)
    avg_gain = table['Change (₹)'].mean()
    kpi_row([
        dict(label="Positive outcomes", value=f"{s['pct_positive']:.1f}%",
             delta=f"{s['positive']:,} up · {s['negative']:,} down", tone=tone_of(s['pct_positive'] - 50),
             icon="✅", accent=GOOD),
        dict(label="Average return", value=pct(s['mean']),
             delta=f"{inr(avg_gain, 0, sign=True)} on {inr(amount)}", tone=tone_of(s['mean']), icon="📐",
             accent=PORT_COLOR),
        dict(label="Spread (std dev σ)", value=f"{s['std']:.2f}%",
             delta=f"90% between {s['p5']:+.1f}% and {s['p95']:+.1f}%", icon="🔔", accent=VIOLET),
        dict(label="Best", value=pct(s['max']),
             delta=f"{s['best_row']['Invest Date']:%d-%m-%Y} → {s['best_row']['Compare Date']:%d-%m-%Y}",
             tone="good", icon="🚀", accent=GOOD),
        dict(label="Worst", value=pct(s['min']),
             delta=f"{s['worst_row']['Invest Date']:%d-%m-%Y} → {s['worst_row']['Compare Date']:%d-%m-%Y}",
             tone="bad", icon="🧯", accent=BAD),
    ])


# =========================================================
# Bell curve
# =========================================================
def _bell_figure(r, mean, std, color, title, height=480):
    lo, hi = float(r.min()), float(r.max())
    n_bins = int(np.clip(np.sqrt(len(r)) * 1.2, 20, 80))
    counts, edges = np.histogram(r, bins=n_bins)
    width = edges[1] - edges[0]
    centers = (edges[:-1] + edges[1:]) / 2
    density = counts / (counts.sum() * width)
    share = counts / counts.sum() * 100

    fig = go.Figure()
    fig.add_vrect(x0=mean - std, x1=mean + std, fillcolor=hex_rgba(VIOLET, 0.10), line_width=0,
                  annotation_text="±1σ", annotation_position="top",
                  annotation_font=dict(color=MUTED, size=11))
    fig.add_trace(go.Bar(
        x=centers, y=density, width=width * 0.92, marker_color=[GOOD if c > 0 else BAD for c in centers],
        marker_line_width=0, opacity=0.85, name="Actual returns",
        customdata=np.stack([edges[:-1], edges[1:], counts, share], axis=1),
        hovertemplate="%{customdata[0]:.1f}% to %{customdata[1]:.1f}%<br>"
                      "%{customdata[2]:,} dates (%{customdata[3]:.1f}%)<extra></extra>"))
    pad = (hi - lo) * 0.08 or 1
    x, y = normal_curve(mean, std, lo - pad, hi + pad)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name="Normal (bell) curve",
                             line=dict(color=INK, width=2.6), hoverinfo='skip'))
    fig.add_vline(x=0, line_color=FAINT, line_dash="dot",
                  annotation_text="0% (break-even)", annotation_position="top right",
                  annotation_font=dict(color=MUTED, size=11))
    fig.add_vline(x=mean, line_color=color, line_width=2,
                  annotation_text=f"mean {mean:+.1f}%", annotation_position="bottom right",
                  annotation_font=dict(color=color, size=12))
    fig.update_layout(**PLOT_LAYOUT, title=title, height=height, bargap=0,
                      xaxis=dict(title="Return (%)", ticksuffix="%"), yaxis=dict(title="Density"),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.5))
    return fig


CHART_H = 400          # one height for every chart so a tab fits on one screen


def _bell_tab(table, s, years):
    r = table['Return (%)']
    within1 = ((r >= s['mean'] - s['std']) & (r <= s['mean'] + s['std'])).mean() * 100
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1, panel():
        _chart(_bell_figure(r, s['mean'], s['std'], "#C4B5FD",
                           f"Total portfolio: {_yrs(years)} same-date returns", height=CHART_H), key="yoy_bell")
    with c2:
        _mini_stats([
            ("Positive", f"{s['pct_positive']:.1f}%", GOOD),
            ("Negative", f"{100 - s['pct_positive']:.1f}%", BAD),
            ("Mean", pct(s['mean']), "#C4B5FD"),
            ("Median", pct(s['median']), "#C4B5FD"),
            ("Within ±1σ", f"{within1:.0f}%  (bell = 68%)", MUTED),
            ("±1σ range", f"{s['mean'] - s['std']:+.1f}% to {s['mean'] + s['std']:+.1f}%", MUTED),
        ])
    st.caption("Green bars = positive returns, red = negative. White line = normal (bell) curve with the same "
               "mean and spread.")


def _mini_stats(rows):
    """Compact stat list beside a chart. rows: (label, value, colour)."""
    html = "".join(
        f'<div style="display:flex;justify-content:space-between;gap:10px;padding:7px 0;'
        f'border-bottom:1px solid rgba(138,148,179,.15)"><span style="color:{MUTED};font-size:.85rem">{lbl}</span>'
        f'<b style="color:{c};font-size:.95rem;text-align:right">{val}</b></div>' for lbl, val, c in rows)
    st.markdown(f'<div style="padding:4px 2px">{html}</div>', unsafe_allow_html=True)


# =========================================================
# Line curve
# =========================================================
def _line_tab(table, years):
    c1, c2 = st.columns([1, 1], vertical_alignment="center")
    with c1:
        x_choice = st.radio("X-axis date", ["Investment date", "Compare date"], horizontal=True,
                            key="yoy_line_x", label_visibility="collapsed")
    with c2:
        smooth = st.toggle("Add 30-day average line", value=True, key="yoy_line_smooth")
    xcol = 'Invest Date' if x_choice == "Investment date" else 'Compare Date'

    t = table
    r = t['Return (%)']
    custom = np.stack([t['Invest Date'].dt.strftime('%d-%m-%Y'), t['Compare Date'].dt.strftime('%d-%m-%Y'),
                       t['Invest Value (₹)'], t['Compare Value (₹)']], axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[xcol], y=r.clip(lower=0), mode='lines', line=dict(width=0),
                             fill='tozeroy', fillcolor=hex_rgba(GOOD, 0.22), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=t[xcol], y=r.clip(upper=0), mode='lines', line=dict(width=0),
                             fill='tozeroy', fillcolor=hex_rgba(BAD, 0.25), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(
        x=t[xcol], y=r, mode='lines', name=f"Portfolio {_yrs(years)} return",
        line=dict(color="#C4B5FD", width=1.6), customdata=custom,
        hovertemplate="Invest %{customdata[0]}: ₹%{customdata[2]:,.0f}<br>"
                      "Value on %{customdata[1]}: ₹%{customdata[3]:,.0f}<br>"
                      "<b>%{y:+.2f}%</b><extra></extra>"))
    if smooth and len(t) > 30:
        fig.add_trace(go.Scatter(x=t[xcol], y=r.rolling(30, min_periods=10).mean(), mode='lines',
                                 name="30-day average", line=dict(color=INK, width=2, dash='dot'),
                                 hovertemplate="avg %{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=0, line_color=FAINT, line_dash="dot")
    fig.add_hline(y=r.mean(), line_color=VIOLET, line_dash="dash",
                  annotation_text=f"average {r.mean():+.1f}%", annotation_position="top left",
                  annotation_font=dict(color=VIOLET, size=11))
    fig.update_layout(**PLOT_LAYOUT, height=CHART_H + 30, hovermode='x unified',
                      title=f"Total portfolio: return after {_yrs(years)} for every {x_choice.lower()}",
                      yaxis=dict(title="Return (%)", ticksuffix="%"),
                      xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.55))
    with panel():
        _chart(fig, key="yoy_line")
    st.caption("Green = that investment date made money after the holding period, red = it lost money. "
               "Drag the slider under the chart to zoom.")


# =========================================================
# Year by year
# =========================================================
def _sign_color(v):
    """Green for a gain (or 0), red for a loss."""
    return GOOD if v >= 0 else BAD


def _year_figure(ys, title):
    """Bar = average return of the year; ▲ best / ▼ worst return, green if a gain, red if a loss."""
    yr = ys['Year'].str.replace('FY ', '', regex=False)
    d = lambda col: ys[col].dt.strftime('%d-%m-%Y')       # noqa: E731
    span = d('From') + " to " + d('To')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yr, y=ys['Average (%)'], name="Average",
        marker_color=[_sign_color(v) for v in ys['Average (%)']], marker_line_width=0, opacity=0.85,
        text=[f"{v:+.1f}%" for v in ys['Average (%)']], textposition='outside',
        customdata=np.stack([span, ys['Days compared'], ys['% Positive']], axis=1),
        hovertemplate="<b>%{x}</b> (invested %{customdata[0]})<br>average %{y:+.2f}% · "
                      "%{customdata[1]} days · %{customdata[2]:.0f}% positive<extra></extra>"))
    for label, col, symbol in (("Best", 'Best', 'triangle-up'), ("Worst", 'Worst', 'triangle-down')):
        vals = ys[f'{col} (%)']
        fig.add_trace(go.Scatter(
            x=yr, y=vals, mode='markers', name=label,
            marker=dict(color=[_sign_color(v) for v in vals], symbol=symbol, size=11,
                        line=dict(color="#FFFFFF", width=1)),
            customdata=np.stack([d(f'{col} Invest Date'), d(f'{col} Compare Date')], axis=1),
            hovertemplate=f"<b>%{{x}}</b> · {label.lower()} %{{y:+.2f}}%<br>"
                          "invested %{customdata[0]} → valued %{customdata[1]}<extra></extra>"))
    fig.add_hline(y=0, line_color=FAINT, line_dash="dot")
    fig.update_layout(**PLOT_LAYOUT, height=CHART_H, title=title,
                      yaxis=dict(ticksuffix="%"), xaxis=dict(type='category', tickangle=-45),
                      legend=dict(orientation='h', yanchor='top', y=-0.22, x=0),
                      margin=dict(l=12, r=12, t=50, b=60))
    return fig


def _yearly_tab(table):
    fy = yearly_summary(table, financial=True)            # April to March
    cy = yearly_summary(table, financial=False)           # January to December
    c1, c2 = st.columns(2)
    with c1, panel():
        _chart(_year_figure(fy, "Financial year (April to March)"), key="yoy_yearly_fy")
    with c2, panel():
        _chart(_year_figure(cy, "Calendar year (January to December)"), key="yoy_yearly_cy")
    st.caption("Bar = average return for investments made in that year. ▲ best and ▼ worst return in that year: "
               "green if it was a gain, red if a loss. Hover for the exact dates. Financial year 2010-11 = "
               "01-04-2010 to 31-03-2011. The first and last years can be partial.")


# =========================================================
# Portfolio vs each asset
# =========================================================
def _vs_assets_tab(table, asset_dfs, w_norm, from_date, to_date, years, trading_only):
    # Each asset on its own, but only on the same dates the portfolio uses (fair comparison)
    dates = pd.Index(table['Invest Date'])
    series = {'Portfolio': (table['Return (%)'].reset_index(drop=True), "#C4B5FD")}
    for a in w_norm:
        t = same_date_returns(asset_dfs[a], from_date, to_date, years, trading_only).set_index('Invest Date')
        series[ASSET_SHORT[a]] = (t.loc[t.index.intersection(dates), 'Return (%)'], COLORS[a])

    lo = min(s.min() for s, _ in series.values())
    hi = max(s.max() for s, _ in series.values())
    fig = go.Figure()
    for name, (r, color) in series.items():
        x, y = normal_curve(r.mean(), r.std(), lo, hi)
        is_p = name == 'Portfolio'
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name, line=dict(color=color, width=4 if is_p else 2),
                                 fill='tozeroy', fillcolor=hex_rgba(color, 0.18 if is_p else 0.06),
                                 hovertemplate=f"{name}: %{{x:.1f}}%<extra></extra>"))
        fig.add_vline(x=r.mean(), line_color=color, line_dash="dash", line_width=1.2)
    fig.add_vline(x=0, line_color=FAINT, line_dash="dot")
    fig.update_layout(**PLOT_LAYOUT, height=CHART_H, title="Bell curves",
                      xaxis=dict(ticksuffix="%"), yaxis=dict(showticklabels=False),
                      legend=dict(orientation='h', yanchor='top', y=-0.12, x=0), margin=dict(l=12, r=12, t=50, b=40))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=table['Invest Date'], y=table['Return (%)'], mode='lines', name='Portfolio',
                              line=dict(color="#C4B5FD", width=3),
                              hovertemplate="Portfolio: %{y:+.2f}%<extra></extra>"))
    for a in w_norm:
        col = f"{ASSET_SHORT[a]} Return (%)"
        fig2.add_trace(go.Scatter(x=table['Invest Date'], y=table[col], mode='lines', name=ASSET_SHORT[a],
                                  line=dict(color=COLORS[a], width=1.2), opacity=0.75,
                                  hovertemplate=f"{ASSET_SHORT[a]}: %{{y:+.2f}}%<extra></extra>"))
    fig2.add_hline(y=0, line_color=FAINT, line_dash="dot")
    fig2.update_layout(**PLOT_LAYOUT, height=CHART_H, hovermode='x unified',
                       title="Line curves", yaxis=dict(ticksuffix="%"),
                       legend=dict(orientation='h', yanchor='top', y=-0.12, x=0), margin=dict(l=12, r=12, t=50, b=40))

    c1, c2 = st.columns(2)
    with c1, panel():
        _chart(fig, key="yoy_vs_bell")
    with c2, panel():
        _chart(fig2, key="yoy_vs_line")

    rows = []
    for name, (r, _) in series.items():
        w = "100%" if name == 'Portfolio' else f"{w_norm[[a for a in w_norm if ASSET_SHORT[a] == name][0]] * 100:.1f}%"
        rows.append({'': name, 'Weight': w, 'Dates': int(r.size), '% Positive': (r > 0).mean() * 100,
                     'Average (%)': r.mean(), 'Median (%)': r.median(), 'Std dev (%)': r.std(),
                     'Worst (%)': r.min(), 'Best (%)': r.max()})
    summ = pd.DataFrame(rows)
    signed = ['Average (%)', 'Median (%)', 'Worst (%)', 'Best (%)']
    fmt = {c: '{:+.2f}%' for c in signed}
    fmt.update({'% Positive': '{:.1f}%', 'Std dev (%)': '{:.2f}%'})
    st.dataframe(style_change_cols(summ.style.format(fmt), signed), width="stretch", hide_index=True)


# =========================================================
# Data & CSV
# =========================================================
def _show_table(t):
    view = t.drop(columns=['Invest Year']).copy()
    for c in ('Invest Date', 'Compare Date'):
        view[c] = view[c].dt.strftime('%d-%m-%Y')
    fmt = {}
    for c in view.columns:
        if c.endswith('Price'):
            fmt[c] = lambda v: inr(v, 2)
        elif c in ('Invest Value (₹)', 'Compare Value (₹)'):
            fmt[c] = lambda v: inr(v, 0)
        elif c == 'Change (₹)':
            fmt[c] = lambda v: inr(v, 0, sign=True)
        elif c.endswith('(%)'):
            fmt[c] = '{:+.2f}%'
    view, fmt = blank_missing(view, fmt)
    colour = [c for c in view.columns if c.endswith('(%)') or c == 'Change (₹)']
    st.dataframe(style_change_cols(view.style.format(fmt), colour), width="stretch", hide_index=True, height=CHART_H - 40)


def _data_tab(table, years):
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"**{len(table):,} rows**, one per investment date: each asset's price on both dates, "
                    f"its own return, and the total portfolio value and return.")
    with c2:
        st.download_button("⬇️  Download CSV", to_csv(table), type="primary", width="stretch",
                           file_name=f"portfolio_same_date_{years}y_returns.csv", mime="text/csv",
                           key="yoy_dl_main")