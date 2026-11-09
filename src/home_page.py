"""🏠 Main Dashboard: same-date, year-over-year returns for the TOTAL PORTFOLIO.

Portfolio = GoldBEES + Nifty 50 + MidCap + SmallCap, always all 4 (weights changeable in the sidebar).

"If I had invested ₹1,00,000 in the portfolio on 01-01-2010, what was it worth on 01-01-2011?
 If I had invested on 02-01-2010, what was it worth on 02-01-2011?"  ... for every day of every year.

Layout (top to bottom):
  1. Hero banner + "How to read this page" strip
  2. Headline cards (always visible)
  3. View switcher -> one view at a time:
       Return spread (bell curve) · Return timeline · Drawdown (underwater) · Alpha · Year by year · Data
Strategy + benchmark live in the sidebar, so the Drawdown and Alpha views use the same portfolio.
"""
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import (
    panel, hero, section, kpi_row, note, explain, steps, start_value, action_bar, sidebar_nav, sidebar_footer, sidebar_label, weight_badge,
    tone_of, inr, pct, hex_rgba,
    PLOT_LAYOUT, STRAT_COLORS, PORTFOLIO, GOOD_TEXT, BAD_TEXT, INDIGO,
    VIOLET, CYAN, AMBER, PINK, BLUE, GOOD, BAD, FAINT, MUTED, INK,
)
from src.asset_data import ASSETS, ASSET_SHORT, ASSET_ICONS, COLORS, style_change_cols, blank_missing
from src.period_picker import today_ist
from src.yoy_returns import (same_date_returns, portfolio_same_date_returns, summary_stats, yearly_summary,
                             normal_curve, to_csv)
from src.alpha import (calculated_investment_values, benchmark_close, period_alpha, alpha_summary,
                       FREQUENCIES as ALPHA_FREQS)
from src.strategies import STRATEGIES, REBAL_FREQ, DIFF_REBAL_FREQ, LOOKBACK_OPTIONS
from src.drawdown import underwater, drawdown_periods, drawdown_summary
from src.tearsheet import metrics as ts_metrics, monthly_table, LAYOUT as TS_LAYOUT, fmt as ts_fmt

DEFAULT_START = date(2010, 1, 1)
PORT_COLOR = INDIGO        # cards about the whole portfolio (charts draw it as the white PORTFOLIO line)


# =========================================================
# Zoom-friendly charts
# =========================================================
ZOOM_CONFIG = {
    "displaylogo": False,
    "scrollZoom": False,                                   # mouse wheel scrolls the PAGE, never zooms
    "responsive": True,                                    # re-measure the chart when the window changes
    "doubleClick": "reset",                                # double-click = back to the full view
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "autoScale2d"],
}


def _chart(fig, key=None, height=None):
    """Show a chart whose box-zoom lands exactly where you drag:
    - drag a rectangle with the zoom tool and it zooms into exactly that rectangle;
    - no slide/fade animation on the chart or its panel (CSS transforms on a parent make Plotly
      put the zoom in the wrong place on some screens);
    - the mouse wheel scrolls the page, it never zooms;
    - double-click (or the house button) brings back the full view."""
    if height:
        fig.update_layout(height=height)
    fig.update_layout(transition=dict(duration=0), dragmode="zoom",
                      uirevision=f"{key}|{fig.layout.xaxis.range}")
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    st.plotly_chart(fig, theme=None, width="stretch", key=key, config=ZOOM_CONFIG)


_NO_ANIMATION_CSS = """<style>
/* Charts on this page: no animations or transforms around them, so zoom/pan coordinates are exact */
[data-testid="stPlotlyChart"], [class*="st-key-panel_"], [data-testid="stVerticalBlockBorderWrapper"] {
  animation: none !important; transform: none !important;
}
</style>"""


def _yrs(y):
    return f"{y} year" + ("s" if y > 1 else "")


# =========================================================
# Cached calculations: each one runs once per set of inputs, so clicking around the page
# (or a second visitor with the same settings) does not redo the heavy maths.
# =========================================================
@st.cache_data(show_spinner="Calculating portfolio returns...", max_entries=32)
def _cached_portfolio_table(asset_dfs, weights, amount, from_date, to_date, years, trading_only):
    return portfolio_same_date_returns(asset_dfs, weights, amount, from_date, to_date, years, trading_only,
                                       short_names=ASSET_SHORT)


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_same_date(df, from_date, to_date, years, trading_only):
    return same_date_returns(df, from_date, to_date, years, trading_only)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_yearly(table, financial):
    return yearly_summary(table, financial=financial)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_csv(table):
    return to_csv(table)


@st.cache_data(show_spinner="Calculating alpha...", max_entries=64)
def _cached_values(asset_dfs, w_norm, amount, from_date, to_date, mode, freq_m, look_m):
    return calculated_investment_values(asset_dfs, w_norm, amount, from_date, to_date, mode, freq_m, look_m)


@st.cache_data(show_spinner=False, max_entries=256)
def _cached_alpha(asset_dfs, w_norm, amount, from_date, to_date, mode, freq_m, look_m, bench, freq):
    values = _cached_values(asset_dfs, w_norm, amount, from_date, to_date, mode, freq_m, look_m)
    if values is None:
        return None
    return period_alpha(values, benchmark_close(asset_dfs, bench), freq)


@st.cache_data(show_spinner="Calculating drawdowns...", max_entries=64)
def _cached_drawdown(asset_dfs, w_norm, amount, from_date, to_date, mode, freq_m, look_m, bench):
    values = _cached_values(asset_dfs, w_norm, amount, from_date, to_date, mode, freq_m, look_m)
    if values is None or len(values.dropna()) < 2:
        return None
    dd = underwater(values)
    b = benchmark_close(asset_dfs, bench)
    b = b.reindex(b.index.union(dd.index)).ffill().reindex(dd.index).dropna()
    return dd, drawdown_periods(dd), underwater(b), values


# =========================================================
# Page
# =========================================================
def render_home_page(asset_dfs, fetch_errors):
    available = [a for a in ASSETS if a in asset_dfs]

    st.markdown(_NO_ANIMATION_CSS, unsafe_allow_html=True)

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
    strat = _strategy_sidebar(available)
    sidebar_footer(fetch_errors, key="home")

    # ---------------- Calculation ----------------
    if not weights:
        hero("Main Dashboard", "Pick at least one asset with a weight above 0% in the sidebar.",
             page="home", eyebrow="Portfolio year-over-year returns")
        note("Choose the portfolio assets and give at least one a weight above 0%.", AMBER, "👈")
        return

    table = _cached_portfolio_table(asset_dfs, weights, amount, from_date, to_date, years, trading_only)
    w_norm = {a: w / sum(weights.values()) for a, w in weights.items()}
    mix = " · ".join(f"{ASSET_ICONS[a]} {ASSET_SHORT[a]} {w * 100:.0f}%" for a, w in w_norm.items())

    # ---------------- Hero ----------------
    hero("Main Dashboard",
         f"How the portfolio did if you invested {inr(amount)} on any single day and checked it "
         f"{_yrs(years)} later, plus how deep it fell from its highs along the way.",
         page="home", eyebrow="Portfolio year-over-year returns",
         chips=[f"💼 {mix}",
                f"📅 {from_date:%d-%m-%Y} → {to_date:%d-%m-%Y}",
                f"⏳ {_yrs(years)} holding",
                f"🧭 {strat['short']}",
                f"🧮 {len(table):,} comparisons"],
         line=_portfolio_index(asset_dfs, w_norm))

    if table.empty:
        note(f"Not enough data. The range must be longer than <b>{_yrs(years)}</b> and every asset in the "
             f"portfolio needs prices in it. Shared data runs from <b>{data_min:%d-%m-%Y}</b> to "
             f"<b>{data_max:%d-%m-%Y}</b>.", AMBER, "⚠️")
        return

    _how_to_read(years)

    if 'SmallCap' in weights and not trading_only:
        note("SmallCap data is <b>monthly</b>, so on most days its previous month's close is used. "
             "Switch <b>Investment dates</b> to <b>Trading days only</b> for cleaner results.", AMBER, "💡")

    _data_quality_note(table)
    stats = summary_stats(table)

    # ---------------- Headline (always visible) ----------------
    _headline(stats, table, amount, w_norm, years)

    # ---------------- Tearsheet: charts in sequence (left) + key metrics (right) ----------------
    settings = (w_norm, amount, from_date, to_date, strat['mode'], strat['freq_m'], strat['look_m'], strat['bench'])
    _tearsheet(asset_dfs, settings, strat, table, stats, years)

    # ---------------- Tables & downloads (full width) ----------------
    section("Details & downloads", "Drawdown periods, alpha table, full data and each asset on its own", "📥", CYAN)
    _details(asset_dfs, settings, strat, table, years, w_norm, from_date, to_date, trading_only)


def _strategy_sidebar(avail):
    """Strategy + benchmark (sidebar). Shared by the Drawdown and Alpha views."""
    sidebar_label("🧭 Strategy & benchmark")
    mode = st.sidebar.selectbox("Strategy", list(STRATEGIES), format_func=STRATEGIES.get, key="al_mode",
                                help="Used by the Drawdown and Alpha views. " + STRATEGY_HELP["bh"])
    freq_m, look_m, extra = 1, 12, ""
    if mode == "periodic":
        start_value("al_rebal_freq", list(REBAL_FREQ)[1], list(REBAL_FREQ))
        f = st.sidebar.selectbox("🔁 Rebalance every", list(REBAL_FREQ), key="al_rebal_freq")
        freq_m, extra = REBAL_FREQ[f], f" · rebalanced {f.lower()}"
    elif mode == "diff":
        start_value("al_diff_lookback", list(LOOKBACKS)[1], list(LOOKBACKS))
        lb = st.sidebar.selectbox("🔍 Compare growth over the previous", list(LOOKBACKS), key="al_diff_lookback")
        f = st.sidebar.selectbox("🔁 Rebalance every", list(DIFF_REBAL_FREQ), key="al_diff_freq")
        freq_m, look_m = DIFF_REBAL_FREQ[f], LOOKBACKS[lb]
        extra = f" · {f.lower()}, growth over the previous {lb}"
    start_value("al_bench", 'LargeCap' if 'LargeCap' in avail else avail[0], avail)
    bench = st.sidebar.selectbox("📏 Benchmark", avail, key="al_bench",
                                 format_func=lambda a: f"{ASSET_ICONS[a]} {ASSET_SHORT[a]}")
    short = STRATEGIES[mode].split(" ", 1)[1].split(" (")[0]
    return dict(mode=mode, freq_m=freq_m, look_m=look_m, bench=bench,
                desc=f"{STRATEGIES[mode]}{extra}", short=short)


def _how_to_read(years):
    """Plain-language guide: what the page is answering, in 4 steps."""
    with st.expander("📖 New here? How to read this page", expanded=False):
        steps([
            ("💼", "Your portfolio", "The 4 assets with the weights from the sidebar, treated as one investment.", INDIGO),
            ("📅", "Every start date", f"Pretend you invested on each day in the range and checked the value "
                                       f"exactly {_yrs(years)} later. Each day = one result.", VIOLET),
            ("⚡", "Headline cards", "Summary of all those results: how often you made money, the average, "
                                    "the spread, the best and the worst.", CYAN),
            ("🔍", "Pick a view", "Bell curve, timeline, drawdown (worst falls), alpha vs benchmark, "
                                 "year-wise and raw data. One at a time, below the cards.", PINK),
        ])


@st.cache_data(show_spinner=False, max_entries=32)
def _portfolio_index(asset_dfs, w_norm):
    """Decorative line for the hero: weighted growth of the portfolio over the last ~3 years."""
    series = []
    for a, w in w_norm.items():
        s = asset_dfs[a].set_index('Date')['Close'].iloc[-800:]
        series.append(w * s / s.iloc[0])
    return pd.concat(series, axis=1, sort=True).ffill().dropna().sum(axis=1).to_numpy()


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
def _headline(s, table, amount, w_norm, years):
    section("Headline", f"Total portfolio · {s['count']:,} investment dates compared", "⚡", CYAN)
    explain(f"Out of {s['count']:,} possible start dates, {s['pct_positive']:.0f}% were worth more {_yrs(years)} "
            f"later. On average {inr(amount)} became {inr(amount * (1 + s['mean'] / 100))}.", CYAN)
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
                  annotation_text="±1σ", annotation_position="bottom",
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
                  annotation_text="0%", annotation_position="top left",
                  annotation_font=dict(color=MUTED, size=11))
    fig.add_vline(x=mean, line_color=color, line_width=2,
                  annotation_text=f"mean {mean:+.1f}%", annotation_position="top right",
                  annotation_font=dict(color=color, size=12))
    fig.update_layout(**PLOT_LAYOUT, title=title, height=height, bargap=0,
                      xaxis=dict(title="Return (%)", ticksuffix="%"), yaxis=dict(title="Density"),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.5))
    return fig


CHART_H = 360          # one height for every chart in the tearsheet


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


# =========================================================
# Portfolio vs each asset
# =========================================================
def _vs_assets_tab(table, asset_dfs, w_norm, from_date, to_date, years, trading_only):
    # Each asset on its own, but only on the same dates the portfolio uses (fair comparison)
    dates = pd.Index(table['Invest Date'])
    series = {'Portfolio': (table['Return (%)'].reset_index(drop=True), PORTFOLIO)}
    for a in w_norm:
        t = _cached_same_date(asset_dfs[a], from_date, to_date, years, trading_only).set_index('Invest Date')
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

    xd = table['Invest Date'].dt.strftime('%Y-%m-%d')      # short date text = smaller page
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=xd, y=table['Return (%)'].round(2), mode='lines', name='Portfolio',
                              line=dict(color=PORTFOLIO, width=3),
                              hovertemplate="Portfolio: %{y:+.2f}%<extra></extra>"))
    for a in w_norm:
        col = f"{ASSET_SHORT[a]} Return (%)"
        fig2.add_trace(go.Scatter(x=xd, y=table[col].round(2), mode='lines', name=ASSET_SHORT[a],
                                  line=dict(color=COLORS[a], width=1.2), opacity=0.75,
                                  hovertemplate=f"{ASSET_SHORT[a]}: %{{y:+.2f}}%<extra></extra>"))
    fig2.add_hline(y=0, line_color=FAINT, line_dash="dot")
    fig2.update_layout(**PLOT_LAYOUT, height=CHART_H, hovermode='x unified',
                       title="Line curves", yaxis=dict(ticksuffix="%"),
                       legend=dict(orientation='h', yanchor='top', y=-0.12, x=0), margin=dict(l=12, r=12, t=50, b=40))

    with panel():
        _chart(fig, key="yoy_vs_bell")
    with panel():
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
# Drawdown: underwater plot
# =========================================================
def _drawdown_figure(dd, bench_dd, b_name, avg, worst, show_bench, height):
    x = dd.index.strftime('%Y-%m-%d')
    y = dd.round(2)
    fig = go.Figure()
    if show_bench and bench_dd is not None and not bench_dd.empty:
        fig.add_trace(go.Scatter(x=bench_dd.index.strftime('%Y-%m-%d'), y=bench_dd.round(2), mode='lines',
                                 name=b_name, line=dict(color=FAINT, width=1.3),
                                 hovertemplate=f"{b_name}: %{{y:.2f}}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', name="Portfolio", fill='tozeroy', fillcolor=hex_rgba(BLUE, 0.20),
        line=dict(color=BLUE, width=1.4),
        customdata=dd.index.strftime('%d-%m-%Y'),
        hovertemplate="%{customdata}<br><b>%{y:.2f}%</b> below the previous high<extra></extra>"))
    fig.add_hline(y=0, line_color=INK, line_dash="dash", line_width=1.2)
    fig.add_hline(y=avg, line_color=BAD, line_dash="dash", line_width=2.2,
                  annotation_text=f"average drawdown {avg:.1f}%", annotation_position="bottom right",
                  annotation_font=dict(color=BAD_TEXT, size=11))
    if worst is not None:
        fig.add_trace(go.Scatter(
            x=[worst['Trough'].strftime('%Y-%m-%d')], y=[round(worst['Depth (%)'], 2)], mode='markers+text',
            marker=dict(color=BAD, size=10, line=dict(color="#FFFFFF", width=1.5)),
            text=[f"max {worst['Depth (%)']:.1f}%"], textposition="bottom center",
            textfont=dict(color=BAD_TEXT, size=11), name="Max drawdown", showlegend=False,
            hovertemplate=f"Max drawdown %{{y:.2f}}%<br>{worst['Trough']:%d-%m-%Y}<extra></extra>"))
    lo = min(dd.min(), bench_dd.min() if show_bench and bench_dd is not None and not bench_dd.empty else 0)
    fig.update_layout(**PLOT_LAYOUT, height=height, hovermode='x unified', title="Underwater plot",
                      yaxis=dict(title="Below previous high", ticksuffix="%", range=[lo * 1.12, 2]),
                      showlegend=show_bench, legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.6))
    return fig


# =========================================================
# Alpha: whole portfolio vs one benchmark, every month
# =========================================================
STRATEGY_HELP = {
    "bh": "Money is split once and never touched again. Each asset is bought on its own first trading "
          "day in the period (if an asset's data starts later, its share waits as cash until then). "
          "Weights drift as prices move.",
    "periodic": "Money is split on the first day, then every month / quarter / half-year / year the portfolio "
                "is brought back to your target weights (sell what grew, buy what fell).",
    "diff": "First buy uses your weights. On every rebalance date each asset's growth over the previous "
            "look-back period is compared: the asset that grew the MOST gets the SMALLEST of your weights, "
            "and the one that grew the LEAST (or fell) gets the BIGGEST.",
}
LOOKBACKS = {**LOOKBACK_OPTIONS, "2 Years": 24, "2.5 Years": 30}


def _alpha_table(tables, b_name, start_day):
    """Every frequency in one table: Frequency column first, biggest period first, newest first inside each."""
    order = [f for f in reversed(list(ALPHA_FREQS)) if tables.get(f) is not None and not tables[f].empty]
    parts = []
    for f in order:
        t = tables[f].iloc[::-1]                                # newest first
        parts.append(pd.DataFrame({
            'Frequency': f,
            'Period': t['Period'] + np.where(t['Partial'], " *", ""),
            'Date': t['To'].dt.strftime('%d-%m-%Y'),
            f'{b_name} (start = 100)': t['Benchmark index'],
            'Portfolio (start = 100)': t['Portfolio index'],
            f'{b_name} return': t['Benchmark (%)'],
            'Portfolio return': t['Portfolio (%)'],
            'Alpha (active return)': t['Alpha (%)'],
        }))
    view = pd.concat(parts, ignore_index=True)

    counts = " · ".join(f"{len(tables[f])} {ALPHA_FREQS[f][0]}s" for f in order)
    st.caption(f"Alpha for every year, half-year, quarter, month and week: {counts}.")
    on = action_bar(download=dict(label="⬇️  Download alpha CSV", data=view.round(4).to_csv(index=False).encode('utf-8-sig'),
                                  file_name=f"alpha_all_periods_vs_{b_name.replace(' ', '_').lower()}.csv",
                                  key="dl_alpha_csv"),
                    toggles=[("📋 Show the alpha table", "al_show_table")])
    if not on["al_show_table"]:
        return
    start_value("al_table_freqs", list(order), order)
    show = st.pills("Show in the table", order, selection_mode="multi", key="al_table_freqs")
    view = view[view['Frequency'].isin(show or order)].reset_index(drop=True)

    ret_cols = [f'{b_name} return', 'Portfolio return']
    fmt = {f'{b_name} (start = 100)': '{:,.1f}', 'Portfolio (start = 100)': '{:,.1f}',
           **{c: '{:+.2f}%' for c in ret_cols}, 'Alpha (active return)': '{:+.2f}%'}

    def alpha_bg(v):
        if pd.isna(v):
            return ''
        return (f'background-color: {hex_rgba(GOOD, 0.16)}; color: {GOOD_TEXT}; font-weight: 700' if v > 0 else
                f'background-color: {hex_rgba(BAD, 0.16)}; color: {BAD_TEXT}; font-weight: 700' if v < 0 else '')

    styler = view.style.format(fmt, na_rep="—")
    styler = style_change_cols(styler, ret_cols)
    styler = (styler.map if hasattr(styler, 'map') else styler.applymap)(alpha_bg, subset=['Alpha (active return)'])
    st.dataframe(styler, width="stretch", hide_index=True, height=min(CHART_H + 120, 38 + 35 * len(view)))
    st.caption(f"All values start at 100 on {pd.Timestamp(start_day):%d-%m-%Y}. (start = 100): what ₹100 put in on "
               f"that day was worth at the end of the period. Return: change during that period. "
               f"Alpha = portfolio return − {b_name} return. * = part period (the dates start or end inside it). "
               f"Use the buttons above the table to hide frequencies.")


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


def _data_tab(table, years, asset_dfs, w_norm, from_date, to_date, trading_only):
    st.caption(f"{len(table):,} rows, one per investment date: each asset's price on both dates, its own return, "
               f"and the total portfolio value and return.")
    on = action_bar(download=dict(label="⬇️  Download CSV", data=_cached_csv(table), key="yoy_dl_main",
                                  file_name=f"portfolio_same_date_{years}y_returns.csv"),
                    toggles=[("🧩 Compare with each asset", "home_show_vs"), ("📋 Show the full table", "home_show_table")])
    if on["home_show_vs"]:
        section("Portfolio vs assets", "The portfolio next to each asset on its own", "🧩", GOOD)
        _vs_assets_tab(table, asset_dfs, w_norm, from_date, to_date, years, trading_only)
    if on["home_show_table"]:
        _show_table(table)


# =========================================================
# Tearsheet: every chart in sequence on the left, key metrics on the right
# =========================================================
_TEAR_CSS = """<style>
/* numbered chart headers */
.blk { display:flex; align-items:baseline; gap:12px; margin: 1.6rem 0 .45rem 0; }
.blk:first-child { margin-top: .3rem; }
.blk-n { font-size:.78rem; font-weight:800; color:#4F46E5; background:#EEF0FF; border-radius:8px; padding:3px 8px;
  letter-spacing:.04em; }
.blk-t { font-size:1.12rem; font-weight:800; color:#0F1B3D; letter-spacing:-.01em; }
.blk-s { font-size:.86rem; color:#64708A; margin-left:auto; text-align:right; }
/* key metrics table (quantstats style) */
.mt { width:100%; border-collapse:collapse; font-size:.86rem; margin:0; }
.mt, .mt tr, .mt td, .mt th { border:none !important; background:none !important; }
.mt th { text-align:right; font-size:.75rem; font-weight:700; color:#64708A; padding:0 0 6px 0;
  border-bottom:2px solid #0F1B3D !important; }
.mt th:first-child { text-align:left; }
.mt td { padding:3px 0; color:#0F1B3D; white-space:nowrap; }
.mt td.v { text-align:right; font-variant-numeric: tabular-nums; font-weight:600; padding-left:10px; }
.mt td.b { text-align:right; font-variant-numeric: tabular-nums; color:#64708A; padding-left:10px; }
.mt tr.gap td { padding:0; height:8px; border-bottom:1px solid #E2E7F0 !important; }
.mt tr.gap + tr td { padding-top:8px; }
.mt .neg { color:#B91C1C; } .mt .pos { color:#15803D; }
.mt-title { font-size:1.12rem; font-weight:800; color:#0F1B3D; margin:.2rem 0 .2rem 0; }
.mt-sub { font-size:.82rem; color:#64708A; margin-bottom:12px; }
/* keep the metrics in view while the charts scroll */
[data-testid="stColumn"]:has(.st-key-tear_metrics) { align-self: stretch; }
[data-testid="stLayoutWrapper"]:has(> .st-key-tear_metrics) { position: sticky; top: 1rem; z-index: 2; }
.st-key-tear_metrics { max-height: calc(100vh - 2rem); overflow-y: auto; scrollbar-width: thin; }
</style>"""


def _block(n, title, sub=""):
    s = f'<div class="blk-s">{sub}</div>' if sub else ""
    st.markdown(f'<div class="blk"><span class="blk-n">{n:02d}</span><span class="blk-t">{title}</span>{s}</div>',
                unsafe_allow_html=True)


@st.cache_data(show_spinner="Calculating tearsheet...", max_entries=64)
def _cached_tearsheet(asset_dfs, w_norm, amount, from_date, to_date, mode, freq_m, look_m, bench):
    values = _cached_values(asset_dfs, w_norm, amount, from_date, to_date, mode, freq_m, look_m)
    if values is None or len(values.dropna()) < 3:
        return None
    values = values.dropna()
    b = benchmark_close(asset_dfs, bench)
    b = b.reindex(b.index.union(values.index)).ffill().reindex(values.index).dropna()
    return values, b, ts_metrics(values), ts_metrics(b), monthly_table(values)


def _fig_cumulative(values, bench, b_name, log=False):
    p = (values / values.iloc[0] - 1) * 100
    q = (bench / bench.iloc[0] - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q.index, y=(q / 100 + 1) if log else q.round(2), mode='lines', name=b_name,
                             line=dict(color=FAINT, width=1.4),
                             hovertemplate=f"{b_name}: %{{customdata:+.1f}}%<extra></extra>", customdata=q))
    fig.add_trace(go.Scatter(x=p.index, y=(p / 100 + 1) if log else p.round(2), mode='lines', name="Portfolio",
                             line=dict(color=BLUE, width=2),
                             hovertemplate="Portfolio: %{customdata:+.1f}%<extra></extra>", customdata=p))
    if log:
        lo, hi = float(min(p.min(), q.min())), float(max(p.max(), q.max()))
        ticks = [t for t in (-90, -75, -50, -25, 0, 25, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000)
                 if lo * 1.1 - 5 <= t <= hi * 1.1 + 5]
        fig.add_hline(y=1, line_color=INK, line_dash="dash", line_width=1.1)
        yaxis = dict(type='log', tickvals=[t / 100 + 1 for t in ticks], ticktext=[f"{t:,}%" for t in ticks])
    else:
        fig.add_hline(y=0, line_color=INK, line_dash="dash", line_width=1.1)
        yaxis = dict(ticksuffix="%")
    fig.update_layout(**PLOT_LAYOUT, height=CHART_H, hovermode='x unified', yaxis=yaxis,
                      margin=dict(l=12, r=12, t=30, b=12),
                      legend=dict(orientation='h', yanchor='bottom', y=1.0, x=0))
    return fig


def _fig_monthly(grid):
    z = grid.to_numpy(dtype=float)
    text = [["" if np.isnan(v) else f"{v:+.1f}" for v in row] for row in z]
    lim = float(np.nanpercentile(np.abs(z[:, :12]), 95)) if np.isfinite(z[:, :12]).any() else 1.0
    fig = go.Figure(go.Heatmap(
        z=z, x=list(grid.columns), y=[str(y) for y in grid.index], text=text, texttemplate="%{text}",
        textfont=dict(size=10), zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2, showscale=False,
        colorscale=[[0, BAD], [0.5, "#FFFFFF"], [1, GOOD]],
        hovertemplate="%{y} %{x}: <b>%{z:+.2f}%</b><extra></extra>"))
    fig.update_layout(**PLOT_LAYOUT, height=max(260, 26 * len(grid) + 70), margin=dict(l=12, r=12, t=10, b=12),
                      yaxis=dict(autorange='reversed', type='category', showgrid=False),
                      xaxis=dict(side='top', showgrid=False))
    return fig


def _fig_timeline(table, years):
    xs = table['Invest Date'].dt.strftime('%Y-%m-%d')
    r = table['Return (%)'].round(2)
    custom = np.stack([table['Invest Date'].dt.strftime('%d-%m-%Y'), table['Compare Date'].dt.strftime('%d-%m-%Y'),
                       table['Invest Value (₹)'].round(0), table['Compare Value (₹)'].round(0)], axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=r.clip(lower=0), mode='lines', line=dict(width=0), fill='tozeroy',
                             fillcolor=hex_rgba(GOOD, 0.22), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=r.clip(upper=0), mode='lines', line=dict(width=0), fill='tozeroy',
                             fillcolor=hex_rgba(BAD, 0.25), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=r, mode='lines', name=f"{_yrs(years)} return",
                             line=dict(color=PORTFOLIO, width=1.4), customdata=custom,
                             hovertemplate="Invest %{customdata[0]}: ₹%{customdata[2]:,.0f}<br>"
                                           "Value on %{customdata[1]}: ₹%{customdata[3]:,.0f}<br>"
                                           "<b>%{y:+.2f}%</b><extra></extra>"))
    if len(table) > 30:
        fig.add_trace(go.Scatter(x=xs, y=r.rolling(30, min_periods=10).mean().round(2), mode='lines',
                                 name="30-day average", line=dict(color=VIOLET, width=2, dash='dot'),
                                 hovertemplate="avg %{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=0, line_color=INK, line_dash="dash", line_width=1.1)
    fig.update_layout(**PLOT_LAYOUT, height=CHART_H, hovermode='x unified', yaxis=dict(ticksuffix="%"),
                      margin=dict(l=12, r=12, t=30, b=12), legend=dict(orientation='h', yanchor='bottom', y=1.0, x=0))
    return fig


def _fig_alpha(t, b_name, unit):
    a = t['Alpha (%)'].round(2)
    custom = np.stack([t['Period'], t['Portfolio (%)'].round(2), t['Benchmark (%)'].round(2)], axis=1)
    fig = go.Figure(go.Bar(
        x=t['To'].dt.strftime('%Y-%m-%d'), y=a, marker_color=[_sign_color(v) for v in a], marker_line_width=0,
        customdata=custom, name="Alpha",
        hovertemplate="<b>%{customdata[0]}</b><br>alpha %{y:+.2f}%<br>portfolio %{customdata[1]:+.2f}% · "
                      + b_name + " %{customdata[2]:+.2f}%<extra></extra>"))
    fig.add_hline(y=0, line_color=INK, line_dash="dash", line_width=1.1)
    fig.add_hline(y=float(a.mean()), line_color=VIOLET, line_dash="dot", line_width=1.5)
    fig.update_layout(**PLOT_LAYOUT, height=CHART_H, showlegend=False, bargap=0.15,
                      yaxis=dict(ticksuffix="%"), margin=dict(l=12, r=12, t=30, b=12))
    return fig


def _metrics_table(pm, bm, b_name, strat):
    def cell(m, key, kind, cls):
        v = m.get(key)
        txt = ts_fmt(v, kind)
        tone = ""
        if kind == 'pct' and isinstance(v, (int, float)) and v == v and key not in ('time_in_market', 'psr', 'vol'):
            tone = " neg" if v < 0 else ""
        return f'<td class="{cls}{tone}">{txt}</td>'

    rows = []
    for g, group in enumerate(TS_LAYOUT):
        if g:
            rows.append('<tr class="gap"><td colspan="3"></td></tr>')
        for label, key, kind in group:
            rows.append(f"<tr><td>{label}</td>{cell(pm, key, kind, 'v')}{cell(bm, key, kind, 'b')}</tr>")
    st.markdown(
        f'<div class="mt-title">Key performance metrics</div>'
        f'<div class="mt-sub">{strat["desc"]} · daily values, 252 trading days a year, risk-free rate 0%</div>'
        f'<table class="mt"><thead><tr><th>Metric</th><th>Portfolio</th><th>{b_name}</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>', unsafe_allow_html=True)


def _tearsheet(asset_dfs, settings, strat, table, stats, years):
    st.markdown(_TEAR_CSS, unsafe_allow_html=True)
    b_name = ASSET_SHORT[strat['bench']]
    amount, from_date = settings[1], settings[2]
    ts = _cached_tearsheet(asset_dfs, *settings)
    ddr = _cached_drawdown(asset_dfs, *settings)

    section("Tearsheet", f"One investment of {inr(amount)} on {from_date:%d-%m-%Y}, held with "
                         f"{strat['short']} · compared with {b_name}", "📊", BLUE)
    if ts is None or ddr is None:
        note("Not enough price data for these dates. Pick a longer period in the sidebar.", AMBER, "⚠️")
        return
    values, bench, pm, bm, grid = ts
    dd, periods, bench_dd, _ = ddr
    s_dd = drawdown_summary(dd, periods)

    left, right = st.columns([2.15, 1], gap="large", vertical_alignment="top")

    with right, st.container(border=True, key="tear_metrics"):
        _metrics_table(pm, bm, b_name, strat)

    with left:
        n = 0

        n += 1
        _block(n, "Cumulative returns", f"portfolio vs {b_name}")
        with panel():
            _chart(_fig_cumulative(values, bench, b_name), key="ts_cum")

        n += 1
        _block(n, "Cumulative returns (log scaled)", "equal % moves look equal")
        with panel():
            _chart(_fig_cumulative(values, bench, b_name, log=True), key="ts_cum_log")

        n += 1
        _block(n, "Underwater plot", f"max {s_dd['max']:.1f}% · average {s_dd['avg']:.1f}%")
        with panel():
            start_value("dd_show_bench", False)
            show_bench = st.toggle(f"Show {b_name} drawdown too", key="dd_show_bench")
            fig = _drawdown_figure(dd, bench_dd, b_name, s_dd['avg'], s_dd['worst'], show_bench, CHART_H)
            fig.update_layout(title=None, margin=dict(l=12, r=12, t=20, b=12))
            _chart(fig, key="dd_underwater")
        st.caption("Black dashed line = new high (0%). Red dashed line = average drawdown. "
                   "Red dot = deepest fall. The worst periods are listed under Details below.")

        n += 1
        _block(n, "Monthly returns (%)", "green = up month, red = down month")
        with panel():
            _chart(_fig_monthly(grid), key="ts_monthly")

        n += 1
        _block(n, f"{_yrs(years).capitalize()} return distribution", f"{stats['count']:,} start dates")
        with panel():
            fig = _bell_figure(table['Return (%)'], stats['mean'], stats['std'], PORTFOLIO, None, height=CHART_H)
            fig.update_layout(margin=dict(l=12, r=12, t=30, b=12))
            _chart(fig, key="yoy_bell")
        st.caption("Each bar = how many start dates gave that return. Green = gain, red = loss. "
                   "Dark line = a normal bell curve with the same average and spread.")

        n += 1
        _block(n, f"{_yrs(years).capitalize()} return for every start date", "rolling, same-date")
        with panel():
            _chart(_fig_timeline(table, years), key="yoy_line")

        n += 1
        _block(n, f"Alpha vs {b_name}", "portfolio return − benchmark return")
        with panel():
            start_value("al_freq", "Monthly", list(ALPHA_FREQS))
            freq = st.segmented_control("Alpha for every", list(ALPHA_FREQS), key="al_freq",
                                        label_visibility="collapsed") or "Monthly"
            t = _cached_alpha(asset_dfs, *settings, freq)
            s = alpha_summary(t)
            if s:
                unit = ALPHA_FREQS[freq][0]
                _chart(_fig_alpha(t, b_name, unit), key="alpha_period_line")
                st.caption(f"Average alpha {pct(s['avg'])} per {unit} (purple dotted line). "
                           f"Beat {b_name} in {s['beat']} of {s['periods']} {unit}s. "
                           f"Best {unit}: {pct(s['best']['Alpha (%)'])} ({s['best']['Period']}), "
                           f"worst: {pct(s['worst']['Alpha (%)'])} ({s['worst']['Period']}).")
            else:
                note("Not enough data for alpha in this period.", AMBER, "⚠️")

        n += 1
        _block(n, "Year by year: financial year", "April to March")
        with panel():
            fig = _year_figure(_cached_yearly(table, True), None)
            _chart(fig, key="yoy_yearly_fy")

        n += 1
        _block(n, "Year by year: calendar year", "January to December")
        with panel():
            fig = _year_figure(_cached_yearly(table, False), None)
            _chart(fig, key="yoy_yearly_cy")
        st.caption("Bar = average return for start dates in that year. ▲ best and ▼ worst start date that year.")


def _details(asset_dfs, settings, strat, table, years, w_norm, from_date, to_date, trading_only):
    b_name = ASSET_SHORT[strat['bench']]
    ddr = _cached_drawdown(asset_dfs, *settings)
    if ddr is not None:
        dd, periods, _, values = ddr
        st.markdown(f"**Worst drawdown periods** · {len(periods)} in total")
        if not periods.empty:
            top = periods.head(10)
            view = pd.DataFrame({
                'Rank': range(1, len(top) + 1), 'Fall (%)': top['Depth (%)'],
                'Peak': top['Peak'].dt.strftime('%d-%m-%Y'), 'Lowest point': top['Trough'].dt.strftime('%d-%m-%Y'),
                'Back to peak': top['Recovery'].dt.strftime('%d-%m-%Y').fillna("not yet"),
                'Days falling': top['Days to trough'],
                'Days recovering': top['Days to recover'].map(lambda v: "—" if pd.isna(v) else f"{int(v):,}"),
                'Total days': top['Days underwater'],
            })
            st.dataframe(style_change_cols(view.style.format({'Fall (%)': '{:.2f}%', 'Days falling': '{:,}',
                                                              'Total days': '{:,}'}), ['Fall (%)']),
                         width="stretch", hide_index=True)
            csv = pd.DataFrame({'Date': dd.index.strftime('%d-%m-%Y'),
                                'Portfolio value (₹)': values.reindex(dd.index).round(2),
                                'Drawdown (%)': dd.round(4)}).to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️  Download daily drawdown CSV", csv, file_name="portfolio_drawdown.csv",
                               mime="text/csv", key="dl_drawdown")

    st.markdown("**Alpha table**")
    tables = {f: _cached_alpha(asset_dfs, *settings, f) for f in ALPHA_FREQS}
    first = next((t for t in tables.values() if t is not None and not t.empty), None)
    if first is not None:
        _alpha_table(tables, b_name, first.attrs.get('start', first['From'].iloc[0]))

    st.markdown("**Year-over-year data**")
    _data_tab(table, years, asset_dfs, w_norm, from_date, to_date, trading_only)