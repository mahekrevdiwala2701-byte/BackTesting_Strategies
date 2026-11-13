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
from plotly.subplots import make_subplots
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
                       alpha_stats, rolling_alpha,
                       FREQUENCIES as ALPHA_FREQS)
from src.strategies import STRATEGIES, REBAL_FREQ, DIFF_REBAL_FREQ, LOOKBACK_OPTIONS
from src.drawdown import underwater, drawdown_periods, drawdown_summary
from src.tearsheet import metrics as ts_metrics, period_returns, PERIODS, LAYOUT as KPI_LAYOUT, fmt as ts_fmt

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
/* View switcher: one even row of buttons across the page, wraps on small screens */
.st-key-home_view_bar { margin: 1.4rem 0 .2rem 0; padding: 6px; border-radius: 14px;
  background: #FFFFFF; border: 1px solid #E2E7F0; box-shadow: 0 1px 2px rgba(15,27,61,.04); }
.st-key-home_view, .st-key-home_view_bar [data-testid="stButtonGroup"] { width: 100% !important; }
.st-key-home_view_bar [role="radiogroup"] { display: flex !important; flex-wrap: wrap; gap: 4px; width: 100% !important; }
.st-key-home_view_bar [role="radiogroup"] > button { flex: 1 1 120px !important; max-width: none !important;
  justify-content: center; font-weight: 600; border-radius: 10px !important; }
.st-key-home_view_bar [role="radiogroup"] > button * { max-width: none !important; overflow: visible !important;
  text-overflow: clip !important; white-space: nowrap !important; }
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
VIEWS = {                       # key -> (button label, section title, subtitle, icon, colour)
    "spread":   ("🔔 Return spread",  "Return spread",  "How often each return happened", "🔔", PINK),
    "timeline": ("📈 Return timeline", "Return timeline", "Return for every investment date", "📈", VIOLET),
    "drawdown": ("🌊 Drawdown",       "Drawdown (underwater plot)",
                 "How far below its previous high the portfolio was, every day", "🌊", BLUE),
    "alpha":    ("🎯 Alpha",          "Alpha", "How much the portfolio beat the benchmark each period", "🎯", AMBER),
    "yearly":   ("🗓️ Year by year",   "Year by year", "Average, best and worst return per investment year", "🗓️", AMBER),
    "kpi":      ("📊 Key performance", "Key performance", "Strategy metrics and returns per week, month, quarter, year",
                 "📊", INDIGO),
    "data":     ("📥 Data",           "Data & more", "Download everything, the full table, each asset", "📥", CYAN),
}
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

    # ---------------- One view at a time ----------------
    start_value("home_view", "spread", list(VIEWS))
    with st.container(key="home_view_bar"):
        view = st.segmented_control("Show", list(VIEWS), format_func=lambda k: VIEWS[k][0],
                                    key="home_view", label_visibility="collapsed",
                                    width="stretch") or "spread"
    _, title, sub, icon, color = VIEWS[view]
    section(title, sub, icon, color)

    settings = (w_norm, amount, from_date, to_date, strat['mode'], strat['freq_m'], strat['look_m'], strat['bench'])
    if view == "spread":
        _bell_tab(table, stats, years)
    elif view == "timeline":
        _line_tab(table, years)
    elif view == "drawdown":
        _drawdown_tab(asset_dfs, settings, strat)
    elif view == "alpha":
        _alpha_tab(asset_dfs, settings, strat)
    elif view == "yearly":
        _yearly_tab(table)
    elif view == "kpi":
        _kpi_tab(asset_dfs, settings, strat)
    else:
        _data_tab(table, years, asset_dfs, w_norm, from_date, to_date, trading_only)


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
        fig = _bell_figure(r, s['mean'], s['std'], PORTFOLIO,
                           f"Total portfolio: {_yrs(years)} same-date returns", height=CHART_H)
        _chart(fig, key="yoy_bell")
    with c2:
        _mini_stats([
            ("Positive", f"{s['pct_positive']:.1f}%", GOOD),
            ("Negative", f"{100 - s['pct_positive']:.1f}%", BAD),
            ("Mean", pct(s['mean']), PORTFOLIO),
            ("Median", pct(s['median']), PORTFOLIO),
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
GAP_DAYS = 10          # dates further apart than this are a hole in the data: the line is broken there


def _line_tab(table, years):
    c1, c2 = st.columns([1, 1], vertical_alignment="center")
    with c1:
        x_choice = st.radio("X-axis date", ["Investment date", "Compare date"], horizontal=True,
                            key="yoy_line_x", label_visibility="collapsed")
    with c2:
        start_value("yoy_line_smooth", True)
        smooth = st.toggle("Add 30-day average line", key="yoy_line_smooth")
    xcol = 'Invest Date' if x_choice == "Investment date" else 'Compare Date'

    # Break the line where dates are missing (holes in the source data), instead of drawing a straight
    # line across them: an empty row is put into every gap longer than GAP_DAYS.
    t = table.sort_values(xcol).reset_index(drop=True)
    gap = t[xcol].diff().dt.days
    holes = gap[gap > GAP_DAYS].index
    starts, ends = t[xcol].shift(1)[holes], t[xcol][holes]
    if len(holes):
        blanks = pd.DataFrame({xcol: starts.values + pd.Timedelta(days=1)}, index=holes - 0.5)
        t = pd.concat([t, blanks]).sort_index().reset_index(drop=True)
    # 30-day average over calendar days, never spanning a hole
    seg = t['Return (%)'].isna().cumsum()
    avg = (t.set_index(xcol).groupby(seg.values)['Return (%)']
           .transform(lambda s: s.rolling('30D', min_periods=10).mean()).to_numpy())

    xs = t[xcol].dt.strftime('%Y-%m-%d')                   # short date text = much smaller page than full timestamps
    r = t['Return (%)'].round(2)                          # 2 decimals is all the chart shows (smaller page)
    custom = np.stack([t['Invest Date'].dt.strftime('%d-%m-%Y').fillna(""),
                       t['Compare Date'].dt.strftime('%d-%m-%Y').fillna(""),
                       t['Invest Value (₹)'].round(0), t['Compare Value (₹)'].round(0)], axis=1)
    fig = go.Figure()
    # green / red shading, drawn separately for each stretch of data so it never bridges a hole
    parts = [g for _, g in r.groupby(seg.values) if g.notna().sum() > 1]
    if len(parts) > 40:                                   # very patchy data: one shape is lighter for the page
        parts = [r]
    for g in parts:
        g = g.dropna()
        for vals, color in ((g.clip(lower=0), hex_rgba(GOOD, 0.22)), (g.clip(upper=0), hex_rgba(BAD, 0.25))):
            fig.add_trace(go.Scatter(x=xs[g.index], y=vals, mode='lines', line=dict(width=0), fill='tozeroy',
                                     fillcolor=color, hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(
        x=xs, y=r, mode='lines', name=f"Portfolio {_yrs(years)} return",
        line=dict(color=PORTFOLIO, width=1.6), customdata=custom,
        hovertemplate="Invest %{customdata[0]}: ₹%{customdata[2]:,.0f}<br>"
                      "Value on %{customdata[1]}: ₹%{customdata[3]:,.0f}<br>"
                      "<b>%{y:+.2f}%</b><extra></extra>"))
    lone = r.groupby(seg.values).transform(lambda g: g.notna().sum()) == 1     # a single date between two holes
    lone &= r.notna()
    if lone.any():
        fig.add_trace(go.Scatter(x=xs[lone], y=r[lone], mode='markers', marker=dict(size=6, color=PORTFOLIO),
                                 customdata=custom[lone.to_numpy()], showlegend=False,
                                 hovertemplate="Invest %{customdata[0]} (isolated date)<br>"
                                               "<b>%{y:+.2f}%</b><extra></extra>"))
    if smooth and len(t) > 30:
        fig.add_trace(go.Scatter(x=xs, y=np.round(avg, 2), mode='lines',
                                 name="30-day average", line=dict(color=INK, width=2, dash='dot'),
                                 hovertemplate="avg %{y:+.2f}%<extra></extra>"))
    bands = []                                             # grey band = no data; nearby holes become one band
    for a_, b_ in zip(starts, ends):
        if (b_ - a_).days > 30:
            if bands and (a_ - bands[-1][1]).days < 120:
                bands[-1][1] = b_
            else:
                bands.append([a_, b_])
    for a_, b_ in bands:
        fig.add_vrect(x0=a_.strftime('%Y-%m-%d'), x1=b_.strftime('%Y-%m-%d'), fillcolor=hex_rgba(FAINT, 0.18),
                      line_width=0, layer="below", annotation_text="missing data",
                      annotation_position="top", annotation_font=dict(color=MUTED, size=10))
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
    fy = _cached_yearly(table, True)                      # April to March
    cy = _cached_yearly(table, False)                     # January to December
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


def _drawdown_tab(asset_dfs, settings, strat):
    w_norm, amount, from_date, to_date = settings[:4]
    res = _cached_drawdown(asset_dfs, *settings)
    if res is None:
        note("Not enough price data for these dates. Pick a longer period in the sidebar.", AMBER, "⚠️")
        return
    dd, periods, bench_dd, values = res
    s = drawdown_summary(dd, periods)
    b_name = ASSET_SHORT[strat['bench']]

    explain(f"One investment of {inr(amount)} on {from_date:%d-%m-%Y}, held with {strat['desc']}. "
            f"0% = the portfolio is at a new high. Anything below 0% = how much it has fallen from its best "
            f"value so far. Deeper and wider blue areas = bigger, longer falls.", BLUE)

    worst, longest = s['worst'], s['longest']
    w_rec = (f"recovered {worst['Recovery']:%d-%m-%Y}" if worst is not None and worst['Recovered']
             else "not recovered yet")
    kpi_row([
        dict(label="Max drawdown", value=pct(s['max']),
             delta=(f"{worst['Peak']:%d-%m-%y} → {worst['Trough']:%d-%m-%y}" if worst is not None else "none"),
             tone="bad" if s['max'] < 0 else "neutral", icon="🧯", accent=BAD),
        dict(label="Current drawdown", value=pct(s['current']),
             delta="at a new high" if s['current'] > -1e-9 else f"last high {s['last_high']:%d-%m-%Y}",
             tone="good" if s['current'] > -1e-9 else "bad", icon="📍", accent=BLUE),
        dict(label="Average drawdown", value=pct(s['avg']),
             delta=f"{s['pct_under']:.0f}% of days below a high", icon="〰️", accent=VIOLET),
        dict(label="Longest underwater",
             value=f"{longest['Days underwater']:,} days" if longest is not None else "0 days",
             delta=(f"{longest['Peak']:%d-%m-%y} → "
                    + (f"{longest['Recovery']:%d-%m-%y}" if longest['Recovered'] else "still open"))
             if longest is not None else "never below a high", icon="⏱️", accent=AMBER),
    ])

    start_value("dd_show_bench", False)
    with panel():
        show_bench = st.toggle(f"Show {b_name} drawdown for comparison", key="dd_show_bench")
        _chart(_drawdown_figure(dd, bench_dd, b_name, s['avg'], worst, show_bench, CHART_H + 20), key="dd_underwater")
    st.caption(f"Black dashed line = 0% (new high). Red dashed line = average drawdown on days below a high "
               f"({s['avg']:.1f}%). Red dot = deepest point ({w_rec}). Drag on the chart to zoom, double-click to reset.")

    if periods.empty:
        return
    st.markdown(f"**Worst drawdown periods** · {s['count']} in total")
    top = periods.head(10).copy()
    view = pd.DataFrame({
        'Rank': range(1, len(top) + 1),
        'Fall (%)': top['Depth (%)'],
        'Peak': top['Peak'].dt.strftime('%d-%m-%Y'),
        'Lowest point': top['Trough'].dt.strftime('%d-%m-%Y'),
        'Back to peak': top['Recovery'].dt.strftime('%d-%m-%Y').fillna("not yet"),
        'Days falling': top['Days to trough'],
        'Days recovering': top['Days to recover'].map(lambda v: "—" if pd.isna(v) else f"{int(v):,}"),
        'Total days': top['Days underwater'],
    })
    fmt = {'Fall (%)': '{:.2f}%', 'Days falling': '{:,}', 'Total days': '{:,}'}
    st.dataframe(style_change_cols(view.style.format(fmt), ['Fall (%)']), width="stretch", hide_index=True)
    csv = pd.DataFrame({'Date': dd.index.strftime('%d-%m-%Y'), 'Portfolio value (₹)': values.reindex(dd.index).round(2),
                        'Drawdown (%)': dd.round(4)}).to_csv(index=False).encode('utf-8-sig')
    st.download_button("⬇️  Download daily drawdown CSV", csv, file_name="portfolio_drawdown.csv",
                       mime="text/csv", key="dl_drawdown")


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


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_growth(asset_dfs, w_norm, amount, from_date, to_date, mode, freq_m, look_m, bench):
    """Portfolio and benchmark every day, both starting at 100."""
    values = _cached_values(asset_dfs, w_norm, amount, from_date, to_date, mode, freq_m, look_m)
    if values is None or len(values.dropna()) < 2:
        return None
    v = values.dropna()
    v = v[~v.index.duplicated(keep='last')].sort_index()
    b = benchmark_close(asset_dfs, bench)
    b = b.reindex(b.index.union(v.index)).ffill().reindex(v.index)
    df = pd.DataFrame({'P': v, 'B': b}).dropna()
    return (df / df.iloc[0] * 100).round(3)


def _fig_growth(g, b_name):
    """Top: ₹100 in the portfolio vs ₹100 in the benchmark. Bottom: the gap between them."""
    gap = (g['P'] - g['B']).round(2)
    x = g.index.strftime('%Y-%m-%d')
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=x, y=g['B'], name=b_name, mode='lines', line=dict(color=FAINT, width=1.6),
                             hovertemplate=f"{b_name}: ₹%{{y:,.1f}}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=g['P'], name="Portfolio", mode='lines', line=dict(color=PORTFOLIO, width=2.2),
                             hovertemplate="Portfolio: ₹%{y:,.1f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=100, line_color=FAINT, line_dash="dot", row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=gap.clip(lower=0), mode='lines', line=dict(width=0), fill='tozeroy',
                             fillcolor=hex_rgba(GOOD, 0.35), hoverinfo='skip', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=gap.clip(upper=0), mode='lines', line=dict(width=0), fill='tozeroy',
                             fillcolor=hex_rgba(BAD, 0.35), hoverinfo='skip', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=gap, mode='lines', line=dict(color=INK, width=1), name="Gap",
                             showlegend=False, hovertemplate="Ahead by ₹%{y:+,.1f} (per ₹100)<extra></extra>"),
                  row=2, col=1)
    fig.add_hline(y=0, line_color=INK, line_dash="dash", line_width=1, row=2, col=1)
    fig.update_layout(**PLOT_LAYOUT, height=CHART_H + 120, hovermode='x unified',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                      margin=dict(l=12, r=12, t=30, b=12))
    fig.update_yaxes(title_text="Value of ₹100", tickprefix="₹", row=1, col=1)
    fig.update_yaxes(title_text="Gap (₹)", row=2, col=1)
    return fig


def _fig_alpha_bars(t, b_name, unit):
    a = t['Alpha (%)'].round(2)
    custom = np.stack([t['Period'], t['Portfolio (%)'].round(2), t['Benchmark (%)'].round(2)], axis=1)
    fig = go.Figure(go.Bar(
        x=t['To'].dt.strftime('%Y-%m-%d'), y=a, marker_color=[_sign_color(v) for v in a], marker_line_width=0,
        customdata=custom, name="Alpha",
        hovertemplate="<b>%{customdata[0]}</b><br>alpha <b>%{y:+.2f}%</b><br>portfolio %{customdata[1]:+.2f}% · "
                      + b_name + " %{customdata[2]:+.2f}%<extra></extra>"))
    fig.add_hline(y=0, line_color=INK, line_width=1)
    fig.add_hline(y=float(a.mean()), line_color=VIOLET, line_dash="dash", line_width=1.6,
                  annotation_text=f"average {a.mean():+.2f}% per {unit}", annotation_position="top left",
                  annotation_font=dict(color=VIOLET, size=11), annotation_bgcolor="rgba(255,255,255,.85)")
    fig.update_layout(**PLOT_LAYOUT, height=CHART_H, showlegend=False, bargap=0.12, hovermode='closest',
                      yaxis=dict(ticksuffix="%", title="Alpha"), margin=dict(l=12, r=12, t=20, b=12))
    return fig


def _fig_rolling(r, b_name):
    x = r.index.strftime('%Y-%m-%d')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=r.clip(lower=0).round(2), mode='lines', line=dict(width=0), fill='tozeroy',
                             fillcolor=hex_rgba(GOOD, 0.28), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=r.clip(upper=0).round(2), mode='lines', line=dict(width=0), fill='tozeroy',
                             fillcolor=hex_rgba(BAD, 0.28), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=r.round(2), mode='lines', line=dict(color=INK, width=1.8), showlegend=False,
                             hovertemplate="12 months to %{x|%b %Y}: <b>%{y:+.2f}%</b> vs " + b_name
                                           + "<extra></extra>"))
    fig.add_hline(y=0, line_color=INK, line_dash="dash", line_width=1)
    fig.update_layout(**PLOT_LAYOUT, height=CHART_H - 40, hovermode='x unified',
                      yaxis=dict(ticksuffix="%", title="1-year alpha"), margin=dict(l=12, r=12, t=20, b=12))
    return fig


def _fig_alpha_heatmap(monthly):
    t = monthly[~monthly['Partial']] if (~monthly['Partial']).any() else monthly
    grid = (pd.DataFrame({'Y': t['To'].dt.year, 'M': t['To'].dt.month, 'A': t['Alpha (%)']})
            .pivot_table(index='Y', columns='M', values='A', aggfunc='sum').reindex(columns=range(1, 13)))
    yearly = t.groupby(t['To'].dt.year).apply(
        lambda d: ((1 + d['Portfolio (%)'] / 100).prod() - (1 + d['Benchmark (%)'] / 100).prod()) * 100)
    grid[13] = yearly
    z = grid.to_numpy(dtype=float)
    lim = float(np.nanpercentile(np.abs(z[:, :12]), 95)) if np.isfinite(z[:, :12]).any() else 1.0
    cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Year']
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=[str(y) for y in grid.index], zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2, showscale=False,
        text=[["" if np.isnan(v) else f"{v:+.1f}" for v in row] for row in z], texttemplate="%{text}",
        textfont=dict(size=10), colorscale=[[0, BAD], [0.5, "#FFFFFF"], [1, GOOD]],
        hovertemplate="%{y} %{x}: alpha <b>%{z:+.2f}%</b><extra></extra>"))
    fig.update_layout(**PLOT_LAYOUT, height=max(CHART_H, 26 * len(grid) + 60), margin=dict(l=12, r=12, t=10, b=12),
                      yaxis=dict(autorange='reversed', type='category', showgrid=False),
                      xaxis=dict(side='top', showgrid=False))
    return fig


def _fig_scatter(t, st_, b_name, unit, height):
    p, b = t['Portfolio (%)'], t['Benchmark (%)']
    lo, hi = float(min(p.min(), b.min())), float(max(p.max(), b.max()))
    pad = (hi - lo) * 0.06
    xs = np.array([lo - pad, hi + pad])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=xs, mode='lines', line=dict(color=FAINT, dash='dot'), name="Same as benchmark",
                             hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=xs, y=st_['alpha_period'] + st_['beta'] * xs, mode='lines',
                             line=dict(color=VIOLET, width=2), name=f"Fit: β = {st_['beta']:.2f}", hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=b.round(2), y=p.round(2), mode='markers', name=f"Each {unit}", customdata=t['Period'],
        marker=dict(size=6 if len(t) > 80 else 8, color=[_sign_color(v) for v in t['Alpha (%)']], opacity=0.75,
                    line=dict(color="#FFFFFF", width=0.6)),
        hovertemplate="<b>%{customdata}</b><br>" + b_name + " %{x:+.2f}% → portfolio %{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=0, line_color=FAINT, line_width=1)
    fig.add_vline(x=0, line_color=FAINT, line_width=1)
    fig.update_layout(**PLOT_LAYOUT, height=height, hovermode='closest',
                      xaxis=dict(title=f"{b_name} return", ticksuffix="%", range=list(xs)),
                      yaxis=dict(title="Portfolio return", ticksuffix="%"),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                      margin=dict(l=12, r=12, t=30, b=12))
    return fig


def _alpha_tab(asset_dfs, settings, strat):
    w_norm = settings[0]
    bench = strat['bench']
    b_name = ASSET_SHORT[bench]
    explain(f"Alpha = how much better (or worse) the portfolio did than {b_name}. Green = beat it, red = lagged. "
            f"Strategy: {strat['desc']}. Strategy and benchmark are set in the sidebar.", AMBER)

    tables = {f: _cached_alpha(asset_dfs, *settings, f) for f in ALPHA_FREQS}     # every frequency, cached
    monthly = tables.get("Monthly")
    growth = _cached_growth(asset_dfs, *settings)
    if growth is None or monthly is None or monthly.empty:
        note("Not enough price data for these dates. Pick a longer period in the sidebar.", AMBER, "⚠️")
        return

    # ---------- 1. headline: proper alpha numbers (monthly returns, the standard) ----------
    ms = alpha_stats(monthly, "Monthly")
    years_ = max((growth.index[-1] - growth.index[0]).days / 365.25, 0.01)
    p_cagr = ((growth['P'].iloc[-1] / 100) ** (1 / years_) - 1) * 100
    b_cagr = ((growth['B'].iloc[-1] / 100) ** (1 / years_) - 1) * 100
    beat = int((monthly['Alpha (%)'] > 0).sum())
    kpi_row([
        dict(label="Extra return / yr", value=f"{p_cagr - b_cagr:+.2f}%",
             delta=f"CAGR {p_cagr:.1f}% vs {b_cagr:.1f}%", tone=tone_of(p_cagr - b_cagr), icon="🎯",
             accent=GOOD if p_cagr >= b_cagr else BAD),
        dict(label="Jensen's alpha", value=pct(ms.get('alpha_ann', np.nan)),
             delta="beta-adjusted", tone=tone_of(ms.get('alpha_ann')), icon="α", accent=VIOLET),
        dict(label="Beta", value=f"{ms.get('beta', np.nan):.2f}",
             delta=f"R² {ms.get('r2', np.nan):.2f}", icon="📐", accent=BLUE),
        dict(label="Info ratio", value=f"{ms.get('info_ratio', np.nan):.2f}",
             delta=f"tracking err {ms.get('tracking_error', np.nan):.1f}%", tone=tone_of(ms.get('info_ratio')),
             icon="⚖️", accent=CYAN),
        dict(label="Months won", value=f"{beat / len(monthly) * 100:.0f}%",
             delta=f"{beat} of {len(monthly)} months", tone=tone_of(beat - len(monthly) / 2), icon="🏁",
             accent=AMBER),
    ])
    st.caption("Jensen's alpha = yearly return the portfolio earned beyond what its beta (how strongly it follows "
               f"{b_name}) explains. Beta 1 = moves like {b_name}; below 1 = calmer. Information ratio = extra "
               "return per unit of extra risk (above 0.5 is good). Measured on monthly returns.")

    # ---------- 2. growth of ₹100 + the gap ----------
    section("Growth of ₹100", f"Portfolio vs {b_name}, and how far ahead (green) or behind (red) it is", "📈", INDIGO)
    with panel():
        _chart(_fig_growth(growth, b_name), key="alpha_growth")

    # ---------- 3. alpha each period ----------
    section("Alpha in each period", "Bars above 0 = the portfolio beat the benchmark in that period", "📊", AMBER)
    with panel():
        start_value("al_freq", "Monthly", list(ALPHA_FREQS))
        freq = st.segmented_control("Alpha for every", list(ALPHA_FREQS), key="al_freq",
                                    label_visibility="collapsed") or "Monthly"
        t = tables[freq]
        s = alpha_summary(t)
        if not s:
            note("Not enough data for this frequency.", AMBER, "⚠️")
        else:
            unit = ALPHA_FREQS[freq][0]
            if freq == "Weekly" and ('SmallCap' in w_norm or bench == 'SmallCap'):
                note("SmallCap prices may be <b>monthly</b>: weekly alpha then jumps once a month. "
                     "Monthly or longer gives a cleaner picture.", AMBER, "💡")
            _chart(_fig_alpha_bars(t, b_name, unit), key=f"alpha_bars_{freq}")
            _mini_stats([
                (f"{unit.capitalize()}s beating {b_name}", f"{s['beat']} of {s['periods']} "
                                                            f"({s['beat'] / s['periods'] * 100:.0f}%)", GOOD),
                (f"Average alpha per {unit}", f"{pct(s['avg'])}  (median {pct(s['median'])})", PORTFOLIO),
                (f"Best {unit}", f"{pct(s['best']['Alpha (%)'])}  ({s['best']['Period']})", GOOD),
                (f"Worst {unit}", f"{pct(s['worst']['Alpha (%)'])}  ({s['worst']['Period']})", BAD),
            ])

    # ---------- 4. rolling 1-year alpha ----------
    roll = rolling_alpha(monthly, 12)
    if not roll.empty:
        section("Rolling 1-year alpha", f"Portfolio return over the last 12 months − {b_name} return over the same "
                                        f"12 months", "🔁", VIOLET)
        with panel():
            _chart(_fig_rolling(roll, b_name), key="alpha_rolling")
        st.caption(f"Positive {(roll > 0).mean() * 100:.0f}% of the time. Latest: {pct(roll.iloc[-1])} "
                   f"(12 months to {roll.index[-1]:%b %Y}). Best {pct(roll.max())}, worst {pct(roll.min())}.")

    # ---------- 5. month-by-month heatmap + scatter ----------
    section("Where the alpha came from", "Alpha every month (left) and every month's returns side by side (right)",
            "🧭", PINK)
    up_c, down_c = ms.get('up_capture', np.nan), ms.get('down_capture', np.nan)
    c1, c2 = st.columns([1.45, 1], gap="medium")
    with c1, panel():
        fig = _fig_alpha_heatmap(monthly)
        _chart(fig, key="alpha_heatmap")
        h = fig.layout.height
    with c2, panel():
        _chart(_fig_scatter(monthly, ms, b_name, "month", h - 150), key="alpha_scatter")
        _mini_stats([
            ("Up-market capture (higher = better)", f"{up_c:.0f}%", GOOD if up_c >= 100 else INK),
            ("Down-market capture (lower = better)", f"{down_c:.0f}%", GOOD if down_c < 100 else BAD),
        ])
    st.caption(f"Heatmap: green month = beat {b_name}, red = lagged; 'Year' = full-year difference. "
               f"Scatter: each dot is a month. Dots above the dotted line beat {b_name}. Up capture 90% = the "
               f"portfolio gained 90% as much as {b_name} in its up months; down capture below 100% = it fell less "
               f"in down months.")

    # ---------- 6. full table ----------
    section("Alpha table", "Every year, half-year, quarter, month and week", "📋", CYAN)
    first = next((x for x in tables.values() if x is not None and not x.empty), None)
    _alpha_table(tables, b_name, first.attrs.get('start', first['From'].iloc[0]))


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
# Key performance: quantstats-style metrics + strategy returns per period
# =========================================================
_KPI_CSS = """<style>
.qs-title { font-size:1.05rem; font-weight:800; color:#0F1B3D; margin:.1rem 0 .7rem 0; }
.qs { width:100%; border-collapse:collapse; font-size:.9rem; margin:0; }
.qs, .qs tr, .qs td, .qs th { border:none !important; }
.qs thead th { background:#EEF1F7 !important; color:#0F1B3D; font-weight:800; padding:8px 6px; text-align:right; }
.qs thead th:first-child { text-align:left; }
.qs td { padding:4px 6px; color:#0F1B3D; white-space:nowrap; background:none !important; }
.qs td.v { text-align:right; font-variant-numeric:tabular-nums; font-weight:600; }
.qs td.b { text-align:right; font-variant-numeric:tabular-nums; color:#64708A; }
.qs td.neg { color:#B91C1C; }
.qs tr.gap td { padding:0; height:9px; border-bottom:1px solid #D5DBE7 !important; }
.qs tr.gap + tr td { padding-top:9px; }
</style>"""

_KPI_NO_TONE = {'rf', 'time_in_market', 'psr', 'vol', 'longest_dd', 'skew', 'kurt', 'start', 'end'}


@st.cache_data(show_spinner="Calculating key performance...", max_entries=64)
def _cached_kpi(asset_dfs, w_norm, amount, from_date, to_date, mode, freq_m, look_m, bench):
    values = _cached_values(asset_dfs, w_norm, amount, from_date, to_date, mode, freq_m, look_m)
    if values is None or len(values.dropna()) < 3:
        return None
    values = values.dropna()
    values = values[~values.index.duplicated(keep='last')].sort_index()
    b = benchmark_close(asset_dfs, bench)
    b = b.reindex(b.index.union(values.index)).ffill().reindex(values.index).dropna()
    rets = {f: (period_returns(values, rule) * 100, period_returns(b, rule) * 100) for f, rule in PERIODS.items()}
    return ts_metrics(values), ts_metrics(b), rets


def _kpi_table(pm, bm, b_name):
    def td(m, key, kind, cls):
        v = m.get(key)
        neg = (" neg" if cls == "v" and key not in _KPI_NO_TONE and isinstance(v, (int, float))
               and v == v and v < 0 else "")
        return f'<td class="{cls}{neg}">{ts_fmt(v, kind)}</td>'

    rows = []
    for g, group in enumerate(KPI_LAYOUT):
        if g:
            rows.append('<tr class="gap"><td colspan="3"></td></tr>')
        rows += [f"<tr><td>{label}</td>{td(pm, k, kind, 'v')}{td(bm, k, kind, 'b')}</tr>" for label, k, kind in group]
    st.markdown(f'<div class="qs-title">Key Performance Metrics</div><table class="qs"><thead><tr><th>Metric</th>'
                f'<th>Strategy</th><th>{b_name}</th></tr></thead><tbody>{"".join(rows)}</tbody></table>',
                unsafe_allow_html=True)


_QUARTER = {1: "Q1 (Jan–Mar)", 2: "Q2 (Apr–Jun)", 3: "Q3 (Jul–Sep)", 4: "Q4 (Oct–Dec)"}
_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _heatmap(p, cols, col_of, height):
    """Year (rows) x month/quarter (cols) grid of returns in %."""
    t = pd.DataFrame({'Y': p.index.year, 'C': [col_of(d) for d in p.index], 'R': p.values})
    grid = t.pivot(index='Y', columns='C', values='R').reindex(columns=cols)
    z = grid.to_numpy(dtype=float)
    lim = float(np.nanpercentile(np.abs(z), 95)) if np.isfinite(z).any() else 1.0
    fig = go.Figure(go.Heatmap(
        z=z, x=[c.split(" ")[0] for c in cols], y=[str(y) for y in grid.index],
        text=[["" if np.isnan(v) else f"{v:+.1f}" for v in row] for row in z], texttemplate="%{text}",
        textfont=dict(size=11), zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2, showscale=False,
        colorscale=[[0, BAD], [0.5, "#FFFFFF"], [1, GOOD]],
        hovertemplate="%{y} %{x}: <b>%{z:+.2f}%</b><extra></extra>"))
    fig.update_layout(**PLOT_LAYOUT, height=max(height, 28 * len(grid) + 60), margin=dict(l=12, r=12, t=10, b=12),
                      yaxis=dict(autorange='reversed', type='category', showgrid=False),
                      xaxis=dict(side='top', showgrid=False))
    return fig


def _period_bars(p, q, b_name, freq, height):
    lab = (p.index.strftime('%d-%b-%y') if freq == "Weekly" else p.index.strftime('%Y'))
    fig = go.Figure(go.Bar(
        x=p.index.strftime('%Y-%m-%d'), y=p.round(2), marker_color=[_sign_color(v) for v in p], marker_line_width=0,
        customdata=np.stack([lab, q.reindex(p.index).round(2)], axis=1),
        text=[f"{v:+.1f}%" for v in p] if freq == "Yearly" else None, textposition='outside',
        hovertemplate="<b>%{customdata[0]}</b><br>strategy %{y:+.2f}%<br>" + b_name
                      + " %{customdata[1]:+.2f}%<extra></extra>"))
    fig.add_hline(y=0, line_color=INK, line_dash="dash", line_width=1)
    fig.update_layout(**PLOT_LAYOUT, height=height, showlegend=False, bargap=0.15, yaxis=dict(ticksuffix="%"),
                      xaxis=dict(type='category' if freq == "Yearly" else 'date'),
                      margin=dict(l=12, r=12, t=20, b=12))
    return fig


def _kpi_tab(asset_dfs, settings, strat):
    st.markdown(_KPI_CSS, unsafe_allow_html=True)
    amount, from_date = settings[1], settings[2]
    b_name = ASSET_SHORT[strat['bench']]
    res = _cached_kpi(asset_dfs, *settings)
    if res is None:
        note("Not enough price data for these dates. Pick a longer period in the sidebar.", AMBER, "⚠️")
        return
    pm, bm, rets = res
    explain(f"One investment of {inr(amount)} on {from_date:%d-%m-%Y}, held with {strat['desc']}, "
            f"next to {b_name}. Daily values, 252 trading days a year. Strategy and benchmark are in the sidebar.",
            INDIGO)

    left, right = st.columns([1, 1.45], gap="medium")
    with left, panel():
        _kpi_table(pm, bm, b_name)

    with right, panel():
        st.markdown('<div class="qs-title">Strategy returns</div>', unsafe_allow_html=True)
        start_value("kpi_freq", "Monthly", list(PERIODS))
        freq = st.segmented_control("Returns for every", list(PERIODS), key="kpi_freq",
                                    label_visibility="collapsed") or "Monthly"
        p, q = rets[freq]
        p = p.dropna()
        unit = {"Weekly": "week", "Monthly": "month", "Quarterly": "quarter", "Yearly": "year"}[freq]
        if freq == "Monthly":
            fig = _heatmap(p, _MONTHS, lambda d: _MONTHS[d.month - 1], CHART_H + 120)
        elif freq == "Quarterly":
            fig = _heatmap(p, list(_QUARTER.values()), lambda d: _QUARTER[(d.month - 1) // 3 + 1], CHART_H + 120)
        else:
            fig = _period_bars(p, q, b_name, freq, CHART_H + 120)
        _chart(fig, key=f"kpi_ret_{freq}")
        beat = int((p > q.reindex(p.index)).sum())
        _mini_stats([
            (f"Positive {unit}s", f"{(p > 0).mean() * 100:.0f}%  ({int((p > 0).sum())} of {len(p)})", GOOD),
            (f"Average {unit}", pct(p.mean()), PORTFOLIO),
            (f"Best {unit}", f"{pct(p.max())}  ({_plabel(p.idxmax(), freq)})", GOOD),
            (f"Worst {unit}", f"{pct(p.min())}  ({_plabel(p.idxmin(), freq)})", BAD),
            (f"{unit.capitalize()}s beating {b_name}", f"{beat} of {len(p)}", MUTED),
        ])

        csv = pd.DataFrame({'Period end': p.index.strftime('%d-%m-%Y'), 'Strategy (%)': p.round(4),
                            f'{b_name} (%)': q.reindex(p.index).round(4)}).to_csv(index=False).encode('utf-8-sig')
        st.download_button(f"⬇️  Download {freq.lower()} returns CSV", csv, key="dl_kpi_rets",
                           file_name=f"strategy_{freq.lower()}_returns.csv", mime="text/csv")


def _plabel(d, freq):
    if freq == "Weekly":
        return f"week to {d:%d-%m-%Y}"
    if freq == "Monthly":
        return f"{d:%b %Y}"
    if freq == "Quarterly":
        return f"Q{(d.month - 1) // 3 + 1} {d:%Y}"
    return f"{d:%Y}"