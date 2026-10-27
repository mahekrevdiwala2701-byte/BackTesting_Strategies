"""💼 Everyday Portfolio: invest an amount split across GoldBEES / LargeCap / MidCap / SmallCap by your weights.

Year-on-year chained logic, per index (default start = 01-01-2015):
  * Year 1 (start → one year later): index value = actual Close price.
  * Every later day d: YoY % = price(d) / price(same day one year earlier) − 1,
    applied to the index value at the end of the previous year.
  * If that day last year was a holiday, the last trading day before it is used.

Each holding = (amount × weight) × index value(d) / index value(first day),
so all 4 lines are in ₹ of YOUR money and can be compared on one chart.
"""
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import (
    PLOT_LAYOUT, VIOLET, CYAN, EMERALD, AMBER, BLUE, BAD, FAINT,
    hero, section, kpi, kpi_row, tone_of, note, chart, inr, inr_short, pct as fmt_pct,
    sidebar_nav, sidebar_footer, sidebar_label, weight_badge,
)
from src.asset_data import ASSETS, ASSET_LABELS, ASSET_SHORT, ASSET_ICONS, COLORS, style_change_cols, blank_missing
from src.period_picker import today_ist

DEFAULT_START = date(2015, 1, 1)
PORTFOLIO_COLOR = "#FFFFFF"


# ---------------------------------------------------------
# Core calculation
# ---------------------------------------------------------
def yoy_chain(prices: pd.Series, start, end) -> pd.DataFrame:
    """prices: Close indexed by Date. Returns frame with Close, Prev Year Close, YoY (%), Index Value, Year No."""
    prices = prices[~prices.index.duplicated(keep='last')].sort_index().dropna()
    start_ts = max(pd.Timestamp(start), prices.index.min())
    end_ts = pd.Timestamp(end)

    p = prices[(prices.index >= start_ts) & (prices.index <= end_ts)]
    if p.empty:
        return pd.DataFrame()

    # Price on the same calendar day one year earlier (or last trading day before it)
    prev_dates = p.index - pd.DateOffset(years=1)
    pos = prices.index.searchsorted(prev_dates, side='right') - 1
    prev = pd.Series(np.where(pos >= 0, prices.values[np.clip(pos, 0, None)], np.nan), index=p.index)

    # Year number since start (0 = base year)
    year_idx = pd.Series(0, index=p.index)
    k = 1
    while start_ts + pd.DateOffset(years=k) <= p.index.max():
        year_idx[p.index >= start_ts + pd.DateOffset(years=k)] = k
        k += 1

    yoy = p / prev - 1
    value = pd.Series(np.nan, index=p.index)
    base = year_idx == 0
    value[base] = p[base]                       # year 1 = real prices
    anchor = p[base].iloc[-1]                   # value at end of previous year
    for y in range(1, int(year_idx.max()) + 1):
        mask = year_idx == y
        if not mask.any():
            continue
        value[mask] = (anchor * (1 + yoy[mask])).ffill().fillna(anchor)
        anchor = value[mask].iloc[-1]

    yoy[base] = np.nan
    return pd.DataFrame({'Close': p, 'Prev Year Close': prev.where(~base),
                         'YoY (%)': yoy * 100, 'Index Value': value, 'Year No.': year_idx})


def value_one_year_ago(s: pd.Series, d: pd.Timestamp):
    past = s[s.index <= d - pd.DateOffset(years=1)]
    return past.iloc[-1] if not past.empty else np.nan


def money(v):
    return "—" if pd.isna(v) else inr(v, 2)


def pct(v):
    return "—" if pd.isna(v) else f"{v:+.2f}%"


# ---------------------------------------------------------
# Page
# ---------------------------------------------------------
def render_everyday_portfolio_page(asset_dfs, fetch_errors):
    today = today_ist()

    # ---------------- Sidebar ----------------
    sidebar_nav("portfolio")
    sidebar_label("📅 Period")
    data_min = min(df['Date'].min() for df in asset_dfs.values()).date()
    min_start = min(data_min, DEFAULT_START)
    start = st.sidebar.date_input("📍 Start date (base year begins)", value=max(DEFAULT_START, min_start),
                                  min_value=min_start, max_value=today, format="DD-MM-YYYY", key="ep_start")
    end = st.sidebar.date_input("🏁 End date", value=today, min_value=min_start, max_value=today,
                                format="DD-MM-YYYY", key="ep_end")
    if start > end:
        st.sidebar.error("Start date must be before End date. Showing them swapped.")
        start, end = end, start

    sidebar_label("💰 Investment")
    amount = st.sidebar.number_input("Total investment (₹)", min_value=1000.0, value=100000.0,
                                     step=5000.0, format="%.0f", key="ep_amount")

    available = [a for a in ASSETS if a in asset_dfs]
    sidebar_label("⚖️ Weights (%)")
    default_w = round(100 / len(available), 2) if available else 0.0
    raw_w = {a: st.sidebar.number_input(f"{ASSET_ICONS[a]} {ASSET_SHORT[a]}", min_value=0.0, max_value=100.0,
                                        value=default_w, step=5.0, key=f"ep_w_{a}") for a in available}
    total_w = sum(raw_w.values())
    with st.sidebar:
        weight_badge(total_w)

    sidebar_label("📈 Chart")
    chart_mode = st.sidebar.radio("Show", ["₹ Value", "% Return"], horizontal=True, key="ep_mode")
    show_total = st.sidebar.checkbox("Show total portfolio line", value=True, key="ep_total")
    sidebar_footer(fetch_errors, key="ep")

    # ---------------- Hero ----------------
    hero("Everyday Portfolio",
         "Year 1 uses real prices. After that, each day's change versus the same day last year is applied "
         "to the previous year-end value, so you can follow your money day by day.",
         page="portfolio", eyebrow="Day-by-day tracker",
         chips=[f"📅 {start:%d %b %Y} → {end:%d %b %Y}", f"💰 {inr_short(amount)} invested",
                "⚙️ Set amount & weights in the sidebar"])

    if total_w <= 0:
        note("Give at least one index a weight above 0%.", BAD, "⛔")
        return
    if abs(total_w - 100) > 0.01:
        note(f"Weights add up to <b>{total_w:.1f}%</b>, so they are scaled to 100%.", AMBER, "⚖️")
    weights = {a: w / total_w for a, w in raw_w.items() if w > 0}

    # ---------------- Calculate ----------------
    results, notes = {}, []
    for a in weights:
        res = yoy_chain(asset_dfs[a].set_index('Date')['Close'], start, end)
        if res.empty:
            notes.append(f"<b>{ASSET_SHORT[a]}</b>: no data in this period, its share stays as cash.")
            continue
        if res.index[0].date() > start:
            notes.append(f"<b>{ASSET_SHORT[a]}</b>: data starts {res.index[0]:%d-%m-%Y}, so its money is invested "
                         f"from that day and its base year starts there.")
        invested = amount * weights[a]
        res['Holding (₹)'] = invested * res['Index Value'] / res['Index Value'].iloc[0]
        results[a] = res
    for n in notes:
        note(n, BLUE)
    if not results:
        note("No data to show for the selected dates.", AMBER, "⚠️")
        return

    all_dates = sorted(set().union(*[r.index for r in results.values()]))
    holdings = pd.DataFrame(index=pd.DatetimeIndex(all_dates))
    for a, res in results.items():
        h = res['Holding (₹)'].reindex(holdings.index).ffill()
        holdings[a] = h.fillna(amount * weights[a])          # before its data starts = cash
    cash = amount * sum(w for a, w in weights.items() if a not in results)
    portfolio = holdings.sum(axis=1) + cash

    # ---------------- Cards ----------------
    last_d = portfolio.index[-1]
    cur_val = portfolio.iloc[-1]
    pl = cur_val - amount
    port_yoy = (cur_val / value_one_year_ago(portfolio, last_d) - 1) * 100
    day_chg = portfolio.diff().iloc[-1] if len(portfolio) > 1 else 0.0

    section("Your portfolio today",
            f"{portfolio.index[0]:%d %b %Y} → {last_d:%d %b %Y} · {len(portfolio):,} trading days", "💼", CYAN)
    kpi_row([
        dict(label="Invested", value=inr(amount), delta=f"across {len(results)} indices", icon="💵", accent=BLUE),
        dict(label="Current value", value=inr(cur_val), delta=fmt_pct(pl / amount * 100), tone=tone_of(pl),
             icon="💎", accent=CYAN),
        dict(label="Total P&L", value=inr(pl, sign=True), delta=f"today {inr(day_chg, sign=True)}",
             tone=tone_of(day_chg), icon="💰", accent=EMERALD if pl >= 0 else BAD),
        dict(label="Portfolio YoY", value=pct(port_yoy), delta="vs same day last year",
             tone=tone_of(port_yoy), icon="🔁", accent=VIOLET),
    ])

    # Per-holding mini cards
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    cols = st.columns(len(results))
    for col, (a, res) in zip(cols, results.items()):
        inv = amount * weights[a]
        cur = holdings[a].iloc[-1]
        with col:
            kpi(f"{ASSET_SHORT[a]} · {weights[a] * 100:.0f}%", inr(cur), fmt_pct((cur / inv - 1) * 100),
                tone_of(cur - inv), ASSET_ICONS[a], COLORS[a])

    # ---------------- Chart ----------------
    section("Growth of each holding", "Dotted vertical lines mark each year boundary", "📈", BLUE)
    fig = go.Figure()
    for a in results:
        y = holdings[a] if chart_mode == "₹ Value" else (holdings[a] / (amount * weights[a]) - 1) * 100
        fig.add_trace(go.Scatter(x=holdings.index, y=y, name=f"{ASSET_SHORT[a]} ({weights[a] * 100:.0f}%)",
                                 line=dict(color=COLORS[a], width=2.2)))
    if show_total:
        y = portfolio if chart_mode == "₹ Value" else (portfolio / amount - 1) * 100
        fig.add_trace(go.Scatter(x=portfolio.index, y=y, name="Total portfolio",
                                 line=dict(color=PORTFOLIO_COLOR, width=3, dash='dot')))
    k = 1
    while pd.Timestamp(start) + pd.DateOffset(years=k) <= last_d:
        fig.add_vline(x=pd.Timestamp(start) + pd.DateOffset(years=k), line_dash="dot",
                      line_color="rgba(138,148,179,0.22)")
        k += 1
    if chart_mode == "% Return":
        fig.add_hline(y=0, line_dash="dash", line_color=FAINT)
    title = "Value of each holding (₹)" if chart_mode == "₹ Value" else "Return since start (%)"
    fig.update_layout(**PLOT_LAYOUT, title=title, height=560, hovermode='x unified',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    if chart_mode == "₹ Value":
        fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
    else:
        fig.update_yaxes(ticksuffix="%")
    with st.container(border=True):
        chart(fig, key="ep_chart")

    # ---------------- Holdings summary ----------------
    section("Holdings", "Latest YoY = change versus the same day last year", "📋", AMBER)
    rows = []
    for a, res in results.items():
        inv = amount * weights[a]
        cur = holdings[a].iloc[-1]
        rows.append({'Index': ASSET_LABELS[a], 'Weight (%)': weights[a] * 100, 'Invested (₹)': inv,
                     'Current Value (₹)': cur, 'P&L (₹)': cur - inv, 'Return (%)': (cur / inv - 1) * 100,
                     'Latest YoY (%)': res['YoY (%)'].iloc[-1], 'Latest Close (₹)': res['Close'].iloc[-1]})
    rows.append({'Index': '🧾 Total portfolio', 'Weight (%)': 100.0, 'Invested (₹)': amount,
                 'Current Value (₹)': cur_val, 'P&L (₹)': pl, 'Return (%)': pl / amount * 100,
                 'Latest YoY (%)': port_yoy, 'Latest Close (₹)': np.nan})
    hold_df = pd.DataFrame(rows)
    hold_fmt = {'Weight (%)': '{:.1f}%', 'Invested (₹)': money, 'Current Value (₹)': money,
                'P&L (₹)': lambda v: inr(v, 2, sign=True), 'Return (%)': pct, 'Latest YoY (%)': pct,
                'Latest Close (₹)': money}
    hold_df, hold_fmt = blank_missing(hold_df, hold_fmt)
    st.dataframe(style_change_cols(hold_df.style.format(hold_fmt), ['P&L (₹)', 'Return (%)', 'Latest YoY (%)']),
                 width="stretch", hide_index=True)

    # ---------------- Year-end values ----------------
    section("Year-end values", "Value of every holding at the end of each investment year", "📅", VIOLET)
    year_no = pd.Series(0, index=holdings.index)
    k = 1
    while pd.Timestamp(start) + pd.DateOffset(years=k) <= last_d:
        year_no[holdings.index >= pd.Timestamp(start) + pd.DateOffset(years=k)] = k
        k += 1
    ye = holdings.assign(**{'Total Portfolio': portfolio}).groupby(year_no.values).last()
    ye.columns = [ASSET_LABELS.get(c, c) for c in ye.columns]
    labels = []
    for i in ye.index:
        y_end = pd.Timestamp(start) + pd.DateOffset(years=i + 1) - pd.Timedelta(days=1)
        labels.append(f"Year {i + 1} (to {min(y_end, last_d):%d-%m-%Y})")

    fig_ye = go.Figure()
    for a in results:
        fig_ye.add_trace(go.Bar(x=[f"Y{i + 1}" for i in ye.index], y=ye[ASSET_LABELS[a]], name=ASSET_SHORT[a],
                                marker_color=COLORS[a], marker_line_width=0, hovertemplate="₹%{y:,.0f}"))
    fig_ye.update_layout(**PLOT_LAYOUT, barmode='stack', height=340, title="Year-end value, stacked by holding",
                         yaxis=dict(tickprefix="₹", tickformat=",.0f"), hovermode='x unified',
                         legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    with st.container(border=True):
        chart(fig_ye, key="ep_ye")

    ye_show = ye.apply(lambda col: col.map(money))
    ye_show.insert(0, 'Period', labels)
    st.dataframe(ye_show, width="stretch", hide_index=True)

    # ---------------- One combined table + CSV ----------------
    section("All indices, day by day", "Close, previous-year close, YoY %, chained index value and holding",
            "📥", CYAN)
    combined = pd.DataFrame(index=holdings.index)
    for a, res in results.items():
        lab = ASSET_SHORT[a]
        r = res.reindex(holdings.index)
        combined[f"{lab} Close"] = r['Close']
        combined[f"{lab} Prev Year Close"] = r['Prev Year Close']
        combined[f"{lab} YoY (%)"] = r['YoY (%)']
        combined[f"{lab} Index Value"] = r['Index Value']
        combined[f"{lab} Holding (₹)"] = holdings[a]
    combined["Total Portfolio (₹)"] = portfolio
    combined["Portfolio Day Change (₹)"] = portfolio.diff()

    show = combined.sort_index(ascending=False)
    show_fmt = pd.DataFrame({'Date': show.index.strftime('%d-%m-%Y')})
    for c in show.columns:
        f = pct if "(%)" in c else money
        show_fmt[c] = [f(v) for v in show[c]]
    st.dataframe(show_fmt, width="stretch", height=420, hide_index=True)

    export = combined.round(4).copy()
    export.index = export.index.strftime('%Y-%m-%d')
    export.index.name = 'Date'
    st.download_button("⬇️  Download all indices + portfolio (CSV)", export.to_csv().encode('utf-8-sig'),
                       file_name=f"everyday_portfolio_{start:%Y%m%d}_{end:%Y%m%d}.csv", mime="text/csv",
                       key="ep_dl")