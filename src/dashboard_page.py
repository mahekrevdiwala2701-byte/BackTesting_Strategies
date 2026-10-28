"""📊 Market Dashboard page: live overview of every asset + analysis tabs."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.ui import (
    panel,
    PLOT_LAYOUT, VIOLET, CYAN, AMBER, PINK, EMERALD, BLUE, BAD, GOOD, FAINT, SURFACE, TEXT, MUTED,
    hero, ticker, section, kpi, kpi_row, tone_of, note, chart, hex_rgba, inr, pct,
    sidebar_nav, sidebar_footer,
)
from src.analytics import compute_technical_indicators, calculate_performance_metrics
from src.asset_data import (ASSETS, ASSET_LABELS, ASSET_SHORT, ASSET_ICONS, COLORS, filter_period,
                             style_change_cols, blank_missing)
from src.period_picker import period_selector


def render_dashboard_page(asset_dfs, fetch_errors):
    # ---------------- Sidebar ----------------
    sidebar_nav("dashboard")
    start_date, end_date = period_selector(asset_dfs, default_preset="Custom Range", key="main")
    sidebar_footer(fetch_errors, key="main")

    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    period_dfs = filter_period(asset_dfs, start_date, end_date)
    available = [a for a in ASSETS if a in period_dfs and not period_dfs[a].empty]
    last_date = max(df['Date'].max() for df in asset_dfs.values())

    # ---------------- Hero ----------------
    hero_line = None
    for n in ('LargeCap', 'MidCap', 'Gold', 'SmallCap'):
        if n in period_dfs and len(period_dfs[n]) > 5:
            hero_line = period_dfs[n]['Close'].to_numpy()
            break
    hero("Market Overview",
         "GoldBEES, Nifty 50, MidCap and SmallCap side by side: prices, trends, correlations and raw data, "
         "synced live from Google Sheets.",
         page="dashboard", eyebrow="Live overview",
         chips=[f"📅 {start_date:%d %b %Y} → {end_date:%d %b %Y}",
                f"🧩 {len(asset_dfs)} of {len(ASSETS)} assets loaded",
                f"🕒 Latest data {last_date:%d %b %Y}"], line=hero_line)

    # ---------------- Ticker tape ----------------
    tape = []
    for n in ASSETS:
        if n in asset_dfs and len(asset_dfs[n]) > 1:
            full = asset_dfs[n]
            chg = full['Change (%)'].iloc[-1]
            tape.append((ASSET_SHORT[n], inr(full['Close'].iloc[-1], 2), 0.0 if pd.isna(chg) else chg, COLORS[n]))
    if tape:
        ticker(tape)

    # ---------------- Snapshot row ----------------
    section("Snapshot", "Latest price and change over the selected period", "⚡", VIOLET)
    cols = st.columns(len(ASSETS))
    for col, name in zip(cols, ASSETS):
        with col:
            d = period_dfs.get(name)
            if d is None or d.empty:
                kpi(ASSET_SHORT[name], "No data", "outside period", "neutral", ASSET_ICONS[name], COLORS[name])
                continue
            first, last = d['Close'].iloc[0], d['Close'].iloc[-1]
            chg = (last / first - 1) * 100 if first else 0.0
            kpi(ASSET_SHORT[name], inr(last, 2), f"{chg:+.2f}% in period", tone_of(chg),
                ASSET_ICONS[name], COLORS[name], spark=d['Close'].to_numpy())

    # ---------------- Per-asset tabs ----------------
    section("Asset deep-dive", "Price history, daily moves and every record for one asset", "🔍", CYAN)
    for tab, name in zip(st.tabs([ASSET_LABELS[a] for a in ASSETS]), ASSETS):
        with tab:
            _asset_tab(name, asset_dfs, period_dfs, fetch_errors, start_date, end_date)

    # ---------------- Analysis ----------------
    section("Analysis", "Technical indicators, cross-asset comparison and data export", "🧪", PINK)
    if not available:
        note("No data in the selected period. Pick a different period in the sidebar.", AMBER, "⚠️")
        return

    labels = ["📈 Technical analysis", "🔗 Compare & correlate", "📥 Data inspector"]
    try:                                                # open "Compare & correlate" first
        t1, t2, t3 = st.tabs(labels, default=labels[1])
    except TypeError:                                   # older Streamlit without `default`
        t2, t1, t3 = st.tabs([labels[1], labels[0], labels[2]])
    with t1:
        _technical_tab(asset_dfs, available, start_ts, end_ts)
    with t2:
        _compare_tab(period_dfs, available)
    with t3:
        _data_tab(period_dfs, available)


# =========================================================
# One asset
# =========================================================
def _asset_tab(name, asset_dfs, period_dfs, fetch_errors, start_date, end_date):
    color = COLORS[name]
    if name not in asset_dfs:
        note(f"No data available for <b>{ASSET_LABELS[name]}</b>. {fetch_errors.get(name, '')}<br>"
             "If the columns show as 'Unnamed', check the sheet has a header row with 'Date' and 'Close' "
             "(or 'Price').", BAD, "⛔")
        return
    d = period_dfs[name]
    if d.empty:
        full = asset_dfs[name]
        note(f"No {ASSET_LABELS[name]} data in the selected period. Available: "
             f"<b>{full['Date'].min():%d %b %Y}</b> to <b>{full['Date'].max():%d %b %Y}</b>.", AMBER, "⚠️")
        return

    latest, first = d.iloc[-1], d.iloc[0]
    day_chg = latest['Change (₹)'] if pd.notna(latest['Change (₹)']) else 0.0
    day_pct = latest['Change (%)'] if pd.notna(latest['Change (%)']) else 0.0
    per_chg = latest['Close'] - first['Close']
    per_pct = (per_chg / first['Close'] * 100) if first['Close'] else 0.0
    hi, lo = d['Close'].max(), d['Close'].min()

    kpi_row([
        dict(label="Latest price", value=inr(latest['Close'], 2), delta=f"on {latest['Date']:%d %b %Y}",
             icon="💹", accent=color),
        dict(label="Day change", value=inr(day_chg, 2, sign=True), delta=pct(day_pct), tone=tone_of(day_chg),
             icon="📆", accent=BLUE),
        dict(label="Change in period", value=inr(per_chg, 2, sign=True), delta=pct(per_pct),
             tone=tone_of(per_chg), icon="📈", accent=EMERALD if per_chg >= 0 else BAD),
        dict(label="Period high / low", value=inr(hi, 2), delta=f"Low {inr(lo, 2)}", icon="↕️", accent=AMBER),
    ])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.72, 0.28],
                        subplot_titles=(f"{ASSET_SHORT[name]} price (₹)", "Daily change (%)"))
    fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'], name='Close', line=dict(color=color, width=2.4),
                             fill='tozeroy', fillcolor=hex_rgba(color, 0.10),
                             hovertemplate="₹%{y:,.2f}<extra></extra>"), row=1, col=1)
    bar_colors = [GOOD if v >= 0 else BAD for v in d['Change (%)'].fillna(0)]
    fig.add_trace(go.Bar(x=d['Date'], y=d['Change (%)'], name='Change', marker_color=bar_colors,
                         marker_line_width=0, hovertemplate="%{y:+.2f}%<extra></extra>"), row=2, col=1)
    pad = (hi - lo) * 0.06 or hi * 0.01
    fig.update_yaxes(range=[lo - pad, hi + pad], tickprefix="₹", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_layout(**PLOT_LAYOUT, height=540, showlegend=False, hovermode='x unified')
    fig.update_annotations(font=dict(size=13, color=MUTED))
    with panel():
        chart(fig, key=f"price_{name}")

    st.markdown(f"**📋 All records** · {first['Date']:%d %b %Y} to {latest['Date']:%d %b %Y} · {len(d):,} rows")
    cols = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change (₹)', 'Change (%)'] if c in d.columns]
    table = d[cols].sort_values('Date', ascending=False).reset_index(drop=True)
    fmt = {'Date': lambda x: x.strftime('%d-%m-%Y'), 'Change (₹)': lambda v: inr(v, 2, sign=True),
           'Change (%)': '{:+.2f}%'}
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in table.columns:
            fmt[c] = lambda v: inr(v, 2)
    if 'Volume' in table.columns:
        fmt['Volume'] = '{:,.0f}'
    table, fmt = blank_missing(table, fmt)
    styler = style_change_cols(table.style.format(fmt), ['Change (₹)', 'Change (%)'])
    st.dataframe(styler, width="stretch", height=380)


# =========================================================
# Analysis tabs
# =========================================================
def _technical_tab(asset_dfs, available, start_ts, end_ts):
    name = st.selectbox("Asset", options=available, format_func=lambda n: ASSET_LABELS[n], key="ta_asset")
    color = COLORS[name]
    df_full = compute_technical_indicators(asset_dfs[name].copy())     # full history so SMA 200 is ready
    df = df_full[(df_full['Date'] >= start_ts) & (df_full['Date'] <= end_ts)]
    m = calculate_performance_metrics(df)

    if m:
        kpi_row([
            dict(label="CAGR", value=pct(m['CAGR'] * 100), delta=f"over {m['Years']:.1f} yrs",
                 tone=tone_of(m['CAGR']), icon="🚀", accent=EMERALD),
            dict(label="Total return", value=pct(m['Total Return'] * 100), tone="neutral", icon="💹",
                 delta=f"{m['Total Trading Days']:,} trading days", accent=color),
            dict(label="Volatility (ann.)", value=pct(m['Annual Volatility'] * 100, sign=False), icon="🌊",
                 delta="std. dev. of daily moves", accent=BLUE),
            dict(label="Sharpe / Sortino", value=f"{m['Sharpe Ratio']:.2f} / {m['Sortino Ratio']:.2f}",
                 delta="risk-free 5%", icon="⚖️", accent=VIOLET),
            dict(label="Max drawdown", value=pct(m['Max Drawdown'] * 100), delta="worst peak-to-trough",
                 tone="bad", icon="📉", accent=BAD),
        ])

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.56, 0.22, 0.22],
                        subplot_titles=(f"{ASSET_SHORT[name]} with moving averages & Bollinger bands",
                                        "RSI (14)", "Drawdown from peak"))
    if 'BB_Upper' in df:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], line=dict(width=0), hoverinfo='skip',
                                 showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], line=dict(width=0), fill='tonexty',
                                 fillcolor=hex_rgba(VIOLET, 0.10), name='Bollinger band',
                                 hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color=color, width=2.2),
                             hovertemplate="₹%{y:,.2f}"), row=1, col=1)
    for col_name, c, dash in (('SMA_50', CYAN, 'solid'), ('SMA_200', PINK, 'solid'), ('EMA_50', AMBER, 'dot')):
        if col_name in df and df[col_name].notna().any():
            fig.add_trace(go.Scatter(x=df['Date'], y=df[col_name], name=col_name.replace('_', ' '),
                                     line=dict(color=c, width=1.5, dash=dash), hovertemplate="₹%{y:,.2f}"),
                          row=1, col=1)
    if 'RSI' in df:
        fig.add_hrect(y0=70, y1=100, fillcolor=hex_rgba(BAD, 0.08), line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor=hex_rgba(GOOD, 0.08), line_width=0, row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color=VIOLET, width=1.6),
                                 hovertemplate="%{y:.1f}"), row=2, col=1)
    dd = (df['Close'] / df['Close'].cummax() - 1) * 100
    fig.add_trace(go.Scatter(x=df['Date'], y=dd, name='Drawdown', line=dict(color=BAD, width=1.4),
                             fill='tozeroy', fillcolor=hex_rgba(BAD, 0.16), hovertemplate="%{y:.2f}%"),
                  row=3, col=1)
    fig.update_yaxes(tickprefix="₹", row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_yaxes(ticksuffix="%", row=3, col=1)
    fig.update_annotations(font=dict(size=13, color=MUTED))
    fig.update_layout(**PLOT_LAYOUT, height=720, hovermode='x unified',
                      legend=dict(orientation='h', yanchor='bottom', y=1.04, x=0))
    with panel():
        chart(fig, key="ta_chart")
    st.caption("RSI above 70 (red zone) = overbought, below 30 (green zone) = oversold. "
               "Moving averages are computed on the full history, so they are ready from the first day.")


def _compare_tab(period_dfs, available):
    picked = st.pills("Assets to compare (click to switch on / off)", options=available, selection_mode="multi",
                      default=available, format_func=lambda n: f"{ASSET_ICONS[n]} {ASSET_SHORT[n]}",
                      key="cmp_assets")
    available = [n for n in available if n in (picked or [])]
    if not available:
        note("Switch on at least one asset above to compare.", AMBER, "👆")
        return
    fig = go.Figure()
    for n in available:
        d = period_dfs[n]
        fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'] / d['Close'].iloc[0] * 100, name=ASSET_SHORT[n],
                                 line=dict(color=COLORS[n], width=2.4), hovertemplate="%{y:.1f}"))
    fig.add_hline(y=100, line_dash="dot", line_color=FAINT)
    fig.update_layout(**PLOT_LAYOUT, title="Growth of 100 (every asset rebased to 100 on day one)", height=460,
                      hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.45))
    with panel():
        chart(fig, key="cmp_norm")

    c1, c2 = st.columns([1.15, 1])
    with c1:
        summary = pd.DataFrame([{
            'Asset': ASSET_LABELS[n],
            'Start price': period_dfs[n]['Close'].iloc[0],
            'End price': period_dfs[n]['Close'].iloc[-1],
            'Change (₹)': period_dfs[n]['Close'].iloc[-1] - period_dfs[n]['Close'].iloc[0],
            'Change (%)': (period_dfs[n]['Close'].iloc[-1] / period_dfs[n]['Close'].iloc[0] - 1) * 100,
        } for n in available])
        st.markdown("**📋 Change in the selected period**")
        st.dataframe(style_change_cols(summary.style.format({
            'Start price': lambda v: inr(v, 2), 'End price': lambda v: inr(v, 2),
            'Change (₹)': lambda v: inr(v, 2, sign=True), 'Change (%)': '{:+.2f}%'}), ['Change (₹)', 'Change (%)']),
            width="stretch", hide_index=True)

        fig_bar = go.Figure(go.Bar(
            x=[ASSET_SHORT[n] for n in available], y=summary['Change (%)'],
            marker_color=[COLORS[n] for n in available], marker_line_width=0,
            text=[f"{v:+.1f}%" for v in summary['Change (%)']], textposition='outside',
            hovertemplate="%{x}: %{y:+.2f}%<extra></extra>"))
        fig_bar.update_layout(**PLOT_LAYOUT, title="Return in period", height=300, yaxis=dict(ticksuffix="%"),
                              showlegend=False)
        chart(fig_bar, key="cmp_bar")
    with c2:
        returns = pd.concat({ASSET_SHORT[n]: period_dfs[n].set_index('Date')['Change (%)'] for n in available}, axis=1)
        corr = returns.corr(min_periods=10)
        st.markdown("**🔗 Correlation of daily returns**")
        if len(available) > 1 and corr.notna().sum().sum() > len(available):
            fig_c = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, xgap=3, ygap=3,
                colorscale=[[0, BAD], [0.5, SURFACE], [1, CYAN]],
                text=[[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in corr.values],
                texttemplate="%{text}", textfont=dict(size=14, color=TEXT),
                hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>", colorbar=dict(thickness=12)))
            fig_c.update_layout(**PLOT_LAYOUT, height=420, yaxis=dict(autorange='reversed'),
                                margin=dict(l=10, r=10, t=20, b=10))
            chart(fig_c, key="cmp_corr")
            st.caption("1 = always move together · 0 = unrelated · −1 = move opposite. "
                       "Low correlation = better diversification.")
        else:
            note("Not enough shared dates in this period to compute correlations.", AMBER, "⚠️")


def _data_tab(period_dfs, available):
    c1, c2 = st.columns([3, 1])
    with c1:
        name = st.selectbox("Asset to export", options=available, format_func=lambda n: ASSET_LABELS[n],
                            key="inspect_select")
    df = period_dfs[name].copy()
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    with c2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.download_button("⬇️  Download CSV", df.to_csv(index=False).encode('utf-8'),
                           file_name=f"{name.lower()}_data.csv", mime="text/csv", width="stretch")
    st.caption(f"{len(df):,} records for {ASSET_LABELS[name]}")
    st.dataframe(df, width="stretch", height=460)