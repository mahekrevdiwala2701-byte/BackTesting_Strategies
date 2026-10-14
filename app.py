import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.ui import apply_custom_css, render_metric_card
from src.analytics import compute_technical_indicators, calculate_performance_metrics
from src.asset_data import (
    ASSETS, ASSET_LABELS, COLORS, PLOT_LAYOUT,
    load_all_assets, filter_period, style_change_cols
)
from src.period_picker import period_selector
from src.investment_page import render_investment_page
from src.monte_carlo_page import render_monte_carlo_page

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multi-Asset Analysis Lab (INR ₹)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
asset_dfs, fetch_errors = load_all_assets()

if not asset_dfs:
    st.error("⚠️ Unable to load dataset for any entity. Please verify Google Sheet permissions.")
    st.stop()


# ---------------------------------------------------------
# Page switcher: show Investment page or Dashboard
# ---------------------------------------------------------
if "view" not in st.session_state:
    st.session_state["view"] = "dashboard"

if st.session_state["view"] == "invest":
    render_investment_page(asset_dfs, fetch_errors)
    st.stop()

if st.session_state["view"] == "montecarlo":
    render_monte_carlo_page(asset_dfs, fetch_errors)
    st.stop()

# =========================================================
# 📊 DASHBOARD PAGE
# =========================================================

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown("## ⚙️ Control Panel")

if st.sidebar.button("💰 Calculated Investment", use_container_width=True, type="primary"):
    st.session_state["view"] = "invest"
    st.rerun()

if st.sidebar.button("🎲 Monte Carlo Simulation", use_container_width=True, type="primary"):
    st.session_state["view"] = "montecarlo"
    st.rerun()

st.sidebar.markdown("---")
start_date, end_date = period_selector(asset_dfs, key="main")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Live Sheet Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

if fetch_errors:
    with st.sidebar.expander("⚠️ Data load issues", expanded=False):
        for name, err in fetch_errors.items():
            st.write(f"**{name}:** {err}")

start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
period_dfs = filter_period(asset_dfs, start_date, end_date)

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown('<div class="header-title">📊 Multi-Asset Analysis Suite (INR ₹)</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Real-time Google Sheets Live Sync: GoldBEES, LargeCap (Nifty 50), MidCap & SmallCap</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Asset Tab Renderer
# ---------------------------------------------------------
def render_asset_tab(name):
    if name not in asset_dfs:
        st.error(f"No data available for {ASSET_LABELS[name]}. {fetch_errors.get(name, '')}")
        st.info("If the columns show as 'Unnamed', check the sheet has a header row with 'Date' and 'Close' (or 'Price').")
        return

    d = period_dfs[name]
    if d.empty:
        full = asset_dfs[name]
        st.warning(f"No {ASSET_LABELS[name]} data in the selected period. "
                   f"Available: {full['Date'].min():%d %b %Y} to {full['Date'].max():%d %b %Y}.")
        return

    latest, first = d.iloc[-1], d.iloc[0]
    day_chg = latest['Change (₹)'] if pd.notna(latest['Change (₹)']) else 0.0
    day_pct = latest['Change (%)'] if pd.notna(latest['Change (%)']) else 0.0
    per_chg = latest['Close'] - first['Close']
    per_pct = (per_chg / first['Close'] * 100) if first['Close'] else 0.0
    hi, lo = d['Close'].max(), d['Close'].min()

    st.caption(f"Period: **{first['Date']:%d %b %Y}** to **{latest['Date']:%d %b %Y}**  ·  {len(d):,} records")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Latest Price", f"₹{latest['Close']:,.2f}", f"{day_pct:+.2f}%", is_positive=day_chg >= 0)
    with c2:
        render_metric_card("Day Change (₹)", f"₹{day_chg:+,.2f}", f"{day_pct:+.2f}%", is_positive=day_chg >= 0)
    with c3:
        render_metric_card("Change in Period", f"₹{per_chg:+,.2f}", f"{per_pct:+.2f}%", is_positive=per_chg >= 0)
    with c4:
        render_metric_card("Period High / Low", f"₹{hi:,.2f}", f"Low ₹{lo:,.2f}", is_positive=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.72, 0.28],
                        subplot_titles=(f"{ASSET_LABELS[name]} Price (₹)", "Daily Change (%)"))
    fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'], name='Close (₹)',
                             line=dict(color=COLORS[name], width=2)), row=1, col=1)
    bar_colors = ['#10B981' if v >= 0 else '#EF4444' for v in d['Change (%)'].fillna(0)]
    fig.add_trace(go.Bar(x=d['Date'], y=d['Change (%)'], name='Change (%)',
                         marker_color=bar_colors), row=2, col=1)
    pad = (hi - lo) * 0.05 or hi * 0.01
    fig.update_yaxes(range=[lo - pad, hi + pad], tickprefix="₹", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_layout(**PLOT_LAYOUT, height=520, showlegend=False, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 All Data (selected period)")
    cols = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change (₹)', 'Change (%)'] if c in d.columns]
    table = d[cols].sort_values('Date', ascending=False).reset_index(drop=True)

    fmt = {'Date': lambda x: x.strftime('%d-%m-%Y'), 'Change (₹)': '₹{:+,.2f}', 'Change (%)': '{:+.2f}%'}
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in table.columns:
            fmt[c] = '₹{:,.2f}'
    if 'Volume' in table.columns:
        fmt['Volume'] = '{:,.0f}'

    styler = style_change_cols(table.style.format(fmt, na_rep='—'), ['Change (₹)', 'Change (%)'])
    st.dataframe(styler, use_container_width=True, height=400)

    export = table.copy()
    export['Date'] = export['Date'].dt.strftime('%Y-%m-%d')
    st.download_button(
        label=f"⬇️ Download {name} data (CSV)",
        data=export.to_csv(index=False).encode('utf-8'),
        file_name=f"{name.lower()}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv",
        mime="text/csv",
        key=f"dl_{name}"
    )


# ---------------------------------------------------------
# Live Asset Overview (4 tabs: GoldBEES open by default)
# ---------------------------------------------------------
st.subheader("📌 Live Asset Overview")
asset_tabs = st.tabs([ASSET_LABELS[a] for a in ASSETS])
for tab, name in zip(asset_tabs, ASSETS):
    with tab:
        render_asset_tab(name)

st.markdown("---")

# ---------------------------------------------------------
# Analysis Tabs
# ---------------------------------------------------------
available = [a for a in ASSETS if a in period_dfs and not period_dfs[a].empty]

tab_individual, tab_multi, tab_data = st.tabs([
    "📈 Individual Entity Analysis",
    "📊 Asset Correlations & Comparison",
    "📥 Data Inspector & Export"
])

with tab_individual:
    st.subheader("📈 Individual Asset Technical & Trend Analysis")

    if not available:
        st.info("No data in the selected period.")
    else:
        selected_entity = st.selectbox("Select Entity to Inspect:", options=available,
                                       format_func=lambda n: ASSET_LABELS[n])
        df_full = compute_technical_indicators(asset_dfs[selected_entity].copy())
        df_ta = df_full[(df_full['Date'] >= start_ts) & (df_full['Date'] <= end_ts)]
        indiv_metrics = calculate_performance_metrics(df_ta)

        fig_indiv = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3],
            subplot_titles=(f"{ASSET_LABELS[selected_entity]} Price History (₹)", "Relative Strength Index (RSI)")
        )
        fig_indiv.add_trace(go.Scatter(x=df_ta['Date'], y=df_ta['Close'], name='Close (₹)',
                                       line=dict(color='#6366F1', width=2)), row=1, col=1)
        if 'SMA_20' in df_ta.columns:
            fig_indiv.add_trace(go.Scatter(x=df_ta['Date'], y=df_ta['SMA_20'], name='SMA 20',
                                           line=dict(color='#3B82F6', width=1.5, dash='dot')), row=1, col=1)
        if 'SMA_50' in df_ta.columns:
            fig_indiv.add_trace(go.Scatter(x=df_ta['Date'], y=df_ta['SMA_50'], name='SMA 50',
                                           line=dict(color='#F59E0B', width=1.5)), row=1, col=1)
        if 'RSI' in df_ta.columns:
            fig_indiv.add_trace(go.Scatter(x=df_ta['Date'], y=df_ta['RSI'], name='RSI (14)',
                                           line=dict(color='#A855F7', width=1.5)), row=2, col=1)
            fig_indiv.add_hline(y=70, line_dash="dash", line_color="#EF4444", row=2, col=1)
            fig_indiv.add_hline(y=30, line_dash="dash", line_color="#10B981", row=2, col=1)

        fig_indiv.update_layout(**PLOT_LAYOUT, height=600)
        fig_indiv.update_yaxes(tickprefix="₹", row=1, col=1)
        st.plotly_chart(fig_indiv, use_container_width=True)

with tab_multi:
    st.subheader("📊 Normalized Multi-Asset Performance Comparison")

    if not available:
        st.info("No data in the selected period.")
    else:
        fig_multi = go.Figure()
        for name in available:
            df_a = period_dfs[name]
            fig_multi.add_trace(go.Scatter(
                x=df_a['Date'], y=df_a['Close'] / df_a['Close'].iloc[0] * 100,
                name=ASSET_LABELS[name], line=dict(color=COLORS[name], width=2)
            ))
        fig_multi.update_layout(**PLOT_LAYOUT, title="Normalized Growth Comparison (Rebased to 100)",
                                height=450, hovermode='x unified')
        st.plotly_chart(fig_multi, use_container_width=True)

        summary_df = pd.DataFrame([{
            'Asset': ASSET_LABELS[n],
            'Start Price (₹)': period_dfs[n]['Close'].iloc[0],
            'End Price (₹)': period_dfs[n]['Close'].iloc[-1],
            'Change (₹)': period_dfs[n]['Close'].iloc[-1] - period_dfs[n]['Close'].iloc[0],
            'Change (%)': (period_dfs[n]['Close'].iloc[-1] / period_dfs[n]['Close'].iloc[0] - 1) * 100,
        } for n in available])
        st.markdown("### 📋 Change in Selected Period")
        st.dataframe(
            style_change_cols(summary_df.style.format({
                'Start Price (₹)': '₹{:,.2f}', 'End Price (₹)': '₹{:,.2f}',
                'Change (₹)': '₹{:+,.2f}', 'Change (%)': '{:+.2f}%'
            }), ['Change (₹)', 'Change (%)']),
            use_container_width=True, hide_index=True
        )

        st.markdown("### 🔗 Asset Returns Correlation Matrix")
        returns = pd.concat({n: period_dfs[n].set_index('Date')['Change (%)'] for n in available}, axis=1)
        corr = returns.corr(min_periods=10)
        if len(available) > 1 and corr.notna().sum().sum() > len(available):
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="Viridis",
                                 title="Daily Returns Correlation Heatmap")
            fig_corr.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough shared dates in this period to compute correlations.")

with tab_data:
    st.subheader("📥 Data Table & Export")

    if not available:
        st.info("No data in the selected period.")
    else:
        selected_inspect_asset = st.selectbox("Select Asset to Export Data:", options=available,
                                              format_func=lambda n: ASSET_LABELS[n], key="inspect_select")
        df_export = period_dfs[selected_inspect_asset].copy()
        df_export['Date'] = df_export['Date'].dt.strftime('%Y-%m-%d')

        col_d1, col_d2 = st.columns([3, 1])
        with col_d1:
            st.write(f"Showing **{len(df_export)}** records for **{ASSET_LABELS[selected_inspect_asset]}**")
        with col_d2:
            st.download_button(
                label="⬇️ Download CSV",
                data=df_export.to_csv(index=False).encode('utf-8'),
                file_name=f"{selected_inspect_asset.lower()}_data.csv",
                mime="text/csv"
            )

        st.dataframe(df_export, use_container_width=True)