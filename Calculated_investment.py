import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import apply_custom_css, render_metric_card
from src.asset_data import (
    ASSETS, ASSET_LABELS, COLORS, PLOT_LAYOUT,
    load_all_assets, period_selector, filter_period, style_change_cols
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Calculated Investment (INR ₹)",
    page_icon="💰",
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
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown("## ⚙️ Control Panel")

if st.sidebar.button("⬅️ Back to Dashboard", use_container_width=True, type="primary"):
    st.switch_page("app.py")

st.sidebar.markdown("---")
start_date, end_date = period_selector(asset_dfs, default_preset="1 Year", key="invest")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Live Sheet Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

if fetch_errors:
    with st.sidebar.expander("⚠️ Data load issues", expanded=False):
        for name, err in fetch_errors.items():
            st.write(f"**{name}:** {err}")

period_dfs = filter_period(asset_dfs, start_date, end_date)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown('<div class="header-title">💰 Calculated Investment</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-subtitle">Enter how much you invest and how to split it across GoldBEES, '
    'LargeCap, MidCap & SmallCap. See what it would be worth today.</div>',
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Input Form
# ---------------------------------------------------------
with st.form("investment_form"):
    st.markdown("### 💵 Investment Amount")
    total_amount = st.number_input("Total amount to invest (₹)", min_value=1000.0,
                                   value=100000.0, step=10000.0, format="%.0f")

    st.markdown("### ⚖️ Weights for each asset (%)")
    w_cols = st.columns(4)
    weights = {}
    for col, name in zip(w_cols, ASSETS):
        with col:
            weights[name] = st.number_input(ASSET_LABELS[name], min_value=0.0, max_value=100.0,
                                            value=25.0, step=5.0, key=f"w_{name}")

    st.caption(f"Investment period: **{start_date:%d %b %Y}** to **{end_date:%d %b %Y}** "
               "(change it from the sidebar). Amount is invested on the first trading day of the period.")

    submitted = st.form_submit_button("🧮 Calculate Investment", use_container_width=True, type="primary")

total_w = sum(weights.values())
if abs(total_w - 100) > 0.01:
    if total_w == 0:
        st.error("Please give at least one asset a weight above 0%.")
        st.stop()
    st.info(f"Your weights add up to **{total_w:.1f}%**. They will be scaled to 100%.")

if not submitted and "invest_done" not in st.session_state:
    st.info("👆 Enter your amount and weights, then press **Calculate Investment**.")
    st.stop()
st.session_state["invest_done"] = True

# ---------------------------------------------------------
# Calculation (lump sum at the start of the period)
# ---------------------------------------------------------
rows = []
value_series = {}
skipped = []

for name in ASSETS:
    w = weights[name]
    if w <= 0:
        continue
    d = period_dfs.get(name)
    if d is None or d.empty:
        skipped.append(name)
        continue

    alloc = total_amount * w / total_w
    buy_price = d['Close'].iloc[0]
    units = alloc / buy_price
    cur_price = d['Close'].iloc[-1]
    cur_value = units * cur_price

    rows.append({
        'Asset': ASSET_LABELS[name],
        'Weight (%)': w / total_w * 100,
        'Invested (₹)': alloc,
        'Buy Date': d['Date'].iloc[0].strftime('%d-%m-%Y'),
        'Buy Price (₹)': buy_price,
        'Units': units,
        'Latest Date': d['Date'].iloc[-1].strftime('%d-%m-%Y'),
        'Latest Price (₹)': cur_price,
        'Current Value (₹)': cur_value,
        'Profit / Loss (₹)': cur_value - alloc,
        'Return (%)': (cur_value / alloc - 1) * 100,
    })
    value_series[name] = d.set_index('Date')['Close'] * units

if skipped:
    st.warning("No data in this period for: " + ", ".join(ASSET_LABELS[s] for s in skipped)
               + ". Their share was left out of the calculation.")

if not rows:
    st.error("None of the selected assets have data in this period. Try a different time period.")
    st.stop()

result = pd.DataFrame(rows)
invested = result['Invested (₹)'].sum()
current = result['Current Value (₹)'].sum()
profit = current - invested
ret_pct = (current / invested - 1) * 100

# Portfolio value over time: each asset carries its last known value forward,
# so the sheets do not need identical dates.
values = pd.concat(value_series, axis=1).sort_index().ffill()
for name in values.columns:
    alloc = result.loc[result['Asset'] == ASSET_LABELS[name], 'Invested (₹)'].iloc[0]
    values[name] = values[name].fillna(alloc)   # before its first price, it's still the cash invested
values['Total'] = values[list(value_series.keys())].sum(axis=1)

years = max((values.index[-1] - values.index[0]).days / 365.25, 1e-9)
cagr = ((current / invested) ** (1 / years) - 1) * 100 if years >= 1 else None

running_max = values['Total'].cummax()
max_dd = ((values['Total'] / running_max - 1) * 100).min()

# ---------------------------------------------------------
# Results: summary cards
# ---------------------------------------------------------
st.markdown("## 📊 Results")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_metric_card("Total Invested", f"₹{invested:,.0f}", is_positive=True)
with c2:
    render_metric_card("Current Value", f"₹{current:,.0f}", f"{ret_pct:+.2f}%", is_positive=profit >= 0)
with c3:
    render_metric_card("Profit / Loss", f"₹{profit:+,.0f}", f"{ret_pct:+.2f}%", is_positive=profit >= 0)
with c4:
    if cagr is not None:
        render_metric_card("CAGR (yearly)", f"{cagr:.2f}%", is_positive=cagr >= 0)
    else:
        render_metric_card("CAGR (yearly)", "—", "Needs 1+ year", is_positive=True)
with c5:
    render_metric_card("Max Drawdown", f"{max_dd:.2f}%", is_positive=False)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=values.index, y=values['Total'], name='Total Portfolio',
                         line=dict(color='#A855F7', width=3)))
for name in value_series:
    fig.add_trace(go.Scatter(x=values.index, y=values[name], name=f"{ASSET_LABELS[name]} (holding)",
                             line=dict(color=COLORS[name], width=2)))
fig.add_hline(y=invested, line_dash="dash", line_color="#94A3B8",
              annotation_text=f"Invested ₹{invested:,.0f}", annotation_position="top left")
fig.update_layout(**PLOT_LAYOUT, title="Total Portfolio & Each Index Holding Over Time (₹)", height=520,
                  hovermode='x unified', yaxis=dict(tickprefix="₹", tickformat=",.0f"),
                  legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Day-by-day change of all 4 indices + portfolio
# ---------------------------------------------------------
names = list(value_series.keys())
day_px = pd.concat({n: period_dfs[n].set_index('Date')['Close'] for n in names}, axis=1).sort_index()
day_px = day_px[~day_px.index.duplicated(keep='last')].reindex(values.index).ffill()
daily_tbl = pd.DataFrame(index=values.index)
for n in names:
    s = ASSET_LABELS[n].split(' ', 1)[1]
    daily_tbl[f'{s} Price (₹)'] = day_px[n]
    daily_tbl[f'{s} Day Change (%)'] = day_px[n].pct_change(fill_method=None) * 100
    daily_tbl[f'{s} Holding (₹)'] = values[n]
    daily_tbl[f'{s} Day P&L (₹)'] = values[n].diff()
daily_tbl['Portfolio Value (₹)'] = values['Total']
daily_tbl['Portfolio Day P&L (₹)'] = values['Total'].diff()
daily_tbl['Portfolio Day Change (%)'] = values['Total'].pct_change() * 100

st.markdown("### 📅 Day-by-Day Change: All 4 Indices + Portfolio")
show_daily = daily_tbl.sort_index(ascending=False).reset_index()
show_daily = show_daily.rename(columns={show_daily.columns[0]: 'Date'})
show_daily['Date'] = pd.to_datetime(show_daily['Date']).dt.strftime('%d-%m-%Y')
fmt = {c: ('{:+.2f}%' if c.endswith('(%)') else '₹{:+,.2f}' if 'P&L' in c else '₹{:,.2f}')
       for c in show_daily.columns if c != 'Date'}
chg_cols = [c for c in show_daily.columns if c.endswith('(%)') or 'P&L' in c]
st.dataframe(style_change_cols(show_daily.style.format(fmt, na_rep='—', subset=list(fmt)), chg_cols),
             use_container_width=True, height=420, hide_index=True)

# ---------------------------------------------------------
# Breakdown table
# ---------------------------------------------------------
st.markdown("### 📋 Asset-wise Breakdown")
total_row = pd.DataFrame([{
    'Asset': '🧾 TOTAL', 'Weight (%)': 100.0, 'Invested (₹)': invested, 'Buy Date': '',
    'Buy Price (₹)': None, 'Units': None, 'Latest Date': '', 'Latest Price (₹)': None,
    'Current Value (₹)': current, 'Profit / Loss (₹)': profit, 'Return (%)': ret_pct,
}])
table = pd.concat([result, total_row], ignore_index=True)

styler = table.style.format({
    'Weight (%)': '{:.2f}%', 'Invested (₹)': '₹{:,.2f}', 'Buy Price (₹)': '₹{:,.2f}',
    'Units': '{:,.4f}', 'Latest Price (₹)': '₹{:,.2f}', 'Current Value (₹)': '₹{:,.2f}',
    'Profit / Loss (₹)': '₹{:+,.2f}', 'Return (%)': '{:+.2f}%'
}, na_rep='—')
st.dataframe(style_change_cols(styler, ['Profit / Loss (₹)', 'Return (%)']),
             use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Download
# ---------------------------------------------------------
dl1, dl2 = st.columns(2)
with dl1:
    st.download_button("⬇️ Download Breakdown (CSV)", table.to_csv(index=False).encode('utf-8'),
                       file_name=f"investment_breakdown_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv",
                       mime="text/csv", use_container_width=True)
with dl2:
    daily = values.reset_index().rename(columns={'index': 'Date'})
    daily.columns = ['Date'] + [ASSET_LABELS.get(c, c) + ' (₹)' if c != 'Total' else 'Total Value (₹)'
                                for c in daily.columns[1:]]
    daily['Date'] = pd.to_datetime(daily['Date']).dt.strftime('%Y-%m-%d')
    st.download_button("⬇️ Download Daily Values (CSV)", daily.to_csv(index=False).encode('utf-8'),
                       file_name=f"investment_daily_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv",
                       mime="text/csv", use_container_width=True)