"""💰 Calculated Investment page (UI only — the maths lives in src/strategies.py)."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import render_metric_card
from src.asset_data import ASSETS, ASSET_LABELS, COLORS, PLOT_LAYOUT, filter_period, style_change_cols
from src.period_picker import period_selector
from src.strategies import (
    STRATEGIES, REBAL_FREQ, DIFF_REBAL_FREQ, LOOKBACK_OPTIONS,
    build_rebalance_dates, simulate_portfolio, simulate_buy_hold, build_diff_schedule,
)

STRATEGY_HELP = {
    "bh": "Money is split once and never touched again. Each asset is bought on its own first trading "
          "day in the period (if an asset's data starts later, its share waits as cash until then). "
          "Weights drift as prices move.",
    "periodic": "Money is split on the first day, then every month/quarter/etc. the portfolio is brought "
                "back to your target weights (sell what grew, buy what fell).",
    "diff": "First buy uses your weights. On every rebalance date each asset's growth over the previous "
            "6 months / 1 year / 1.5 years is compared: the asset that grew the MOST gets the SMALLEST of "
            "your weights, and the one that grew the LEAST (or fell) gets the BIGGEST.",
}


def _short(name):
    """'🥇 GoldBEES' → 'GoldBEES'"""
    return ASSET_LABELS[name].split(' ', 1)[1]


def render_investment_page(asset_dfs, fetch_errors):
    # ---------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------
    st.sidebar.markdown("## ⚙️ Control Panel")
    if st.sidebar.button("⬅️ Back to Dashboard", use_container_width=True, type="primary"):
        st.session_state["view"] = "dashboard"
        st.rerun()
    if st.sidebar.button("🎲 Monte Carlo Simulation", use_container_width=True, key="invest_to_mc"):
        st.session_state["view"] = "montecarlo"
        st.rerun()

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
        'LargeCap, MidCap & SmallCap. Choose buy & hold, same-weight rebalancing, or different weights '
        'based on past growth.</div>',
        unsafe_allow_html=True
    )
    st.caption("🏷️ Code version: **v4**: fixed date reading (data_loader) + start on the chosen month")
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Strategy choice (outside the form so the inputs below change instantly)
    # ---------------------------------------------------------
    st.markdown("### 🧭 Investment Strategy")
    mode = st.radio("Strategy", list(STRATEGIES.keys()), format_func=lambda k: STRATEGIES[k],
                    horizontal=True, key="invest_mode", label_visibility="collapsed")
    st.caption(STRATEGY_HELP[mode])

    # ---------------------------------------------------------
    # Input form
    # ---------------------------------------------------------
    freq_label, lookback_label = None, None
    with st.form("investment_form"):
        st.markdown("### 💵 Investment Amount")
        total_amount = st.number_input("Total amount to invest (₹)", min_value=1000.0,
                                       value=100000.0, step=10000.0, format="%.0f")

        st.markdown("### ⚖️ Your weights for each asset (%)")
        weights = {}
        for col, name in zip(st.columns(4), ASSETS):
            with col:
                weights[name] = st.number_input(ASSET_LABELS[name], min_value=0.0, max_value=100.0,
                                                value=25.0, step=5.0, key=f"w_{name}")

        if mode == "periodic":
            freq_label = st.selectbox("🔁 Rebalance every", list(REBAL_FREQ.keys()), index=1,
                                      key="rebal_freq")
        elif mode == "diff":
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔍 Compare growth with previous")
                lookback_label = st.radio("Look-back period", list(LOOKBACK_OPTIONS.keys()), index=1,
                                          horizontal=True, key="diff_lookback", label_visibility="collapsed")
            with c2:
                st.markdown("#### 🔁 Rebalance every")
                freq_label = st.radio("Rebalance frequency", list(DIFF_REBAL_FREQ.keys()), index=1,
                                      horizontal=True, key="diff_freq", label_visibility="collapsed")

        st.caption(f"Investment period: **{start_date:%b %Y}** to **{end_date:%b %Y}** "
                   "(change it from the sidebar).")
        submitted = st.form_submit_button("🧮 Calculate Investment", use_container_width=True, type="primary")

    if not submitted and "invest_done" not in st.session_state:
        st.info("👆 Choose a strategy, enter your amount and weights, then press **Calculate Investment**.")
        st.stop()
    st.session_state["invest_done"] = True

    # ---------------------------------------------------------
    # Weights check
    # ---------------------------------------------------------
    total_w = sum(weights.values())
    if total_w <= 0:
        st.error("Please give at least one asset a weight above 0%.")
        st.stop()
    if abs(total_w - 100) > 0.01:
        st.info(f"Your weights add up to **{total_w:.1f}%**. They will be scaled to 100%.")

    used = [n for n in ASSETS if weights[n] > 0]
    missing = [n for n in used if asset_dfs.get(n) is None
               or (asset_dfs[n]['Date'] > pd.Timestamp(end_date)).all()]   # no price on/before the end
    if missing:
        st.error("No data in this period for: " + ", ".join(ASSET_LABELS[m] for m in missing)
                 + ". Set its weight to 0% or choose a different period.")
        st.stop()
    w0 = {n: weights[n] / total_w for n in used}

    # ---------------------------------------------------------
    # Price table: one column per asset, missing days filled with the previous price
    # ---------------------------------------------------------
    # Built from the FULL history and then cut to the period, so an asset whose sheet has a gap
    # at the start of the period still has a price (its last known one) on the first day.
    # Before, only rows inside the period were used, so one asset with a gap pushed the first
    # buy (and every rebalance after it) forward — e.g. From Feb 2017 started in 2019.
    p_start, p_end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    full_prices = pd.concat({n: asset_dfs[n].set_index('Date')['Close'] for n in used}, axis=1, sort=True)
    full_prices = full_prices.loc[:p_end]
    trade_days = full_prices.index[full_prices.index >= p_start]
    raw_prices = full_prices.ffill().loc[trade_days]
    first_dates = {n: raw_prices[n].first_valid_index() for n in used}
    common_start = max(first_dates.values())

    if mode == "bh":
        prices = raw_prices.loc[min(first_dates.values()):]
    else:   # rebalancing needs a price for every asset
        prices = raw_prices.loc[common_start:]

    if len(prices) < 2:
        st.error("Not enough price data in this period. Try a longer period.")
        st.stop()

    # Warn when a price is being carried forward over a long gap in the sheet
    last_real = {n: full_prices[n].dropna().index for n in used}
    stale = []
    for n in used:
        real = last_real[n]
        on_first = real[real <= prices.index[0]]
        if len(on_first) and (prices.index[0] - on_first[-1]).days > 40:
            stale.append(f"**{ASSET_LABELS[n]}** (last price before the start is from {on_first[-1]:%d %b %Y})")
        gaps = real[real >= prices.index[0]].to_series().diff().dt.days
        if len(gaps) and gaps.max() > 40:
            stale.append(f"**{ASSET_LABELS[n]}** has no data between {gaps.idxmax() - pd.Timedelta(days=int(gaps.max())):%d %b %Y} "
                         f"and {gaps.idxmax():%d %b %Y}")
    if stale:
        st.warning("⚠️ Gaps in the sheet data, the last known price is used through them: " + "; ".join(stale)
                   + ". Fill these dates in the Google Sheet for exact results.")

    late = [n for n in used if first_dates[n] > prices.index[0] + pd.Timedelta(days=7)]
    if mode == "bh" and late:
        st.info("Bought later because their data starts later (their money waits as cash until then): "
                + ", ".join(f"**{ASSET_LABELS[n]}** on {first_dates[n]:%d %b %Y}" for n in late))
    elif mode != "bh" and common_start > pd.Timestamp(start_date) + pd.Timedelta(days=7):
        st.info(f"Investing starts on **{common_start:%d %b %Y}**, the first date on which all selected "
                "assets have a price.")

    # ---------------------------------------------------------
    # Run the chosen strategy
    # ---------------------------------------------------------
    decisions = None
    if mode == "bh":
        schedule = [(prices.index[0], w0)]
        units_df, values, trade_log = simulate_buy_hold(prices, w0, total_amount)
    elif mode == "periodic":
        schedule = [(d, w0) for d in build_rebalance_dates(prices.index, REBAL_FREQ[freq_label])]
        units_df, values, trade_log = simulate_portfolio(prices, schedule, total_amount)
    else:   # diff
        full_close = {n: asset_dfs[n].set_index('Date')['Close'] for n in used}   # full history for look-back
        schedule, decisions = build_diff_schedule(prices, full_close, w0, DIFF_REBAL_FREQ[freq_label],
                                                  LOOKBACK_OPTIONS[lookback_label])
        units_df, values, trade_log = simulate_portfolio(prices, schedule, total_amount)

    rebal_dates = [d for d, _ in schedule]
    bh_values = None
    if mode != "bh":
        _, bh_values, _ = simulate_buy_hold(prices, w0, total_amount)

    # ---------------------------------------------------------
    # Result numbers
    # ---------------------------------------------------------
    invested = total_amount
    current = values['Total'].iloc[-1]
    profit = current - invested
    ret_pct = (current / invested - 1) * 100
    years = max((values.index[-1] - values.index[0]).days / 365.25, 1e-9)
    cagr = ((current / invested) ** (1 / years) - 1) * 100
    max_dd = (values['Total'] / values['Total'].cummax() - 1).min() * 100

    net_invested = trade_log.groupby('Asset')['Trade Amount (₹)'].sum()
    latest = prices.iloc[-1]
    result = pd.DataFrame([{
        'Asset': ASSET_LABELS[n],
        'Starting Weight (%)': w0[n] * 100,
        'Latest Target Weight (%)': schedule[-1][1][n] * 100,
        'Current Weight (%)': values[n].iloc[-1] / current * 100,
        'Net Invested (₹)': net_invested[n],
        'Units Held': units_df[n].iloc[-1],
        'Latest Date': prices.index[-1].strftime('%d-%m-%Y'),
        'Latest Price (₹)': latest[n],
        'Current Value (₹)': values[n].iloc[-1],
        'Profit / Loss (₹)': values[n].iloc[-1] - net_invested[n],
        'Return (%)': (values[n].iloc[-1] / net_invested[n] - 1) * 100 if net_invested[n] > 0 else float('nan'),
    } for n in used])

    # ---------------------------------------------------------
    # Summary cards
    # ---------------------------------------------------------
    st.markdown("## 📊 Results")
    if mode == "bh":
        strat_text = STRATEGIES[mode]
        detail = "bought once, never rebalanced"
    else:
        strat_text = STRATEGIES[mode] + f" · {freq_label.lower()}"
        if mode == "diff":
            strat_text += f" · compare previous {lookback_label}"
        n_reb = len(schedule) - 1
        detail = f"first buy + {n_reb} rebalance{'s' if n_reb != 1 else ''}"
    st.caption(f"Strategy: **{strat_text}**  ·  {values.index[0]:%d %b %Y} to {values.index[-1]:%d %b %Y}  ·  {detail}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card("Total Invested", f"₹{invested:,.0f}", is_positive=True)
    with c2:
        render_metric_card("Current Value", f"₹{current:,.0f}", f"{ret_pct:+.2f}%", is_positive=profit >= 0)
    with c3:
        render_metric_card("Profit / Loss", f"₹{profit:+,.0f}", f"{ret_pct:+.2f}%", is_positive=profit >= 0)
    with c4:
        render_metric_card("CAGR (Whole Period)", f"{cagr:.2f}%", f"over {years:.2f} years", is_positive=cagr >= 0)
    with c5:
        render_metric_card("Max Drawdown", f"{max_dd:.2f}%", is_positive=False)

    if bh_values is not None:
        bh_cur = bh_values['Total'].iloc[-1]
        bh_cagr = ((bh_cur / invested) ** (1 / years) - 1) * 100
        bh_dd = (bh_values['Total'] / bh_values['Total'].cummax() - 1).min() * 100
        diff = current - bh_cur
        st.markdown("#### ⚖️ Compared with Buy & Hold (your weights, never rebalanced)")
        cmp_df = pd.DataFrame([
            {'Strategy': strat_text, 'Final Value (₹)': current, 'Return (%)': ret_pct,
             'CAGR (%)': cagr, 'Max Drawdown (%)': max_dd},
            {'Strategy': STRATEGIES['bh'], 'Final Value (₹)': bh_cur, 'Return (%)': (bh_cur / invested - 1) * 100,
             'CAGR (%)': bh_cagr, 'Max Drawdown (%)': bh_dd},
        ])
        st.dataframe(cmp_df.style.format({'Final Value (₹)': '₹{:,.2f}', 'Return (%)': '{:+.2f}%',
                                          'CAGR (%)': '{:.2f}%', 'Max Drawdown (%)': '{:.2f}%'}),
                     use_container_width=True, hide_index=True)
        st.caption(f"Your strategy ended **₹{diff:+,.0f}** {'ahead of' if diff >= 0 else 'behind'} buy & hold.")

    # ---------------------------------------------------------
    # Different Weights: how each date's weights were decided
    # ---------------------------------------------------------
    if decisions is not None:
        _render_weight_decisions(decisions, lookback_label)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Charts
    # ---------------------------------------------------------
    left, right = st.columns([2, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=values.index, y=values['Total'], name='Total Value',
                                 line=dict(color='#A855F7', width=3)))
        if bh_values is not None:
            fig.add_trace(go.Scatter(x=bh_values.index, y=bh_values['Total'], name='Buy & Hold (compare)',
                                     line=dict(color='#94A3B8', width=1.5, dash='dash')))
        for n in used:
            fig.add_trace(go.Scatter(x=values.index, y=values[n], name=ASSET_LABELS[n],
                                     line=dict(color=COLORS[n], width=1.2, dash='dot'), visible='legendonly'))
        if len(rebal_dates) > 1:
            fig.add_trace(go.Scatter(x=rebal_dates[1:], y=values['Total'].loc[rebal_dates[1:]], mode='markers',
                                     name='Rebalance', marker=dict(color='#F8FAFC', size=7, symbol='diamond')))
        fig.add_hline(y=invested, line_dash="dash", line_color="#64748B",
                      annotation_text=f"Invested ₹{invested:,.0f}", annotation_position="top left")
        fig.update_layout(**PLOT_LAYOUT, title="Investment Value Over Time (₹)", height=450,
                          hovermode='x unified', yaxis=dict(tickprefix="₹"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Click an asset in the legend to show its own value line.")

    with right:
        pie_df = pd.DataFrame({'Asset': [ASSET_LABELS[n] for n in used],
                               'Current Value': [values[n].iloc[-1] for n in used]})
        fig_pie = px.pie(pie_df, names='Asset', values='Current Value', hole=0.5,
                         color='Asset', color_discrete_map={ASSET_LABELS[k]: v for k, v in COLORS.items()},
                         title="Current Allocation")
        fig_pie.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', height=450,
                              margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation='h'))
        st.plotly_chart(fig_pie, use_container_width=True)

    wt = values[used].div(values['Total'], axis=0) * 100
    fig_w = go.Figure()
    for n in used:
        fig_w.add_trace(go.Scatter(x=wt.index, y=wt[n], name=ASSET_LABELS[n], stackgroup='one',
                                   line=dict(width=0.5, color=COLORS[n])))
    fig_w.update_layout(**PLOT_LAYOUT, title="Actual Weight of Each Asset Over Time (%)", height=350,
                        hovermode='x unified', yaxis=dict(ticksuffix="%", range=[0, 100]))
    st.plotly_chart(fig_w, use_container_width=True)
    st.caption("Weights drift as prices move. Each rebalance sets them to that date's target.")

    # ---------------------------------------------------------
    # Breakdown table
    # ---------------------------------------------------------
    st.markdown("### 📋 Asset-wise Breakdown")
    total_row = pd.DataFrame([{
        'Asset': '🧾 TOTAL', 'Starting Weight (%)': 100.0, 'Latest Target Weight (%)': 100.0,
        'Current Weight (%)': 100.0, 'Net Invested (₹)': invested, 'Units Held': None, 'Latest Date': '',
        'Latest Price (₹)': None, 'Current Value (₹)': current, 'Profit / Loss (₹)': profit, 'Return (%)': ret_pct,
    }])
    table = pd.concat([result, total_row], ignore_index=True)
    if mode != "diff":
        table = table.drop(columns=['Latest Target Weight (%)'])
    styler = table.style.format({
        'Starting Weight (%)': '{:.2f}%', 'Latest Target Weight (%)': '{:.2f}%', 'Current Weight (%)': '{:.2f}%',
        'Net Invested (₹)': '₹{:,.2f}', 'Units Held': '{:,.4f}', 'Latest Price (₹)': '₹{:,.2f}',
        'Current Value (₹)': '₹{:,.2f}', 'Profit / Loss (₹)': '₹{:+,.2f}', 'Return (%)': '{:+.2f}%'
    }, na_rep='—', subset=[c for c in table.columns if c not in ('Asset', 'Latest Date')])
    st.dataframe(style_change_cols(styler, ['Profit / Loss (₹)', 'Return (%)']),
                 use_container_width=True, hide_index=True)
    if mode != "bh":
        st.caption("**Net Invested** = money put into the asset minus money taken out of it at rebalances. "
                   "Profit / Loss = Current Value − Net Invested.")

    # ---------------------------------------------------------
    # Rebalance data (table + CSV)
    # ---------------------------------------------------------
    _render_rebalance_data(schedule, trade_log, used, prices, units_df, values, current,
                           start_date, end_date, decisions, lookback_label)


def _render_weight_decisions(decisions, lookback_label):
    """Table showing, for every rebalance date, each asset's growth, its rank and the weight it got."""
    st.markdown(f"#### 📐 How the weights were decided (growth over previous {lookback_label})")
    st.caption("Rank 1 = highest growth → gets the smallest of your weights. "
               "The lowest growth gets the biggest weight. "
               "If an asset has no data that far back, it keeps your weight.")

    show = decisions.copy()
    show['Date'] = show['Date'].dt.strftime('%d-%m-%Y')
    show['Compare From'] = show['Compare From'].apply(lambda d: d.strftime('%d-%m-%Y') if pd.notna(d) else '—')
    show['Asset'] = show['Asset'].map(_short)
    show = show.drop(columns=['Rebalance #'])
    num_cols = ['Price Then (₹)', 'Price Now (₹)', 'Growth (%)', 'Growth Rank', 'Your Weight (%)', 'New Weight (%)']
    styler = show.style.format({
        'Price Then (₹)': '₹{:,.2f}', 'Price Now (₹)': '₹{:,.2f}', 'Growth (%)': '{:+.2f}%',
        'Growth Rank': '{:.0f}', 'Your Weight (%)': '{:.2f}%', 'New Weight (%)': '{:.2f}%',
    }, na_rep='—', subset=num_cols)
    st.dataframe(style_change_cols(styler, ['Growth (%)']), use_container_width=True, hide_index=True,
                 height=min(420, 38 + 35 * len(show)))
    st.download_button("⬇️ Download Weight Decisions (CSV)",
                       show.to_csv(index=False).encode('utf-8-sig'),
                       file_name="weight_decisions.csv", mime="text/csv")


def _render_rebalance_data(schedule, trade_log, used, prices, units_df, values, current,
                           start_date, end_date, decisions=None, lookback_label=None):
    st.markdown("### 📥 Rebalance Data")

    # Different Weights: previous price + % change for every (rebalance #, asset)
    cmp = {}
    if decisions is not None:
        for _, d in decisions.iterrows():
            cmp[(d['Rebalance #'], d['Asset'])] = d
    cf_col = f'Compare From ({lookback_label})'

    rows = []
    for k in range(len(schedule)):
        step = trade_log[trade_log['Rebalance #'] == k + 1]
        for n in used:
            r = step[step['Asset'] == n].iloc[0]
            pv = r['Portfolio Value (₹)']
            row = {
                'Date': r['Date'].strftime('%d-%m-%Y'),
                'Type': 'First Buy' if k == 0 else 'Rebalance',
                'Asset': _short(n),
                'Price (₹)': round(r['Price (₹)'], 2),
            }
            if decisions is not None:
                d = cmp[(k + 1, n)]
                row[cf_col] = d['Compare From'].strftime('%d-%m-%Y') if pd.notna(d['Compare From']) else None
                row['Prev Price (₹)'] = round(d['Price Then (₹)'], 2) if pd.notna(d['Price Then (₹)']) else None
                row['Change vs Prev (%)'] = round(d['Growth (%)'], 2) if pd.notna(d['Growth (%)']) else None
                row['Growth Rank'] = int(d['Growth Rank']) if pd.notna(d['Growth Rank']) else None
            row.update({
                'Units Before': round(r['Units Before'], 4),
                'Value Before (₹)': round(r['Value Before (₹)'], 2),
                'Weight Before (%)': round(r['Value Before (₹)'] / pv * 100, 2) if k > 0 else 0.0,
                'Units Bought / Sold': round(r['Units Bought (+) / Sold (−)'], 4),
                'Units After': round(r['Units After'], 4),
                'Value After (₹)': round(r['Target Value (₹)'], 2),
                'Weight After (%)': round(r['Target Weight (%)'], 2),
                'Portfolio Value (₹)': round(pv, 2),
            })
            rows.append(row)
    end_dt = prices.index[-1]
    for n in used:
        end_row = {'Date': end_dt.strftime('%d-%m-%Y'), 'Type': 'End Value', 'Asset': _short(n),
                   'Price (₹)': round(prices[n].iloc[-1], 2)}
        if decisions is not None:
            end_row.update({cf_col: None, 'Prev Price (₹)': None, 'Change vs Prev (%)': None, 'Growth Rank': None})
        rows.append(end_row)
        end_row.update({
            'Units Before': round(units_df[n].iloc[-1], 4),
            'Value Before (₹)': round(values[n].iloc[-1], 2),
            'Weight Before (%)': round(values[n].iloc[-1] / current * 100, 2),
            'Units Bought / Sold': None, 'Units After': None, 'Value After (₹)': None, 'Weight After (%)': None,
            'Portfolio Value (₹)': round(current, 2),
        })
    reb = pd.DataFrame(rows)
    st.caption("On each rebalance date: the weights before (drifted) and after (the new target), with the "
               "price and units for every asset. The last rows show the value at the end of the period.")
    if decisions is not None:
        st.caption(f"**Prev Price** = the asset's price {lookback_label} before that date (on the "
                   f"*Compare From* date). **Change vs Prev** = % growth from Prev Price to Price. "
                   "Rank 1 = highest growth.")
    st.dataframe(reb, use_container_width=True, height=320, hide_index=True)
    st.download_button("⬇️ Download Rebalance Data (CSV)", reb.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f"rebalance_data_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv",
                       mime="text/csv", type="primary")