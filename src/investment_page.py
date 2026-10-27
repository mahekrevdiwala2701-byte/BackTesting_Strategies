"""💰 Calculated Investment page (UI only: the maths lives in src/strategies.py)."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.ui import (
    panel,
    PLOT_LAYOUT, STRAT_COLORS, VIOLET, CYAN, EMERALD, AMBER, BLUE, BAD, FAINT, TEXT, MUTED,
    hero, section, kpi_row, tone_of, note, explain, chart, hex_rgba, inr, inr_short, pct,
    sidebar_nav, sidebar_footer,
)
from src.asset_data import ASSETS, ASSET_LABELS, ASSET_SHORT, ASSET_ICONS, COLORS, style_change_cols, blank_missing
from src.period_picker import period_selector
from src.strategies import (
    STRATEGIES, REBAL_FREQ, DIFF_REBAL_FREQ, LOOKBACK_OPTIONS,
    build_rebalance_dates, simulate_portfolio, simulate_buy_hold, build_diff_schedule,
)


def _extend_lookbacks(options):
    """Add '2 Years' and '2.5 Years' to the look-back choices, in the same unit (months)
    that strategies.py uses."""
    opts = dict(options)
    for label, months in {"2 Years": 24, "2.5 Years": 30}.items():
        opts.setdefault(label, months)
    return opts


LOOKBACKS = _extend_lookbacks(LOOKBACK_OPTIONS)

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


def _short(name):
    return ASSET_SHORT[name]


def render_investment_page(asset_dfs, fetch_errors):
    # ---------------- Sidebar ----------------
    sidebar_nav("invest")
    start_date, end_date = period_selector(asset_dfs, default_preset="Custom Range", key="invest")
    sidebar_footer(fetch_errors, key="invest")

    # ---------------- Hero ----------------
    hero("Calculated Investment",
         "Enter an amount, split it across GoldBEES, Nifty 50, MidCap and SmallCap, and see exactly what it "
         "would be worth today with buy & hold, regular rebalancing, or growth-based weights.",
         page="invest", eyebrow="Backtest your split",
         chips=[f"📅 {start_date:%d %b %Y} → {end_date:%d %b %Y}", "⚙️ Change the period in the sidebar"])

    # ---------------- Step 1: strategy ----------------
    section("Step 1 · Choose a strategy", "Changes the options below instantly", "🧭", VIOLET)
    mode = st.radio("Strategy", list(STRATEGIES.keys()), format_func=lambda k: STRATEGIES[k],
                    horizontal=True, key="invest_mode", label_visibility="collapsed")
    s_color = STRAT_COLORS[mode]
    explain(STRATEGY_HELP[mode], s_color)

    # ---------------- Step 2: inputs ----------------
    section("Step 2 · Amount & weights", "Weights that don't add up to 100% are scaled automatically", "💵",
            EMERALD)
    freq_label, lookback_label = None, None
    with st.form("investment_form"):
        a1, a2 = st.columns([1.2, 2])
        with a1:
            total_amount = st.number_input("💰 Total amount to invest (₹)", min_value=1000.0, value=100000.0,
                                           step=10000.0, format="%.0f")
            st.caption(f"= {inr_short(total_amount)}")
        with a2:
            if mode == "periodic":
                freq_label = st.selectbox("🔁 Rebalance every", list(REBAL_FREQ.keys()), index=1, key="rebal_freq")
            elif mode == "diff":
                lookback_label = st.radio("🔍 Compare growth over the previous", list(LOOKBACKS.keys()), index=1,
                                          horizontal=True, key="diff_lookback")
                freq_label = st.radio("🔁 Rebalance every", list(DIFF_REBAL_FREQ.keys()), index=1,
                                      horizontal=True, key="diff_freq")
            else:
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                note("Buy & Hold has no extra settings: money is invested once on the first day.", CYAN, "🧺")

        st.markdown("**⚖️ Weight of each asset (%)**")
        weights = {}
        for col, name in zip(st.columns(4), ASSETS):
            with col:
                weights[name] = st.number_input(f"{ASSET_ICONS[name]} {ASSET_SHORT[name]}", min_value=0.0,
                                                max_value=100.0, value=25.0, step=5.0, key=f"w_{name}")
        submitted = st.form_submit_button("🧮  Calculate investment", width="stretch", type="primary")

    if not submitted and "invest_done" not in st.session_state:
        note("Choose a strategy, enter your amount and weights, then press <b>Calculate investment</b>.", VIOLET, "👆")
        st.stop()
    st.session_state["invest_done"] = True

    # ---------------- Weights check ----------------
    total_w = sum(weights.values())
    if total_w <= 0:
        note("Please give at least one asset a weight above 0%.", BAD, "⛔")
        st.stop()
    if abs(total_w - 100) > 0.01:
        note(f"Your weights add up to <b>{total_w:.1f}%</b>, so they were scaled to 100%.", AMBER, "⚖️")

    used = [n for n in ASSETS if weights[n] > 0]
    missing = [n for n in used if asset_dfs.get(n) is None
               or (asset_dfs[n]['Date'] > pd.Timestamp(end_date)).all()]   # no price on/before the end
    if missing:
        note("No data in this period for: <b>" + ", ".join(ASSET_LABELS[m] for m in missing)
             + "</b>. Set its weight to 0% or choose a different period.", BAD, "⛔")
        st.stop()
    w0 = {n: weights[n] / total_w for n in used}

    # ---------------- Price table ----------------
    # Built from the FULL history and then cut to the period, so an asset whose sheet has a gap
    # at the start of the period still has a price (its last known one) on the first day.
    p_start, p_end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    full_prices = pd.concat({n: asset_dfs[n].set_index('Date')['Close'] for n in used}, axis=1, sort=True)
    full_prices = full_prices.loc[:p_end]
    trade_days = full_prices.index[full_prices.index >= p_start]
    raw_prices = full_prices.ffill().loc[trade_days]
    first_dates = {n: raw_prices[n].first_valid_index() for n in used}
    if any(d is None for d in first_dates.values()):
        note("Not enough price data in this period. Try a longer period.", BAD, "⛔")
        st.stop()
    common_start = max(first_dates.values())
    prices = raw_prices.loc[min(first_dates.values()):] if mode == "bh" else raw_prices.loc[common_start:]

    if len(prices) < 2:
        note("Not enough price data in this period. Try a longer period.", BAD, "⛔")
        st.stop()

    # Warn when a price is carried forward over a long gap in the sheet
    stale = []
    for n in used:
        real = full_prices[n].dropna().index
        on_first = real[real <= prices.index[0]]
        if len(on_first) and (prices.index[0] - on_first[-1]).days > 40:
            stale.append(f"<b>{ASSET_SHORT[n]}</b> (last price before the start is from {on_first[-1]:%d %b %Y})")
        gaps = real[real >= prices.index[0]].to_series().diff().dt.days
        if len(gaps) and gaps.max() > 40:
            stale.append(f"<b>{ASSET_SHORT[n]}</b> has no data between "
                         f"{gaps.idxmax() - pd.Timedelta(days=int(gaps.max())):%d %b %Y} and {gaps.idxmax():%d %b %Y}")
    if stale:
        note("Gaps in the sheet data; the last known price is used through them: " + "; ".join(stale)
             + ". Fill these dates in the Google Sheet for exact results.", AMBER, "⚠️")

    late = [n for n in used if first_dates[n] > prices.index[0] + pd.Timedelta(days=7)]
    if mode == "bh" and late:
        note("Bought later because their data starts later (their money waits as cash until then): "
             + ", ".join(f"<b>{ASSET_SHORT[n]}</b> on {first_dates[n]:%d %b %Y}" for n in late), BLUE)
    elif mode != "bh" and common_start > pd.Timestamp(start_date) + pd.Timedelta(days=7):
        note(f"Investing starts on <b>{common_start:%d %b %Y}</b>, the first date on which all selected assets "
             "have a price.", BLUE)

    # ---------------- Run the strategy ----------------
    decisions = None
    if mode == "bh":
        schedule = [(prices.index[0], w0)]
        units_df, values, trade_log = simulate_buy_hold(prices, w0, total_amount)
    elif mode == "periodic":
        schedule = [(d, w0) for d in build_rebalance_dates(prices.index, REBAL_FREQ[freq_label])]
        units_df, values, trade_log = simulate_portfolio(prices, schedule, total_amount)
    else:
        full_close = {n: asset_dfs[n].set_index('Date')['Close'] for n in used}   # full history for look-back
        schedule, decisions = build_diff_schedule(prices, full_close, w0, DIFF_REBAL_FREQ[freq_label],
                                                  LOOKBACKS[lookback_label])
        units_df, values, trade_log = simulate_portfolio(prices, schedule, total_amount)

    rebal_dates = [d for d, _ in schedule]
    bh_values = None
    if mode != "bh":
        _, bh_values, _ = simulate_buy_hold(prices, w0, total_amount)

    # ---------------- Result numbers ----------------
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

    # ---------------- Results ----------------
    if mode == "bh":
        strat_text, detail = STRATEGIES[mode], "bought once, never rebalanced"
    else:
        strat_text = STRATEGIES[mode] + f" · {freq_label.lower()}"
        if mode == "diff":
            strat_text += f" · compare previous {lookback_label}"
        n_reb = len(schedule) - 1
        detail = f"first buy + {n_reb} rebalance{'s' if n_reb != 1 else ''}"

    section("Results", f"{values.index[0]:%d %b %Y} → {values.index[-1]:%d %b %Y} · {detail}", "📊", s_color)
    kpi_row(_result_cards(invested, current, profit, ret_pct, cagr, years, max_dd, s_color))

    if bh_values is not None:
        bh_cur = bh_values['Total'].iloc[-1]
        bh_cagr = ((bh_cur / invested) ** (1 / years) - 1) * 100
        bh_dd = (bh_values['Total'] / bh_values['Total'].cummax() - 1).min() * 100
        bh_ret = (bh_cur / invested - 1) * 100
        diff = current - bh_cur

        section("Head-to-head with Buy & Hold", "Same amount, same weights, never rebalanced", "🤝", CYAN)
        kpi_row(_result_cards(invested, bh_cur, bh_cur - invested, bh_ret, bh_cagr, years, bh_dd, CYAN))
        verdict_color = EMERALD if diff >= 0 else BAD
        note(f"Your strategy ended <b>{inr(diff, sign=True)}</b> "
             f"{'ahead of' if diff >= 0 else 'behind'} buy & hold "
             f"(CAGR {cagr:.2f}% vs {bh_cagr:.2f}%, max drawdown {max_dd:.1f}% vs {bh_dd:.1f}%).",
             verdict_color, "🏆" if diff >= 0 else "📉")

    if decisions is not None:
        _render_weight_decisions(decisions, lookback_label)

    # ---------------- Charts ----------------
    section("Growth over time", "Your total next to buy & hold, and how each holding moved", "📈", BLUE)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.55, 0.45],
                        subplot_titles=("Total portfolio value (₹)", "Value of each holding inside it (₹)"))
    fig.add_trace(go.Scatter(x=values.index, y=values['Total'], name='Total portfolio',
                             line=dict(color=s_color, width=3), fill='tozeroy', fillcolor=hex_rgba(s_color, 0.10),
                             hovertemplate="₹%{y:,.0f}"), row=1, col=1)
    if bh_values is not None:
        fig.add_trace(go.Scatter(x=bh_values.index, y=bh_values['Total'], name='Buy & Hold (compare)',
                                 line=dict(color=MUTED, width=1.6, dash='dash'), hovertemplate="₹%{y:,.0f}"),
                      row=1, col=1)
    if len(rebal_dates) > 1:
        fig.add_trace(go.Scatter(x=rebal_dates[1:], y=values['Total'].loc[rebal_dates[1:]], mode='markers',
                                 name='Rebalance', hoverinfo='skip',
                                 marker=dict(color=TEXT, size=9, symbol='diamond', line=dict(color=s_color, width=1.5))),
                      row=1, col=1)
    fig.add_hline(y=invested, line_dash="dot", line_color=FAINT, annotation_text=f"Invested {inr(invested)}",
                  annotation_position="bottom right", annotation_font_color=MUTED, row=1, col=1)
    for n in used:
        fig.add_trace(go.Scatter(x=values.index, y=values[n], name=ASSET_SHORT[n],
                                 line=dict(color=COLORS[n], width=2), hovertemplate="₹%{y:,.0f}"), row=2, col=1)
    lo_t = min(values['Total'].min(), invested, bh_values['Total'].min() if bh_values is not None else invested)
    hi_t = max(values['Total'].max(), invested, bh_values['Total'].max() if bh_values is not None else invested)
    pad = (hi_t - lo_t) * 0.08 or invested * 0.02
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
    fig.update_yaxes(range=[lo_t - pad, hi_t + pad], row=1, col=1)
    fig.update_annotations(font=dict(size=13, color=MUTED))
    fig.update_layout(**PLOT_LAYOUT, height=640, hovermode='x unified',
                      legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='right', x=1))
    with panel():
        chart(fig, key="inv_main")
    st.caption("Bottom panel: the ₹ value of each asset inside your portfolio; together they add up to the total "
               "above. Click a legend item to hide / show it.")

    wt = values[used].div(values['Total'], axis=0) * 100
    fig_w = go.Figure()
    for n in used:
        fig_w.add_trace(go.Scatter(x=wt.index, y=wt[n], name=ASSET_SHORT[n], stackgroup='one',
                                   line=dict(width=0.6, color=COLORS[n]), hovertemplate="%{y:.1f}%"))
    fig_w.update_layout(**PLOT_LAYOUT, title="Actual weight of each asset over time (%)", height=340,
                        hovermode='x unified', yaxis=dict(ticksuffix="%", range=[0, 100]),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    with panel():
        chart(fig_w, key="inv_weights")

    # ---------------- Breakdown table ----------------
    section("Asset-wise breakdown", "Net invested = money put in minus money taken out at rebalances", "📋",
            AMBER)
    total_row = pd.DataFrame([{
        'Asset': '🧾 TOTAL', 'Starting Weight (%)': 100.0, 'Latest Target Weight (%)': 100.0,
        'Current Weight (%)': 100.0, 'Net Invested (₹)': invested, 'Units Held': float('nan'), 'Latest Date': '',
        'Latest Price (₹)': float('nan'), 'Current Value (₹)': current, 'Profit / Loss (₹)': profit, 'Return (%)': ret_pct,
    }])
    table = pd.concat([result, total_row], ignore_index=True)
    if mode != "diff":
        table = table.drop(columns=['Latest Target Weight (%)'])
    money2 = lambda v: inr(v, 2)  # noqa: E731
    tfmt = {
        'Starting Weight (%)': '{:.2f}%', 'Latest Target Weight (%)': '{:.2f}%', 'Current Weight (%)': '{:.2f}%',
        'Net Invested (₹)': money2, 'Units Held': '{:,.4f}', 'Latest Price (₹)': money2,
        'Current Value (₹)': money2, 'Profit / Loss (₹)': lambda v: inr(v, 2, sign=True), 'Return (%)': '{:+.2f}%'}
    tfmt = {k: v for k, v in tfmt.items() if k in table.columns}
    table, tfmt = blank_missing(table, tfmt)
    styler = table.style.format(tfmt)
    st.dataframe(style_change_cols(styler, ['Profit / Loss (₹)', 'Return (%)']), width="stretch", hide_index=True)

    _render_rebalance_data(schedule, trade_log, used, prices, units_df, values, current,
                           start_date, end_date, decisions, lookback_label)


def _result_cards(invested, current, profit, ret_pct, cagr, years, max_dd, accent):
    return [
        dict(label="Total invested", value=inr(invested), icon="💵", accent=BLUE),
        dict(label="Current value", value=inr(current), delta=pct(ret_pct), tone=tone_of(profit), icon="💎",
             accent=accent),
        dict(label="Profit / loss", value=inr(profit, sign=True), delta=pct(ret_pct), tone=tone_of(profit),
             icon="💰", accent=EMERALD if profit >= 0 else BAD),
        dict(label="CAGR", value=pct(cagr, sign=False), delta=f"over {years:.2f} years", icon="🚀", accent=VIOLET),
        dict(label="Max drawdown", value=pct(max_dd, sign=False), delta="worst fall from a peak", tone="bad",
             icon="📉", accent=BAD),
    ]


def _render_weight_decisions(decisions, lookback_label):
    """For every rebalance date: each asset's growth, its rank and the weight it got."""
    section("How the weights were decided", f"Growth over the previous {lookback_label}. Rank 1 = highest "
            "growth → smallest weight. No older data → keeps your weight.", "📐", STRAT_COLORS['diff'])
    show = decisions.copy()
    show['Date'] = show['Date'].dt.strftime('%d-%m-%Y')
    show['Compare From'] = show['Compare From'].apply(lambda d: d.strftime('%d-%m-%Y') if pd.notna(d) else '—')
    show['Asset'] = show['Asset'].map(_short)
    show = show.drop(columns=['Rebalance #'])
    dfmt = {'Price Then (₹)': lambda v: inr(v, 2), 'Price Now (₹)': lambda v: inr(v, 2), 'Growth (%)': '{:+.2f}%',
            'Growth Rank': '{:.0f}', 'Your Weight (%)': '{:.2f}%', 'New Weight (%)': '{:.2f}%'}
    show, dfmt = blank_missing(show, dfmt)
    styler = show.style.format(dfmt)
    st.dataframe(style_change_cols(styler, ['Growth (%)']), width="stretch", hide_index=True,
                 height=min(420, 38 + 35 * len(show)))
    st.download_button("⬇️  Download weight decisions (CSV)", show.to_csv(index=False).encode('utf-8-sig'),
                       file_name="weight_decisions.csv", mime="text/csv")


def _render_rebalance_data(schedule, trade_log, used, prices, units_df, values, current,
                           start_date, end_date, decisions=None, lookback_label=None):
    section("Rebalance log", "Weights before (drifted) and after (new target) on every rebalance date, plus "
            "the value at the end", "📥", CYAN)

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
        end_row.update({
            'Units Before': round(units_df[n].iloc[-1], 4),
            'Value Before (₹)': round(values[n].iloc[-1], 2),
            'Weight Before (%)': round(values[n].iloc[-1] / current * 100, 2),
            'Units Bought / Sold': None, 'Units After': None, 'Value After (₹)': None, 'Weight After (%)': None,
            'Portfolio Value (₹)': round(current, 2),
        })
        rows.append(end_row)
    reb = pd.DataFrame(rows)
    if decisions is not None:
        st.caption(f"**Prev Price** = the asset's price {lookback_label} before that date (on the *Compare From* "
                   "date). **Change vs Prev** = % growth from Prev Price to Price. Rank 1 = highest growth.")
    st.dataframe(blank_missing(reb)[0], width="stretch", height=340, hide_index=True)
    st.download_button("⬇️  Download rebalance log (CSV)", reb.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f"rebalance_data_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv", mime="text/csv")