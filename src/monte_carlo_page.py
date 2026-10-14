"""🎲 Monte Carlo page.

Same 3 options as the Investment page (Buy & Hold / Rebalance / Different Weights).
Pick one and its dashboard appears: every possible setting of that strategy is simulated
automatically, so there is nothing to configure on the page itself.
The maths lives in src/monte_carlo.py.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import render_metric_card
from src.asset_data import ASSETS, ASSET_LABELS, COLORS
from src.strategies import STRATEGIES, REBAL_FREQ, DIFF_REBAL_FREQ, LOOKBACK_OPTIONS
from src import monte_carlo as mc

# Fixed simulation settings (no options on the page)
N_SIMS = 5000          # simulated futures
BLOCK = 12             # chunks of 12 real months (keeps yearly trends & crashes intact)
SEED = 42              # same seed = same result every time
AMOUNT = 100000.0      # ₹1,00,000
YEARS = 10             # horizon
COMBO_SIMS = 1000      # Buy & Hold / Rebalance: futures per weight combination (287 combinations)
DIFF_SIMS = 2000       # Different Weights: futures per weight set & setting (24 sets x 6 settings)
WEIGHT_STEP = 10       # Different Weights: every split in 10% steps

# One colour per strategy (checked for colour-blind safety on the dark background)
STRAT_COLORS = {'bh': '#0891B2', 'periodic': '#A855F7', 'diff': '#D97706'}
BAD, GOOD, NEUTRAL, INK, MUTED = '#EF4444', '#10B981', '#94A3B8', '#F8FAFC', '#64748B'
GRID = 'rgba(148,163,184,0.12)'

# Same look as the app's dark charts, but without copying plotly's whole dark template for every chart
MC_LAYOUT = dict(template='none', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15, 23, 42, 0.6)',
                 font=dict(color='#CBD5E1', family='Inter, sans-serif', size=12),
                 margin=dict(l=20, r=20, t=40, b=20),
                 hoverlabel=dict(bgcolor='#1E293B', bordercolor='#334155', font=dict(color='#F8FAFC')))

STRATEGY_HELP = {
    "bh": "Money is split once and never touched again. Not your weights: EVERY possible split of GoldBEES / "
          "LargeCap / MidCap / SmallCap in 10% steps (100/0/0/0 … 0/0/0/100, plus 25/25/25/25 = 287 "
          "combinations) is simulated on the same futures.",
    "periodic": "Brought back to the same weights every month/quarter/half-year/year. Not your weights: EVERY "
                "possible split in 10% steps (287 combinations) × ALL 4 rebalance frequencies (= 1,148 "
                "combinations) is simulated on the same futures.",
    "diff": "On each rebalance, the asset that grew the MOST gets the SMALLEST weight. Here it is NOT your "
            "weights: EVERY possible split in 10% steps (100/0/0/0, 90/10/0/0 … 30/30/20/20, plus 25/25/25/25 "
            "= 24 weight sets) × ALL 6 settings (half-yearly / yearly rebalance × 6 months / 1 year / 1.5 years "
            "look-back) is simulated on the same futures.",
}


def _inr(v):
    """₹ in lakh/crore style: ₹3.45 L, ₹1.20 Cr."""
    if abs(v) >= 1e7:
        return f"₹{v / 1e7:,.2f} Cr"
    if abs(v) >= 1e5:
        return f"₹{v / 1e5:,.2f} L"
    return f"₹{v:,.0f}"


@st.cache_data(show_spinner=False, persist="disk")
def _simulate_combos(rets, mode):
    settings = {'Buy & Hold': None} if mode == 'bh' else dict(REBAL_FREQ)
    return mc.run_combos(rets, AMOUNT, YEARS, COMBO_SIMS, min(BLOCK, len(rets)), settings, WEIGHT_STEP, SEED)


@st.cache_data(show_spinner=False, persist="disk")
def _simulate_diff_all(rets):
    return mc.run_diff_all_weights(rets, AMOUNT, YEARS, DIFF_SIMS, min(BLOCK, len(rets)),
                                   DIFF_REBAL_FREQ, LOOKBACK_OPTIONS, WEIGHT_STEP, SEED)


def _kde(x, grid):
    """Smooth bell-curve line (Gaussian kernel density)."""
    x = np.asarray(x, dtype=float)
    std = x.std()
    if std == 0 or len(x) < 2:
        return np.zeros_like(grid)
    bw = 1.06 * std * len(x) ** (-1 / 5)
    if len(x) > 3000:
        x = np.random.default_rng(0).choice(x, 3000, replace=False)
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(x) * bw * np.sqrt(2 * np.pi))


# =========================================================
# Page
# =========================================================
def render_monte_carlo_page(asset_dfs, fetch_errors):
    # ---------------- Sidebar ----------------
    st.sidebar.markdown("## ⚙️ Control Panel")
    if st.sidebar.button("⬅️ Back to Dashboard", use_container_width=True, type="primary", key="mc_back"):
        st.session_state["view"] = "dashboard"
        st.rerun()
    if st.sidebar.button("💰 Calculated Investment", use_container_width=True, key="mc_to_invest"):
        st.session_state["view"] = "invest"
        st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Live Sheet Data", use_container_width=True, key="mc_refresh"):
        st.cache_data.clear()
        st.rerun()
    if fetch_errors:
        with st.sidebar.expander("⚠️ Data load issues", expanded=False):
            for name, err in fetch_errors.items():
                st.write(f"**{name}:** {err}")

    # ---------------- Header ----------------
    st.markdown('<div class="header-title">Monte Carlo Simulation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="header-subtitle">Thousands of possible futures, built from real past months of '
        'GoldBEES, LargeCap, MidCap & SmallCap. Pick a strategy to see every possible outcome: the bell '
        'curve, the mean, the median, the top 5% and the bottom 5%.</div>',
        unsafe_allow_html=True)

    # ---------------- Data (fixed: ₹1,00,000, 10 years, all history) ----------------
    amount, years = AMOUNT, YEARS
    used = [n for n in ASSETS if n in asset_dfs]
    if not used:
        st.error("No asset data loaded.")
        st.stop()
    rets, dropped = mc.monthly_returns(asset_dfs, used)
    if len(rets) < 24:
        st.error(f"Only {len(rets)} months where all assets have data. At least 2 years are needed.")
        st.stop()

    # ---------------- The 3 options ----------------
    st.markdown("### 🧭 Investment Strategy")
    mode = st.radio("Strategy", list(STRATEGIES.keys()), format_func=lambda k: STRATEGIES[k],
                    horizontal=True, key="mc_mode", label_visibility="collapsed")
    st.caption(STRATEGY_HELP[mode])

    if dropped:
        st.warning("⚠️ Skipped months with a move above ±60% (looks like a data error in the sheet): "
                   + ", ".join(d.strftime('%b %Y') for d in dropped))
    hist = f"**{len(rets)} real months** ({rets.index[0]:%b %Y} – {rets.index[-1]:%b %Y})"
    n_combos = len(mc.all_weight_combos(len(used), WEIGHT_STEP))
    if mode == 'diff':
        st.caption(f"Amount **{_inr(amount)}** · **{years} years** · weights: **all 24 weight sets** · "
                   f"**{DIFF_SIMS:,}** futures for each of the 144 combinations "
                   f"(= {DIFF_SIMS * 144:,} outcomes) built from {hist}")
    else:
        n_set = 1 if mode == 'bh' else len(REBAL_FREQ)
        st.caption(f"Amount **{_inr(amount)}** · **{years} years** · weights: **all {n_combos} combinations** "
                   f"· **{COMBO_SIMS:,}** futures for each of the {n_combos * n_set:,} combinations "
                   f"(= {COMBO_SIMS * n_combos * n_set:,} outcomes) built from {hist}")
    st.markdown("---")

    with st.spinner("Simulating every weight combination (a few seconds the first time, then saved)…"):
        bh = _simulate_combos(rets, 'bh')
    if mode == 'bh':
        _combos_dashboard('bh', *bh, amount, years, bh_ref=None)
    elif mode == 'periodic':
        with st.spinner("Simulating every weight combination × every rebalance frequency…"):
            pr = _simulate_combos(rets, 'periodic')
        _combos_dashboard('periodic', *pr, amount, years, bh_ref=bh[0]['All settings'])
    else:
        with st.spinner("Simulating every weight set × every setting…"):
            pooled, paths, combos, sets = _simulate_diff_all(rets)
        _diff_all_weights(pooled, paths, combos, sets, bh[0]['All settings'], amount, years)


# =========================================================
# Buy & Hold / Rebalance: every weight combination (x every frequency)
# =========================================================
def _combos_dashboard(mode, pooled, paths, combos, combo_list, amount, years, bh_ref=None):
    color = STRAT_COLORS[mode]
    names = [n for n in pooled if n != 'All settings']
    asset_cols = [c for c in combos.columns if c.endswith(' (%)') and c.split(' (%)')[0] in ASSETS]
    n_c = len(combo_list)
    multi = len(names) > 1
    what = f"{n_c} weight combinations" + (f" × {len(names)} frequencies" if multi else "")

    # ---- 1) everything together ----
    st.markdown(f"### 🌐 All possibilities together: {what}")
    _single_dashboard(mode, 'All combinations', paths['All settings'], pooled['All settings'],
                      amount, years, key=f"{mode}_all", show_title=False)

    # ---- 2) best combinations ----
    st.markdown("---")
    st.markdown("### 🏆 Best combinations")

    def label(row):
        w = " / ".join(f"{row[c]:g}" for c in asset_cols)
        return w + (f" · {row['Setting']}" if multi else "")

    best_med = combos.loc[combos['Median (₹)'].idxmax()]
    best_p5 = combos.loc[combos['Bottom 5% (₹)'].idxmax()]
    least = combos.sort_values(['Chance of loss (%)', 'Bottom 5% (₹)'], ascending=[True, False]).iloc[0]
    worst = combos.loc[combos['Median (₹)'].idxmin()]
    c = st.columns(4)
    with c[0]:
        render_metric_card("Best typical (median)", _inr(best_med['Median (₹)']), label(best_med))
    with c[1]:
        render_metric_card("Safest bottom 5%", _inr(best_p5['Bottom 5% (₹)']), label(best_p5))
    with c[2]:
        render_metric_card("Lowest chance of loss", f"{least['Chance of loss (%)']:.1f}%", label(least))
    with c[3]:
        render_metric_card("Worst typical (median)", _inr(worst['Median (₹)']), label(worst), is_positive=False)
    st.caption("Weights are written in the order GoldBEES / LargeCap / MidCap / SmallCap.")

    # ---- 3) risk vs return: every combination as a dot ----
    hover = combos.apply(lambda r: "<br>".join(f"{c.split(' (%)')[0]}: {r[c]:g}%" for c in asset_cols)
                         + (f"<br>{r['Setting']}" if multi else ""), axis=1)
    fig = go.Figure(go.Scatter(
        x=combos['Bottom 5% (₹)'], y=combos['Median (₹)'], mode='markers',
        marker=dict(color=color, size=8, opacity=0.75, line=dict(color='#0F172A', width=1)),
        customdata=np.stack([hover, combos['Chance of loss (%)'], combos['Top 5% (₹)']], axis=1),
        hovertemplate=("%{customdata[0]}<br>Median: ₹%{y:,.0f}<br>Bottom 5%: ₹%{x:,.0f}"
                       "<br>Top 5%: ₹%{customdata[2]:,.0f}<br>Chance of loss: %{customdata[1]:.1f}%<extra></extra>")))
    for row, txt in ((best_med, "Best median"), (best_p5, "Safest bottom 5%")):
        fig.add_trace(go.Scatter(x=[row['Bottom 5% (₹)']], y=[row['Median (₹)']], mode='markers+text',
                                 marker=dict(color=INK, size=13, symbol='star'), text=[f"{txt}: {label(row)} "],
                                 textposition='middle left' if txt == "Safest bottom 5%" else 'top left',
                                 textfont=dict(color=INK, size=11), hoverinfo='skip'))
    fig.add_vline(x=amount, line_dash='dot', line_color=MUTED, annotation_text="Bottom 5% = invested",
                  annotation_position='top left', annotation_font_color=NEUTRAL)
    fig.update_layout(**MC_LAYOUT, height=520, showlegend=False, hovermode='closest',
                      title=f"Every combination: typical result vs bad-case result ({len(combos):,} dots)",
                      xaxis=dict(title="Bottom 5% final value (₹) → safer to the right", tickprefix="₹",
                                 tickformat=',.0f', gridcolor=GRID),
                      yaxis=dict(title="Median final value (₹) → better upward", tickprefix="₹",
                                 tickformat=',.0f', gridcolor=GRID))
    st.plotly_chart(fig, use_container_width=True, key=f"scatter_{mode}")
    st.caption("Each dot is one weight combination" + (" with one rebalance frequency" if multi else "")
               + ". Top-right = good typical result AND a safer bad case. Hover a dot to see its weights.")

    # ---- 4) how each asset's weight changes the result ----
    fig2 = go.Figure()
    for a_col in asset_cols:
        asset = a_col.split(' (%)')[0]
        g = combos.groupby(a_col)['Median (₹)'].mean()
        g = g[g.index.isin(range(0, 101, WEIGHT_STEP))]
        fig2.add_trace(go.Scatter(x=g.index, y=g.values, mode='lines+markers', name=ASSET_LABELS[asset],
                                  line=dict(color=COLORS[asset], width=2.5), marker=dict(size=7),
                                  hovertemplate=f"{ASSET_LABELS[asset]} at %{{x}}%<br>Average median: ₹%{{y:,.0f}}<extra></extra>"))
    fig2.update_layout(**MC_LAYOUT, height=420, hovermode='x unified',
                       title="How each asset's weight changes the typical result (average over all other splits)",
                       xaxis=dict(title="Weight of the asset (%)", ticksuffix="%", dtick=10, gridcolor=GRID),
                       yaxis=dict(title="Average median final value (₹)", tickprefix="₹", tickformat=',.0f',
                                  gridcolor=GRID),
                       legend=dict(orientation='h', y=-0.22))
    st.plotly_chart(fig2, use_container_width=True, key=f"effect_{mode}")
    st.caption("A rising line means: giving this asset more weight raised the typical final value.")

    # ---- 5) frequencies compared (Rebalance only) ----
    if multi:
        st.markdown("---")
        _all_possibilities(mode, {n: (paths[n], pooled[n]) for n in names}, bh_ref, amount, years,
                           title=f"### ⚙️ The {len(names)} rebalance frequencies compared (each over all {n_c} combinations)",
                           bh_label="Buy & Hold, all combinations (reference)")

    # ---- 6) every combination ----
    st.markdown("##### 📋 Every combination (sorted by median, best first)")
    tbl = combos.sort_values('Median (₹)', ascending=False).reset_index(drop=True)
    if not multi:
        tbl = tbl.drop(columns=['Setting'])
    money = [c for c in tbl.columns if '(₹)' in c]
    pct = [c for c in tbl.columns if '(%)' in c and c not in asset_cols]
    st.dataframe(tbl.style.format({**{c: '₹{:,.0f}' for c in money}, **{c: '{:.2f}%' for c in pct},
                                   **{c: '{:g}%' for c in asset_cols}}),
                 use_container_width=True, hide_index=True, height=420)
    st.download_button("⬇️ Download all combinations (CSV)", tbl.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f"monte_carlo_{mode}_all_combinations.csv", mime="text/csv",
                       key=f"dl_{mode}_combos")

    # ---- 7) full dashboard per frequency (Rebalance only) ----
    if multi:
        st.markdown("---")
        st.markdown(f"### 🔎 Full dashboard for each frequency (all {n_c} combinations together)")
        for i, (tab, name) in enumerate(zip(st.tabs(names), names)):
            with tab:
                _single_dashboard(mode, name, paths[name], pooled[name], amount, years, key=f"{mode}_{i}")


# =========================================================
# Different Weights: every weight set x every setting
# =========================================================
def _diff_all_weights(pooled, paths, combos, sets, bh, amount, years):
    names = [n for n in pooled if n != 'All settings']

    # ---- 1) everything together ----
    st.markdown(f"### 🌐 All possibilities together: {len(sets)} weight sets × {len(names)} settings")
    _single_dashboard('diff', 'All weight sets & all settings', paths['All settings'], pooled['All settings'],
                      amount, years, key='diff_all', show_title=False)

    # ---- 2) best combinations ----
    st.markdown("---")
    st.markdown("### 🏆 Best combinations")
    best_med = combos.loc[combos['Median (₹)'].idxmax()]
    best_p5 = combos.loc[combos['Bottom 5% (₹)'].idxmax()]
    least = combos.sort_values(['Chance of loss (%)', 'Bottom 5% (₹)'], ascending=[True, False]).iloc[0]
    worst = combos.loc[combos['Median (₹)'].idxmin()]
    c = st.columns(4)
    with c[0]:
        render_metric_card("Best typical (median)", _inr(best_med['Median (₹)']),
                           f"{best_med['Weights']} · {best_med['Setting']}")
    with c[1]:
        render_metric_card("Safest bottom 5%", _inr(best_p5['Bottom 5% (₹)']),
                           f"{best_p5['Weights']} · {best_p5['Setting']}")
    with c[2]:
        render_metric_card("Lowest chance of loss", f"{least['Chance of loss (%)']:.1f}%",
                           f"{least['Weights']} · {least['Setting']}")
    with c[3]:
        render_metric_card("Worst typical (median)", _inr(worst['Median (₹)']),
                           f"{worst['Weights']} · {worst['Setting']}", is_positive=False)
    st.caption("Weights are written biggest first. On every rebalance the biggest weight goes to the asset that "
               "grew the least, the smallest to the one that grew the most.")

    # ---- 3) heatmap: weight set x setting ----
    piv = combos.pivot(index='Weights', columns='Setting', values='Median (₹)')
    order = [" / ".join(f"{x:g}" for x in ws) for ws in sets]
    piv = piv.reindex(index=order, columns=names)
    p5 = combos.pivot(index='Weights', columns='Setting', values='Bottom 5% (₹)').reindex(index=order, columns=names)
    loss = combos.pivot(index='Weights', columns='Setting', values='Chance of loss (%)').reindex(index=order, columns=names)
    custom = np.dstack([p5.to_numpy(), loss.to_numpy()])
    fig = go.Figure(go.Heatmap(
        z=piv.to_numpy(), x=[n.replace(' · ', '<br>') for n in names], y=order,
        colorscale=[[0, '#292018'], [0.5, '#92400E'], [1, '#F59E0B']], xgap=2, ygap=2,
        text=[[_inr(v) for v in row] for row in piv.to_numpy()], texttemplate="%{text}",
        textfont=dict(size=11, color='#F8FAFC'), customdata=custom,
        colorbar=dict(title="Median", tickprefix="₹", tickformat=',.0f'),
        hovertemplate=("Weights %{y}<br>%{x}<br>Median: ₹%{z:,.0f}<br>Bottom 5%: ₹%{customdata[0]:,.0f}"
                       "<br>Chance of loss: %{customdata[1]:.1f}%<extra></extra>")))
    fig.update_layout(**{**MC_LAYOUT, 'margin': dict(l=20, r=20, t=130, b=20)}, height=170 + 30 * len(order),
                      title=dict(text="Typical (median) final value for every weight set × setting", y=0.99),
                      xaxis=dict(side='top'), yaxis=dict(autorange='reversed', title="Weights (biggest first)"))
    st.plotly_chart(fig, use_container_width=True, key="diff_heat")
    st.caption("Brighter = higher typical final value. Hover any cell for its bottom 5% and chance of loss.")

    # ---- 4) settings compared (pooled over all weight sets) ----
    st.markdown("---")
    _all_possibilities('diff', {n: (paths[n], pooled[n]) for n in names}, bh, amount, years,
                       title=f"### ⚙️ The 6 settings compared (each over all {len(sets)} weight sets)")

    # ---- 5) every combination ----
    st.markdown("##### 📋 Every combination (sorted by median, best first)")
    tbl = combos.sort_values('Median (₹)', ascending=False).reset_index(drop=True)
    money = [c for c in tbl.columns if '(₹)' in c]
    pct = [c for c in tbl.columns if '(%)' in c]
    st.dataframe(tbl.style.format({**{c: '₹{:,.0f}' for c in money}, **{c: '{:.2f}%' for c in pct}}),
                 use_container_width=True, hide_index=True, height=420)
    st.download_button("⬇️ Download all combinations (CSV)", tbl.to_csv(index=False).encode('utf-8-sig'),
                       file_name="monte_carlo_different_weights_all.csv", mime="text/csv", key="dl_diff_combos")

    # ---- 6) full dashboard per setting ----
    st.markdown("---")
    st.markdown("### 🔎 Full dashboard for each setting (all weight sets together)")
    for i, (tab, name) in enumerate(zip(st.tabs(names), names)):
        with tab:
            _single_dashboard('diff', name, paths[name], pooled[name], amount, years, key=f"diff_{i}")


# =========================================================
# "All possibilities" overview (Rebalance & Different Weights)
# =========================================================
def _all_possibilities(mode, variants, bh, amount, years, title=None,
                       bh_label="Buy & Hold, all combinations (reference)"):
    color = STRAT_COLORS[mode]
    names = list(variants.keys())
    stats = {n: variants[n][1] for n in names}
    st.markdown(title or f"### 🌐 All possibilities: {len(names)} settings of {STRATEGIES[mode]}")

    best_med = max(names, key=lambda n: stats[n]['median'])
    best_p5 = max(names, key=lambda n: stats[n]['p5'])
    least_loss = min(names, key=lambda n: stats[n]['prob_loss'])
    best_mean = max(names, key=lambda n: stats[n]['mean'])
    c = st.columns(4)
    with c[0]:
        render_metric_card("Best typical (median)", _inr(stats[best_med]['median']), best_med)
    with c[1]:
        render_metric_card("Best average (mean)", _inr(stats[best_mean]['mean']), best_mean)
    with c[2]:
        render_metric_card("Safest bottom 5%", _inr(stats[best_p5]['p5']), best_p5)
    with c[3]:
        render_metric_card("Lowest chance of loss", f"{stats[least_loss]['prob_loss']:.1f}%", least_loss)
    st.markdown("<br>", unsafe_allow_html=True)

    # ---- range chart: every setting on one chart ----
    rows = [(bh_label, bh, MUTED)] + [(n, stats[n], color) for n in names]
    rows = rows[::-1]                                  # first setting at the top
    labels = [r[0] for r in rows]
    fig = go.Figure()
    notes = []
    for lbl, s, col in rows:
        p25, p75 = np.percentile(s['final'], [25, 75])
        fig.add_trace(go.Scatter(x=[s['p5'], s['p95']], y=[lbl, lbl], mode='lines',
                                 line=dict(color=col, width=3), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[p25, p75], y=[lbl, lbl], mode='lines',
                                 line=dict(color=col, width=14), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=[s['median']], y=[lbl], mode='markers', showlegend=False,
            marker=dict(color=INK, size=11, line=dict(color='#0F172A', width=2)),
            hovertemplate=(f"<b>{lbl}</b><br>Bottom 5%: {_inr(s['p5'])}<br>Middle 50%: {_inr(p25)} – {_inr(p75)}"
                           f"<br>Median: {_inr(s['median'])}<br>Mean: {_inr(s['mean'])}"
                           f"<br>Top 5%: {_inr(s['p95'])}<br>Chance of loss: {s['prob_loss']:.1f}%<extra></extra>")))
        fig.add_trace(go.Scatter(x=[s['mean']], y=[lbl], mode='markers', showlegend=False, hoverinfo='skip',
                                 marker=dict(color=INK, size=10, symbol='diamond-open', line=dict(width=2))))
        notes += [dict(x=s['p5'], y=lbl, text=_inr(s['p5']), showarrow=False, xanchor='right', xshift=-6,
                       font=dict(color=NEUTRAL, size=11)),
                  dict(x=s['p95'], y=lbl, text=_inr(s['p95']), showarrow=False, xanchor='left', xshift=6,
                       font=dict(color=NEUTRAL, size=11))]
    fig.update_layout(annotations=notes)
    fig.add_vline(x=amount, line_dash='dot', line_color=MUTED, annotation_text=f"Invested {_inr(amount)}",
                  annotation_position='top', annotation_font_color=NEUTRAL)
    lo = min(r[1]['p5'] for r in rows)
    hi = max(r[1]['p95'] for r in rows)
    pad = (hi - lo) * 0.12
    fig.update_layout(**MC_LAYOUT, height=120 + 52 * len(rows), hovermode='closest',
                      title="Range of final value for every setting",
                      xaxis=dict(title="Final value (₹)", tickprefix="₹", tickformat=',.0f', gridcolor=GRID,
                                 range=[max(0, lo - pad), hi + pad]),
                      yaxis=dict(categoryorder='array', categoryarray=labels, gridcolor=GRID))
    st.plotly_chart(fig, use_container_width=True, key=f"range_{mode}")
    st.caption("Each row is one setting. Thin line = bottom 5% to top 5% (90% of futures). "
               "Thick bar = middle 50%. ● = median (typical), ◇ = mean (average). Hover a row for all numbers. "
               "The grey row is Buy & Hold over all weight combinations, for reference.")

    # ---- table of all settings ----
    trs = []
    for n, s in list(stats.items()) + [(bh_label, bh)]:
        trs.append({
            'Setting': n, 'Mean (₹)': s['mean'], 'Median (₹)': s['median'], 'Bottom 5% (₹)': s['p5'],
            'Top 5% (₹)': s['p95'], 'Avg of worst 5% (₹)': s['cvar5'], 'Chance of loss (%)': s['prob_loss'],
            'Chance to double (%)': s['prob_double'], 'Median CAGR (%)': s['cagr_median'],
            'Typical max fall (%)': s['dd_median'], 'Worst-5% max fall (%)': s['dd_p5'],
        })
    df = pd.DataFrame(trs)
    money = [c for c in df.columns if '(₹)' in c]
    pct = [c for c in df.columns if '(%)' in c]
    st.markdown("##### 📊 All settings side by side")
    st.dataframe(df.style.format({**{c: '₹{:,.0f}' for c in money}, **{c: '{:.2f}%' for c in pct}}),
                 use_container_width=True, hide_index=True)


# =========================================================
# One setting's full dashboard
# =========================================================
def _single_dashboard(mode, name, path, s, amount, years, key, show_title=True):
    color = STRAT_COLORS[mode]
    if mode != 'bh' and show_title:
        st.markdown(f"#### {STRATEGIES[mode]} · {name}")

    # ---- headline cards ----
    c = st.columns(5)
    with c[0]:
        render_metric_card("Mean (average)", _inr(s['mean']), f"{(s['mean'] / amount - 1) * 100:+.1f}%",
                           is_positive=s['mean'] >= amount)
    with c[1]:
        render_metric_card("Median (typical)", _inr(s['median']), f"{(s['median'] / amount - 1) * 100:+.1f}%",
                           is_positive=s['median'] >= amount)
    with c[2]:
        render_metric_card("Top 5% (best)", _inr(s['p95']), f"{(s['p95'] / amount - 1) * 100:+.1f}% or more",
                           is_positive=True)
    with c[3]:
        render_metric_card("Bottom 5% (worst)", _inr(s['p5']), f"{(s['p5'] / amount - 1) * 100:+.1f}% or less",
                           is_positive=s['p5'] >= amount)
    with c[4]:
        render_metric_card("Chance of loss", f"{s['prob_loss']:.1f}%", f"ends below {_inr(amount)}",
                           is_positive=s['prob_loss'] < 10)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c = st.columns(5)
    with c[0]:
        render_metric_card("Median CAGR", f"{s['cagr_median']:.2f}%",
                           f"5%–95%: {s['cagr_p5']:.1f}% to {s['cagr_p95']:.1f}%", is_positive=s['cagr_median'] >= 0)
    with c[1]:
        render_metric_card("Chance to double", f"{s['prob_double']:.1f}%", f"ends at {_inr(2 * amount)}+",
                           is_positive=True)
    with c[2]:
        render_metric_card("Avg of worst 5%", _inr(s['cvar5']), f"{(s['cvar5'] / amount - 1) * 100:+.1f}%",
                           is_positive=s['cvar5'] >= amount)
    with c[3]:
        render_metric_card("Typical max fall", f"{s['dd_median']:.1f}%", "median max drawdown", is_positive=False)
    with c[4]:
        render_metric_card("Worst-5% max fall", f"{s['dd_p5']:.1f}%", "1 in 20 futures fall this much",
                           is_positive=False)
    st.markdown("<br>", unsafe_allow_html=True)

    # ---- bell curves (all three, no options) ----
    st.plotly_chart(_bell_chart(s['final'], amount, _inr, "Final value (₹)", color,
                                "Bell curve of final value: all possible outcomes"),
                    use_container_width=True, key=f"bell_final_{key}")
    st.caption("🟥 bars = bottom 5% of futures · 🟩 bars = top 5% · middle bars = the other 90%. "
               "Hover any bar to see how many futures landed there. The rarest 0.5% at each end are left "
               "out so the shape stays readable.")
    b1, b2 = st.columns(2)
    with b1:
        st.plotly_chart(_bell_chart(s['cagr'], 0.0, lambda v: f"{v:.1f}%", "CAGR (% per year)", color,
                                    "Bell curve of CAGR (yearly return)", height=420),
                        use_container_width=True, key=f"bell_cagr_{key}")
    with b2:
        st.plotly_chart(_bell_chart(s['max_dd'], None, lambda v: f"{v:.1f}%", "Max drawdown (%)", color,
                                    "Bell curve of biggest fall (closer to 0 is better)", height=420),
                        use_container_width=True, key=f"bell_dd_{key}")

    # ---- fan chart ----
    st.plotly_chart(_fan_chart(path, s, amount, years, color), use_container_width=True, key=f"fan_{key}")
    st.caption("The darker band holds the middle 50% of futures, the lighter band the middle 90%. "
               "Thin lines are 20 individual simulated futures.")

    # ---- percentile table ----
    st.markdown("##### 📋 All possibilities by percentile")
    tbl = mc.percentile_table(s, amount, years)
    st.dataframe(tbl.style.format({'Final Value (₹)': '₹{:,.0f}', 'Total Return (%)': '{:+.1f}%',
                                   'CAGR (%)': '{:+.2f}%'}),
                 use_container_width=True, hide_index=True)
    st.caption("Read it as: *in X% of the simulated futures the final value was below this number*. "
               "The 5th percentile is the bottom-5% line, the 95th the top-5% line.")


def _bell_chart(data, ref, fmt, x_title, color, title, height=480):
    data = np.asarray(data, dtype=float)
    lo, hi = np.percentile(data, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1e-9
    shown = data[(data >= lo) & (data <= hi)]
    p5, p50, p95, mean = np.percentile(data, 5), np.median(data), np.percentile(data, 95), data.mean()

    counts, edges = np.histogram(shown, bins=60)
    mids = (edges[:-1] + edges[1:]) / 2
    pct = counts / len(data) * 100
    bar_colors = [BAD if m < p5 else GOOD if m > p95 else color for m in mids]
    hover = [f"{fmt(a)} to {fmt(b)}<br>{c:,} futures ({p:.1f}%)" for a, b, c, p in zip(edges[:-1], edges[1:], counts, pct)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=mids, y=pct, width=(edges[1] - edges[0]) * 0.92, marker_color=bar_colors,
                         marker_line_width=0, hovertext=hover, hoverinfo="text", showlegend=False))
    grid = np.linspace(lo, hi, 300)
    dens = _kde(shown, grid) * (edges[1] - edges[0]) * 100 * len(shown) / len(data)
    fig.add_trace(go.Scatter(x=grid, y=dens, mode='lines', line=dict(color=INK, width=2),
                             hoverinfo='skip', showlegend=False))

    ymax = max(pct.max(), dens.max()) * 1.08
    marks = [(p5, "Bottom 5%", BAD, 'dash'), (p50, "Median", INK, 'solid'),
             (mean, "Mean", NEUTRAL, 'dot'), (p95, "Top 5%", GOOD, 'dash')]
    if ref is not None and lo <= ref <= hi:
        marks.append((ref, "Invested" if ref else "0%", MUTED, 'dot'))
    # labels above the plot, on 3 staggered rows so close lines never overlap
    shapes, notes = [], []
    for i, (x, label, col, dash) in enumerate(sorted(marks, key=lambda m: m[0])):
        level = i % 3
        shapes.append(dict(type='line', x0=x, x1=x, y0=0, y1=1.0 + level * 0.11, yref='paper',
                           line=dict(color=col, width=2, dash=dash)))
        notes.append(dict(x=x, y=1.0 + level * 0.11, yref='paper', yanchor='bottom',
                          text=f"<b>{label}</b> {fmt(x)}", showarrow=False, font=dict(color=INK, size=11),
                          bgcolor='rgba(15,23,42,0.92)', bordercolor=col, borderwidth=1, borderpad=3))
    layout = {**MC_LAYOUT, 'margin': dict(l=20, r=20, t=120, b=20)}
    x_fmt = {'tickprefix': '₹', 'tickformat': ',.0f'} if '₹' in x_title else {'ticksuffix': '%'}
    fig.update_layout(**layout, height=height, bargap=0, hovermode='closest', shapes=shapes, annotations=notes,
                      title=dict(text=title, y=0.98),
                      xaxis=dict(title=x_title, gridcolor=GRID, **x_fmt),
                      yaxis=dict(title="% of futures", ticksuffix="%", range=[0, ymax], gridcolor=GRID))
    return fig


def _fan_chart(path, s, amount, years, color):
    x = np.arange(path.shape[1]) / 12
    fan = s['fan']
    rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=fan[95], line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=fan[5], fill='tonexty', fillcolor=f'rgba{rgb + (0.18,)}', line=dict(width=0),
                             name='Middle 90% (5%–95%)', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=fan[75], line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=fan[25], fill='tonexty', fillcolor=f'rgba{rgb + (0.38,)}', line=dict(width=0),
                             name='Middle 50% (25%–75%)', hoverinfo='skip'))
    sample = np.random.default_rng(1).choice(path.shape[0], min(20, path.shape[0]), replace=False)
    xs = np.concatenate([np.append(x, np.nan) for _ in sample])
    ys = np.concatenate([np.append(path[i], np.nan) for i in sample])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='rgba(148,163,184,0.25)', width=1),
                             hoverinfo='skip', name='Sample futures', connectgaps=False))
    for p, nm, col, dash in ((95, 'Top 5% line', GOOD, 'dash'), (50, 'Median', INK, 'solid'),
                             (5, 'Bottom 5% line', BAD, 'dash')):
        fig.add_trace(go.Scatter(x=x, y=fan[p], mode='lines', line=dict(color=col, width=2, dash=dash), name=nm,
                                 hovertemplate=f"{nm}<br>Year %{{x:.1f}}: ₹%{{y:,.0f}}<extra></extra>"))
    fig.add_hline(y=amount, line_dash='dot', line_color=MUTED,
                  annotation_text=f"Invested {_inr(amount)}", annotation_position='top left')
    fig.update_layout(**MC_LAYOUT, height=440, hovermode='x unified',
                      title=f"Portfolio value over {years} years: range of all futures",
                      xaxis=dict(title="Years from today", gridcolor=GRID),
                      yaxis=dict(title="Portfolio value (₹)", tickprefix="₹", tickformat=',.0f',
                                 range=[0, float(fan[95].max()) * 1.2], gridcolor=GRID),
                      legend=dict(orientation='h', y=-0.28))
    return fig