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

from src.ui import (
    PLOT_LAYOUT, STRAT_COLORS, GOOD, BAD, TEXT, MUTED, FAINT, BG, SURFACE_2,
    VIOLET, CYAN, EMERALD, AMBER, BLUE, PINK, ORANGE,
    hero, section, kpi_row, note, explain, chart, hex_rgba, inr_short,
    sidebar_nav, sidebar_footer, sidebar_label,
)
from src.asset_data import ASSETS, ASSET_SHORT, COLORS
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
WEIGHT_STEP = 10       # every split in 10% steps

STRATEGY_HELP = {
    "bh": "Money is split once and never touched again. Not just your weights: EVERY possible split of "
          "GoldBEES / Nifty 50 / MidCap / SmallCap in 10% steps (100/0/0/0 … 0/0/0/100, plus 25/25/25/25 = 287 "
          "combinations) is simulated on the same futures.",
    "periodic": "Brought back to the same weights every month / quarter / half-year / year. EVERY possible split "
                "in 10% steps (287 combinations) × ALL 4 rebalance frequencies (= 1,148 combinations) is "
                "simulated on the same futures.",
    "diff": "On each rebalance, the asset that grew the MOST gets the SMALLEST weight. EVERY possible weight set "
            "in 10% steps (24 sets) × ALL 6 settings (half-yearly / yearly rebalance × 6 months / 1 year / "
            "1.5 years look-back) is simulated on the same futures.",
}


def _inr(v):
    return inr_short(v)


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
    sidebar_nav("montecarlo")
    sidebar_label("🎛️ Simulation (fixed)")
    st.sidebar.markdown(f"- Amount **{_inr(AMOUNT)}**\n- Horizon **{YEARS} years**\n"
                        f"- Blocks of **{BLOCK} real months**\n- Seed **{SEED}** (repeatable)")
    sidebar_footer(fetch_errors, key="mc")

    amount, years = AMOUNT, YEARS
    used = [n for n in ASSETS if n in asset_dfs]
    if not used:
        note("No asset data loaded.", BAD, "⛔")
        st.stop()
    rets, dropped = mc.monthly_returns(asset_dfs, used)

    hero("Monte Carlo Simulation",
         "Thousands of possible futures, built by shuffling real 12-month chunks of history. "
         "See the full range of outcomes: the bell curve, median, mean, the best 5% and the worst 5%.",
         page="montecarlo", eyebrow="What could happen next",
         chips=[f"💰 {_inr(amount)} invested", f"⏳ {years}-year horizon",
                f"📚 {len(rets)} real months of history"])

    if len(rets) < 24:
        note(f"Only {len(rets)} months where all assets have data. At least 2 years are needed.", BAD, "⛔")
        st.stop()

    section("Choose a strategy", "Every weight combination of it is simulated automatically", "🧭", VIOLET)
    mode = st.radio("Strategy", list(STRATEGIES.keys()), format_func=lambda k: STRATEGIES[k],
                    horizontal=True, key="mc_mode", label_visibility="collapsed")
    explain(STRATEGY_HELP[mode], STRAT_COLORS[mode])

    if dropped:
        note("Skipped months with a move above ±60% (looks like a data error in the sheet): "
             + ", ".join(d.strftime('%b %Y') for d in dropped), AMBER, "⚠️")
    hist = f"<b>{len(rets)} real months</b> ({rets.index[0]:%b %Y} – {rets.index[-1]:%b %Y})"
    n_combos = len(mc.all_weight_combos(len(used), WEIGHT_STEP))
    if mode == 'diff':
        note(f"<b>All 24 weight sets</b> × 6 settings · <b>{DIFF_SIMS:,}</b> futures each "
             f"(= {DIFF_SIMS * 144:,} outcomes) built from {hist}", STRAT_COLORS[mode], "🧮")
    else:
        n_set = 1 if mode == 'bh' else len(REBAL_FREQ)
        note(f"<b>All {n_combos} weight combinations</b>{' × 4 frequencies' if n_set > 1 else ''} · "
             f"<b>{COMBO_SIMS:,}</b> futures each (= {COMBO_SIMS * n_combos * n_set:,} outcomes) built from {hist}",
             STRAT_COLORS[mode], "🧮")

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


def _best_cards(best_med, best_p5, least, worst, label):
    kpi_row([
        dict(label="Best typical (median)", value=_inr(best_med['Median (₹)']), delta=label(best_med),
             icon="🏆", accent=EMERALD),
        dict(label="Safest bottom 5%", value=_inr(best_p5['Bottom 5% (₹)']), delta=label(best_p5),
             icon="🛡️", accent=CYAN),
        dict(label="Lowest chance of loss", value=f"{least['Chance of loss (%)']:.1f}%", delta=label(least),
             icon="🎯", accent=VIOLET),
        dict(label="Worst typical (median)", value=_inr(worst['Median (₹)']), delta=label(worst), tone="bad",
             icon="⚠️", accent=BAD),
    ])


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

    section("All possibilities together", what, "🌐", color)
    _single_dashboard(mode, 'All combinations', paths['All settings'], pooled['All settings'],
                      amount, years, key=f"{mode}_all", show_title=False)

    section("Best combinations", "Weights in the order GoldBEES / Nifty 50 / MidCap / SmallCap", "🏆", EMERALD)

    def label(row):
        w = " / ".join(f"{row[c]:g}" for c in asset_cols)
        return w + (f" · {row['Setting']}" if multi else "")

    best_med = combos.loc[combos['Median (₹)'].idxmax()]
    best_p5 = combos.loc[combos['Bottom 5% (₹)'].idxmax()]
    least = combos.sort_values(['Chance of loss (%)', 'Bottom 5% (₹)'], ascending=[True, False]).iloc[0]
    worst = combos.loc[combos['Median (₹)'].idxmin()]
    _best_cards(best_med, best_p5, least, worst, label)

    # ---- risk vs return: every combination as a dot, coloured by chance of loss ----
    hover = combos.apply(lambda r: "<br>".join(f"{ASSET_SHORT.get(c.split(' (%)')[0], c)}: {r[c]:g}%"
                                               for c in asset_cols)
                         + (f"<br>{r['Setting']}" if multi else ""), axis=1)
    fig = go.Figure(go.Scatter(
        x=combos['Bottom 5% (₹)'], y=combos['Median (₹)'], mode='markers',
        marker=dict(color=combos['Chance of loss (%)'], colorscale=[[0, EMERALD], [0.5, AMBER], [1, BAD]],
                    size=9, opacity=0.85, line=dict(color=BG, width=1),
                    colorbar=dict(title="Chance<br>of loss", ticksuffix="%", thickness=12)),
        customdata=np.stack([hover, combos['Chance of loss (%)'], combos['Top 5% (₹)']], axis=1),
        hovertemplate=("%{customdata[0]}<br>Median: ₹%{y:,.0f}<br>Bottom 5%: ₹%{x:,.0f}"
                       "<br>Top 5%: ₹%{customdata[2]:,.0f}<br>Chance of loss: %{customdata[1]:.1f}%<extra></extra>")))
    for row, txt in ((best_med, "Best median"), (best_p5, "Safest bottom 5%")):
        fig.add_trace(go.Scatter(x=[row['Bottom 5% (₹)']], y=[row['Median (₹)']], mode='markers+text',
                                 marker=dict(color='#fff', size=15, symbol='star', line=dict(color=color, width=1.5)),
                                 text=[f"{txt}: {label(row)} "],
                                 textposition='middle left' if txt == "Safest bottom 5%" else 'top left',
                                 textfont=dict(color=TEXT, size=11), hoverinfo='skip'))
    fig.add_vline(x=amount, line_dash='dot', line_color=FAINT, annotation_text="Bottom 5% = invested",
                  annotation_position='top left', annotation_font_color=MUTED)
    fig.update_layout(**PLOT_LAYOUT, height=540, showlegend=False, hovermode='closest',
                      title=f"Typical result vs bad-case result · {len(combos):,} combinations",
                      xaxis=dict(title="Bottom 5% final value (₹) → safer to the right", tickprefix="₹",
                                 tickformat=',.0f'),
                      yaxis=dict(title="Median final value (₹) → better upward", tickprefix="₹", tickformat=',.0f'))
    with st.container(border=True):
        chart(fig, key=f"scatter_{mode}")
    st.caption("Each dot is one weight combination" + (" with one rebalance frequency" if multi else "")
               + ". Top-right = good typical result AND a safer bad case. Green dots rarely lose money.")

    # ---- how each asset's weight changes the result ----
    fig2 = go.Figure()
    for a_col in asset_cols:
        asset = a_col.split(' (%)')[0]
        g = combos.groupby(a_col)['Median (₹)'].mean()
        g = g[g.index.isin(range(0, 101, WEIGHT_STEP))]
        fig2.add_trace(go.Scatter(x=g.index, y=g.values, mode='lines+markers', name=ASSET_SHORT[asset],
                                  line=dict(color=COLORS[asset], width=2.8), marker=dict(size=8),
                                  hovertemplate=f"{ASSET_SHORT[asset]} at %{{x}}%: ₹%{{y:,.0f}}<extra></extra>"))
    fig2.update_layout(**PLOT_LAYOUT, height=420, hovermode='x unified',
                       title="How each asset's weight moves the typical result",
                       xaxis=dict(title="Weight of the asset (%)", ticksuffix="%", dtick=10),
                       yaxis=dict(title="Average median final value (₹)", tickprefix="₹", tickformat=',.0f'),
                       legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    with st.container(border=True):
        chart(fig2, key=f"effect_{mode}")
    st.caption("A rising line means giving this asset more weight raised the typical final value "
               "(averaged over every split of the other assets).")

    if multi:
        _all_possibilities(mode, {n: (paths[n], pooled[n]) for n in names}, bh_ref, amount, years,
                           title=f"The {len(names)} rebalance frequencies compared",
                           subtitle=f"Each pooled over all {n_c} weight combinations",
                           bh_label="Buy & Hold, all combinations (reference)")

    section("Every combination", "Sorted by median, best first", "📋", AMBER)
    tbl = combos.sort_values('Median (₹)', ascending=False).reset_index(drop=True)
    if not multi:
        tbl = tbl.drop(columns=['Setting'])
    money = [c for c in tbl.columns if '(₹)' in c]
    pct_cols = [c for c in tbl.columns if '(%)' in c and c not in asset_cols]
    st.dataframe(tbl.style.format({**{c: '₹{:,.0f}' for c in money}, **{c: '{:.2f}%' for c in pct_cols},
                                   **{c: '{:g}%' for c in asset_cols}}),
                 width="stretch", hide_index=True, height=420)
    st.download_button("⬇️  Download all combinations (CSV)", tbl.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f"monte_carlo_{mode}_all_combinations.csv", mime="text/csv",
                       key=f"dl_{mode}_combos")

    if multi:
        section("Full dashboard for each frequency", f"All {n_c} combinations together", "🔎", BLUE)
        for i, (tab, name) in enumerate(zip(st.tabs(names), names)):
            with tab:
                _single_dashboard(mode, name, paths[name], pooled[name], amount, years, key=f"{mode}_{i}")


# =========================================================
# Different Weights: every weight set x every setting
# =========================================================
def _diff_all_weights(pooled, paths, combos, sets, bh, amount, years):
    color = STRAT_COLORS['diff']
    names = [n for n in pooled if n != 'All settings']

    section("All possibilities together", f"{len(sets)} weight sets × {len(names)} settings", "🌐", color)
    _single_dashboard('diff', 'All weight sets & all settings', paths['All settings'], pooled['All settings'],
                      amount, years, key='diff_all', show_title=False)

    section("Best combinations", "Weights written biggest first: the biggest goes to the asset that grew the "
            "least, the smallest to the one that grew the most", "🏆", EMERALD)
    best_med = combos.loc[combos['Median (₹)'].idxmax()]
    best_p5 = combos.loc[combos['Bottom 5% (₹)'].idxmax()]
    least = combos.sort_values(['Chance of loss (%)', 'Bottom 5% (₹)'], ascending=[True, False]).iloc[0]
    worst = combos.loc[combos['Median (₹)'].idxmin()]
    _best_cards(best_med, best_p5, least, worst, lambda r: f"{r['Weights']} · {r['Setting']}")

    # ---- heatmap: weight set x setting ----
    piv = combos.pivot(index='Weights', columns='Setting', values='Median (₹)')
    order = [" / ".join(f"{x:g}" for x in ws) for ws in sets]
    piv = piv.reindex(index=order, columns=names)
    p5 = combos.pivot(index='Weights', columns='Setting', values='Bottom 5% (₹)').reindex(index=order, columns=names)
    loss = combos.pivot(index='Weights', columns='Setting', values='Chance of loss (%)').reindex(index=order,
                                                                                              columns=names)
    custom = np.dstack([p5.to_numpy(), loss.to_numpy()])
    fig = go.Figure(go.Heatmap(
        z=piv.to_numpy(), x=[n.replace(' · ', '<br>') for n in names], y=order,
        colorscale=[[0, '#1E1B4B'], [0.35, '#7C3AED'], [0.7, '#EC4899'], [1, '#FBBF24']], xgap=3, ygap=3,
        text=[[_inr(v) for v in row] for row in piv.to_numpy()], texttemplate="%{text}",
        textfont=dict(size=11, color='#fff'), customdata=custom,
        colorbar=dict(title="Median", tickprefix="₹", tickformat=',.0f', thickness=12),
        hovertemplate=("Weights %{y}<br>%{x}<br>Median: ₹%{z:,.0f}<br>Bottom 5%: ₹%{customdata[0]:,.0f}"
                       "<br>Chance of loss: %{customdata[1]:.1f}%<extra></extra>")))
    fig.update_layout(**PLOT_LAYOUT, height=170 + 30 * len(order),
                      title=dict(text="Typical (median) final value for every weight set × setting", y=0.99),
                      xaxis=dict(side='top'), yaxis=dict(autorange='reversed', title="Weights (biggest first)"))
    fig.update_layout(margin=dict(l=12, r=12, t=130, b=12))
    with st.container(border=True):
        chart(fig, key="diff_heat")
    st.caption("Brighter = higher typical final value. Hover any cell for its bottom 5% and chance of loss.")

    _all_possibilities('diff', {n: (paths[n], pooled[n]) for n in names}, bh, amount, years,
                       title="The 6 settings compared", subtitle=f"Each pooled over all {len(sets)} weight sets")

    section("Every combination", "Sorted by median, best first", "📋", AMBER)
    tbl = combos.sort_values('Median (₹)', ascending=False).reset_index(drop=True)
    money = [c for c in tbl.columns if '(₹)' in c]
    pct_cols = [c for c in tbl.columns if '(%)' in c]
    st.dataframe(tbl.style.format({**{c: '₹{:,.0f}' for c in money}, **{c: '{:.2f}%' for c in pct_cols}}),
                 width="stretch", hide_index=True, height=420)
    st.download_button("⬇️  Download all combinations (CSV)", tbl.to_csv(index=False).encode('utf-8-sig'),
                       file_name="monte_carlo_different_weights_all.csv", mime="text/csv", key="dl_diff_combos")

    section("Full dashboard for each setting", "All weight sets together", "🔎", BLUE)
    for i, (tab, name) in enumerate(zip(st.tabs(names), names)):
        with tab:
            _single_dashboard('diff', name, paths[name], pooled[name], amount, years, key=f"diff_{i}")


# =========================================================
# Settings compared (Rebalance & Different Weights)
# =========================================================
def _all_possibilities(mode, variants, bh, amount, years, title, subtitle="",
                       bh_label="Buy & Hold, all combinations (reference)"):
    color = STRAT_COLORS[mode]
    names = list(variants.keys())
    stats = {n: variants[n][1] for n in names}
    section(title, subtitle, "⚙️", color)

    best_med = max(names, key=lambda n: stats[n]['median'])
    best_p5 = max(names, key=lambda n: stats[n]['p5'])
    least_loss = min(names, key=lambda n: stats[n]['prob_loss'])
    best_mean = max(names, key=lambda n: stats[n]['mean'])
    kpi_row([
        dict(label="Best typical (median)", value=_inr(stats[best_med]['median']), delta=best_med, icon="🏆",
             accent=EMERALD),
        dict(label="Best average (mean)", value=_inr(stats[best_mean]['mean']), delta=best_mean, icon="📊",
             accent=BLUE),
        dict(label="Safest bottom 5%", value=_inr(stats[best_p5]['p5']), delta=best_p5, icon="🛡️", accent=CYAN),
        dict(label="Lowest chance of loss", value=f"{stats[least_loss]['prob_loss']:.1f}%", delta=least_loss,
             icon="🎯", accent=VIOLET),
    ])

    # ---- range chart: every setting on one chart ----
    rows = [(bh_label, bh, MUTED)] + [(n, stats[n], color) for n in names]
    rows = rows[::-1]
    labels = [r[0] for r in rows]
    fig = go.Figure()
    notes = []
    for lbl, s, col in rows:
        p25, p75 = np.percentile(s['final'], [25, 75])
        fig.add_trace(go.Scatter(x=[s['p5'], s['p95']], y=[lbl, lbl], mode='lines',
                                 line=dict(color=hex_rgba(col, 0.55), width=3), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[p25, p75], y=[lbl, lbl], mode='lines',
                                 line=dict(color=col, width=16), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=[s['median']], y=[lbl], mode='markers', showlegend=False,
            marker=dict(color='#fff', size=12, line=dict(color=BG, width=2)),
            hovertemplate=(f"<b>{lbl}</b><br>Bottom 5%: {_inr(s['p5'])}<br>Middle 50%: {_inr(p25)} – {_inr(p75)}"
                           f"<br>Median: {_inr(s['median'])}<br>Mean: {_inr(s['mean'])}"
                           f"<br>Top 5%: {_inr(s['p95'])}<br>Chance of loss: {s['prob_loss']:.1f}%<extra></extra>")))
        fig.add_trace(go.Scatter(x=[s['mean']], y=[lbl], mode='markers', showlegend=False, hoverinfo='skip',
                                 marker=dict(color='#fff', size=10, symbol='diamond-open', line=dict(width=2))))
        notes += [dict(x=s['p5'], y=lbl, text=_inr(s['p5']), showarrow=False, xanchor='right', xshift=-6,
                       font=dict(color=BAD, size=11)),
                  dict(x=s['p95'], y=lbl, text=_inr(s['p95']), showarrow=False, xanchor='left', xshift=6,
                       font=dict(color=GOOD, size=11))]
    lo = min(r[1]['p5'] for r in rows)
    hi = max(r[1]['p95'] for r in rows)
    pad = (hi - lo) * 0.12
    fig.update_layout(**PLOT_LAYOUT, annotations=notes, height=130 + 54 * len(rows), hovermode='closest',
                      title="Range of final value for every setting",
                      xaxis=dict(title="Final value (₹)", tickprefix="₹", tickformat=',.0f',
                                 range=[max(0, lo - pad), hi + pad]),
                      yaxis=dict(categoryorder='array', categoryarray=labels))
    fig.add_vline(x=amount, line_dash='dot', line_color=FAINT, annotation_text=f"Invested {_inr(amount)}",
                  annotation_position='top', annotation_font_color=MUTED)
    with st.container(border=True):
        chart(fig, key=f"range_{mode}")
    st.caption("Thin line = bottom 5% to top 5% (90% of futures) · thick bar = middle 50% · "
               "● = median · ◇ = mean. The grey row is Buy & Hold over all weight combinations, for reference.")

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
    pct_cols = [c for c in df.columns if '(%)' in c]
    st.markdown("**📊 All settings side by side**")
    st.dataframe(df.style.format({**{c: '₹{:,.0f}' for c in money}, **{c: '{:.2f}%' for c in pct_cols}}),
                 width="stretch", hide_index=True)


# =========================================================
# One setting's full dashboard
# =========================================================
def _single_dashboard(mode, name, path, s, amount, years, key, show_title=True):
    color = STRAT_COLORS[mode]
    if mode != 'bh' and show_title:
        st.markdown(f"#### {STRATEGIES[mode]} · {name}")

    def ch(v):
        return f"{(v / amount - 1) * 100:+.1f}%"

    kpi_row([
        dict(label="Mean (average)", value=_inr(s['mean']), delta=ch(s['mean']),
             tone="good" if s['mean'] >= amount else "bad", icon="📊", accent=BLUE),
        dict(label="Median (typical)", value=_inr(s['median']), delta=ch(s['median']),
             tone="good" if s['median'] >= amount else "bad", icon="🎯", accent=color),
        dict(label="Top 5% (best)", value=_inr(s['p95']), delta=ch(s['p95']) + " or more", tone="good",
             icon="🚀", accent=EMERALD),
        dict(label="Bottom 5% (worst)", value=_inr(s['p5']), delta=ch(s['p5']) + " or less",
             tone="good" if s['p5'] >= amount else "bad", icon="🧯", accent=BAD),
        dict(label="Chance of loss", value=f"{s['prob_loss']:.1f}%", delta=f"ends below {_inr(amount)}",
             tone="good" if s['prob_loss'] < 10 else "bad", icon="🎲", accent=AMBER),
    ])
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    kpi_row([
        dict(label="Median CAGR", value=f"{s['cagr_median']:.2f}%",
             delta=f"5–95%: {s['cagr_p5']:.1f}% to {s['cagr_p95']:.1f}%", icon="📈", accent=VIOLET),
        dict(label="Chance to double", value=f"{s['prob_double']:.1f}%", delta=f"ends at {_inr(2 * amount)}+",
             icon="💹", accent=CYAN),
        dict(label="Avg of worst 5%", value=_inr(s['cvar5']), delta=ch(s['cvar5']),
             tone="good" if s['cvar5'] >= amount else "bad", icon="🌧️", accent=PINK),
        dict(label="Typical max fall", value=f"{s['dd_median']:.1f}%", delta="median max drawdown", icon="📉",
             accent=ORANGE),
        dict(label="Worst 5% max fall", value=f"{s['dd_p5']:.1f}%", delta="1 in 20 futures",
             icon="⛈️", accent=BAD),
    ])

    with st.container(border=True):
        chart(_bell_chart(s['final'], amount, _inr, "Final value (₹)", color,
                          "Bell curve of final value: all possible outcomes"), key=f"bell_final_{key}")
    st.caption("Red bars = bottom 5% of futures · green bars = top 5% · coloured bars = the middle 90%. "
               "The rarest 0.5% at each end are left out so the shape stays readable.")
    b1, b2 = st.columns(2)
    with b1:
        with st.container(border=True):
            chart(_bell_chart(s['cagr'], 0.0, lambda v: f"{v:.1f}%", "CAGR (% per year)", VIOLET,
                              "Bell curve of CAGR (yearly return)", height=420), key=f"bell_cagr_{key}")
    with b2:
        with st.container(border=True):
            chart(_bell_chart(s['max_dd'], None, lambda v: f"{v:.1f}%", "Max drawdown (%)", PINK,
                              "Biggest fall (closer to 0 is better)", height=420), key=f"bell_dd_{key}")

    with st.container(border=True):
        chart(_fan_chart(path, s, amount, years, color), key=f"fan_{key}")
    st.caption("Darker band = middle 50% of futures, lighter band = middle 90%. Thin lines are 20 sample futures.")

    st.markdown("**📋 All possibilities by percentile**")
    tbl = mc.percentile_table(s, amount, years)
    st.dataframe(tbl.style.format({'Final Value (₹)': '₹{:,.0f}', 'Total Return (%)': '{:+.1f}%',
                                   'CAGR (%)': '{:+.2f}%'}), width="stretch", hide_index=True)
    st.caption("Read it as: *in X% of the simulated futures the final value was below this number*.")


def _bell_chart(data, ref, fmt, x_title, color, title, height=480):
    data = np.asarray(data, dtype=float)
    lo, hi = np.percentile(data, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1e-9
    shown = data[(data >= lo) & (data <= hi)]
    p5, p50, p95, mean = np.percentile(data, 5), np.median(data), np.percentile(data, 95), data.mean()

    counts, edges = np.histogram(shown, bins=60)
    mids = (edges[:-1] + edges[1:]) / 2
    pct_ = counts / len(data) * 100
    bar_colors = [hex_rgba(BAD, 0.85) if m < p5 else hex_rgba(GOOD, 0.85) if m > p95 else hex_rgba(color, 0.75)
                  for m in mids]
    hover = [f"{fmt(a)} to {fmt(b)}<br>{c:,} futures ({p:.1f}%)"
             for a, b, c, p in zip(edges[:-1], edges[1:], counts, pct_)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=mids, y=pct_, width=(edges[1] - edges[0]) * 0.9, marker_color=bar_colors,
                         marker_line_width=0, hovertext=hover, hoverinfo="text", showlegend=False))
    grid = np.linspace(lo, hi, 300)
    dens = _kde(shown, grid) * (edges[1] - edges[0]) * 100 * len(shown) / len(data)
    fig.add_trace(go.Scatter(x=grid, y=dens, mode='lines', line=dict(color='#fff', width=2.2),
                             hoverinfo='skip', showlegend=False))

    ymax = max(pct_.max(), dens.max()) * 1.08
    marks = [(p5, "Bottom 5%", BAD, 'dash'), (p50, "Median", '#fff', 'solid'),
             (mean, "Mean", CYAN, 'dot'), (p95, "Top 5%", GOOD, 'dash')]
    if ref is not None and lo <= ref <= hi:
        marks.append((ref, "Invested" if ref else "0%", MUTED, 'dot'))
    shapes, notes = [], []
    for i, (x, label, col, dash) in enumerate(sorted(marks, key=lambda m: m[0])):
        level = i % 3
        shapes.append(dict(type='line', x0=x, x1=x, y0=0, y1=1.0 + level * 0.11, yref='paper',
                           line=dict(color=col, width=2, dash=dash)))
        notes.append(dict(x=x, y=1.0 + level * 0.11, yref='paper', yanchor='bottom',
                          text=f"<b>{label}</b> {fmt(x)}", showarrow=False, font=dict(color=TEXT, size=11),
                          bgcolor=SURFACE_2, bordercolor=col, borderwidth=1, borderpad=4))
    x_fmt = {'tickprefix': '₹', 'tickformat': ',.0f'} if '₹' in x_title else {'ticksuffix': '%'}
    fig.update_layout(**PLOT_LAYOUT, height=height, bargap=0, hovermode='closest', shapes=shapes, annotations=notes,
                      title=dict(text=title, y=0.98),
                      xaxis=dict(title=x_title, **x_fmt),
                      yaxis=dict(title="% of futures", ticksuffix="%", range=[0, ymax]))
    fig.update_layout(margin=dict(l=12, r=12, t=120, b=12))
    return fig


def _fan_chart(path, s, amount, years, color):
    x = np.arange(path.shape[1]) / 12
    fan = s['fan']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=fan[95], line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=fan[5], fill='tonexty', fillcolor=hex_rgba(color, 0.16), line=dict(width=0),
                             name='Middle 90% (5–95%)', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=fan[75], line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=fan[25], fill='tonexty', fillcolor=hex_rgba(color, 0.36), line=dict(width=0),
                             name='Middle 50% (25–75%)', hoverinfo='skip'))
    sample = np.random.default_rng(1).choice(path.shape[0], min(20, path.shape[0]), replace=False)
    xs = np.concatenate([np.append(x, np.nan) for _ in sample])
    ys = np.concatenate([np.append(path[i], np.nan) for i in sample])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='rgba(200,210,235,0.18)', width=1),
                             hoverinfo='skip', name='Sample futures', connectgaps=False))
    for p, nm, col, dash in ((95, 'Top 5% line', GOOD, 'dash'), (50, 'Median', '#fff', 'solid'),
                             (5, 'Bottom 5% line', BAD, 'dash')):
        fig.add_trace(go.Scatter(x=x, y=fan[p], mode='lines', line=dict(color=col, width=2.2, dash=dash), name=nm,
                                 hovertemplate=f"{nm}<br>Year %{{x:.1f}}: ₹%{{y:,.0f}}<extra></extra>"))
    fig.add_hline(y=amount, line_dash='dot', line_color=FAINT,
                  annotation_text=f"Invested {_inr(amount)}", annotation_position='top left',
                  annotation_font_color=MUTED)
    fig.update_layout(**PLOT_LAYOUT, height=460, hovermode='x unified',
                      title=f"Portfolio value over {years} years: range of all futures",
                      xaxis=dict(title="Years from today"),
                      yaxis=dict(title="Portfolio value (₹)", tickprefix="₹", tickformat=',.0f',
                                 range=[0, float(fan[95].max()) * 1.2]),
                      legend=dict(orientation='h', yanchor='top', y=-0.18, x=0))
    return fig