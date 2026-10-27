"""🎲 Monte Carlo page, built to be understood without a finance degree.

Flow of the page:
  1. How it works          : 4 simple steps
  2. Pick a strategy       : Buy & Hold / Rebalance / Different Weights
  3. Your mix              : sliders for the 4 weights (+ rebalance settings)
  4. The answer in words   : typical / bad case / great case / chance of loss
  5. 100 possible futures  : a dot grid showing how many futures end where
  6. Charts                : path over time, bell curve, yearly growth, worst falls, percentile table
  7. Explore every mix     : best combinations, weight effects, settings compared, full tables
  8. Glossary

The maths lives in src/monte_carlo.py.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import (
    panel,
    PLOT_LAYOUT, STRAT_COLORS, GOOD, BAD, TEXT, INK, MUTED, FAINT,
    VIOLET, CYAN, EMERALD, AMBER, BLUE, PINK, ORANGE,
    hero, section, kpi_row, note, explain, steps, story, waffle, glossary, chart, hex_rgba, inr_short,
    weight_badge, sidebar_nav, sidebar_footer, sidebar_label,
)
from src.asset_data import ASSETS, ASSET_SHORT, ASSET_ICONS, COLORS
from src.strategies import STRATEGIES, REBAL_FREQ, DIFF_REBAL_FREQ, LOOKBACK_OPTIONS
from src import monte_carlo as mc

# Fixed simulation settings
N_SIMS = 5000          # futures for YOUR mix
BLOCK = 12             # chunks of 12 real months (keeps yearly trends & crashes intact)
SEED = 42              # same seed = same result every time
AMOUNT = 100000.0      # ₹1,00,000
YEARS = 10             # horizon
COMBO_SIMS = 1000      # Buy & Hold / Rebalance: futures per weight combination (287 combinations)
DIFF_SIMS = 2000       # Different Weights: futures per weight set & setting (24 sets x 6 settings)
WEIGHT_STEP = 10       # every split in 10% steps

STRATEGY_HELP = {
    "bh": "Split the money once and never touch it again. Winners grow into a bigger slice over time.",
    "periodic": "Every month / quarter / half-year / year, sell a little of what grew and buy what fell, "
                "so the split goes back to your weights.",
    "diff": "On each rebalance date, look at how much each asset grew recently. The one that grew the MOST gets "
            "your SMALLEST weight, the one that grew the LEAST gets your BIGGEST weight (buy the laggards).",
}

GLOSSARY = [
    ("Typical (median)", "Half of all simulated futures ended above this, half below. The best single guess."),
    ("Average (mean)", "Add up every ending and divide. A few huge winners pull it above the typical value."),
    ("Bad case (5th percentile)", "Only 1 future in 20 ended lower than this. A realistic 'things went badly'."),
    ("Great case (95th percentile)", "Only 1 future in 20 ended higher. A realistic 'things went very well'."),
    ("Chance of losing money", "Share of futures that ended with less than the amount you put in."),
    ("CAGR", "The steady yearly growth rate that would give the same final amount."),
    ("Max drawdown (worst fall)", "The biggest drop from a high point to a later low during the 10 years."),
    ("Block bootstrap", "Building new futures from real 12-month chunks of history, shuffled in random order."),
]


def _inr(v):
    return inr_short(v)


def _mult(v):
    return f"{v / AMOUNT:.1f}×"


# =========================================================
# Cached simulations
# =========================================================
@st.cache_data(show_spinner=False)
def _simulate_mix(rets, weights, mode, every, lookback):
    """Your own mix on N_SIMS futures. Returns (paths, stats)."""
    n_months = YEARS * 12
    warmup = max(18, lookback or 0)
    growth = mc.bootstrap_paths(rets, N_SIMS, warmup + n_months, min(BLOCK, len(rets)), SEED)
    path = mc.run_strategy(growth, np.asarray(weights, dtype=float), AMOUNT, mode, every or 12, lookback or 12,
                           warmup)
    return path, mc.summarize(path, AMOUNT, YEARS)


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


def _out_of_100(final):
    """Bucket the final values into 5 groups and round to whole dots that add up to 100."""
    edges = [(-np.inf, 1), (1, 2), (2, 3), (3, 5), (5, np.inf)]
    labels = ["Ended with LESS than you put in", "Grew, but less than double (1–2×)", "Doubled to tripled (2–3×)",
              "Tripled to 5× (3–5×)", "More than 5×"]
    colors = [BAD, AMBER, BLUE, VIOLET, EMERALD]
    m = np.asarray(final) / AMOUNT
    shares = np.array([((m >= lo) & (m < hi)).mean() * 100 for lo, hi in edges])
    counts = np.floor(shares).astype(int)
    for i in np.argsort(-(shares - counts))[:100 - counts.sum()]:
        counts[i] += 1
    return [(lbl, n, c) for lbl, n, c in zip(labels, counts, colors)]


# =========================================================
# Page
# =========================================================
def render_monte_carlo_page(asset_dfs, fetch_errors):
    sidebar_nav("montecarlo")
    sidebar_label("🎛️ Simulation settings")
    st.sidebar.markdown(f"- Amount **{_inr(AMOUNT)}**\n- Horizon **{YEARS} years**\n"
                        f"- **{N_SIMS:,}** futures for your mix\n- Built from **{BLOCK}-month** real chunks\n"
                        f"- Seed **{SEED}** (same result every time)")
    sidebar_footer(fetch_errors, key="mc")

    used = [n for n in ASSETS if n in asset_dfs]
    if not used:
        note("No asset data loaded.", BAD, "⛔")
        st.stop()
    rets, dropped = mc.monthly_returns(asset_dfs, used)

    hero("Monte Carlo Simulation",
         "What could ₹1 lakh become in 10 years? We replay real market history in thousands of different orders "
         "and count how often each outcome happens.",
         page="montecarlo", eyebrow="What could happen next",
         chips=[f"💰 {_inr(AMOUNT)} invested", f"⏳ {YEARS}-year horizon", f"📚 {len(rets)} real months of history"])

    if len(rets) < 24:
        note(f"Only {len(rets)} months where all assets have data. At least 2 years are needed.", BAD, "⛔")
        st.stop()

    # ---------------- 1. How it works ----------------
    section("How it works", "No crystal ball: just real history, reshuffled", "💡", AMBER)
    steps([
        ("📚", "Take real history", f"{len(rets)} real months of all {len(used)} assets "
         f"({rets.index[0]:%b %Y} – {rets.index[-1]:%b %Y}), moving together exactly as they did.", BLUE),
        ("🔀", "Shuffle into new futures", f"Pick random {BLOCK}-month chunks and glue them into a new {YEARS}-year "
         f"path. Repeat {N_SIMS:,} times.", VIOLET),
        ("⚖️", "Apply your strategy", "Run your weights and strategy on every one of those futures.", PINK),
        ("📊", "Count the outcomes", "Line up all the endings: the typical one, the bad and great cases, "
         "and the odds of losing money.", EMERALD),
    ])
    if dropped:
        note("Skipped months with a move above ±60% (looks like a data error in the sheet): "
             + ", ".join(d.strftime('%b %Y') for d in dropped), AMBER, "⚠️")

    # ---------------- 2. Strategy ----------------
    section("Pick a strategy", "How the money is managed during the 10 years", "🧭", VIOLET)
    mode = st.radio("Strategy", list(STRATEGIES.keys()), format_func=lambda k: STRATEGIES[k],
                    horizontal=True, key="mc_mode", label_visibility="collapsed")
    color = STRAT_COLORS[mode]
    explain(STRATEGY_HELP[mode], color)

    # ---------------- 3. Your mix ----------------
    section("Your mix", "Move the sliders: the results below update instantly", "🎚️", PINK)
    every, lookback, setting_label = None, None, ""
    with panel():
        cols = st.columns(len(used))
        raw = {}
        for col, n in zip(cols, used):
            with col:
                raw[n] = st.slider(f"{ASSET_ICONS[n]} {ASSET_SHORT[n]}", 0, 100, 100 // len(used), 5,
                                   format="%d%%", key=f"mc_w_{n}")
        total = sum(raw.values())
        weight_badge(total)
        if mode == "periodic":
            f_lbl = st.radio("🔁 Rebalance every", list(REBAL_FREQ), index=3, horizontal=True, key="mc_freq")
            every, setting_label = REBAL_FREQ[f_lbl], f" · rebalanced {f_lbl.lower()}"
        elif mode == "diff":
            c1, c2 = st.columns(2)
            with c1:
                f_lbl = st.radio("🔁 Rebalance every", list(DIFF_REBAL_FREQ), index=1, horizontal=True,
                                 key="mc_dfreq")
            with c2:
                l_lbl = st.radio("🔍 Compare growth over the previous", list(LOOKBACK_OPTIONS), index=1,
                                 horizontal=True, key="mc_look")
            every, lookback = DIFF_REBAL_FREQ[f_lbl], LOOKBACK_OPTIONS[l_lbl]
            setting_label = f" · {f_lbl.lower()} · look-back {l_lbl}"
    if total <= 0:
        note("Give at least one asset a weight above 0%.", BAD, "⛔")
        st.stop()
    weights = tuple(raw[n] / total for n in used)
    mix_label = " / ".join(f"{raw[n] / total * 100:.0f}" for n in used)

    with st.spinner(f"Simulating {N_SIMS:,} futures for your mix…"):
        path, s = _simulate_mix(rets, weights, mode, every, lookback)

    # ---------------- 4. The answer ----------------
    section("The answer, in plain words", f"Your mix {mix_label} ({' / '.join(ASSET_SHORT[n] for n in used)})"
            f"{setting_label}", "🗣️", ORANGE)
    loss_n = s['prob_loss']
    story(
        f"If you invest {_inr(AMOUNT)} today and wait {YEARS} years…",
        f"You will most likely end up with about <span class='big'>{_inr(s['median'])}</span>, "
        f"that is <b>{_mult(s['median'])}</b> your money, or roughly <b>{s['cagr_median']:.1f}% a year</b>.",
        [("😟 Bad case · 1 in 20", _inr(s['p5']), f"{_mult(s['p5'])} your money"),
         ("🤩 Great case · 1 in 20", _inr(s['p95']), f"{_mult(s['p95'])} your money"),
         ("🎲 Chance of losing money", f"{loss_n:.0f} in 100" if loss_n >= 1 else "Less than 1 in 100",
          f"Typical worst fall on the way: {s['dd_median']:.0f}%")],
        c1={"bh": "#0E7490", "periodic": "#6D28D9", "diff": "#C2410C"}[mode], c2="#DB2777" if mode == "diff" else "#2563EB")

    # ---------------- 5. 100 futures ----------------
    section("100 possible futures", "Each dot is one future. Where would your ₹1 lakh end up?", "🔮", BLUE)
    waffle(_out_of_100(s['final']), caption=f"Out of 100 futures, after {YEARS} years you would have…")

    # ---------------- 6. Charts ----------------
    section("See it in charts", "Each chart says how to read it underneath", "📈", CYAN)
    _outcome_tabs(path, s, color, key="mine")

    # ---------------- 7. Explore every mix ----------------
    section("Explore every other mix", "Every weight split in 10% steps, simulated on the same futures", "🧪", EMERALD)
    with st.spinner("Simulating every weight combination (a few seconds the first time, then saved)…"):
        bh = _simulate_combos(rets, 'bh')
        if mode == 'bh':
            _combos_explorer('bh', *bh, bh_ref=None, mine=(s, mix_label))
        elif mode == 'periodic':
            pr = _simulate_combos(rets, 'periodic')
            _combos_explorer('periodic', *pr, bh_ref=bh[0]['All settings'], mine=(s, mix_label))
        else:
            pooled, paths, combos, sets = _simulate_diff_all(rets)
            _diff_explorer(pooled, paths, combos, sets, bh[0]['All settings'], mine=(s, mix_label))

    # ---------------- 8. Glossary ----------------
    section("What the words mean", "Plain-language glossary", "📖", MUTED)
    glossary(GLOSSARY)


# =========================================================
# Outcome charts for one set of futures
# =========================================================
def _outcome_tabs(path, s, color, key):
    t1, t2, t3, t4 = st.tabs(["🛣️ Path over time", "🔔 Where the money ends", "📆 Yearly growth & worst falls",
                              "📋 Percentile table"])
    with t1:
        with panel():
            chart(_fan_chart(path, s, color), key=f"fan_{key}")
        note("<b>How to read it:</b> the dark band holds the middle half of all futures, the light band 9 out of 10. "
             "The thick line is the typical path. Thin grey lines are 20 individual futures: real markets wiggle "
             "a lot more than the typical line.", color, "👀")
    with t2:
        with panel():
            chart(_bell_chart(s['final'], AMOUNT, _inr, "Final value after 10 years (₹)", color,
                              "Where the money ends up"), key=f"bell_{key}")
        note("<b>How to read it:</b> the taller the bar, the more futures ended there. Red bars are the worst 1 in 20, "
             "green bars the best 1 in 20. The long tail to the right is a few futures that did extremely well.",
             color, "👀")
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            with panel():
                chart(_bell_chart(s['cagr'], 0.0, lambda v: f"{v:.1f}%", "Yearly growth, CAGR (%)", VIOLET,
                                  "Yearly growth rate", height=420), key=f"cagr_{key}")
        with c2:
            with panel():
                chart(_bell_chart(s['max_dd'], None, lambda v: f"{v:.1f}%", "Worst fall from a peak (%)", PINK,
                                  "Worst fall along the way", height=420), key=f"dd_{key}")
        note("<b>How to read it:</b> left = the steady yearly growth each future works out to. Right = the biggest "
             "drop from a high point you'd have had to sit through. Closer to 0% is calmer.", color, "👀")
    with t4:
        tbl = mc.percentile_table(s, AMOUNT, YEARS)
        st.dataframe(tbl.style.format({'Final Value (₹)': '₹{:,.0f}', 'Total Return (%)': '{:+.1f}%',
                                       'CAGR (%)': '{:+.2f}%'}), width="stretch", hide_index=True)
        st.caption("Read a row as: *in X% of futures, the final value was below this number*.")


def _stat_cards(s, color):
    def ch(v):
        return f"{(v / AMOUNT - 1) * 100:+.0f}%"
    kpi_row([
        dict(label="Typical (median)", value=_inr(s['median']), delta=ch(s['median']),
             tone="good" if s['median'] >= AMOUNT else "bad", icon="🎯", accent=color),
        dict(label="Average (mean)", value=_inr(s['mean']), delta=ch(s['mean']),
             tone="good" if s['mean'] >= AMOUNT else "bad", icon="📊", accent=BLUE),
        dict(label="Bad case · 1 in 20", value=_inr(s['p5']), delta=ch(s['p5']),
             tone="good" if s['p5'] >= AMOUNT else "bad", icon="😟", accent=BAD),
        dict(label="Great case · 1 in 20", value=_inr(s['p95']), delta=ch(s['p95']), tone="good", icon="🤩",
             accent=EMERALD),
        dict(label="Chance of loss", value=f"{s['prob_loss']:.1f}%", delta=f"ends below {_inr(AMOUNT)}",
             tone="good" if s['prob_loss'] < 10 else "bad", icon="🎲", accent=AMBER),
    ])


# =========================================================
# Buy & Hold / Rebalance: every weight combination
# =========================================================
def _combos_explorer(mode, pooled, paths, combos, combo_list, bh_ref, mine):
    color = STRAT_COLORS[mode]
    my_s, my_label = mine
    names = [n for n in pooled if n != 'All settings']
    asset_cols = [c for c in combos.columns if c.endswith(' (%)') and c.split(' (%)')[0] in ASSETS]
    multi = len(names) > 1

    def label(row):
        w = " / ".join(f"{row[c]:g}" for c in asset_cols)
        return w + (f" · {row['Setting']}" if multi else "")

    tabs = ["🏆 Best mixes", "🧭 Does each asset help?"] + (["⚙️ Rebalance frequencies"] if multi else []) \
        + ["🌐 All mixes together", "📋 Every mix (table)"]
    tab = dict(zip(tabs, st.tabs(tabs)))

    with tab["🏆 Best mixes"]:
        best_med = combos.loc[combos['Median (₹)'].idxmax()]
        best_p5 = combos.loc[combos['Bottom 5% (₹)'].idxmax()]
        least = combos.sort_values(['Chance of loss (%)', 'Bottom 5% (₹)'], ascending=[True, False]).iloc[0]
        worst = combos.loc[combos['Median (₹)'].idxmin()]
        _best_cards(best_med, best_p5, least, worst, label)
        st.caption("Weights are in the order " + " / ".join(ASSET_SHORT[a.split(' (%)')[0]] for a in asset_cols) + ".")

        hover = combos.apply(lambda r: "<br>".join(f"{ASSET_SHORT.get(c.split(' (%)')[0], c)}: {r[c]:g}%"
                                                   for c in asset_cols) + (f"<br>{r['Setting']}" if multi else ""),
                             axis=1)
        fig = go.Figure(go.Scatter(
            x=combos['Bottom 5% (₹)'], y=combos['Median (₹)'], mode='markers', name='Every mix',
            marker=dict(color=combos['Chance of loss (%)'], colorscale=[[0, EMERALD], [0.5, AMBER], [1, BAD]],
                        size=9, opacity=0.85, line=dict(color='#0A0F1E', width=1),
                        colorbar=dict(title="Chance<br>of loss", ticksuffix="%", thickness=12)),
            customdata=np.stack([hover, combos['Chance of loss (%)'], combos['Top 5% (₹)']], axis=1),
            hovertemplate=("%{customdata[0]}<br>Typical: ₹%{y:,.0f}<br>Bad case: ₹%{x:,.0f}"
                           "<br>Great case: ₹%{customdata[2]:,.0f}<br>Chance of loss: %{customdata[1]:.1f}%"
                           "<extra></extra>")))
        fig.add_trace(go.Scatter(x=[my_s['p5']], y=[my_s['median']], mode='markers+text', name='Your mix',
                                 marker=dict(color=PINK, size=20, symbol='diamond', line=dict(color='#fff', width=2)),
                                 text=[f"  Your mix {my_label}"], textposition='middle right',
                                 textfont=dict(color=PINK, size=13), hoverinfo='skip'))
        for row, txt in ((best_med, "Best typical"), (best_p5, "Safest")):
            fig.add_trace(go.Scatter(x=[row['Bottom 5% (₹)']], y=[row['Median (₹)']], mode='markers+text',
                                     marker=dict(color=AMBER, size=16, symbol='star', line=dict(color='#0A0F1E', width=1)),
                                     text=[f"{txt}: {label(row)}  "], textposition='middle left',
                                     textfont=dict(color=INK, size=11), hoverinfo='skip', showlegend=False))
        fig.add_vline(x=AMOUNT, line_dash='dot', line_color=FAINT, annotation_text="Bad case = money back",
                      annotation_position='top left', annotation_font_color=MUTED)
        fig.update_layout(**PLOT_LAYOUT, height=560, showlegend=False, hovermode='closest',
                          title="Typical result vs bad case for every mix (top-right is best)",
                          xaxis=dict(title="Bad case (1 in 20) → safer to the right", tickprefix="₹", tickformat=',.0f'),
                          yaxis=dict(title="Typical result → better upward", tickprefix="₹", tickformat=',.0f'))
        with panel():
            chart(fig, key=f"scatter_{mode}")
        note("<b>How to read it:</b> every dot is one way to split your money. Higher = better typical result, "
             "further right = less painful bad case. Green dots almost never lose money. The pink diamond is your mix.",
             color, "👀")

    with tab["🧭 Does each asset help?"]:
        fig2 = go.Figure()
        for a_col in asset_cols:
            asset = a_col.split(' (%)')[0]
            g = combos.groupby(a_col)['Median (₹)'].mean()
            g = g[g.index.isin(range(0, 101, WEIGHT_STEP))]
            fig2.add_trace(go.Scatter(x=g.index, y=g.values, mode='lines+markers', name=ASSET_SHORT[asset],
                                      line=dict(color=COLORS[asset], width=3), marker=dict(size=9),
                                      hovertemplate=f"{ASSET_SHORT[asset]} at %{{x}}%: ₹%{{y:,.0f}}<extra></extra>"))
        fig2.update_layout(**PLOT_LAYOUT, height=440, hovermode='x unified',
                           title="Typical result as you give each asset more weight",
                           xaxis=dict(title="Weight of the asset (%)", ticksuffix="%", dtick=10),
                           yaxis=dict(title="Typical final value (₹)", tickprefix="₹", tickformat=',.0f'),
                           legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        with panel():
            chart(fig2, key=f"effect_{mode}")
        note("<b>How to read it:</b> a line going UP means adding more of that asset raised the typical result "
             "(averaged over every way of splitting the rest). A line going DOWN means more of it hurt.", color, "👀")

    if multi:
        with tab["⚙️ Rebalance frequencies"]:
            _settings_compared(mode, {n: (paths[n], pooled[n]) for n in names}, bh_ref,
                               bh_label="Buy & Hold (reference)")
            pick = st.selectbox("Open the full picture for one frequency", names, key=f"one_{mode}")
            _stat_cards(pooled[pick], color)
            _outcome_tabs(paths[pick], pooled[pick], color, key=f"{mode}_one")

    with tab["🌐 All mixes together"]:
        st.caption(f"All {len(combo_list)} weight mixes{' × ' + str(len(names)) + ' frequencies' if multi else ''} "
                   "pooled into one big set of futures.")
        _stat_cards(pooled['All settings'], color)
        _outcome_tabs(paths['All settings'], pooled['All settings'], color, key=f"{mode}_all")

    with tab["📋 Every mix (table)"]:
        tbl = combos.sort_values('Median (₹)', ascending=False).reset_index(drop=True)
        if not multi:
            tbl = tbl.drop(columns=['Setting'])
        money = [c for c in tbl.columns if '(₹)' in c]
        pct_cols = [c for c in tbl.columns if '(%)' in c and c not in asset_cols]
        st.dataframe(tbl.style.format({**{c: '₹{:,.0f}' for c in money}, **{c: '{:.2f}%' for c in pct_cols},
                                       **{c: '{:g}%' for c in asset_cols}}),
                     width="stretch", hide_index=True, height=440)
        st.download_button("⬇️  Download every mix (CSV)", tbl.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f"monte_carlo_{mode}_all_combinations.csv", mime="text/csv",
                           key=f"dl_{mode}_combos")


def _best_cards(best_med, best_p5, least, worst, label):
    kpi_row([
        dict(label="Best typical result", value=_inr(best_med['Median (₹)']), delta=label(best_med),
             icon="🏆", accent=EMERALD),
        dict(label="Safest bad case", value=_inr(best_p5['Bottom 5% (₹)']), delta=label(best_p5),
             icon="🛡️", accent=CYAN),
        dict(label="Lowest chance of loss", value=f"{least['Chance of loss (%)']:.1f}%", delta=label(least),
             icon="🎯", accent=VIOLET),
        dict(label="Worst typical result", value=_inr(worst['Median (₹)']), delta=label(worst), tone="bad",
             icon="⚠️", accent=BAD),
    ])


# =========================================================
# Different Weights: every weight set x every setting
# =========================================================
def _diff_explorer(pooled, paths, combos, sets, bh, mine):
    color = STRAT_COLORS['diff']
    names = [n for n in pooled if n != 'All settings']
    t_best, t_heat, t_set, t_all, t_tbl = st.tabs(["🏆 Best weight sets", "🗺️ Heatmap", "⚙️ The 6 settings",
                                                   "🌐 All together", "📋 Every combination"])
    with t_best:
        best_med = combos.loc[combos['Median (₹)'].idxmax()]
        best_p5 = combos.loc[combos['Bottom 5% (₹)'].idxmax()]
        least = combos.sort_values(['Chance of loss (%)', 'Bottom 5% (₹)'], ascending=[True, False]).iloc[0]
        worst = combos.loc[combos['Median (₹)'].idxmin()]
        _best_cards(best_med, best_p5, least, worst, lambda r: f"{r['Weights']} · {r['Setting']}")
        st.caption("Weights are written biggest first: on every rebalance the biggest goes to the asset that grew "
                   "the least, the smallest to the one that grew the most.")

    with t_heat:
        piv = combos.pivot(index='Weights', columns='Setting', values='Median (₹)')
        order = [" / ".join(f"{x:g}" for x in ws) for ws in sets]
        piv = piv.reindex(index=order, columns=names)
        p5 = combos.pivot(index='Weights', columns='Setting', values='Bottom 5% (₹)').reindex(index=order,
                                                                                            columns=names)
        loss = combos.pivot(index='Weights', columns='Setting', values='Chance of loss (%)').reindex(index=order,
                                                                                                  columns=names)
        fig = go.Figure(go.Heatmap(
            z=piv.to_numpy(), x=[n.replace(' · ', '<br>') for n in names], y=order,
            colorscale=[[0, '#1E1B4B'], [0.35, '#7C3AED'], [0.7, '#EC4899'], [1, '#FBBF24']], xgap=3, ygap=3,
            text=[[_inr(v) for v in row] for row in piv.to_numpy()], texttemplate="%{text}",
            textfont=dict(size=11, color='#fff'), customdata=np.dstack([p5.to_numpy(), loss.to_numpy()]),
            colorbar=dict(title="Typical", tickprefix="₹", tickformat=',.0f', thickness=12),
            hovertemplate=("Weights %{y}<br>%{x}<br>Typical: ₹%{z:,.0f}<br>Bad case: ₹%{customdata[0]:,.0f}"
                           "<br>Chance of loss: %{customdata[1]:.1f}%<extra></extra>")))
        fig.update_layout(**PLOT_LAYOUT, height=170 + 30 * len(order),
                          title=dict(text="Typical final value for every weight set × setting", y=0.99),
                          xaxis=dict(side='top'), yaxis=dict(autorange='reversed', title="Weights (biggest first)"))
        fig.update_layout(margin=dict(l=12, r=12, t=130, b=12))
        with panel():
            chart(fig, key="diff_heat")
        note("<b>How to read it:</b> warmer / brighter cells = a better typical result. Each row is a set of weights, "
             "each column a rebalance setting. Hover any cell for its bad case and chance of loss.", color, "👀")

    with t_set:
        _settings_compared('diff', {n: (paths[n], pooled[n]) for n in names}, bh, bh_label="Buy & Hold (reference)")
        pick = st.selectbox("Open the full picture for one setting", names, key="one_diff")
        _stat_cards(pooled[pick], color)
        _outcome_tabs(paths[pick], pooled[pick], color, key="diff_one")

    with t_all:
        st.caption(f"All {len(sets)} weight sets × {len(names)} settings pooled into one big set of futures.")
        _stat_cards(pooled['All settings'], color)
        _outcome_tabs(paths['All settings'], pooled['All settings'], color, key="diff_all")

    with t_tbl:
        tbl = combos.sort_values('Median (₹)', ascending=False).reset_index(drop=True)
        money = [c for c in tbl.columns if '(₹)' in c]
        pct_cols = [c for c in tbl.columns if '(%)' in c]
        st.dataframe(tbl.style.format({**{c: '₹{:,.0f}' for c in money}, **{c: '{:.2f}%' for c in pct_cols}}),
                     width="stretch", hide_index=True, height=440)
        st.download_button("⬇️  Download every combination (CSV)", tbl.to_csv(index=False).encode('utf-8-sig'),
                           file_name="monte_carlo_different_weights_all.csv", mime="text/csv", key="dl_diff_combos")


# =========================================================
# Settings compared (range chart + table)
# =========================================================
def _settings_compared(mode, variants, bh, bh_label):
    color = STRAT_COLORS[mode]
    names = list(variants.keys())
    stats = {n: variants[n][1] for n in names}

    rows = [(bh_label, bh, FAINT)] + [(n, stats[n], color) for n in names]
    rows = rows[::-1]
    labels = [r[0] for r in rows]
    fig = go.Figure()
    notes = []
    for lbl, s, col in rows:
        p25, p75 = np.percentile(s['final'], [25, 75])
        fig.add_trace(go.Scatter(x=[s['p5'], s['p95']], y=[lbl, lbl], mode='lines',
                                 line=dict(color=hex_rgba(col, 0.45), width=4), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[p25, p75], y=[lbl, lbl], mode='lines',
                                 line=dict(color=col, width=18), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=[s['median']], y=[lbl], mode='markers', showlegend=False,
            marker=dict(color='#fff', size=13, line=dict(color='#0A0F1E', width=2.5)),
            hovertemplate=(f"<b>{lbl}</b><br>Bad case: {_inr(s['p5'])}<br>Middle half: {_inr(p25)} – {_inr(p75)}"
                           f"<br>Typical: {_inr(s['median'])}<br>Average: {_inr(s['mean'])}"
                           f"<br>Great case: {_inr(s['p95'])}<br>Chance of loss: {s['prob_loss']:.1f}%<extra></extra>")))
        notes += [dict(x=s['p5'], y=lbl, text=_inr(s['p5']), showarrow=False, xanchor='right', xshift=-6,
                       font=dict(color=BAD, size=11)),
                  dict(x=s['p95'], y=lbl, text=_inr(s['p95']), showarrow=False, xanchor='left', xshift=6,
                       font=dict(color=GOOD, size=11))]
    lo = min(r[1]['p5'] for r in rows)
    hi = max(r[1]['p95'] for r in rows)
    pad = (hi - lo) * 0.12
    fig.update_layout(**PLOT_LAYOUT, annotations=notes, height=130 + 56 * len(rows), hovermode='closest',
                      title="Range of outcomes for every setting",
                      xaxis=dict(title="Final value (₹)", tickprefix="₹", tickformat=',.0f',
                                 range=[max(0, lo - pad), hi + pad]),
                      yaxis=dict(categoryorder='array', categoryarray=labels))
    fig.add_vline(x=AMOUNT, line_dash='dot', line_color=FAINT, annotation_text=f"Invested {_inr(AMOUNT)}",
                  annotation_position='top', annotation_font_color=MUTED)
    with panel():
        chart(fig, key=f"range_{mode}")
    note("<b>How to read it:</b> each row is one setting. Thin line = bad case to great case (9 in 10 futures), "
         "thick bar = the middle half, white dot = typical result. The grey row is Buy & Hold for reference.",
         color, "👀")

    trs = [{'Setting': n, 'Typical (₹)': s['median'], 'Average (₹)': s['mean'], 'Bad case (₹)': s['p5'],
            'Great case (₹)': s['p95'], 'Chance of loss (%)': s['prob_loss'], 'Chance to double (%)': s['prob_double'],
            'Typical CAGR (%)': s['cagr_median'], 'Typical worst fall (%)': s['dd_median']}
           for n, s in list(stats.items()) + [(bh_label, bh)]]
    df = pd.DataFrame(trs)
    money = [c for c in df.columns if '(₹)' in c]
    pct_cols = [c for c in df.columns if '(%)' in c]
    st.dataframe(df.style.format({**{c: '₹{:,.0f}' for c in money}, **{c: '{:.2f}%' for c in pct_cols}}),
                 width="stretch", hide_index=True)


# =========================================================
# Charts
# =========================================================
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
    bar_colors = [hex_rgba(BAD, 0.85) if m < p5 else hex_rgba(GOOD, 0.85) if m > p95 else hex_rgba(color, 0.7)
                  for m in mids]
    hover = [f"{fmt(a)} to {fmt(b)}<br>{c:,} futures ({p:.1f}%)"
             for a, b, c, p in zip(edges[:-1], edges[1:], counts, pct_)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=mids, y=pct_, width=(edges[1] - edges[0]) * 0.88, marker_color=bar_colors,
                         marker_line_width=0, hovertext=hover, hoverinfo="text", showlegend=False))
    grid = np.linspace(lo, hi, 300)
    dens = _kde(shown, grid) * (edges[1] - edges[0]) * 100 * len(shown) / len(data)
    fig.add_trace(go.Scatter(x=grid, y=dens, mode='lines', line=dict(color=INK, width=2.2),
                             hoverinfo='skip', showlegend=False))

    ymax = max(pct_.max(), dens.max()) * 1.08
    marks = [(p5, "Bad case", BAD, 'dash'), (p50, "Typical", INK, 'solid'),
             (mean, "Average", BLUE, 'dot'), (p95, "Great case", GOOD, 'dash')]
    if ref is not None and lo <= ref <= hi:
        marks.append((ref, "Invested" if ref else "0%", MUTED, 'dot'))
    shapes, notes = [], []
    for i, (x, label, col, dash) in enumerate(sorted(marks, key=lambda m: m[0])):
        level = i % 3
        shapes.append(dict(type='line', x0=x, x1=x, y0=0, y1=1.0 + level * 0.11, yref='paper',
                           line=dict(color=col, width=2, dash=dash)))
        notes.append(dict(x=x, y=1.0 + level * 0.11, yref='paper', yanchor='bottom',
                          text=f"<b>{label}</b> {fmt(x)}", showarrow=False, font=dict(color=TEXT, size=11),
                          bgcolor='#182240', bordercolor=col, borderwidth=1.5, borderpad=4))
    x_fmt = {'tickprefix': '₹', 'tickformat': ',.0f'} if '₹' in x_title else {'ticksuffix': '%'}
    fig.update_layout(**PLOT_LAYOUT, height=height, bargap=0, hovermode='closest', shapes=shapes, annotations=notes,
                      title=dict(text=title, y=0.98),
                      xaxis=dict(title=x_title, **x_fmt),
                      yaxis=dict(title="% of futures", ticksuffix="%", range=[0, ymax]))
    fig.update_layout(margin=dict(l=12, r=12, t=120, b=12))
    return fig


def _fan_chart(path, s, color):
    x = np.arange(path.shape[1]) / 12
    fan = s['fan']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=fan[95], line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=fan[5], fill='tonexty', fillcolor=hex_rgba(color, 0.14), line=dict(width=0),
                             name='9 in 10 futures', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=fan[75], line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=fan[25], fill='tonexty', fillcolor=hex_rgba(color, 0.32), line=dict(width=0),
                             name='Middle half of futures', hoverinfo='skip'))
    sample = np.random.default_rng(1).choice(path.shape[0], min(20, path.shape[0]), replace=False)
    xs = np.concatenate([np.append(x, np.nan) for _ in sample])
    ys = np.concatenate([np.append(path[i], np.nan) for i in sample])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='rgba(200,210,235,0.16)', width=1),
                             hoverinfo='skip', name='Sample futures', connectgaps=False))
    for p, nm, col, dash, w in ((95, 'Great case', GOOD, 'dash', 2), (50, 'Typical', INK, 'solid', 3.2),
                                (5, 'Bad case', BAD, 'dash', 2)):
        fig.add_trace(go.Scatter(x=x, y=fan[p], mode='lines', line=dict(color=col, width=w, dash=dash), name=nm,
                                 hovertemplate=f"{nm}<br>Year %{{x:.1f}}: ₹%{{y:,.0f}}<extra></extra>"))
    fig.add_hline(y=AMOUNT, line_dash='dot', line_color=FAINT,
                  annotation_text=f"Invested {_inr(AMOUNT)}", annotation_position='top left',
                  annotation_font_color=MUTED)
    fig.update_layout(**PLOT_LAYOUT, height=480, hovermode='x unified',
                      title=f"Your money over {YEARS} years: the range of futures",
                      xaxis=dict(title="Years from today"),
                      yaxis=dict(title="Portfolio value (₹)", tickprefix="₹", tickformat=',.0f',
                                 range=[0, float(fan[95].max()) * 1.15]),
                      legend=dict(orientation='h', yanchor='top', y=-0.18, x=0))
    return fig