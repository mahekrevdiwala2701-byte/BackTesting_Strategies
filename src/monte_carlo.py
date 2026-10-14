"""Monte Carlo simulation of the 3 investment strategies (no Streamlit code here).

How it works
------------
1. Every asset's price history is turned into MONTHLY returns (last price of each month).
   Only months where all selected assets have data are used, so the assets move together
   exactly as they did in real life (gold up while small caps crash, etc.).
2. "Block bootstrap": a future path is built by picking random chunks of real history
   (e.g. 12 months at a time) and gluing them together until the horizon is filled.
   Chunks keep real trends and crashes intact; random order gives a new possible future.
3. The SAME random paths are run through all three strategies, so the comparison is fair:
     bh       Buy & Hold            - split once, never touched
     periodic Same-weight rebalance - back to your weights every N months
     diff     Different Weights     - on each rebalance, the asset that grew MOST over the
                                      look-back gets the SMALLEST of your weights
4. For every path we record the final value, CAGR and max drawdown, then report the
   mean, median, top 5% / bottom 5% and the chance of losing money.
"""
import numpy as np
import pandas as pd

MAX_MONTHLY_MOVE = 0.60     # a monthly index move above ±60% is treated as a data error


# ---------------------------------------------------------
# 1) Monthly returns from the price history
# ---------------------------------------------------------
def monthly_returns(asset_dfs, names, start_date=None, end_date=None):
    """Monthly % returns (as fractions) for `names`, one column each, months where all have data.

    Returns (returns DataFrame, list of dropped months that looked like data errors).
    """
    closes = {}
    for n in names:
        s = asset_dfs[n].set_index('Date')['Close'].sort_index()
        s = s[~s.index.duplicated(keep='last')]
        closes[n] = s.resample('ME').last()
    m = pd.concat(closes, axis=1, sort=True)
    rets = m.pct_change(fill_method=None)
    rets = rets.dropna(how='any')
    if start_date is not None:
        rets = rets[rets.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        rets = rets[rets.index <= pd.Timestamp(end_date) + pd.offsets.MonthEnd(0)]

    bad = (rets.abs() > MAX_MONTHLY_MOVE).any(axis=1)
    dropped = list(rets.index[bad])
    return rets[~bad], dropped


# ---------------------------------------------------------
# 2) Random future paths (block bootstrap)
# ---------------------------------------------------------
def bootstrap_paths(rets, n_sims, n_months, block, seed=42):
    """Array of shape (n_sims, n_months, n_assets) with simulated monthly growth factors (1 + r)."""
    r = rets.to_numpy(dtype=float)
    T = len(r)
    block = int(max(1, min(block, T)))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_months / block))
    starts = rng.integers(0, T - block + 1, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)[:, :n_months]
    return 1.0 + r[idx]


# ---------------------------------------------------------
# 3) Run one strategy on all paths at once
# ---------------------------------------------------------
def run_strategy(growth, weights, amount, mode, every_months=12, lookback_months=12, warmup=0):
    """Simulate a strategy on every path.

    growth  : (n_sims, warmup + n_months, n_assets) monthly growth factors.
              The first `warmup` months are history only (used by 'diff' to look back),
              money is invested from month `warmup` onwards.
    weights : array of the user's weights (fractions, same order as the assets)
    mode    : 'bh' | 'periodic' | 'diff'
    Returns total portfolio value for every path and month: (n_sims, n_months + 1).
    """
    n_sims, total_m, n_assets = growth.shape
    n_months = total_m - warmup
    w = np.asarray(weights, dtype=float)
    w_big_first = np.sort(w)[::-1]

    # price index of every asset (starts at 1) - used for the look-back growth
    price = np.concatenate([np.ones((n_sims, 1, n_assets)), np.cumprod(growth, axis=1)], axis=1)

    hold = np.tile(amount * w, (n_sims, 1))          # ₹ in each asset
    path = np.empty((n_sims, n_months + 1))
    path[:, 0] = amount
    rows = np.arange(n_sims)[:, None]

    for t in range(1, n_months + 1):
        hold = hold * growth[:, warmup + t - 1, :]
        total = hold.sum(axis=1)
        if mode != 'bh' and t % every_months == 0 and t < n_months:
            if mode == 'periodic':
                hold = total[:, None] * w[None, :]
            else:   # diff
                now = warmup + t
                back = max(0, now - lookback_months)
                g = price[:, now, :] / price[:, back, :]            # growth over the look-back
                lowest_first = np.argsort(g, axis=1, kind='stable')  # lowest growth first
                new_w = np.empty_like(g)
                new_w[rows, lowest_first] = w_big_first[None, :]     # lowest growth -> biggest weight
                hold = total[:, None] * new_w
        path[:, t] = total
    return path


# ---------------------------------------------------------
# 4) Statistics
# ---------------------------------------------------------
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def summarize(path, amount, years):
    """Numbers for one strategy's dashboard."""
    final = path[:, -1]
    cagr = (np.maximum(final, 1e-9) / amount) ** (1 / years) - 1
    peak = np.maximum.accumulate(path, axis=1)
    max_dd = (path / peak - 1).min(axis=1)
    return {
        'final': final,
        'cagr': cagr * 100,
        'max_dd': max_dd * 100,
        'mean': final.mean(),
        'median': np.median(final),
        'p5': np.percentile(final, 5),
        'p95': np.percentile(final, 95),
        'std': final.std(),
        'prob_loss': (final < amount).mean() * 100,
        'prob_double': (final >= 2 * amount).mean() * 100,
        'cagr_mean': cagr.mean() * 100,
        'cagr_median': np.median(cagr) * 100,
        'cagr_p5': np.percentile(cagr, 5) * 100,
        'cagr_p95': np.percentile(cagr, 95) * 100,
        'dd_median': np.median(max_dd) * 100,
        'dd_p5': np.percentile(max_dd, 5) * 100,         # worst 5% drawdown
        'cvar5': final[final <= np.percentile(final, 5)].mean(),   # average of the worst 5%
        'fan': {p: np.percentile(path, p, axis=0) for p in (5, 25, 50, 75, 95)},
    }


def percentile_table(stats, amount, years):
    rows = []
    for p in PERCENTILES:
        v = np.percentile(stats['final'], p)
        rows.append({
            'Percentile': f"{p}{'st' if p % 10 == 1 and p != 11 else 'th'}",
            'Meaning': ('worst case' if p == 1 else 'bad case' if p <= 10 else 'below typical' if p < 50
                        else 'typical (median)' if p == 50 else 'above typical' if p < 90
                        else 'good case' if p < 99 else 'best case'),
            'Final Value (₹)': v,
            'Total Return (%)': (v / amount - 1) * 100,
            'CAGR (%)': ((max(v, 1e-9) / amount) ** (1 / years) - 1) * 100,
        })
    return pd.DataFrame(rows)


def run_all(rets, weights, amount, years, n_sims, block, periodic_every, diff_every, diff_lookback, seed=42):
    """Run all 3 strategies on the same random paths. Returns {mode: (path, stats)}."""
    n_months = int(round(years * 12))
    warmup = max(18, diff_lookback)                     # history in front of every path for the look-back
    growth = bootstrap_paths(rets, n_sims, warmup + n_months, block, seed)
    out = {}
    for mode, every in (('bh', 12), ('periodic', periodic_every), ('diff', diff_every)):
        path = run_strategy(growth, weights, amount, mode, every, diff_lookback, warmup)
        out[mode] = (path, summarize(path, amount, years))
    return out


def run_variants(rets, weights, amount, years, n_sims, block, rebal_freq, diff_freq, lookbacks, seed=42):
    """Every possible setting of every strategy, all on the SAME random futures.

    rebal_freq : {label: months}  e.g. {"Monthly": 1, "Quarterly": 3, ...}
    diff_freq  : {label: months}  e.g. {"Half-Yearly": 6, "Yearly": 12}
    lookbacks  : {label: months}  e.g. {"6 Months": 6, "1 Year": 12, "1.5 Years": 18}
    Returns {'bh': {name: (path, stats)}, 'periodic': {...}, 'diff': {...}}
    """
    n_months = int(round(years * 12))
    warmup = max(18, max(lookbacks.values()))
    growth = bootstrap_paths(rets, n_sims, warmup + n_months, block, seed)
    w = np.asarray(weights, dtype=float)
    out = {'bh': {}, 'periodic': {}, 'diff': {}}

    path = run_strategy(growth, w, amount, 'bh', 12, 12, warmup)
    out['bh']['Buy & Hold'] = (path, summarize(path, amount, years))
    for f_lbl, f in rebal_freq.items():
        path = run_strategy(growth, w, amount, 'periodic', f, 12, warmup)
        out['periodic'][f_lbl] = (path, summarize(path, amount, years))
    for f_lbl, f in diff_freq.items():
        for l_lbl, lb in lookbacks.items():
            path = run_strategy(growth, w, amount, 'diff', f, lb, warmup)
            out['diff'][f"{f_lbl} · look-back {l_lbl}"] = (path, summarize(path, amount, years))
    return out


# ---------------------------------------------------------
# Different Weights with ALL possible weight splits
# ---------------------------------------------------------
def all_weight_sets(step=10):
    """Every way to split 100% over 4 assets in `step`% steps, biggest weight first.

    For Different Weights only the SET of weights matters (which asset gets which weight is
    decided by growth on every rebalance), so 40/30/20/10 and 10/20/30/40 are the same set.
    25/25/25/25 is added as well.
    """
    units = 100 // step
    sets = []
    for a in range(units, -1, -1):
        for b in range(min(a, units - a), -1, -1):
            for c in range(min(b, units - a - b), -1, -1):
                d = units - a - b - c
                if 0 <= d <= c:
                    sets.append((a * step, b * step, c * step, d * step))
    if (25, 25, 25, 25) not in sets:
        sets.append((25, 25, 25, 25))
    return sets


def _summary_from_arrays(final, max_dd, sample_paths, amount, years):
    cagr = (np.maximum(final, 1e-9) / amount) ** (1 / years) - 1
    return {
        'final': final, 'cagr': cagr * 100, 'max_dd': max_dd * 100,
        'mean': final.mean(), 'median': np.median(final),
        'p5': np.percentile(final, 5), 'p95': np.percentile(final, 95), 'std': final.std(),
        'prob_loss': (final < amount).mean() * 100, 'prob_double': (final >= 2 * amount).mean() * 100,
        'cagr_mean': cagr.mean() * 100, 'cagr_median': np.median(cagr) * 100,
        'cagr_p5': np.percentile(cagr, 5) * 100, 'cagr_p95': np.percentile(cagr, 95) * 100,
        'dd_median': np.median(max_dd) * 100, 'dd_p5': np.percentile(max_dd, 5) * 100,
        'cvar5': final[final <= np.percentile(final, 5)].mean(),
        'fan': {p: np.percentile(sample_paths, p, axis=0) for p in (5, 25, 50, 75, 95)},
    }


def run_diff_many_weights(growth, price, weight_sets, amount, every_months, lookback_months, warmup):
    """Different Weights for MANY weight sets at once (same futures, one pass over the months).

    The growth ranking on a rebalance date depends only on the future, not on the weights, so it is
    computed once and every weight set just hands out its own weights by that ranking.
    Returns paths of shape (n_sims, n_sets, n_months + 1).
    """
    n_sims, total_m, n_assets = growth.shape
    n_months = total_m - warmup
    W = np.asarray(weight_sets, dtype=float)                     # (n_sets, n_assets)
    W_big_first = -np.sort(-W, axis=1)                           # each set, biggest weight first
    hold = amount * np.broadcast_to(W, (n_sims,) + W.shape).copy()   # (n_sims, n_sets, n_assets)
    path = np.empty((n_sims, W.shape[0], n_months + 1))
    path[:, :, 0] = amount
    for t in range(1, n_months + 1):
        hold *= growth[:, warmup + t - 1, None, :]
        total = hold.sum(axis=2)                                 # (n_sims, n_sets)
        if t % every_months == 0 and t < n_months:
            now = warmup + t
            back = max(0, now - lookback_months)
            g = price[:, now, :] / price[:, back, :]             # (n_sims, n_assets)
            rank = np.argsort(np.argsort(g, axis=1, kind='stable'), axis=1)   # 0 = lowest growth
            # asset with rank r (0 = lowest growth) gets the r-th biggest weight of each set
            new_w = np.take_along_axis(np.broadcast_to(W_big_first, (n_sims,) + W.shape),
                                       np.broadcast_to(rank[:, None, :], (n_sims,) + W.shape), axis=2)
            hold = total[:, :, None] * new_w
        path[:, :, t] = total
    return path


def run_diff_all_weights(rets, amount, years, n_sims, block, diff_freq, lookbacks, step=10, seed=42,
                         keep_paths=150):
    """Different Weights for EVERY weight set x EVERY setting, on the same random futures.

    Returns
      pooled : {setting: summary over all weight sets together}   (+ 'All settings' over everything)
      paths  : {setting: sample of paths for the fan chart}
      combos : DataFrame, one row per (weight set, setting) with its key numbers
    """
    n_months = int(round(years * 12))
    warmup = max(18, max(lookbacks.values()))
    n_assets = rets.shape[1]
    growth = bootstrap_paths(rets, n_sims, warmup + n_months, block, seed)
    price = np.concatenate([np.ones((n_sims, 1, n_assets)), np.cumprod(growth, axis=1)], axis=1)
    sets = [ws for ws in all_weight_sets(step) if sum(ws[:n_assets]) > 0]
    rng = np.random.default_rng(seed + 1)

    pooled, paths, rows = {}, {}, []
    every = {'final': [], 'dd': [], 'paths': []}
    for f_lbl, f in diff_freq.items():
        for l_lbl, lb in lookbacks.items():
            name = f"{f_lbl} · look-back {l_lbl}"
            finals, dds, samp = [], [], []
            W = np.array([np.array(ws[:n_assets], dtype=float) / sum(ws[:n_assets]) for ws in sets])
            allp = run_diff_many_weights(growth, price, W, amount, f, lb, warmup)   # (sims, sets, months+1)
            FIN = allp[:, :, -1]                                                     # (sims, sets)
            DD = (allp / np.maximum.accumulate(allp, axis=2) - 1).min(axis=2)        # (sims, sets)
            CAGR = (np.maximum(FIN, 1e-9) / amount) ** (1 / years) - 1
            q_fin = np.percentile(FIN, [5, 50, 95], axis=0)
            med_cagr, med_dd = np.median(CAGR, axis=0), np.median(DD, axis=0)
            for j, ws in enumerate(sets):
                finals.append(FIN[:, j])
                dds.append(DD[:, j])
                samp.append(allp[rng.choice(n_sims, min(keep_paths, n_sims), replace=False), j, :])
                rows.append({
                    'Weights': " / ".join(f"{x:g}" for x in ws), 'Setting': name,
                    'Mean (₹)': FIN[:, j].mean(), 'Median (₹)': q_fin[1, j],
                    'Bottom 5% (₹)': q_fin[0, j], 'Top 5% (₹)': q_fin[2, j],
                    'Chance of loss (%)': (FIN[:, j] < amount).mean() * 100,
                    'Median CAGR (%)': med_cagr[j] * 100,
                    'Typical max fall (%)': med_dd[j] * 100,
                })
            fin_all, dd_all, sp = np.concatenate(finals), np.concatenate(dds), np.vstack(samp)
            pooled[name] = _summary_from_arrays(fin_all, dd_all, sp, amount, years)
            paths[name] = sp
            every['final'].append(fin_all)
            every['dd'].append(dd_all)
            every['paths'].append(sp[::len(diff_freq) * len(lookbacks)])
    sp = np.vstack(every['paths'])
    pooled['All settings'] = _summary_from_arrays(np.concatenate(every['final']), np.concatenate(every['dd']),
                                                  sp, amount, years)
    paths['All settings'] = sp
    return pooled, paths, pd.DataFrame(rows), sets


# ---------------------------------------------------------
# Buy & Hold / Rebalance with ALL weight combinations
# ---------------------------------------------------------
def all_weight_combos(n_assets=4, step=10):
    """Every split of 100% over the assets in `step`% steps, in asset order (Gold, Large, Mid, Small).

    Here the order matters (40% Gold is not 40% SmallCap), so 10% steps over 4 assets give 286
    combinations; 25/25/25/25 is added too.
    """
    units = 100 // step
    combos = []

    def rec(prefix, left, k):
        if k == 1:
            combos.append(tuple(prefix + [left * step]))
            return
        for x in range(left, -1, -1):
            rec(prefix + [x * step], left - x, k - 1)

    rec([], units, n_assets)
    eq = tuple([100 / n_assets] * n_assets)
    if 100 % n_assets == 0 and eq not in combos:
        combos.append(eq)
    return combos


def run_linear_many_weights(price, W, amount, every_months, warmup, n_months, sample_idx, chunk=12):
    """Buy & Hold (every_months=None) or same-weight rebalancing for MANY weight sets at once.

    Between two rebalances the portfolio is linear in the weights:
        value(t) = value(start) * Σ_i w_i * price_i(t) / price_i(start)
    so every weight set is one matrix product. Only what is needed is kept:
    final value, max drawdown, and full paths for a few sample futures (for the fan chart).
    Returns FIN (sims, sets), DD (sims, sets), SAMPLE (len(sample_idx), sets, n_months + 1).
    """
    n_sims = price.shape[0]
    n_sets = W.shape[0]
    total = np.full((n_sims, n_sets), float(amount))
    peak = total.copy()
    min_dd = np.zeros((n_sims, n_sets))
    sample = np.empty((len(sample_idx), n_sets, n_months + 1))
    sample[:, :, 0] = amount
    every = n_months if not every_months else every_months

    s = 0
    while s < n_months:
        e = min(s + every, n_months)                       # one rebalance period: months s+1 .. e
        base = price[:, warmup + s, :]                     # prices on the rebalance date
        start_total = total
        c = s
        while c < e:                                       # walk the period in small chunks (memory)
            d = min(c + chunk, e)
            ratio = price[:, warmup + c + 1: warmup + d + 1, :] / base[:, None, :]      # (sims, L, assets)
            vals = start_total[:, None, :] * np.einsum('sla,ka->slk', ratio, W)         # (sims, L, sets)
            run_peak = np.maximum(peak[:, None, :], np.maximum.accumulate(vals, axis=1))
            min_dd = np.minimum(min_dd, (vals / run_peak - 1).min(axis=1))
            peak = run_peak[:, -1, :]
            sample[:, :, c + 1: d + 1] = vals[sample_idx].transpose(0, 2, 1)
            total = vals[:, -1, :]
            c = d
        s = e
    return total, min_dd, sample


def run_combos(rets, amount, years, n_sims, block, settings, step=10, seed=42, n_sample=12):
    """Buy & Hold or Rebalance for EVERY weight combination x EVERY setting, same random futures.

    settings : {name: rebalance every N months, or None for Buy & Hold}
    Returns pooled {name: summary over all combos} (+ 'All settings'), sample paths, combos table, weight list.
    """
    n_months = int(round(years * 12))
    warmup = 18
    n_assets = rets.shape[1]
    growth = bootstrap_paths(rets, n_sims, warmup + n_months, block, seed)
    price = np.concatenate([np.ones((n_sims, 1, n_assets)), np.cumprod(growth, axis=1)], axis=1)
    combos_w = all_weight_combos(n_assets, step)
    W = np.array(combos_w, dtype=float) / 100.0
    sample_idx = np.random.default_rng(seed + 2).choice(n_sims, min(n_sample, n_sims), replace=False)
    names = list(rets.columns)

    pooled, paths, rows = {}, {}, []
    every = {'final': [], 'dd': [], 'paths': []}
    for name, k in settings.items():
        FIN, DD, SAMPLE = run_linear_many_weights(price, W, amount, k, warmup, n_months, sample_idx)
        CAGR = (np.maximum(FIN, 1e-9) / amount) ** (1 / years) - 1
        q = np.percentile(FIN, [5, 50, 95], axis=0)
        med_cagr, med_dd = np.median(CAGR, axis=0), np.median(DD, axis=0)
        mean, loss = FIN.mean(axis=0), (FIN < amount).mean(axis=0) * 100
        for j, ws in enumerate(combos_w):
            row = {'Setting': name}
            row.update({f"{a} (%)": round(x, 1) for a, x in zip(names, ws)})
            row.update({
                'Mean (₹)': mean[j], 'Median (₹)': q[1, j], 'Bottom 5% (₹)': q[0, j], 'Top 5% (₹)': q[2, j],
                'Chance of loss (%)': loss[j], 'Median CAGR (%)': med_cagr[j] * 100,
                'Typical max fall (%)': med_dd[j] * 100,
            })
            rows.append(row)
        fin_all, dd_all = FIN.ravel(), DD.ravel()
        sp = SAMPLE.reshape(-1, n_months + 1)
        pooled[name] = _summary_from_arrays(fin_all, dd_all, sp, amount, years)
        paths[name] = sp
        every['final'].append(fin_all)
        every['dd'].append(dd_all)
        every['paths'].append(sp[::max(1, len(settings))])
    if len(settings) > 1:
        sp = np.vstack(every['paths'])
        pooled['All settings'] = _summary_from_arrays(np.concatenate(every['final']), np.concatenate(every['dd']),
                                                      sp, amount, years)
        paths['All settings'] = sp
    else:
        only = next(iter(settings))
        pooled['All settings'], paths['All settings'] = pooled[only], paths[only]
    return pooled, paths, pd.DataFrame(rows), combos_w