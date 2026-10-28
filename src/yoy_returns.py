"""Same-date, year-over-year returns of the TOTAL PORTFOLIO under the 3 strategies
of the Calculated Investment page (no Streamlit code here).

For every trading day D (the investment date), the whole portfolio is bought on D and valued on the
same date N years later (D + N years; if that is a holiday, the last trading day before it):

    Buy & Hold        : split once by your weights, never touched
                        return = Σ w_i × (P_i(end) / P_i(D)) − 1
    Rebalancing       : back to your weights every `periodic_months` months after D
                        return = Π over each rebalance period of Σ w_i × (P_i(next) / P_i(this)) − 1
    Different Weights : every `diff_months` months after D, each asset's growth over the previous
                        `lookback_months` is ranked: highest growth gets the SMALLEST of your weights,
                        lowest growth the BIGGEST (same rule as src/strategies.py)

Trading days only: a day is used when every asset in the portfolio has a recent real price
(within 7 days for daily data, ~1.5 months for monthly data such as SmallCap).
Obviously wrong prices (e.g. ₹7 in a ~₹10,000 index) are removed first and reported.
"""
import numpy as np
import pandas as pd

STRATEGY_KEYS = ("bh", "periodic", "diff")
STRATEGY_NAMES = {"bh": "Buy & Hold", "periodic": "Rebalancing", "diff": "Different Weights"}


# ---------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------
def clean_prices(df, window_days=365, low=0.3, high=3.3):
    """Drop a row when its close is below 30% or above 330% of the median close in the surrounding
    ±window_days (wide enough that real crashes such as 2008 or 2020 are kept).
    Returns (clean Series indexed by Date, removed rows DataFrame)."""
    d = df[['Date', 'Close']].dropna().sort_values('Date').drop_duplicates('Date', keep='last')
    d = d.reset_index(drop=True)
    if len(d) < 5:
        return d.set_index('Date')['Close'], d.iloc[0:0]
    med = d.set_index('Date')['Close'].rolling(f'{2 * window_days}D', center=True, min_periods=1).median()
    ratio = d['Close'].to_numpy() / med.to_numpy()
    bad = (ratio < low) | (ratio > high)
    return d[~bad].set_index('Date')['Close'], d[bad].reset_index(drop=True)


def _max_age_days(series):
    """How old a price may be on a trading day: 7 days for daily data, ~1.5 months for monthly data."""
    spacing = pd.Series(series.index).diff().dt.days.median()
    spacing = 1 if pd.isna(spacing) else spacing
    return int(max(7, 1.5 * spacing + 3))


def price_panel(asset_dfs, assets):
    """Trading-day price table for the portfolio.
    Returns (prices DataFrame on valid trading days, clean full-history series, data-check info)."""
    clean, removed, ages = {}, [], {}
    for a in assets:
        s, bad = clean_prices(asset_dfs[a])
        s.index = s.index.astype('datetime64[ns]')
        clean[a] = s
        ages[a] = _max_age_days(s)
        removed += [[a, d.strftime('%Y-%m-%d'), float(c)] for d, c in zip(bad['Date'], bad['Close'])]

    calendar = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in clean.values()])))
    prices = pd.DataFrame({a: s.reindex(calendar).ffill() for a, s in clean.items()})
    valid = prices.notna().all(axis=1)
    for a, s in clean.items():                       # price must be recent (not across a hole in the data)
        last_real = pd.Series(s.index, index=s.index).reindex(calendar).ffill()
        age = (calendar - pd.DatetimeIndex(last_real)).days
        valid &= pd.Series(age <= ages[a], index=calendar).fillna(False).astype(bool)
    info = {'removed_prices': removed, 'excluded_days': int((~valid & prices.notna().all(axis=1)).sum())}
    return prices[valid], clean, info


# ---------------------------------------------------------
# Strategy maths (vectorised over every investment date)
# ---------------------------------------------------------
def _rebalance_positions(T, start_pos, end_pos, every_months):
    """Matrix of positions [start, rebalance 1, ..., end] for every investment date.
    Rebalance k = first trading day on/after start + k × every_months months (same rule as
    strategies.build_rebalance_dates); rebalances after the end collapse onto the end (no effect)."""
    starts = T[start_pos]
    end_dates = T[end_pos]
    cols = [start_pos]
    k = 1
    while True:
        target = starts + pd.DateOffset(months=every_months * k)
        if (target > end_dates).all():
            break
        pos = np.searchsorted(T.values, target.values, side='left')
        pos = np.where(target <= end_dates, np.minimum(pos, end_pos), end_pos)
        cols.append(pos)
        k += 1
    cols.append(end_pos)
    return np.column_stack(cols)


def _segment_growth(P, R, W):
    """Portfolio growth factor through the rebalance positions R with weights W (one row per period)."""
    g = np.ones(R.shape[0])
    for k in range(R.shape[1] - 1):
        a, b = R[:, k], R[:, k + 1]
        w = W[k] if W.ndim == 3 else W
        g *= np.sum(w * P[b] / P[a], axis=1)
    return g


def _inverse_growth_weights(growth, base):
    """Same rule as strategies.inverse_growth_weights, for one row of growth numbers.
    Highest growth -> smallest weight, lowest growth -> biggest. NaN growth keeps its own weight."""
    new = base.copy()
    ok = ~np.isnan(growth)
    if ok.sum() == 0:
        return new
    idx = np.where(ok)[0]
    lowest_first = idx[np.lexsort((idx, growth[idx]))]          # ties keep the original order
    new[lowest_first] = np.sort(base[idx])[::-1]
    return new


def _lookback_growth(clean, assets, dates, months):
    """Growth (%) of each asset over the `months` before each date, from the full clean history."""
    out = np.full((len(dates), len(assets)), np.nan)
    back = dates - pd.DateOffset(months=months)
    for j, a in enumerate(assets):
        s = clean[a]
        idx = s.index.values
        now_i = np.searchsorted(idx, dates.values, side='right') - 1
        then_i = np.searchsorted(idx, back.values, side='right') - 1
        ok = (now_i >= 0) & (then_i >= 0)
        v = s.values
        out[ok, j] = (v[now_i[ok]] / v[then_i[ok]] - 1) * 100
    return out


def strategy_yoy(asset_dfs, weights, amount, start_date, end_date, years=1,
                 periodic_months=3, diff_months=6, lookback_months=12, short_names=None):
    """Year-over-year table: one row per investment (trading) day, with every asset's prices and the
    total-portfolio value and return under all 3 strategies."""
    assets = [a for a, w in weights.items() if w > 0 and a in asset_dfs]
    if not assets:
        return pd.DataFrame()
    names = short_names or {a: a for a in assets}
    base = np.array([weights[a] for a in assets], dtype=float)
    base = base / base.sum()

    prices, clean, info = price_panel(asset_dfs, assets)
    if prices.empty:
        return pd.DataFrame()
    T = prices.index
    P = prices.to_numpy()

    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    start_pos = np.where((T >= start_ts) & (T <= end_ts))[0]
    if start_pos.size == 0:
        return pd.DataFrame()
    targets = T[start_pos] + pd.DateOffset(years=years)            # same date, N years later
    end_pos = np.searchsorted(T.values, targets.values, side='right') - 1
    close_enough = (targets - T[end_pos]).days <= 7                 # not across a hole in the data
    keep = (targets <= min(end_ts, T[-1])) & close_enough & (end_pos > start_pos)
    info['excluded_days'] += int(((targets <= min(end_ts, T[-1])) & ~close_enough).sum())
    start_pos, end_pos, targets = start_pos[keep], end_pos[keep], targets[keep]
    if start_pos.size == 0:
        return pd.DataFrame()

    out = pd.DataFrame({'Invest Date': T[start_pos], 'Compare Date': targets,
                        'Compare Trading Day': T[end_pos]})
    for j, a in enumerate(assets):
        n = names[a]
        out[f'{n} Invest Price'] = P[start_pos, j]
        out[f'{n} Compare Price'] = P[end_pos, j]
        out[f'{n} Return (%)'] = (P[end_pos, j] / P[start_pos, j] - 1) * 100

    growth = {}
    # Buy & Hold
    growth['bh'] = np.sum(base * P[end_pos] / P[start_pos], axis=1)
    # Rebalancing to the same weights
    R = _rebalance_positions(T, start_pos, end_pos, periodic_months)
    growth['periodic'] = _segment_growth(P, R, base)
    # Different Weights (growth-based)
    R = _rebalance_positions(T, start_pos, end_pos, diff_months)
    W = np.empty((R.shape[1] - 1, R.shape[0], len(assets)))
    W[0] = base
    for k in range(1, R.shape[1] - 1):
        g = _lookback_growth(clean, assets, T[R[:, k]], lookback_months)
        W[k] = np.vstack([_inverse_growth_weights(row, base) for row in g])
    growth['diff'] = _segment_growth(P, R, W)

    for key in STRATEGY_KEYS:
        label = STRATEGY_NAMES[key]
        r = (growth[key] - 1) * 100
        out[f'{label} Value (₹)'] = amount * growth[key]
        out[f'{label} Return (%)'] = r
        if years > 1:
            out[f'{label} CAGR (%)'] = (growth[key] ** (1 / years) - 1) * 100
        out[f'{label} Result'] = np.where(r > 0, 'Positive', np.where(r < 0, 'Negative', 'Flat'))
    out['Invest Year'] = out['Invest Date'].dt.year

    out.attrs.update(info)
    out.attrs['assets'] = [names[a] for a in assets]
    out.attrs['weights'] = {names[a]: float(w) for a, w in zip(assets, base)}
    return out


# ---------------------------------------------------------
# Summaries
# ---------------------------------------------------------
def summary_stats(r):
    """Headline numbers for one column of returns (%)."""
    r = pd.Series(r).dropna()
    if r.empty:
        return {}
    return {
        'count': int(r.size), 'positive': int((r > 0).sum()), 'negative': int((r < 0).sum()),
        'pct_positive': float((r > 0).mean() * 100), 'mean': float(r.mean()), 'median': float(r.median()),
        'std': float(r.std(ddof=1)) if r.size > 1 else 0.0, 'min': float(r.min()), 'max': float(r.max()),
        'p5': float(r.quantile(0.05)), 'p95': float(r.quantile(0.95)),
    }


def yearly_summary(table):
    """Average return per investment year for each strategy, plus % positive."""
    rows = []
    for year, g in table.groupby('Invest Year'):
        row = {'Year': year, 'Days': len(g)}
        for key in STRATEGY_KEYS:
            r = g[f'{STRATEGY_NAMES[key]} Return (%)']
            row[f'{STRATEGY_NAMES[key]} Avg (%)'] = r.mean()
            row[f'{STRATEGY_NAMES[key]} % Positive'] = (r > 0).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


def normal_curve(mean, std, lo, hi, n=400):
    """x, y points of the normal (bell) curve with the given mean and standard deviation."""
    x = np.linspace(lo, hi, n)
    if std <= 0:
        return x, np.zeros_like(x)
    return x, np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def to_csv(table):
    """CSV text: dates as DD-MM-YYYY, numbers rounded, Excel-friendly (₹ shows correctly)."""
    t = table.drop(columns=['Invest Year'], errors='ignore').copy()
    for c in ('Invest Date', 'Compare Date', 'Compare Trading Day'):
        if c in t.columns:
            t[c] = pd.to_datetime(t[c]).dt.strftime('%d-%m-%Y')
    for c in t.columns:
        if pd.api.types.is_float_dtype(t[c]):
            t[c] = t[c].round(3 if '%' in c else 2)
    return t.to_csv(index=False).encode('utf-8-sig')