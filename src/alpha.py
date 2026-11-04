"""Monthly alpha of a whole portfolio against one benchmark index (no Streamlit code here).

The portfolio is built with one of the 3 strategies of the Calculated Investment page:

    bh        Buy & Hold          : split once, never touched
    periodic  Rebalance           : back to the same weights every month / quarter / half-year / year
    diff      Different Weights   : weights re-assigned by past growth on every rebalance date

For every month:
    portfolio return = value at month end / value at the end of the previous month − 1
    benchmark return = benchmark close at month end / close at the end of the previous month − 1
    alpha (%)        = portfolio return − benchmark return
    alpha (₹)        = alpha (%) × portfolio value at the start of the month
"""
import pandas as pd

from src.yoy_returns import clean_prices, _max_gap_days
from src.strategies import build_rebalance_dates, simulate_portfolio, simulate_buy_hold, build_diff_schedule


# ---------------------------------------------------------
# Prices
# ---------------------------------------------------------
def _clean_close(df):
    """Close prices of one asset with obviously wrong rows removed, indexed by date."""
    d, _ = clean_prices(df)
    s = d.set_index('Date')['Close']
    s.index = pd.to_datetime(s.index).normalize()
    return s[~s.index.duplicated(keep='last')].sort_index(), _max_gap_days(d)


def benchmark_close(asset_dfs, bench):
    """Clean close prices of the benchmark (full history)."""
    return _clean_close(asset_dfs[bench])[0]


# ---------------------------------------------------------
# Portfolio value every day
# ---------------------------------------------------------
def calculated_investment_values(asset_dfs, weights, amount, start_date, end_date, mode="bh",
                                 freq_months=3, lookback_months=12):
    """Calculated Investment: same price table and strategies as that page (with obviously wrong prices
    removed first, like the Main Dashboard does).
    mode: 'bh' (Buy & Hold), 'periodic' (rebalance every freq_months), 'diff' (Different Weights)."""
    used = [a for a, w in weights.items() if w > 0 and a in asset_dfs]
    if not used:
        return None
    total = sum(weights[a] for a in used)
    w0 = {a: weights[a] / total for a in used}

    p_start, p_end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    clean = {a: _clean_close(asset_dfs[a])[0] for a in used}          # obviously wrong prices removed
    full_prices = pd.concat(clean, axis=1, sort=True).loc[:p_end]
    trade_days = full_prices.index[full_prices.index >= p_start]
    raw = full_prices.ffill().loc[trade_days]
    first = {a: raw[a].first_valid_index() for a in used}
    if any(d is None for d in first.values()):
        return None
    prices = raw.loc[min(first.values()):] if mode == "bh" else raw.loc[max(first.values()):]
    if len(prices) < 2:
        return None

    if mode == "bh":
        _, values, _ = simulate_buy_hold(prices, w0, amount)
    elif mode == "periodic":
        schedule = [(d, w0) for d in build_rebalance_dates(prices.index, freq_months)]
        _, values, _ = simulate_portfolio(prices, schedule, amount)
    else:
        schedule, _ = build_diff_schedule(prices, clean, w0, freq_months, lookback_months)
        _, values, _ = simulate_portfolio(prices, schedule, amount)
    return values['Total']


# ---------------------------------------------------------
# Monthly alpha
# ---------------------------------------------------------
def monthly_alpha(values, bench):
    """One row per month: portfolio and benchmark return, alpha in % and ₹, and the alpha built up so far.

    values : portfolio value (₹) every day
    bench  : benchmark close prices (any dates; the last close on or before each day is used)
    """
    values = values.dropna()
    values = values[~values.index.duplicated(keep='last')].sort_index()
    if len(values) < 2:
        return pd.DataFrame()
    b = bench.reindex(bench.index.union(values.index)).ffill().reindex(values.index)
    df = pd.DataFrame({'P': values, 'B': b}).dropna()
    if len(df) < 2:
        return pd.DataFrame()

    start = df.iloc[0]
    month = df.index.to_period('M')
    ends = df.groupby(month).last()
    end_dates = pd.Series(df.index, index=df.index).groupby(month).last()

    p_prev, b_prev, d_prev = start['P'], start['B'], df.index[0]
    rows = []
    for (per, row), d_end in zip(ends.iterrows(), end_dates):
        if d_end == d_prev:                                     # first month holds only the starting day
            continue
        p_ret = row['P'] / p_prev - 1
        b_ret = row['B'] / b_prev - 1
        part = []
        if d_prev == df.index[0] and d_prev.day > 7:
            part.append(f"from {d_prev:%d-%m}")
        if d_end < per.end_time.normalize() - pd.Timedelta(days=4):
            part.append(f"till {d_end:%d-%m}")
        rows.append({
            'Month': per.strftime('%b %Y'),
            'From': d_prev, 'To': d_end,
            'Value at start (₹)': p_prev, 'Value at end (₹)': row['P'],
            'Portfolio (%)': p_ret * 100,
            'Benchmark (%)': b_ret * 100,
            'Alpha (%)': (p_ret - b_ret) * 100,
            'Alpha (₹)': (p_ret - b_ret) * p_prev,
            'Total alpha so far (%)': ((row['P'] / start['P']) / (row['B'] / start['B']) - 1) * 100,
            'Part month': " · ".join(part),
        })
        p_prev, b_prev, d_prev = row['P'], row['B'], d_end
    out = pd.DataFrame(rows)
    out.attrs['start'] = {'date': df.index[0], 'value': start['P'], 'bench': start['B']}
    out.attrs['end'] = {'date': df.index[-1], 'value': df['P'].iloc[-1], 'bench': df['B'].iloc[-1]}
    return out


def alpha_summary(m):
    """Whole-period numbers for a monthly_alpha() table."""
    if m is None or m.empty:
        return {}
    s, e = m.attrs['start'], m.attrs['end']
    years = max((e['date'] - s['date']).days / 365.25, 1e-9)
    p_growth, b_growth = e['value'] / s['value'], e['bench'] / s['bench']
    a = m['Alpha (%)']
    return {
        'months': int(len(m)),
        'years': years,
        'port_total': (p_growth - 1) * 100,
        'bench_total': (b_growth - 1) * 100,
        'alpha_total': (p_growth / b_growth - 1) * 100,
        'alpha_rupees': e['value'] - s['value'] * b_growth,      # vs the same money in the benchmark
        'port_cagr': (p_growth ** (1 / years) - 1) * 100,
        'bench_cagr': (b_growth ** (1 / years) - 1) * 100,
        'beat': int((a > 0).sum()),
        'avg_month': float(a.mean()),
        'best': m.loc[a.idxmax()],
        'worst': m.loc[a.idxmin()],
    }