"""Alpha of a whole portfolio against one benchmark index (no Streamlit code here).

The portfolio is built with one of the 3 strategies of the Calculated Investment page:

    bh        Buy & Hold          : split once, never touched
    periodic  Rebalance           : back to the same weights every month / quarter / half-year / year
    diff      Different Weights   : weights re-assigned by past growth on every rebalance date

For every week / month / quarter / half-year / year:
    portfolio return = value at the period end / value at the end of the previous period − 1
    benchmark return = benchmark close at the period end / close at the end of the previous period − 1
    alpha (%)        = portfolio return − benchmark return   (the "active return")
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
# Alpha per week / month / quarter / half-year / year
# ---------------------------------------------------------
FREQUENCIES = {                 # label -> (unit, pandas period code)
    "Weekly": ("week", "W-FRI"),
    "Monthly": ("month", "M"),
    "Quarterly": ("quarter", "Q"),
    "Half-yearly": ("half-year", "H"),
    "Yearly": ("year", "Y"),
}


def _period_keys(dates, freq):
    """For each date: (period label, first day of the period, last day of the period)."""
    code = FREQUENCIES[freq][1]
    if code == "H":                                            # Jan-Jun and Jul-Dec
        first_half = dates.month <= 6
        labels = [f"{'Jan–Jun' if h else 'Jul–Dec'} {y}" for h, y in zip(first_half, dates.year)]
        starts = [pd.Timestamp(y, 1 if h else 7, 1) for h, y in zip(first_half, dates.year)]
        ends = [pd.Timestamp(y, 6, 30) if h else pd.Timestamp(y, 12, 31) for h, y in zip(first_half, dates.year)]
        return pd.Index(labels), pd.DatetimeIndex(starts), pd.DatetimeIndex(ends)
    p = dates.to_period(code)
    starts, ends = p.start_time.normalize(), p.end_time.normalize()
    if code == "W-FRI":
        labels = ["Week to " + e.strftime('%d-%b-%y') for e in ends]
    elif code == "M":
        labels = list(p.strftime('%b-%y'))
    elif code == "Q":
        labels = [f"{s:%b}–{e:%b} {e:%Y}" for s, e in zip(starts, ends)]
    else:
        labels = [str(y) for y in p.year]
    return pd.Index(labels), starts, ends


def period_alpha(values, bench, freq="Monthly"):
    """One row per period, oldest first:

        Benchmark index / Portfolio index : value at the end of the period, both starting at 100
        Benchmark (%) / Portfolio (%)     : return in that period
        Alpha (%)                         : Portfolio (%) − Benchmark (%)   (active return)

    values : portfolio value (₹) every day
    bench  : benchmark close prices (the last close on or before each day is used)
    A period that is cut short by the start or end date is kept and marked 'Partial'.
    """
    values = values.dropna()
    values = values[~values.index.duplicated(keep='last')].sort_index()
    if len(values) < 2:
        return pd.DataFrame()
    b = bench.reindex(bench.index.union(values.index)).ffill().reindex(values.index)
    df = pd.DataFrame({'P': values, 'B': b}).dropna()
    if len(df) < 2:
        return pd.DataFrame()

    labels, starts, ends = _period_keys(df.index, freq)
    key = pd.Series(labels, index=df.index)
    last = df.groupby(key.values, sort=False).last()
    last_date = pd.Series(df.index, index=df.index).groupby(key.values, sort=False).last()
    p_start = pd.Series(starts, index=df.index).groupby(key.values, sort=False).first()
    p_end = pd.Series(ends, index=df.index).groupby(key.values, sort=False).first()

    tol = pd.Timedelta(days=2 if freq == "Weekly" else 4)
    first_day = df.index[0]
    p0, b0 = df['P'].iloc[0], df['B'].iloc[0]
    p_prev, b_prev, d_prev = p0, b0, first_day
    rows = []
    for lab, row in last.iterrows():
        d_end = last_date[lab]
        if d_end == first_day:                                   # this period holds only the starting day
            continue
        p_ret, b_ret = row['P'] / p_prev - 1, row['B'] / b_prev - 1
        partial = ((d_prev == first_day and first_day > p_start[lab] + tol)
                   or d_end < p_end[lab] - tol)
        rows.append({
            'Period': lab, 'From': d_prev, 'To': d_end,
            'Benchmark index': row['B'] / b0 * 100,
            'Portfolio index': row['P'] / p0 * 100,
            'Benchmark (%)': b_ret * 100,
            'Portfolio (%)': p_ret * 100,
            'Alpha (%)': (p_ret - b_ret) * 100,
            'Partial': bool(partial),
        })
        p_prev, b_prev, d_prev = row['P'], row['B'], d_end
    out = pd.DataFrame(rows)
    out.attrs['start'] = first_day
    return out


def alpha_summary(t):
    """Headline numbers for a period_alpha() table."""
    if t is None or t.empty:
        return {}
    a = t['Alpha (%)']
    return {
        'periods': int(len(t)),
        'beat': int((a > 0).sum()),
        'avg': float(a.mean()),
        'median': float(a.median()),
        'best': t.loc[a.idxmax()],
        'worst': t.loc[a.idxmin()],
        'port_total': float(t['Portfolio index'].iloc[-1] - 100),
        'bench_total': float(t['Benchmark index'].iloc[-1] - 100),
    }