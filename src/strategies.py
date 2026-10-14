"""Investment strategy calculations (no Streamlit code here).

Strategies:
  bh       Buy & Hold          : split once, never touched.
  periodic Same-weight rebalance: back to the user's weights every month/quarter/half-year/year.
  diff     Different Weights    : on each rebalance date, compare every asset's growth over the
                                  previous 6 months / 1 year / 1.5 years. The asset that grew the
                                  MOST gets the SMALLEST of the user's weights, the one that grew
                                  the LEAST (or fell) gets the BIGGEST.
"""
import math

import pandas as pd

STRATEGIES = {
    "bh": "🧺 Buy & Hold (no rebalancing)",
    "periodic": "🔁 Rebalance to same weights regularly",
    "diff": "📐 Different Weights (based on past growth)",
}
REBAL_FREQ = {"Monthly": 1, "Quarterly": 3, "Half-Yearly": 6, "Yearly": 12}   # same-weight rebalance
DIFF_REBAL_FREQ = {"Half-Yearly": 6, "Yearly": 12}                            # Different Weights
LOOKBACK_OPTIONS = {"6 Months": 6, "1 Year": 12, "1.5 Years": 18}             # Different Weights


# ---------------------------------------------------------
# Dates
# ---------------------------------------------------------
def to_trading_day(index, d):
    """First trading day on or after date d (None if after the last date)."""
    i = index.searchsorted(pd.Timestamp(d))
    return index[i] if i < len(index) else None


def build_rebalance_dates(index, every_months):
    """Trading days from index[0], then every `every_months` months (first trading day on/after each)."""
    dates, k = [], 0
    while True:
        target = index[0] + pd.DateOffset(months=every_months * k)
        if target > index[-1]:
            break
        td = to_trading_day(index, target)
        if td is not None and (not dates or td != dates[-1]):
            dates.append(td)
        k += 1
    return dates


# ---------------------------------------------------------
# Simulation engines
# ---------------------------------------------------------
def simulate_portfolio(prices, schedule, amount):
    """Rebalancing engine (weights may be different on every date).

    prices   : DataFrame (Date index, one column per asset), forward-filled, no NaN
    schedule : list of (trading_date, {asset: weight as fraction})
    amount   : starting money (₹)

    On each schedule date:  Portfolio value = Σ (units held × price)   (first date: the starting money)
                            New units       = Portfolio value × weight ÷ price
    Returns units_df (units held every day), values (₹ every day + Total) and a trade log.
    """
    cols = list(prices.columns)
    units = {n: 0.0 for n in cols}
    units_at, log = {}, []
    for k, (dt, w) in enumerate(schedule):
        p = prices.loc[dt]
        pv = amount if k == 0 else sum(units[n] * p[n] for n in cols)
        new_units = {}
        for n in cols:
            new_units[n] = pv * w.get(n, 0.0) / p[n]
            log.append({
                'Rebalance #': k + 1, 'Date': dt, 'Asset': n, 'Price (₹)': p[n],
                'Units Before': units[n], 'Value Before (₹)': units[n] * p[n],
                'Portfolio Value (₹)': pv, 'Target Weight (%)': w.get(n, 0.0) * 100,
                'Target Value (₹)': pv * w.get(n, 0.0), 'Units After': new_units[n],
                'Units Bought (+) / Sold (−)': new_units[n] - units[n],
                'Trade Amount (₹)': (new_units[n] - units[n]) * p[n],
            })
        units = new_units
        units_at[dt] = dict(units)
    units_df = pd.DataFrame(units_at).T.reindex(prices.index).ffill()
    values = units_df * prices
    values['Total'] = values[cols].sum(axis=1)
    return units_df, values, pd.DataFrame(log)


def simulate_buy_hold(prices, weights, amount):
    """Buy & Hold: each asset is bought once, on ITS OWN first trading day.
    Until then its share of the money waits as cash."""
    cols = list(prices.columns)
    units_df = pd.DataFrame(0.0, index=prices.index, columns=cols)
    values = pd.DataFrame(index=prices.index)
    log = []
    for n in cols:
        alloc = amount * weights.get(n, 0.0)
        dt = prices[n].first_valid_index()
        p = prices.at[dt, n]
        u = alloc / p
        units_df.loc[dt:, n] = u
        values[n] = (prices[n] * u).fillna(alloc)   # before its first price, still cash
        log.append({
            'Rebalance #': 1, 'Date': dt, 'Asset': n, 'Price (₹)': p,
            'Units Before': 0.0, 'Value Before (₹)': 0.0, 'Portfolio Value (₹)': amount,
            'Target Weight (%)': weights.get(n, 0.0) * 100, 'Target Value (₹)': alloc,
            'Units After': u, 'Units Bought (+) / Sold (−)': u, 'Trade Amount (₹)': alloc,
        })
    values['Total'] = values[cols].sum(axis=1)
    return units_df, values, pd.DataFrame(log)


# ---------------------------------------------------------
# Different Weights: growth comparison + inverse ranking
# ---------------------------------------------------------
def lookback_growth(full_close, on_date, months):
    """Growth of each asset over the `months` before `on_date`, using FULL history
    (so a rebalance at the start of the period can still look back before it).

    full_close : {asset: Series of Close prices with Date index}
    Returns {asset: {'from': date, 'p_then': price, 'to': date, 'p_now': price, 'growth': % or NaN}}
    """
    on_ts = pd.Timestamp(on_date)
    back_ts = on_ts - pd.DateOffset(months=months)
    out = {}
    for n, s in full_close.items():
        now_s, then_s = s.loc[:on_ts], s.loc[:back_ts]
        if now_s.empty or then_s.empty:
            out[n] = {'from': None, 'p_then': math.nan, 'to': None, 'p_now': math.nan, 'growth': math.nan}
            continue
        p_then, p_now = float(then_s.iloc[-1]), float(now_s.iloc[-1])
        out[n] = {'from': then_s.index[-1], 'p_then': p_then, 'to': now_s.index[-1], 'p_now': p_now,
                  'growth': (p_now / p_then - 1) * 100}
    return out


def inverse_growth_weights(growth, base_weights):
    """Re-assign the user's own weights by growth: highest growth → smallest weight,
    lowest growth → biggest weight.

    Example: growth  Small 12%, Mid 7%, Large 5%, Gold −1%
             weights 20, 30, 15, 35  → sorted biggest first: 35, 30, 20, 15
             Gold (lowest growth) 35, Large 30, Mid 20, Small (highest growth) 15

    Assets with no growth number (not enough older data) keep their own weight.
    Ties keep the original asset order.
    """
    order = list(base_weights.keys())
    ranked = [n for n in order if not math.isnan(growth.get(n, math.nan))]
    new = {n: base_weights[n] for n in order if n not in ranked}
    lowest_first = sorted(ranked, key=lambda n: (growth[n], order.index(n)))
    biggest_first = sorted((base_weights[n] for n in ranked), reverse=True)
    for n, w in zip(lowest_first, biggest_first):
        new[n] = w
    return {n: new[n] for n in order}


def build_diff_schedule(prices, full_close, base_weights, every_months, lookback_months):
    """Schedule for the Different Weights strategy.

    First date : the user's weights as given.
    Every later rebalance date: weights from inverse_growth_weights() using the growth over the
    previous `lookback_months`.
    Returns (schedule, decisions DataFrame explaining every date's weights).
    """
    dates = build_rebalance_dates(prices.index, every_months)
    schedule, rows = [], []
    for k, d in enumerate(dates):
        g = lookback_growth({n: full_close[n] for n in base_weights}, d, lookback_months)
        growth = {n: g[n]['growth'] for n in base_weights}
        w = dict(base_weights) if k == 0 else inverse_growth_weights(growth, base_weights)
        schedule.append((d, w))

        ranked = sorted([n for n in base_weights if not math.isnan(growth[n])],
                        key=lambda n: -growth[n])
        for n in base_weights:
            rows.append({
                'Rebalance #': k + 1,
                'Date': d,
                'Type': 'First Buy (your weights)' if k == 0 else 'Rebalance',
                'Asset': n,
                'Compare From': g[n]['from'],
                'Price Then (₹)': g[n]['p_then'],
                'Price Now (₹)': g[n]['p_now'],
                'Growth (%)': growth[n],
                'Growth Rank': ranked.index(n) + 1 if n in ranked else None,
                'Your Weight (%)': base_weights[n] * 100,
                'New Weight (%)': w[n] * 100,
            })
    return schedule, pd.DataFrame(rows)