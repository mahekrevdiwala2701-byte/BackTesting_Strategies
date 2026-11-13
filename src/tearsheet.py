"""Tearsheet metrics for a daily value series (no Streamlit code here).

Same idea as a quantstats tearsheet: cumulative return, CAGR, Sharpe, Sortino, drawdowns, risk,
expected returns. Annualised with 252 trading days, risk-free rate 0 unless given.
"""
import math

import numpy as np
import pandas as pd

from src.drawdown import underwater, drawdown_periods

DAYS = 252


def _clean(v):
    v = v.dropna()
    return v[~v.index.duplicated(keep='last')].sort_index()


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def period_returns(v, rule):
    """Compounded returns per calendar period ('ME' month, 'YE' year). First period starts at v[0]."""
    v = _clean(v)
    last = v.resample(rule).last().dropna()
    prev = last.shift(1)
    prev.iloc[0] = v.iloc[0]
    return last / prev - 1


def monthly_table(v):
    """Year x month table of monthly returns in %, plus a 'Year' total column."""
    m = period_returns(v, 'ME') * 100
    t = pd.DataFrame({'Year': m.index.year, 'Month': m.index.month, 'R': m.values})
    grid = t.pivot(index='Year', columns='Month', values='R').reindex(columns=range(1, 13))
    grid.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    y = period_returns(v, 'YE') * 100
    grid['Year'] = pd.Series(y.values, index=y.index.year)
    return grid


def _autocorr_penalty(r):
    """quantstats 'smart' penalty: sqrt(1 + 2·Σ (n−i)/n · ρ^i), ρ = lag-1 autocorrelation."""
    n = len(r)
    if n < 3:
        return 1.0
    rho = abs(float(np.corrcoef(r.values[:-1], r.values[1:])[0, 1]))
    if not math.isfinite(rho) or rho == 0:
        return 1.0
    i = np.arange(1, n)
    return math.sqrt(1 + 2 * float(np.sum((n - i) / n * rho ** i)))


PERIODS = {"Weekly": "W-FRI", "Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}


def metrics(v, rf=0.0):
    """Dict of numbers for one value series (₹ per day)."""
    v = _clean(v)
    if len(v) < 3:
        return {}
    r = v.pct_change().dropna()
    n = len(r)
    years = max((v.index[-1] - v.index[0]).days / 365.25, 1 / 365.25)
    cum = v.iloc[-1] / v.iloc[0] - 1
    cagr = (1 + cum) ** (1 / years) - 1 if cum > -1 else -1.0
    ex = r - rf / DAYS
    std = r.std()
    vol = std * math.sqrt(DAYS)
    sharpe = ex.mean() / std * math.sqrt(DAYS) if std > 0 else np.nan
    skew, kurt = float(r.skew()), float(r.kurt())

    # Probabilistic Sharpe ratio: chance the true Sharpe is above 0
    sr = ex.mean() / std if std > 0 else 0.0
    denom = 1 - skew * sr + (kurt + 2) / 4 * sr ** 2          # kurt here is excess kurtosis -> +3 -1
    psr = _norm_cdf(sr * math.sqrt(n - 1) / math.sqrt(denom)) if denom > 0 else np.nan

    downside = math.sqrt((r[r < 0] ** 2).sum() / n) if n else 0
    sortino = ex.mean() / downside * math.sqrt(DAYS) if downside > 0 else np.nan
    penalty = _autocorr_penalty(r)
    smart_sharpe = sharpe / penalty if sharpe == sharpe else np.nan
    smart_sortino = sortino / penalty if sortino == sortino else np.nan
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    omega = gains / losses if losses > 0 else np.nan

    dd = underwater(v)
    periods = drawdown_periods(dd)
    max_dd = float(dd.min()) / 100
    longest = int(periods['Days underwater'].max()) if not periods.empty else 0
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    m = period_returns(v, 'ME')
    y = period_returns(v, 'YE')
    geo = lambda s: (1 + s).prod() ** (1 / len(s)) - 1 if len(s) else np.nan   # noqa: E731

    wins, lose = r[r > 0], r[r < 0]
    win_rate = len(wins) / max(len(wins) + len(lose), 1)
    payoff = wins.mean() / abs(lose.mean()) if len(wins) and len(lose) else np.nan
    kelly = win_rate - (1 - win_rate) / payoff if payoff and payoff == payoff else np.nan

    var = r.mean() - 1.645 * std                                     # 95% one-day, normal
    cvar = r[r <= var].mean() if (r <= var).any() else var

    return {
        'start': v.index[0], 'end': v.index[-1], 'years': years, 'rf': rf,
        'smart_sharpe': smart_sharpe, 'smart_sortino': smart_sortino,
        'smart_sortino_r2': smart_sortino / math.sqrt(2) if smart_sortino == smart_sortino else np.nan,
        'time_in_market': float((r != 0).mean()),
        'cum': cum, 'cagr': cagr,
        'sharpe': sharpe, 'psr': psr, 'sortino': sortino, 'sortino_r2': sortino / math.sqrt(2), 'omega': omega,
        'max_dd': max_dd, 'longest_dd': longest, 'vol': vol, 'calmar': calmar, 'skew': skew, 'kurt': kurt,
        'exp_day': geo(r), 'exp_month': geo(m), 'exp_year': geo(y), 'kelly': kelly,
        'var': var, 'cvar': cvar,
        'best_day': r.max(), 'worst_day': r.min(),
        'best_month': m.max(), 'worst_month': m.min(),
        'best_year': y.max(), 'worst_year': y.min(),
        'up_from_high': float((dd >= -1e-9).mean()),
        'win_days': win_rate, 'win_months': float((m > 0).mean()), 'win_years': float((y > 0).mean()),
    }


# quantstats-style table: groups separated by a line. (label, key, format)
LAYOUT = [
    [("Start period", 'start', 'date'), ("End period", 'end', 'date'),
     ("Risk-free rate", 'rf', 'pct'), ("Time in market", 'time_in_market', 'pct')],
    [("Cumulative return", 'cum', 'pct'), ("CAGR", 'cagr', 'pct')],
    [("Sharpe", 'sharpe', 'num'), ("Prob. Sharpe ratio", 'psr', 'pct'), ("Smart Sharpe", 'smart_sharpe', 'num'),
     ("Sortino", 'sortino', 'num'), ("Smart Sortino", 'smart_sortino', 'num'), ("Sortino/√2", 'sortino_r2', 'num'),
     ("Omega", 'omega', 'num')],
    [("Max drawdown", 'max_dd', 'pct'), ("Longest DD days", 'longest_dd', 'int'), ("Volatility (ann.)", 'vol', 'pct'),
     ("Calmar", 'calmar', 'num'), ("Skew", 'skew', 'num'), ("Kurtosis", 'kurt', 'num')],
    [("Expected daily", 'exp_day', 'pct'), ("Expected monthly", 'exp_month', 'pct'),
     ("Expected yearly", 'exp_year', 'pct'), ("Daily value-at-risk", 'var', 'pct')],
    [("Best day", 'best_day', 'pct'), ("Worst day", 'worst_day', 'pct'), ("Best month", 'best_month', 'pct'),
     ("Worst month", 'worst_month', 'pct'), ("Best year", 'best_year', 'pct'), ("Worst year", 'worst_year', 'pct')],
    [("Win days", 'win_days', 'pct'), ("Win months", 'win_months', 'pct'), ("Win years", 'win_years', 'pct')],
]


def fmt(value, kind):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if kind == 'pct':
        return f"{value * 100:,.2f}%"
    if kind == 'int':
        return f"{int(value):,}"
    if kind == 'date':
        return pd.Timestamp(value).strftime('%d-%m-%Y')
    return f"{value:,.2f}"