"""Drawdown / underwater calculations (no Streamlit code here).

Drawdown on a day = how far the portfolio value is below its highest value so far:

    drawdown (%) = (value today / highest value up to today − 1) × 100      (0% = at a new high)

The "underwater plot" draws this line for every day. It is always 0% or below.

A drawdown period ("episode") runs from a peak, down to the lowest point (trough), and back up to
the old peak (recovery). If the value has not climbed back yet, the period is still open.
"""
import numpy as np
import pandas as pd


def underwater(values: pd.Series) -> pd.Series:
    """Drawdown in % for every day (0 or negative). values: portfolio value (₹) indexed by date."""
    v = values.dropna()
    v = v[~v.index.duplicated(keep='last')].sort_index()
    if v.empty:
        return pd.Series(dtype=float)
    return (v / v.cummax() - 1) * 100


def drawdown_periods(dd: pd.Series) -> pd.DataFrame:
    """One row per drawdown period, deepest first.

    Columns: Peak, Trough, Recovery (NaT if not recovered), Depth (%), Days to trough,
             Days to recover, Days underwater, Recovered.
    """
    if dd.empty:
        return pd.DataFrame()
    below = dd < -1e-9
    run_id = (below != below.shift()).cumsum()
    dates = dd.index
    rows = []
    for _, chunk in dd[below].groupby(run_id[below]):
        first_i = dates.get_loc(chunk.index[0])
        peak = dates[first_i - 1] if first_i > 0 else chunk.index[0]
        trough = chunk.idxmin()
        last_i = dates.get_loc(chunk.index[-1])
        recovered = last_i + 1 < len(dates)
        recovery = dates[last_i + 1] if recovered else pd.NaT
        end = recovery if recovered else dates[-1]
        rows.append({
            'Peak': peak,
            'Trough': trough,
            'Recovery': recovery,
            'Depth (%)': float(chunk.min()),
            'Days to trough': int((trough - peak).days),
            'Days to recover': int((recovery - trough).days) if recovered else np.nan,
            'Days underwater': int((end - peak).days),
            'Recovered': recovered,
        })
    out = pd.DataFrame(rows)
    return out.sort_values('Depth (%)').reset_index(drop=True)


def drawdown_summary(dd: pd.Series, periods: pd.DataFrame) -> dict:
    """Headline numbers for the underwater plot."""
    if dd.empty:
        return {}
    in_dd = dd[dd < -1e-9]
    worst = periods.iloc[0] if not periods.empty else None
    longest = periods.loc[periods['Days underwater'].idxmax()] if not periods.empty else None
    return {
        'max': float(dd.min()),
        'current': float(dd.iloc[-1]),
        'avg': float(in_dd.mean()) if len(in_dd) else 0.0,       # average depth on days below a high
        'pct_under': float((dd < -1e-9).mean() * 100),           # share of days below the last high
        'count': int(len(periods)),
        'worst': worst,
        'longest': longest,
        'last_high': dd[dd >= -1e-9].index[-1] if (dd >= -1e-9).any() else dd.index[0],
    }