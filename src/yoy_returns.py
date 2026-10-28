"""Same-date, year-over-year returns (no Streamlit code here).

Example (holding period = 1 year):
    invest on 01-01-2010  -> compare the price on 01-01-2011 with the price on 01-01-2010
    invest on 02-01-2010  -> compare 02-01-2011 with 02-01-2010
    ...
and so on for every day of every year.

If a date is a holiday / weekend (no price that day), the last available closing price
ON OR BEFORE that date is used, and the actual price date is kept in the table.
Obviously wrong prices (e.g. ₹7 in a ~₹10,000 index) are removed first and reported.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------
def clean_prices(df, window_days=365, low=0.3, high=3.3):
    """Remove obviously wrong prices.
    A row is dropped when its close is below 30% or above 330% of the median close in the
    surrounding ±window_days (wide enough that genuine crashes such as 2008 or 2020 are kept).
    Returns (clean_df, removed_rows)."""
    d = df[['Date', 'Close']].dropna().sort_values('Date').drop_duplicates('Date', keep='last')
    d = d.reset_index(drop=True)
    if len(d) < 5:
        return d, d.iloc[0:0]
    med = d.set_index('Date')['Close'].rolling(f'{2 * window_days}D', center=True, min_periods=1).median()
    ratio = d['Close'].to_numpy() / med.to_numpy()
    bad = (ratio < low) | (ratio > high)
    return d[~bad].reset_index(drop=True), d[bad].reset_index(drop=True)


def _max_gap_days(df):
    """How old a 'previous close' may be: 7 days for daily data, ~1.5 months for monthly data."""
    spacing = df['Date'].diff().dt.days.median()
    spacing = 1 if pd.isna(spacing) else spacing
    return int(max(7, 1.5 * spacing + 3))


def _price_on_or_before(df, dates):
    """For each target date, return (price_date, close) of the last row on or before it."""
    prices = df[['Date', 'Close']].sort_values('Date').copy()
    prices['Date'] = prices['Date'].astype('datetime64[ns]')          # same resolution on both sides
    left = pd.DataFrame({'Target': pd.to_datetime(pd.Series(dates)).astype('datetime64[ns]')})
    left['_order'] = np.arange(len(left))
    left = left.sort_values('Target')
    merged = pd.merge_asof(left, prices, left_on='Target', right_on='Date', direction='backward')
    return merged.sort_values('_order').reset_index(drop=True)


# ---------------------------------------------------------
# One asset
# ---------------------------------------------------------
def same_date_returns(df, start_date, end_date, years=1, trading_days_only=False):
    """Year-over-year table for one asset.

    df                : DataFrame with 'Date' and 'Close' columns (one asset)
    start_date        : first investment date
    end_date          : last date a price may be read (investment date + `years` must be <= this)
    years             : holding period in years (1 = compare with the same date next year)
    trading_days_only : True  -> only dates that have a price are used as investment dates
                        False -> every calendar day (holidays use the previous close)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df, removed = clean_prices(df)
    max_gap = _max_gap_days(df)
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    start_ts = max(start_ts, df['Date'].min().normalize())
    end_ts = min(end_ts, df['Date'].max())
    last_invest = end_ts - pd.DateOffset(years=years)              # compare date must still have data
    if last_invest < start_ts:
        return pd.DataFrame()

    if trading_days_only:
        invest_dates = df.loc[(df['Date'] >= start_ts) & (df['Date'] <= last_invest), 'Date'].dt.normalize()
    else:
        invest_dates = pd.Series(pd.date_range(start_ts, last_invest, freq='D'))
    invest_dates = pd.Series(pd.to_datetime(invest_dates).values).drop_duplicates().reset_index(drop=True)
    if invest_dates.empty:
        return pd.DataFrame()

    compare_dates = invest_dates + pd.DateOffset(years=years)      # 29-Feb -> 28-Feb in non-leap years
    buy = _price_on_or_before(df, invest_dates)
    sell = _price_on_or_before(df, compare_dates)

    out = pd.DataFrame({
        'Invest Date': invest_dates.values,
        'Invest Price Date': buy['Date'].values,
        'Invest Price': buy['Close'].values,
        'Compare Date': compare_dates.values,
        'Compare Price Date': sell['Date'].values,
        'Compare Price': sell['Close'].values,
    })
    out = out.dropna(subset=['Invest Price', 'Compare Price'])

    # Skip dates whose nearest previous price is too old (a hole in the data, not just a holiday)
    gap_buy = (out['Invest Date'] - out['Invest Price Date']).dt.days
    gap_sell = (out['Compare Date'] - out['Compare Price Date']).dt.days
    stale = (gap_buy > max_gap) | (gap_sell > max_gap)
    skipped = int(stale.sum())
    out = out[~stale].reset_index(drop=True)

    out['Change (₹)'] = out['Compare Price'] - out['Invest Price']
    out['Return (%)'] = (out['Compare Price'] / out['Invest Price'] - 1) * 100
    if years > 1:
        out['CAGR (%)'] = ((out['Compare Price'] / out['Invest Price']) ** (1 / years) - 1) * 100
    out['Result'] = np.where(out['Return (%)'] > 0, 'Positive',
                             np.where(out['Return (%)'] < 0, 'Negative', 'Flat'))
    out['Invest Year'] = out['Invest Date'].dt.year

    out.attrs['removed_prices'] = [[d.strftime('%Y-%m-%d'), float(c)]      # JSON-friendly (concat-safe)
                                   for d, c in zip(removed['Date'], removed['Close'])]
    out.attrs['skipped_dates'] = skipped
    out.attrs['max_gap_days'] = max_gap
    return out


# ---------------------------------------------------------
# Whole portfolio
# ---------------------------------------------------------
def portfolio_same_date_returns(asset_dfs, weights, amount, start_date, end_date, years=1,
                                trading_days_only=False, short_names=None):
    """Year-over-year returns for the whole portfolio (buy and hold, no rebalancing):

        On every investment date `amount` is split by `weights` and each part buys its asset.
        Portfolio return = Σ weight_i × return_i

    A date is used only when EVERY asset in the portfolio has a valid price on both dates.
    """
    total = sum(w for w in weights.values() if w > 0)
    weights = {a: w / total for a, w in weights.items() if w > 0 and a in asset_dfs} if total else {}
    if not weights:
        return pd.DataFrame()
    names = short_names or {a: a for a in weights}

    per_asset, removed, skipped = {}, [], 0
    for a in weights:
        t = same_date_returns(asset_dfs[a], start_date, end_date, years, trading_days_only)
        if t.empty:
            return pd.DataFrame()
        removed += [[names[a]] + r for r in t.attrs.get('removed_prices', [])]
        skipped += t.attrs.get('skipped_dates', 0)
        per_asset[a] = t.set_index('Invest Date')

    common = None
    for t in per_asset.values():
        common = t.index if common is None else common.intersection(t.index)
    if common is None or common.empty:
        return pd.DataFrame()
    common = common.sort_values()

    first = per_asset[next(iter(weights))].loc[common]
    out = pd.DataFrame({'Invest Date': common, 'Compare Date': first['Compare Date'].values})
    port_ret = np.zeros(len(common))
    for a, w in weights.items():
        t = per_asset[a].loc[common]
        n = names[a]
        out[f'{n} Invest Price'] = t['Invest Price'].values
        out[f'{n} Compare Price'] = t['Compare Price'].values
        out[f'{n} Return (%)'] = t['Return (%)'].values
        port_ret += w * t['Return (%)'].values

    out['Invest Value (₹)'] = float(amount)
    out['Compare Value (₹)'] = amount * (1 + port_ret / 100)
    out['Change (₹)'] = out['Compare Value (₹)'] - out['Invest Value (₹)']
    out['Return (%)'] = port_ret
    if years > 1:
        out['CAGR (%)'] = ((1 + port_ret / 100) ** (1 / years) - 1) * 100
    out['Result'] = np.where(port_ret > 0, 'Positive', np.where(port_ret < 0, 'Negative', 'Flat'))
    out['Invest Year'] = pd.to_datetime(out['Invest Date']).dt.year

    out.attrs['removed_prices'] = removed                  # [[asset, 'YYYY-MM-DD', close], ...]
    out.attrs['skipped_dates'] = skipped
    out.attrs['dropped_not_common'] = int(max(len(t) for t in per_asset.values()) - len(common))
    out.attrs['weights'] = {names[a]: float(w) for a, w in weights.items()}
    return out


# ---------------------------------------------------------
# Summaries
# ---------------------------------------------------------
def summary_stats(table):
    """Headline numbers for a year-over-year table (uses its 'Return (%)' column)."""
    r = table['Return (%)'] if table is not None and not table.empty else pd.Series(dtype=float)
    if r.empty:
        return {}
    return {
        'count': int(r.size),
        'positive': int((r > 0).sum()),
        'negative': int((r < 0).sum()),
        'pct_positive': float((r > 0).mean() * 100),
        'mean': float(r.mean()),
        'median': float(r.median()),
        'std': float(r.std(ddof=1)) if r.size > 1 else 0.0,
        'min': float(r.min()),
        'max': float(r.max()),
        'p5': float(r.quantile(0.05)),
        'p95': float(r.quantile(0.95)),
        'best_row': table.loc[r.idxmax()],
        'worst_row': table.loc[r.idxmin()],
    }


def yearly_summary(table):
    """One row per investment year: how many dates were positive / negative and the range of returns."""
    if table is None or table.empty:
        return pd.DataFrame()
    g = table.groupby('Invest Year')['Return (%)']
    return pd.DataFrame({
        'Days compared': g.size(),
        'Positive days': g.apply(lambda s: int((s > 0).sum())),
        'Negative days': g.apply(lambda s: int((s < 0).sum())),
        '% Positive': g.apply(lambda s: (s > 0).mean() * 100),
        'Average (%)': g.mean(),
        'Median (%)': g.median(),
        'Worst (%)': g.min(),
        'Best (%)': g.max(),
    }).reset_index()


def normal_curve(mean, std, lo, hi, n=400):
    """x, y points of the normal (bell) curve with the given mean and standard deviation."""
    x = np.linspace(lo, hi, n)
    if std <= 0:
        return x, np.zeros_like(x)
    return x, np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def to_csv(table, asset_label=None):
    """CSV text with dates written as DD-MM-YYYY and numbers rounded (Excel shows ₹ correctly)."""
    t = table.drop(columns=['Invest Year'], errors='ignore').copy()
    if asset_label is not None and 'Asset' not in t.columns:
        t.insert(0, 'Asset', asset_label)
    for c in ('Invest Date', 'Invest Price Date', 'Compare Date', 'Compare Price Date'):
        if c in t.columns:
            t[c] = pd.to_datetime(t[c]).dt.strftime('%d-%m-%Y')
    for c in t.columns:
        if pd.api.types.is_float_dtype(t[c]):
            t[c] = t[c].round(3 if '%' in c else 2)
    return t.to_csv(index=False).encode('utf-8-sig')