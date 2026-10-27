"""Shared data loading, cleaning and table helpers used by every page."""
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

from src.ui import PLOT_LAYOUT as _UI_PLOT_LAYOUT, GOOD_TEXT, BAD_TEXT

from src.data_loader import (
    load_csv_data,
    get_available_datasets,
    parse_google_sheet_url,
    download_sheet_csv,
    parse_dates,
)

ASSETS = ['Gold', 'LargeCap', 'MidCap', 'SmallCap']

ASSET_LABELS = {
    'Gold': '🥇 GoldBEES',
    'LargeCap': '🏢 LargeCap (Nifty 50)',
    'MidCap': '🏬 MidCap',
    'SmallCap': '🏪 SmallCap'
}
ASSET_SHORT = {'Gold': 'GoldBEES', 'LargeCap': 'Nifty 50', 'MidCap': 'MidCap', 'SmallCap': 'SmallCap'}
ASSET_ICONS = {'Gold': '🥇', 'LargeCap': '🏢', 'MidCap': '🏬', 'SmallCap': '🏪'}

# One vivid colour per asset, used on every chart, card and chip
COLORS = {'Gold': '#FBBF24', 'LargeCap': '#60A5FA', 'MidCap': '#34D399', 'SmallCap': '#F472B6'}

# Google Sheet Live URLs
SHEET_URLS = {
    'Gold': 'https://docs.google.com/spreadsheets/d/1kiOxwDzxY5NnyuwRkWi1MPvJcJvF2lIw9Rngyw4FkR0/edit?usp=drive_link',
    'LargeCap': 'https://docs.google.com/spreadsheets/d/1EHVn5f1isy3SjP4NF_MdZ-IG5NCNXRMSju8a3Sn27ic/edit?usp=drive_link',
    'MidCap': 'https://docs.google.com/spreadsheets/d/1By1KN8wLjiL84tEp7_XinAWGAAXExABYBLsJz5TiB-s/edit?usp=drive_link',
    'SmallCap': 'https://docs.google.com/spreadsheets/d/1xwzl_75YssaiWQXJy8S24XOINThnnorSTU5tsZX48Bo/edit?usp=drive_link'
}

PLOT_LAYOUT = _UI_PLOT_LAYOUT

DATE_NAMES = ('date', 'datetime', 'timestamp', 'time', 'day')
CLOSE_NAMES = ('close', 'close price', 'closing price', 'price', 'adj close', 'ltp', 'last price')


# ---------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------
def _to_number(series):
    """Convert strings like '₹1,234.50' into floats."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = series.astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce')


def _parse_dates(raw):
    """Parse dates in any common format. Uses the same parser as src/data_loader.py."""
    return parse_dates(raw)


def _map_columns(cols):
    rename = {}
    for c in cols:
        lc = str(c).strip().lower()
        if lc in DATE_NAMES and 'Date' not in rename.values():
            rename[c] = 'Date'
        elif lc in CLOSE_NAMES and 'Close' not in rename.values():
            rename[c] = 'Close'
        elif lc in ('open', 'high', 'low', 'volume'):
            rename[c] = lc.capitalize()
    return rename


def _fix_header_row(df):
    """If blank/title rows sit above the real header (columns show as 'Unnamed: 0'...),
    find the row containing 'Date' and 'Close' and use it as the header."""
    rename = _map_columns(df.columns)
    if 'Date' in rename.values() and 'Close' in rename.values():
        return df
    for i in range(min(20, len(df))):
        row = [str(v).strip() for v in df.iloc[i].tolist()]
        mapped = _map_columns(row)
        if 'Date' in mapped.values() and 'Close' in mapped.values():
            new_df = df.iloc[i + 1:].copy()
            new_df.columns = row
            new_df = new_df.loc[:, [c for c in new_df.columns if c and c.lower() != 'nan']]
            return new_df.reset_index(drop=True)
    return df


def normalize_asset_df(df):
    """Standardise columns, dates and prices; add daily change columns."""
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = _fix_header_row(df)
    df = df.rename(columns=_map_columns(df.columns))

    if 'Date' not in df.columns or 'Close' not in df.columns:
        return None

    df['Date'] = _parse_dates(df['Date'])
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = _to_number(df[col])

    df = df.dropna(subset=['Date', 'Close'])
    df = df[df['Close'] > 0]
    df = df.sort_values('Date').drop_duplicates(subset='Date', keep='last').reset_index(drop=True)

    df['Change (₹)'] = df['Close'].diff()
    df['Change (%)'] = df['Close'].pct_change() * 100
    return df if not df.empty else None


def blank_missing(df, fmt=None):
    """Streamlit shows empty numeric cells as 'None' even when a Styler formats them.
    For every column that has empty cells, write the formatted text ('—' when empty) into the
    column itself. Returns (df, remaining_fmt) — pass remaining_fmt to df.style.format()."""
    df = df.copy()
    fmt = dict(fmt or {})
    for col in df.columns:
        if not df[col].isna().any() or df[col].dtype == object and df[col].map(lambda v: isinstance(v, str)).all():
            continue
        f = fmt.pop(col, None)
        if f is None:
            f = (lambda v: f"{v:,.4f}".rstrip('0').rstrip('.')) if pd.api.types.is_float_dtype(df[col]) else str
        df[col] = ['—' if (v is None or (isinstance(v, float) and v != v)) else
                   (f(v) if callable(f) else f.format(v)) for v in df[col]]
    return df, fmt


def _as_number(v):
    if isinstance(v, str):
        s = v.replace('−', '-').replace(',', '')
        s = re.sub(r'[^0-9.\-]', '', s)
        try:
            return float(s)
        except ValueError:
            return float('nan')
    return v


def style_change_cols(styler, cols):
    """Colour positive values green and negative values red (also works on formatted text)."""
    def color(v):
        v = _as_number(v)
        if pd.isna(v):
            return ''
        if v > 0:
            return f'color: {GOOD_TEXT}; font-weight: 600'
        return f'color: {BAD_TEXT}; font-weight: 600' if v < 0 else ''
    apply_fn = styler.map if hasattr(styler, 'map') else styler.applymap
    return apply_fn(color, subset=cols)


# ---------------------------------------------------------
# Loading
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading data from Google Sheets...", ttl=600)
def load_all_assets():
    """Returns (asset_dfs, fetch_errors)."""
    asset_dfs, fetch_errors = {}, {}

    # Download all 4 sheets AT THE SAME TIME (before: one after another, up to 12 s each)
    urls = {name: parse_google_sheet_url(raw_url) for name, raw_url in SHEET_URLS.items()}
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = {name: pool.submit(download_sheet_csv, url) for name, url in urls.items()}
        downloaded = {name: f.result() for name, f in futures.items()}

    for name in SHEET_URLS:
        df_raw, err = downloaded[name]
        df_clean = normalize_asset_df(df_raw)
        if df_clean is not None:
            asset_dfs[name] = df_clean
        else:
            found = list(df_raw.columns) if df_raw is not None else []
            fetch_errors[name] = err or f"Couldn't find a 'Date' and 'Close' column. Columns found: {found}"

    # Fallback to local CSV files
    local_datasets = get_available_datasets(".")
    for name in ASSETS:
        if name not in asset_dfs:
            for loc_name, path in local_datasets.items():
                if name.lower() in loc_name.lower():
                    df_local = normalize_asset_df(load_csv_data(path))
                    if df_local is not None:
                        asset_dfs[name] = df_local
                        fetch_errors.pop(name, None)
                    break

    return asset_dfs, fetch_errors


def filter_period(asset_dfs, start_date, end_date):
    s, e = pd.Timestamp(start_date), pd.Timestamp(end_date)
    return {n: df[(df['Date'] >= s) & (df['Date'] <= e)].reset_index(drop=True) for n, df in asset_dfs.items()}