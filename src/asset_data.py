"""Shared data loading, cleaning and table helpers used by every page."""
import os
import re
import tempfile
import threading
import time
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
COLORS = {'Gold': '#C8901A', 'LargeCap': '#2563EB', 'MidCap': '#7C3AED', 'SmallCap': '#0D9488'}

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
# How the data loads (fast):
#  1. The last good download of each Google Sheet is kept on disk, so the app opens instantly.
#  2. When that copy is older than LIVE_MAX_AGE, the sheets are downloaded again IN THE BACKGROUND:
#     nobody waits; the next click shows the new prices.
#  3. Only the very first start (no saved copy yet) waits for Google.
#  4. A sheet that cannot be loaded at all falls back to its backup CSV, with a warning on every page.
LIVE_MAX_AGE = 3600            # seconds: saved copy younger than this is used as is
BACKUP_RETRY_SECONDS = 120     # while a backup CSV is shown, try the live sheets again (in background) this often
CACHE_DIR = os.path.join(tempfile.gettempdir(), "multi_asset_lab_sheets")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # folder with app.py + backup CSVs

_refresh_lock = threading.Lock()
_refresh_state = {"running": False, "last_try": 0.0}


def _friendly_error(err):
    """Short reason a person can act on (technical detail stays after a ' | ')."""
    if err == "UNAUTHORIZED":
        return ("sheet not shared publicly: in Google Sheets use Share → General access → "
                "'Anyone with the link' → Viewer")
    e = str(err or "")
    if e.startswith("Connection failed"):
        return "could not connect to Google Sheets (no internet, or Google blocked on this server) | " + e
    if e.startswith("HTTP"):
        return "Google Sheets gave an error | " + e
    return e


# ---------- saved copy on disk ----------
def _cache_path(name):
    return os.path.join(CACHE_DIR, f"{name}.pkl")


def _save_to_disk(dfs):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        for name, df in dfs.items():
            tmp = _cache_path(name) + ".tmp"
            df.to_pickle(tmp)
            os.replace(tmp, _cache_path(name))              # atomic: a half-written file is never read
    except OSError:
        pass                                                # read-only server: just keep it in memory


def _read_from_disk():
    """{name: df} saved earlier and the time of the OLDEST of them (None if nothing saved)."""
    dfs, times = {}, []
    for name in SHEET_URLS:
        path = _cache_path(name)
        try:
            dfs[name] = pd.read_pickle(path)
            times.append(os.path.getmtime(path))
        except Exception:                                   # missing or unreadable: download it
            continue
    return dfs, (min(times) if times else None)


def data_updated_at():
    """When the live sheet data on screen was downloaded (epoch seconds), or None."""
    return _read_from_disk()[1]


# ---------- download ----------
def _download_live():
    """Download all 4 sheets at the same time. Returns ({name: df}, {name: error})."""
    urls = {name: parse_google_sheet_url(raw_url) for name, raw_url in SHEET_URLS.items()}
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = {name: pool.submit(download_sheet_csv, url) for name, url in urls.items()}
        downloaded = {name: f.result() for name, f in futures.items()}
    dfs, errors = {}, {}
    for name, (df_raw, err) in downloaded.items():
        df_clean = normalize_asset_df(df_raw)
        if df_clean is not None:
            dfs[name] = df_clean
        else:
            found = list(df_raw.columns) if df_raw is not None else []
            errors[name] = _friendly_error(err) or f"Couldn't find a 'Date' and 'Close' column. Columns found: {found}"
    return dfs, errors


def _refresh_in_background():
    """Download the sheets in a background thread (at most one at a time). The page is not blocked;
    when it finishes, the next click shows the fresh data."""
    with _refresh_lock:
        if _refresh_state["running"]:
            return
        _refresh_state.update(running=True, last_try=time.time())

    def run():
        try:
            dfs, _ = _download_live()
            if dfs:
                _save_to_disk(dfs)
                _assemble.clear()
        finally:
            _refresh_state["running"] = False

    threading.Thread(target=run, daemon=True, name="sheet-refresh").start()


# ---------- putting it together ----------
@st.cache_data(show_spinner="Loading data from Google Sheets (first start only)...", ttl=300)
def _assemble(force=False):
    """Returns (asset_dfs, fetch_errors, backup_used, saved_at)."""
    asset_dfs, saved_at = ({}, None) if force else _read_from_disk()
    fetch_errors, backup_used = {}, {}

    missing = [n for n in SHEET_URLS if n not in asset_dfs]
    if missing:                                             # first start (or refresh button): must wait once
        live, errors = _download_live()
        _save_to_disk(live)
        asset_dfs.update(live)
        fetch_errors.update({n: e for n, e in errors.items() if n not in asset_dfs})
        saved_at = time.time() if live and not saved_at else saved_at

    # Backup: local CSV files, only for the sheets that could not be loaded
    local_datasets = get_available_datasets(PROJECT_DIR)
    for name in ASSETS:
        if name in asset_dfs:
            continue
        for loc_name, path in local_datasets.items():
            if name.lower() in loc_name.lower():
                df_local = normalize_asset_df(load_csv_data(path))
                if df_local is not None:
                    asset_dfs[name] = df_local
                    backup_used[name] = os.path.basename(str(path))
                    fetch_errors[name] = (f"Live Google Sheet not loaded ({fetch_errors.get(name)}). "
                                          f"Showing the backup file {backup_used[name]} instead: its prices can "
                                          f"be old or wrong.")
                break
    return asset_dfs, fetch_errors, backup_used, saved_at


def load_all_assets(force_refresh=False):
    """Returns (asset_dfs, fetch_errors). Fast: uses the saved copy of the live sheets and refreshes it in
    the background. force_refresh=True (the sidebar button) downloads from Google right now."""
    if force_refresh:
        _assemble.clear()
    asset_dfs, fetch_errors, backup_used, saved_at = _assemble(force_refresh)
    if force_refresh:
        _assemble.clear()                                   # later runs use the normal (disk) path again

    stale = saved_at is None or time.time() - saved_at > LIVE_MAX_AGE
    retry_backup = backup_used and time.time() - _refresh_state["last_try"] > BACKUP_RETRY_SECONDS
    if (stale and not backup_used) or retry_backup:
        _refresh_in_background()
    return asset_dfs, dict(fetch_errors)


def short_reason(err):
    """The part of a fetch error meant for the banner (no technical detail)."""
    e = str(err).replace("Live Google Sheet not loaded (", "", 1)
    return e.split(" | ")[0].split("). Showing the backup")[0]


def backup_assets(fetch_errors):
    """Names of the assets that are showing backup CSV data instead of the live sheet."""
    return [n for n, e in fetch_errors.items() if str(e).startswith("Live Google Sheet not loaded")]


def filter_period(asset_dfs, start_date, end_date):
    s, e = pd.Timestamp(start_date), pd.Timestamp(end_date)
    return {n: df[(df['Date'] >= s) & (df['Date'] <= e)].reset_index(drop=True) for n, df in asset_dfs.items()}