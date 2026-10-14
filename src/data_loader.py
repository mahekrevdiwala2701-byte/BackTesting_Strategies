import os
import re
import pandas as pd
import requests
import io
import streamlit as st


# ---------------------------------------------------------
# Date parsing (shared by data_loader and asset_data)
# ---------------------------------------------------------
def _build_dates(year, month, day):
    """Build dates from number columns (bad combinations become NaT)."""
    return pd.to_datetime(pd.DataFrame({'year': year, 'month': month, 'day': day}), errors='coerce')


def _date_score(dates):
    d = dates.dropna()
    if len(d) < 2:
        return len(d)
    gaps = d.diff().abs().dropna()
    return len(d) + (gaps <= pd.Timedelta(days=7)).mean()


def _month_spread(dates):
    """How many different (year, month) pairs the dates cover.
    A correct reading of daily/monthly data covers many months; a day/month swap squeezes
    everything into the first 12 days of a few months, so it covers far fewer."""
    d = dates.dropna()
    return d.dt.to_period('M').nunique() if len(d) else 0


def parse_dates(raw):
    """Parse a column of dates written in any common format, row by row.

    - yyyy-mm-dd / yyyy/mm/dd (with or without a time) is always year-month-day.
    - a/b/yyyy or a-b-yyyy: the order is decided for the whole column. A value above 12 in the
      first part means day-first, above 12 in the second part means month-first; if still
      unclear, the reading that covers more different months wins.
    - anything else ("1 Jan 2018", "Jan 1, 2018", real datetime values) is left to pandas.

    Plain pd.to_datetime() guesses ONE format from the first row and silently turns every row
    that does not match into NaT, which then got dropped (that is what made years of data vanish).
    """
    if pd.api.types.is_datetime64_any_dtype(raw):
        out = pd.Series(raw)
        if getattr(out.dt, 'tz', None) is not None:
            out = out.dt.tz_localize(None)
        return out.dt.normalize()

    raw = pd.Series(raw).astype(str).str.strip()
    result = pd.Series(pd.NaT, index=raw.index, dtype='datetime64[ns]')

    # 1) yyyy-mm-dd
    iso = raw.str.extract(r'^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:\D|$)').astype(float)
    iso_ok = iso[0].notna()
    if iso_ok.any():
        result[iso_ok] = _build_dates(iso.loc[iso_ok, 0], iso.loc[iso_ok, 1], iso.loc[iso_ok, 2])

    # 2) a/b/yyyy
    ab = raw.str.extract(r'^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})(?:\D|$)').astype(float)
    ab_ok = ab[0].notna() & ~iso_ok
    dayfirst = True
    if ab_ok.any():
        a, b, y = ab.loc[ab_ok, 0], ab.loc[ab_ok, 1], ab.loc[ab_ok, 2]
        y = y.where(y >= 100, y + 2000)                       # 2-digit years -> 20xx
        day_first = _build_dates(y, b, a)                     # dd/mm/yyyy
        month_first = _build_dates(y, a, b)                   # mm/dd/yyyy
        if (a > 12).any() and not (b > 12).any():
            dayfirst = True
        elif (b > 12).any() and not (a > 12).any():
            dayfirst = False
        else:
            s_d = (_month_spread(day_first), _date_score(day_first))
            s_m = (_month_spread(month_first), _date_score(month_first))
            dayfirst = s_d >= s_m
        result[ab_ok] = day_first if dayfirst else month_first

    # 3) anything else, one row at a time
    rest = result.isna() & raw.ne('') & ~raw.str.lower().isin(['nan', 'nat', 'none'])
    if rest.any():
        other = pd.to_datetime(raw[rest], errors='coerce', dayfirst=dayfirst, format='mixed')
        if getattr(other.dt, 'tz', None) is not None:
            other = other.dt.tz_localize(None)
        result[rest] = other

    return result.dt.normalize()


def parse_google_sheet_url(url: str) -> str:
    """
    Convert any standard Google Sheet URL (edit, view, share, or published link)
    into a direct CSV export endpoint.
    """
    url = str(url).strip()
    
    if "export?format=csv" in url or "pub?output=csv" in url:
        return url
        
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if match:
        sheet_id = match.group(1)
        gid_match = re.search(r"[?&]gid=([0-9]+)", url)
        if gid_match:
            gid = gid_match.group(1)
            return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
    return url

def robust_parse_financial_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly extract Date and Close columns from raw DataFrame, handling
    metadata headers, title rows, and varied column naming schemes.
    """
    if df_raw.empty:
        return pd.DataFrame()

    # Strategy 1: Header row search in top 10 rows
    for header_row in range(min(10, len(df_raw))):
        try:
            temp = df_raw.iloc[header_row:].copy()
            temp.columns = [str(c).strip() for c in temp.iloc[0]]
            temp = temp.iloc[1:].reset_index(drop=True)
            
            date_col = None
            close_col = None
            for col in temp.columns:
                col_lower = str(col).lower()
                if 'date' in col_lower or 'time' in col_lower:
                    date_col = col
                elif any(k in col_lower for k in ['close', 'price', 'nav', 'value', 'goldbees', 'nifty']):
                    close_col = col
                    
            if date_col and close_col:
                res = pd.DataFrame()
                res['Date'] = parse_dates(temp[date_col])
                res['Close'] = pd.to_numeric(temp[close_col].astype(str).str.replace(',', ''), errors='coerce')
                res = res.dropna(subset=['Date', 'Close']).sort_values('Date').reset_index(drop=True)
                if len(res) > 5:
                    res['Daily_Return'] = res['Close'].pct_change()
                    res['Cumulative_Return'] = (1 + res['Daily_Return'].fillna(0)).cumprod() - 1
                    return res
        except Exception:
            pass

    # Strategy 2: Scan all column pairs for dates & numeric prices
    for col in df_raw.columns:
        s = df_raw[col].astype(str)
        sample_dates = parse_dates(s)
        if sample_dates.notnull().sum() > 20:
            for col2 in df_raw.columns:
                if col != col2:
                    sample_num = pd.to_numeric(df_raw[col2].astype(str).str.replace(',', ''), errors='coerce')
                    if sample_num.notnull().sum() > 20:
                        res = pd.DataFrame({'Date': sample_dates, 'Close': sample_num})
                        res = res.dropna(subset=['Date', 'Close']).sort_values('Date').reset_index(drop=True)
                        if len(res) > 5:
                            res['Daily_Return'] = res['Close'].pct_change()
                            res['Cumulative_Return'] = (1 + res['Daily_Return'].fillna(0)).cumprod() - 1
                            return res

    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=120)
def fetch_google_sheet_csv(csv_url: str) -> tuple[pd.DataFrame | None, str | None]:
    """Cached version of download_sheet_csv()."""
    return download_sheet_csv(csv_url)


def download_sheet_csv(csv_url: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Fetch CSV directly from Google Sheets via HTTP requests with proper User-Agent.
    Returns (DataFrame, error_message). Not cached, so it can run in parallel threads.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(csv_url, headers=headers, timeout=12)
        if response.status_code in [401, 403]:
            return None, "UNAUTHORIZED"
            
        if response.status_code != 200:
            return None, f"HTTP Error {response.status_code}: Unable to reach Google Sheet."

        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type and '<html' in response.text.lower():
            return None, "UNAUTHORIZED"
            
        raw_df = pd.read_csv(io.StringIO(response.text), header=None)
        parsed_df = robust_parse_financial_dataframe(raw_df)
        
        if parsed_df.empty:
            return None, "Unable to extract Date and Close price columns from sheet."
            
        return parsed_df, None
        
    except Exception as e:
        return None, f"Connection failed: {str(e)}"

@st.cache_data(show_spinner=False)
def load_csv_data(filepath_or_buffer) -> pd.DataFrame:
    """Load financial CSV data from local file or buffer."""
    raw_df = pd.read_csv(filepath_or_buffer, header=None)
    parsed_df = robust_parse_financial_dataframe(raw_df)
    if parsed_df.empty:
        # Fallback to standard read_csv
        df = pd.read_csv(filepath_or_buffer)
        df.columns = [str(c).strip() for c in df.columns]
        if 'Date' in df.columns:
            df['Date'] = parse_dates(df['Date'])
        if 'Close' in df.columns:
            df['Close'] = pd.to_numeric(df['Close'].astype(str).str.replace(',', ''), errors='coerce')
        df = df.dropna(subset=['Date', 'Close']).sort_values('Date').reset_index(drop=True)
        df['Daily_Return'] = df['Close'].pct_change()
        df['Cumulative_Return'] = (1 + df['Daily_Return'].fillna(0)).cumprod() - 1
        return df.copy()
    return parsed_df.copy()

def get_available_datasets(directory: str = ".") -> dict:
    """Find all CSV dataset files available in workspace or parent directory."""
    datasets = {}
    search_dirs = [directory, os.path.dirname(os.path.abspath(directory))]
    for d in search_dirs:
        if os.path.exists(d):
            for filename in os.listdir(d):
                if filename.endswith(".csv") and not filename.startswith("."):
                    name = filename.replace("_data.csv", "").replace(".csv", "").replace("_", " ").title()
                    datasets[name] = os.path.join(d, filename)
    return datasets