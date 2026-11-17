"""📊 Mutual Funds page: full NAV history (from each fund's launch) of 18 Large / Mid / Small Cap funds.

Where the data comes from:
  Google Sheet "MF_NAV_History" (Drive folder BackTesting): one tab per fund
      Date | NAV - Regular Growth | NAV - Direct Growth
  plus an "Index" tab with each fund's name, category and AMFI scheme codes.
  History = AMFI's daily NAV files (from 3-Apr-2006); the sheet's Apps Script adds each new day at ~6 AM IST
  from api.mfapi.in. The sheet must be shared as "Anyone with the link → Viewer".
  The last good download is kept on disk and used if Google Sheets can't be reached.
"""
import html
import io
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.ui import panel, hero, note, sidebar_nav, BAD, AMBER
from src.data_loader import parse_dates

MF_SHEET_ID = "1kTBWCUK0Sk4qESVATUitjr6KAgN0V3tnbprGl0KtK-g"          # MF_NAV_History
MF_SHEET_LINK = f"https://docs.google.com/spreadsheets/d/{MF_SHEET_ID}/edit"
INDEX_GID = 737861081
# tab name -> gid (tab ids in the MF_NAV_History sheet)
FUND_TABS = {
    "LC - Mirae Asset": 1215407722, "LC - SBI": 1603216962, "LC - Invesco India": 1875256281,
    "LC - Nippon India": 885908979, "LC - ICICI Prudential": 895240370, "LC - Canara Robeco": 1088794199,
    "MC - Kotak": 1586544532, "MC - Invesco India": 137465654, "MC - HDFC": 895093236,
    "MC - Edelweiss": 920137933, "MC - Tata": 1934635721, "MC - Axis": 749543225,
    "SC - Nippon India": 1422255950, "SC - SBI": 1915893857, "SC - Axis": 930035628,
    "SC - HDFC": 2135183426, "SC - HSBC": 1414816242, "SC - Kotak": 2019375923,
}
GROUP_OF_PREFIX = {"LC": "Large Cap", "MC": "Mid Cap", "SC": "Small Cap"}
CACHE_FILE = os.path.join(tempfile.gettempdir(), "multi_asset_lab_sheets", "mf_nav_history.pkl")



# ---------------------------------------------------------
# Loading
# ---------------------------------------------------------
def _csv(gid: int) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{MF_SHEET_ID}/export?format=csv&gid={gid}"
    r = requests.get(url, timeout=(6, 60), headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code in (401, 403) or "text/html" in r.headers.get("Content-Type", ""):
        raise ValueError("sheet not shared: Share → General access → 'Anyone with the link' → Viewer")
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), header=None, dtype=str)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _index_info(raw: pd.DataFrame) -> dict:
    """{tab: (fund name, group, current regular code)} from the Index tab (header row has 'Tab')."""
    hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "Tab"]
    if not len(hdr):
        return {}
    t = raw.iloc[hdr[0] + 1:].copy()
    t.columns = [str(c).strip() for c in raw.iloc[hdr[0]]]
    out = {}
    for _, r in t.iterrows():
        tab = str(r.get("Tab", "")).strip()
        if tab in FUND_TABS:
            name = re.sub(r"\s*\(.*?\)", "", str(r.get("Fund", tab))).strip()
            codes = re.findall(r"\d+", str(r.get("AMFI code - Regular", "")))
            out[tab] = (name, str(r.get("Category", "")).strip(), int(codes[-1]) if codes else 0)
    return out


def _fund_frame(tab: str, raw: pd.DataFrame, info: dict) -> pd.DataFrame:
    body = raw.iloc[1:]
    df = pd.DataFrame({"Date": parse_dates(body.iloc[:, 0]),
                       "NAV_Regular": _num(body.iloc[:, 1]),
                       "NAV_Direct": _num(body.iloc[:, 2]) if body.shape[1] > 2 else np.nan})
    df = df.dropna(subset=["Date"])
    name, group, code = info.get(tab, (None, None, 0))
    prefix = tab.split(" - ")[0]
    df["Fund"] = name or tab.split(" - ", 1)[-1] + " " + GROUP_OF_PREFIX.get(prefix, "")
    df["Group"] = group or GROUP_OF_PREFIX.get(prefix, "Other")
    df["Code"] = code
    return df


@st.cache_data(show_spinner="Loading mutual fund NAVs from Google Sheets...", ttl=3600)
def load_mf_history():
    """All 18 funds, both plans. Returns (df, source, error).
    df columns: Date, Fund, Group, Code, NAV_Regular, NAV_Direct."""
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            idx_f = pool.submit(_csv, INDEX_GID)
            tab_f = {tab: pool.submit(_csv, gid) for tab, gid in FUND_TABS.items()}
            info = _index_info(idx_f.result())
            frames = [_fund_frame(tab, f.result(), info) for tab, f in tab_f.items()]
        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            raise ValueError("the fund tabs are empty")
        try:
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            df.to_pickle(CACHE_FILE)
        except OSError:
            pass
        return df, "live", None
    except Exception as e:                                   # noqa: BLE001  (shown to the user)
        err = str(e)[:160]
    try:
        return pd.read_pickle(CACHE_FILE), "saved", err
    except Exception:                                        # noqa: BLE001
        return pd.DataFrame(columns=["Date", "Fund", "Group", "Code", "NAV_Regular", "NAV_Direct"]), "none", err


# ---------------------------------------------------------
# Page: just the data, no sidebar
# ---------------------------------------------------------
def _table(full: pd.DataFrame, col: str) -> pd.DataFrame:
    """Date down the side, one column per fund (Large → Mid → Small Cap), newest date first."""
    wide = full.pivot_table(index="Date", columns="Fund", values=col, aggfunc="last")
    rank = {"Large Cap": 0, "Mid Cap": 1, "Small Cap": 2}
    group_of = full.groupby("Fund")["Group"].first()
    cols = sorted(wide.columns, key=lambda f: (rank.get(group_of.get(f), 9), f))
    wide = wide[cols].dropna(how="all").sort_index(ascending=False)
    wide.index = wide.index.date
    wide.index.name = "Date"
    wide.columns.name = None
    return wide.reset_index()


def render_mf_page(asset_dfs, fetch_errors):
    sidebar_nav("mf")
    st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],"
                "[data-testid='stExpandSidebarButton']{display:none !important}</style>", unsafe_allow_html=True)

    full, source, err = load_mf_history()
    hero("Mutual Funds", "Daily NAV of 18 Large, Mid and Small Cap funds, from their first day in AMFI's records.",
         page="mf", eyebrow="NAV data")
    if full.empty:
        note(f"Couldn't load the <a href='{MF_SHEET_LINK}' target='_blank'>MF_NAV_History sheet</a> "
             f"({html.escape(str(err))}). Share it as <b>Anyone with the link → Viewer</b>.", BAD, "⛔")
        st.stop()
    if source == "saved":
        note(f"Google Sheets couldn't be reached ({html.escape(str(err))}); showing the last copy downloaded.",
             AMBER, "⚠️")

    t_reg, t_dir = st.tabs(["Regular plan · Growth", "Direct plan · Growth"])
    for tab, col, fname in ((t_reg, "NAV_Regular", "mf_nav_regular.csv"), (t_dir, "NAV_Direct", "mf_nav_direct.csv")):
        with tab:
            table = _table(full, col)
            with panel():
                st.dataframe(table, hide_index=True, width="stretch", height=640,
                             column_config={"Date": st.column_config.DateColumn(format="DD-MM-YYYY", pinned=True),
                                            **{c: st.column_config.NumberColumn(format="%.4f")
                                               for c in table.columns if c != "Date"}})
            st.download_button("⬇️  Download CSV", table.to_csv(index=False).encode("utf-8-sig"),
                               file_name=fname, key=f"dl_{col}")