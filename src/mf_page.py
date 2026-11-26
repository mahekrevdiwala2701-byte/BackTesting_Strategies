"""📊 Mutual Funds page: full NAV history (from launch) of ONE fund, picked with ONLY_FUND below.
(The sheet has 18 Large / Mid / Small Cap funds; set ONLY_FUND to another tab name to switch.)

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

import plotly.graph_objects as go

from src.ui import (panel, hero, note, section, kpi_row, chart, sidebar_nav, PLOT_LAYOUT,
                    BAD, AMBER, VIOLET, EMERALD, INDIGO, TEAL)
from src.asset_data import ASSET_SHORT, COLORS
from src.everyday_portfolio_page import (drop_glitches, sharpe_at, rolling_return_at, week_dates_back,
                                         VOL_WEEKS)
from src.ui import hex_rgba, GOOD, FAINT
from src.data_loader import parse_dates
from src.mf_recovery import clean_nav, as_years_months
from src.period_picker import LAST_DATE          # 15-10-2026: last date shown anywhere in the app
from src.period_picker import DEFAULT_START      # 01-01-2018: default From date on every page

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
# The page shows only this one fund. Use any tab name from FUND_TABS above.
ONLY_FUND = "LC - Mirae Asset"
GROUP_OF_PREFIX = {"LC": "Large Cap", "MC": "Mid Cap", "SC": "Small Cap"}
CACHE_FILE = os.path.join(tempfile.gettempdir(), "multi_asset_lab_sheets", f"mf_nav_{ONLY_FUND.replace(' ', '')}.pkl")



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
    """ONLY_FUND, both plans. Returns (df, source, error).
    df columns: Date, Fund, Group, Code, NAV_Regular, NAV_Direct."""
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            idx_f = pool.submit(_csv, INDEX_GID)
            tab_f = {ONLY_FUND: pool.submit(_csv, FUND_TABS[ONLY_FUND])}
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
# Weekly log returns & Information ratio
# ---------------------------------------------------------
def _weekly_log_returns(series: pd.Series, dates, weeks: int = VOL_WEEKS) -> np.ndarray:
    """For each date d: the `weeks` weekly log returns ln(P(d−7(k−1)) / P(d−7k)), k = 1…weeks
    (a holiday uses the last trading day before it). Rows with an incomplete look-back are NaN."""
    s = series.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s[s > 0]
    dates = pd.DatetimeIndex(dates)
    if len(s) < 2 or len(dates) == 0:
        return np.full((len(dates), weeks), np.nan)
    logp = np.log(s.to_numpy(dtype=float))
    idx = s.index.values.astype("datetime64[ns]")
    offsets = np.arange(weeks + 1) * np.timedelta64(7, "D")
    targets = dates.values.astype("datetime64[ns]")[:, None] - offsets[None, :]
    pos = np.searchsorted(idx, targets.ravel(), side="right").reshape(targets.shape) - 1
    ok = (pos >= 0).all(axis=1) & (dates.values <= idx[-1])
    lp = logp[np.clip(pos, 0, None)]
    rets = lp[:, :-1] - lp[:, 1:]
    rets[~ok] = np.nan
    return rets


def ir_at(fund: pd.Series, bench: pd.Series, dates) -> pd.DataFrame:
    """Information ratio = (fund rolling 1Y return − benchmark rolling 1Y return) ÷ tracking error, at each date.
       Tracking error = annualised std (×√52) of the 52 weekly log return differences (fund − benchmark).
    Same 1-year return and 52-week window as the Sharpe ratio, so the two charts line up."""
    dates = pd.DatetimeIndex(dates)
    excess = rolling_return_at(fund, dates, 1) - rolling_return_at(bench, dates, 1)
    active = _weekly_log_returns(fund, dates) - _weekly_log_returns(bench, dates)
    te = pd.Series(np.nanstd(active, axis=1, ddof=1) * np.sqrt(52) * 100, index=dates)
    te[np.isnan(active).any(axis=1)] = np.nan
    return pd.DataFrame({"Excess 1Y Return (%)": excess, "Tracking error (%)": te,
                         "Information ratio": excess / te.where(te > 0)}, index=dates)



# ---------------------------------------------------------
# Weekly Sharpe ratio: SAME logic as the index Sharpe ratio on the Main Dashboard
#   Sharpe = (rolling 1-year return − risk-free) ÷ 52-week annualised log σ, shown every week
#   (src.everyday_portfolio_page.sharpe_at; full history is used for each 1-year look-back)
# ---------------------------------------------------------
INDEX_ORDER = ["Gold", "LargeCap", "MidCap", "SmallCap"]
CATEGORIES = ["Large Cap", "Mid Cap", "Small Cap"]                       # still imported by mf_recovery_page.py
CAT_COLORS = {"Large Cap": INDIGO, "Mid Cap": "#EA580C", "Small Cap": TEAL}


@st.cache_data(show_spinner="Calculating the weekly Sharpe ratio...", max_entries=16)
def fund_sharpe_weekly(nav: pd.Series, fund: str, asset_dfs: dict, d_from, d_to, rf: float):
    """({name: frame}, week dates). Fund first, then GoldBEES, Nifty 50, MidCap, SmallCap.
    Weeks = last date, 1 week before, 2 weeks before … back to d_from (same as the index chart)."""
    nav = clean_nav(nav)
    weeks = week_dates_back(min(nav.index[-1], pd.Timestamp(d_to)), max(pd.Timestamp(d_from), nav.index[0]))
    out = {fund: sharpe_at(nav, weeks, rf).dropna(how="all")}
    for a in INDEX_ORDER:
        if asset_dfs.get(a) is not None:
            clean, _ = drop_glitches(asset_dfs[a].set_index("Date")["Close"])
            out[ASSET_SHORT[a]] = sharpe_at(clean, weeks, rf).dropna(how="all")
    return {k: v for k, v in out.items() if len(v)}, weeks


def _line_chart(main: str, m: pd.DataFrame, metric: str, title: str, color: str, others: dict, key: str,
                hover_cols: tuple, hover_names: tuple, level: float = 0.0):
    """Same look as the index Sharpe chart: green fill above 0, red below, dashed average,
    other lines hidden in the legend (click to show), range slider underneath."""
    m = m.dropna(subset=[metric]).sort_index()
    xs, y = m.index.strftime("%Y-%m-%d"), m[metric].round(3)
    fig = go.Figure()
    for vals, fill in ((y.clip(lower=0), hex_rgba(GOOD, 0.22)), (y.clip(upper=0), hex_rgba(BAD, 0.25))):
        fig.add_trace(go.Scatter(x=xs, y=vals, mode="lines", line=dict(width=0), fill="tozeroy",
                                 fillcolor=fill, hoverinfo="skip", showlegend=False))
    c1, c2 = hover_cols
    n1, n2 = hover_names
    fig.add_trace(go.Scatter(
        x=xs, y=y, mode="lines+markers", name=f"{main[:28]} {metric.split()[0]}", marker=dict(size=3),
        line=dict(color=color, width=1.8),
        customdata=np.stack([m.index.strftime("%d-%m-%Y"), m[c1].round(2), m[c2].round(2)], axis=1),
        hovertemplate="%{customdata[0]}<br>" + n1 + " %{customdata[1]:+.2f}% · " + n2
                      + " %{customdata[2]:.2f}%<br><b>" + metric + " %{y:.2f}</b><extra>" + main[:30] + "</extra>"))
    for name, (series, col) in others.items():
        s = series.dropna().sort_index()
        fig.add_trace(go.Scatter(x=s.index.strftime("%Y-%m-%d"), y=s.round(3), mode="lines", name=name,
                                 line=dict(color=col, width=1.3), visible="legendonly",
                                 hovertemplate="<b>%{y:.2f}</b><extra>" + name + "</extra>"))
    fig.add_hline(y=0, line_color=FAINT, line_dash="dot")
    fig.add_hline(y=y.mean(), line_color=VIOLET, line_dash="dash", annotation_text=f"average {y.mean():.2f}",
                  annotation_position="top left", annotation_font=dict(color=VIOLET, size=11))
    fig.update_layout(**PLOT_LAYOUT, height=430, hovermode="x unified", title=title,
                      yaxis=dict(title=metric), xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.45))
    with panel():
        chart(fig, key=key)


def render_sharpe(nav_all: pd.DataFrame, asset_dfs: dict, fund: str, d_from, d_to):
    section("Sharpe ratio",
            "(Rolling 1-year return − risk-free) ÷ 52-week annualised log σ, shown every week", "⚖️", GOOD)
    c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
    plan = c1.segmented_control("Plan", ["Regular", "Direct"], default="Regular", key="mf_plan") or "Regular"
    rf = c2.number_input("Risk-free rate (% a year)", min_value=0.0, max_value=20.0, value=0.0, step=0.25,
                         format="%.2f", key="mf_rf")
    nav_col = "NAV_Regular" if plan == "Regular" else "NAV_Direct"
    nav = nav_all.set_index("Date")[nav_col].dropna()
    if len(nav) < 300:
        note(f"The {plan} plan doesn't have enough NAV history.", AMBER, "⏳")
        return None, plan, rf
    weekly, weeks = fund_sharpe_weekly(nav, fund, asset_dfs, d_from, d_to, rf)
    if fund not in weekly:
        note("Not enough history yet: the Sharpe ratio needs 1 year of returns and 52 weeks of prices "
             "before each date.", AMBER, "⏳")
        return None, plan, rf
    m = weekly[fund].sort_index()
    sh = m["Sharpe"].dropna()
    nifty = weekly.get("Nifty 50", pd.DataFrame(columns=["Sharpe"]))["Sharpe"].reindex(sh.index)
    beat = (sh > nifty).sum() / max(nifty.notna().sum(), 1) * 100
    kpi_row([
        dict(label="Sharpe now", value=f"{sh.iloc[-1]:.2f}", delta=f"week of {sh.index[-1]:%d %b %Y}",
             tone="good" if sh.iloc[-1] > 0 else "bad", icon="⚖️", accent=EMERALD),
        dict(label="Average Sharpe", value=f"{sh.mean():.2f}",
             delta=f"Nifty 50: {nifty.mean():.2f}" if nifty.notna().any() else None,
             tone="good" if nifty.notna().any() and sh.mean() > nifty.mean() else "neutral", icon="📊", accent=INDIGO),
        dict(label="Weeks above 0", value=f"{(sh > 0).mean() * 100:.0f}%", delta=f"{len(sh)} weeks",
             icon="🟩", accent=TEAL),
        dict(label="Weeks beating Nifty 50", value=f"{beat:.0f}%", icon="🏁", accent=VIOLET),
    ])

    others = {k: (v["Sharpe"], COLORS[a]) for a in INDEX_ORDER
              for k, v in weekly.items() if k == ASSET_SHORT[a]}
    _line_chart(fund, m, "Sharpe", f"{fund}: Sharpe ratio every week (risk-free {rf:.2f}%)",
                "#0F1B3D", others, "mf_sh_line", ("Rolling 1Y Return (%)", "Annualised σ (%)"), ("Rp", "σ"))
    st.caption("Green = Sharpe above 0, red = below 0. Click an index in the legend to add its line. "
               "Drag the slider under the chart to zoom.")

    table = pd.DataFrame(index=weeks)
    for k, v in weekly.items():
        w = v.reindex(weeks)
        table[f"{k} Rp 1Y (%)"] = w["Rolling 1Y Return (%)"]
        table[f"{k} σ (%)"] = w["Annualised σ (%)"]
        table[f"{k} Sharpe"] = w["Sharpe"]
    table = table.dropna(how="all")
    show = table.copy()
    show.insert(0, "Week", range(len(show)))
    show.insert(1, "Date", show.index.strftime("%d-%m-%Y"))
    fmt = {c: ("{:+.2f}%" if "Rp" in c else "{:.2f}%" if "σ" in c else "{:.2f}") for c in table.columns}
    with st.expander("📋 Weekly Sharpe table", expanded=False):
        st.dataframe(show.style.format(fmt, na_rep="—"), width="stretch", hide_index=True, height=400)
    export = table.round(4).copy()
    export.index = export.index.strftime("%Y-%m-%d")
    export.index.name = "Date"
    st.download_button("⬇️  Download weekly Sharpe table (CSV)", export.to_csv().encode("utf-8-sig"),
                       file_name=f"weekly_sharpe_{re.sub(r'[^A-Za-z0-9]+', '_', fund)}_{plan.lower()}.csv",
                       mime="text/csv", key="mf_sh_dl")
    return m, plan, rf


# ---------------------------------------------------------
# Information ratio every week (vs Nifty 50, same 1-year / 52-week window as the Sharpe)
# ---------------------------------------------------------
def render_ir(nav_all: pd.DataFrame, asset_dfs: dict, fund: str, weeks, plan: str):
    section("Information ratio", "(Fund 1-year return − Nifty 50 1-year return) ÷ 52-week tracking error, "
            "shown every week", "🎯", VIOLET)
    if asset_dfs.get("LargeCap") is None:
        note("Nifty 50 data isn't loaded, so the Information ratio can't be calculated.", AMBER, "⏳")
        return
    bench = drop_glitches(asset_dfs["LargeCap"].set_index("Date")["Close"])[0]
    nav = clean_nav(nav_all.set_index("Date")["NAV_Regular" if plan == "Regular" else "NAV_Direct"])
    ir = ir_at(nav, bench, weeks).dropna(subset=["Information ratio"])
    if ir.empty:
        note("Not enough history for the Information ratio in these dates.", AMBER, "⏳")
        return
    _line_chart(fund, ir, "Information ratio", f"{fund}: Information ratio every week (vs Nifty 50)", INDIGO, {},
                "mf_ir_line", ("Excess 1Y Return (%)", "Tracking error (%)"), ("Excess", "TE"))
    st.caption("Above 0 = the fund beat Nifty 50 over that year; the higher, the more consistently.")


# ---------------------------------------------------------
# Recovery based on the Sharpe ratio:
#   the Sharpe drops below a level (0 by default) → its lowest point → the first week back at/above the level
# ---------------------------------------------------------
_FALL_C, _RISE_C = "#DC2626", "#16A34A"


def sharpe_dips(sh: pd.Series, level: float = 0.0) -> pd.DataFrame:
    """One row per stretch of weeks with Sharpe below `level`, oldest first."""
    s = sh.dropna().sort_index()
    cols = ["Below from", "Lowest on", "Lowest Sharpe", "Back on", "Weeks below", "Weeks to recover", "Recovered"]
    below = (s < level).to_numpy()
    rows, i, n = [], 0, len(s)
    while i < n:
        if not below[i]:
            i += 1
            continue
        start = i
        while i < n and below[i]:
            i += 1
        bot = start + int(s.iloc[start:i].to_numpy().argmin())
        ok = i < n
        end = s.index[i] if ok else s.index[-1]
        rows.append({"Below from": s.index[start], "Lowest on": s.index[bot], "Lowest Sharpe": s.iloc[bot],
                     "Back on": s.index[i] if ok else pd.NaT,
                     "Weeks below": int(round((end - s.index[start]).days / 7)),
                     "Weeks to recover": int(round((end - s.index[bot]).days / 7)),
                     "Recovered": ok})
    return pd.DataFrame(rows, columns=cols)


def _span(w, ok) -> str:
    txt = f"{int(w)} wk" if w < 9 else as_years_months(w * 7)
    return txt + ("" if ok else " (not back yet)")


def render_sharpe_recovery(m: pd.DataFrame, fund: str, plan: str):
    section("Recovery time (Sharpe ratio)",
            "When the Sharpe ratio drops below a level, how long until it climbs back?", "⛰️", TEAL)
    level = st.number_input("Recovery level (Sharpe)", min_value=-3.0, max_value=5.0, value=0.0, step=0.25,
                            format="%.2f", key="mf_rec_level",
                            help="0 = the fund earned less than the risk-free rate over the past year. "
                                 "Try 1.0 to see how long it took to get back to a 'good' Sharpe.")
    sh = m["Sharpe"].dropna().sort_index()
    d = sharpe_dips(sh, level)
    if d.empty:
        note(f"The Sharpe ratio never went below {level:.2f} in these dates.", GOOD, "✅")
        return
    worst = d.loc[d["Lowest Sharpe"].idxmin()]
    longest = d.loc[d["Weeks below"].idxmax()]
    rec = d[d["Recovered"]]
    now_below = not d.iloc[-1]["Recovered"]
    kpi_row([
        dict(label="Times it fell below", value=f"{len(d)}", delta=f"level {level:.2f}", icon="🔁", accent=INDIGO),
        dict(label="Lowest Sharpe", value=f"{worst['Lowest Sharpe']:.2f}", delta=f"{worst['Lowest on']:%d %b %Y}",
             tone="bad", icon="📉", accent=BAD),
        dict(label="Longest time below", value=_span(longest["Weeks below"], longest["Recovered"]),
             delta=f"from {longest['Below from']:%d %b %Y}", tone="bad", icon="⏳", accent=AMBER),
        dict(label="Average recovery", value=_span(rec["Weeks to recover"].mean(), True) if len(rec) else "—",
             delta="lowest point → back above level", icon="⏱️", accent=TEAL),
        dict(label="Right now", value="below" if now_below else "above", tone="bad" if now_below else "good",
             delta=f"Sharpe {sh.iloc[-1]:.2f}", icon="📍", accent=EMERALD),
    ])

    # chart: Sharpe line, red = falling to its lowest point, green = climbing back above the level
    xs = sh.index
    fig = go.Figure(go.Scatter(x=xs, y=sh.round(3), mode="lines", line=dict(color="#0F1B3D", width=1.8),
                               name="Sharpe", hovertemplate="%{x|%d %b %Y}<br>Sharpe %{y:.2f}<extra></extra>"))
    for _, r in d.iterrows():
        end = r["Back on"] if r["Recovered"] else xs[-1]
        fig.add_vrect(x0=r["Below from"], x1=r["Lowest on"], fillcolor=hex_rgba(_FALL_C, 0.15), line_width=0)
        fig.add_vrect(x0=r["Lowest on"], x1=end, fillcolor=hex_rgba(_RISE_C, 0.14), line_width=0)
        fig.add_trace(go.Scatter(x=[r["Lowest on"]], y=[r["Lowest Sharpe"]], mode="markers", showlegend=False,
                                 marker=dict(color=_FALL_C, size=7),
                                 hovertemplate=f"Lowest {r['Lowest Sharpe']:.2f}<br>recovered in "
                                               f"{_span(r['Weeks to recover'], r['Recovered'])}<extra></extra>"))
    fig.add_hline(y=level, line_color=VIOLET, line_dash="dash", annotation_text=f"level {level:.2f}",
                  annotation_position="top left", annotation_font=dict(color=VIOLET, size=11))
    fig.update_layout(**PLOT_LAYOUT, height=420, showlegend=False, hovermode="x unified",
                      title=f"{fund}: Sharpe ratio recovery ({plan} plan)", yaxis=dict(title="Sharpe ratio"),
                      xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)))
    with panel():
        chart(fig, key="mf_rec_chart")
    st.caption(f"🟥 Red = Sharpe falling below {level:.2f} down to its lowest point. "
               f"🟩 Green = climbing back to {level:.2f}. Red dot = lowest Sharpe of each dip.")

    # bar chart: how long each dip lasted
    lab = [f"{r['Below from']:%b %Y}" for _, r in d.iterrows()]
    bars = go.Figure()
    bars.add_trace(go.Bar(x=lab, y=d["Weeks below"] - d["Weeks to recover"], name="Falling (weeks)",
                          marker_color=hex_rgba(_FALL_C, 0.75)))
    bars.add_trace(go.Bar(x=lab, y=d["Weeks to recover"], name="Recovering (weeks)",
                          marker_color=[hex_rgba(_RISE_C, 0.8) if ok else hex_rgba(AMBER, 0.8)
                                        for ok in d["Recovered"]],
                          text=[_span(w, ok) for w, ok in zip(d["Weeks below"], d["Recovered"])],
                          textposition="outside", cliponaxis=False))
    bars.update_layout(**PLOT_LAYOUT, barmode="stack", height=320, yaxis=dict(title="Weeks below the level"),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                       margin=dict(l=12, r=12, t=40, b=12))
    with panel():
        chart(bars, key="mf_rec_bars")
    st.caption("Each bar = one dip, named by the month it started. Total height = weeks below the level; "
               "green part = weeks from the lowest point back to the level (amber = not back yet).")

    view = pd.DataFrame({
        "Below from": d["Below from"].dt.date, "Lowest on": d["Lowest on"].dt.date,
        "Lowest Sharpe": d["Lowest Sharpe"],
        "Back above on": [x.date() if ok else None for x, ok in zip(d["Back on"], d["Recovered"])],
        "Time below": [_span(w, ok) for w, ok in zip(d["Weeks below"], d["Recovered"])],
        "Recovery (lowest → back)": [_span(w, ok) for w, ok in zip(d["Weeks to recover"], d["Recovered"])],
    }).iloc[::-1]
    with panel():
        st.dataframe(view.style.format({"Lowest Sharpe": "{:.2f}"}, na_rep="not yet"), hide_index=True,
                     width="stretch",
                     column_config={c: st.column_config.DateColumn(format="DD-MM-YYYY")
                                    for c in ("Below from", "Lowest on", "Back above on")})
    st.download_button("⬇️  Download (CSV)", view.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"mf_sharpe_recovery_{plan.lower()}.csv", key="mf_rec_dl")


# ---------------------------------------------------------
# Page
# ---------------------------------------------------------
def render_mf_page(asset_dfs, fetch_errors):
    sidebar_nav("mf")
    st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],"
                "[data-testid='stExpandSidebarButton']{display:none !important}</style>", unsafe_allow_html=True)

    full, source, err = load_mf_history()
    if full.empty:
        hero("Mutual Fund", "No data loaded.", page="mf", eyebrow="NAV data")
        note(f"Couldn't load the <a href='{MF_SHEET_LINK}' target='_blank'>MF_NAV_History sheet</a> "
             f"({html.escape(str(err))}). Share it as <b>Anyone with the link → Viewer</b>.", BAD, "⛔")
        st.stop()
    fund = full["Fund"].iloc[0]
    full = full[full["Fund"] == fund].sort_values("Date")        # one fund only (also guards an old saved copy)
    hero(fund, f"{full['Group'].iloc[0]} · daily NAV from its first day in AMFI's records.",
         page="mf", eyebrow="Mutual Fund")
    if source == "saved":
        note(f"Google Sheets couldn't be reached ({html.escape(str(err))}); showing the last copy downloaded.",
             AMBER, "⚠️")

    # ---------------- date filter ----------------
    first_day = full["Date"].min().date()
    last_day = min(LAST_DATE, full["Date"].max().date())
    if "mf_from" not in st.session_state:
        st.session_state["mf_from"] = min(max(DEFAULT_START, first_day), last_day)
        st.session_state["mf_to"] = last_day
    with panel():
        c1, c2, c3 = st.columns([1, 1, 2], vertical_alignment="bottom")
        d_from = c1.date_input("📍 From date", min_value=first_day, max_value=last_day, format="DD-MM-YYYY",
                               key="mf_from")
        d_to = c2.date_input("🏁 To date", min_value=first_day, max_value=last_day, format="DD-MM-YYYY",
                             key="mf_to")
        c3.caption(f"Starts on **{max(DEFAULT_START, first_day):%d %b %Y}** by default. You can go back to "
                   f"**{first_day:%d %b %Y}** (first NAV of {fund}). The 1-year look-back before the From date "
                   "uses the full history, same as the index charts.")
    if d_from > d_to:
        d_from, d_to = d_to, d_from
    nav_all = full[full["Date"] <= pd.Timestamp(d_to)]             # history up to To (look-back needs earlier data)
    full = full[(full["Date"] >= pd.Timestamp(d_from)) & (full["Date"] <= pd.Timestamp(d_to))]

    m, plan, rf = render_sharpe(nav_all, asset_dfs, fund, d_from, d_to)
    if m is not None:
        render_ir(nav_all, asset_dfs, fund, m.index, plan)
        render_sharpe_recovery(m, fund, plan)

    section("NAV data", f"Daily NAV of {fund} for the dates above", "📋", INDIGO)
    table = (full[["Date", "NAV_Regular", "NAV_Direct"]].dropna(subset=["NAV_Regular", "NAV_Direct"], how="all")
             .rename(columns={"NAV_Regular": "Regular plan · Growth", "NAV_Direct": "Direct plan · Growth"}))
    table["Date"] = table["Date"].dt.date
    with panel():
        st.dataframe(table, hide_index=True, width="stretch", height=640,
                     column_config={"Date": st.column_config.DateColumn(format="DD-MM-YYYY", pinned=True),
                                    **{c: st.column_config.NumberColumn(format="%.4f")
                                       for c in table.columns if c != "Date"}})
    st.download_button("⬇️  Download CSV", table.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"mf_nav_{re.sub(r'[^A-Za-z0-9]+', '_', fund).lower()}.csv", key="mf_nav_dl")