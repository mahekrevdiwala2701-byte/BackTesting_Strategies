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

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.ui import (panel, hero, note, section, kpi_row, chart, sidebar_nav, PLOT_LAYOUT,
                    BAD, AMBER, VIOLET, EMERALD, INDIGO, TEAL, ORANGE, GOOD_TEXT, BAD_TEXT, MUTED)
from src.asset_data import ASSET_SHORT
from src.everyday_portfolio_page import (drop_glitches, sharpe_at, rolling_return_at, week_dates_back,
                                         VOL_WEEKS)
from src.ui import hex_rgba, GOOD, FAINT
from src.data_loader import parse_dates
from src.mf_recovery import clean_nav, falls, biggest_falls, as_years_months
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
# Sharpe ratio & Information ratio
# ---------------------------------------------------------
CATEGORIES = ["Large Cap", "Mid Cap", "Small Cap"]
CAT_COLORS = {"Large Cap": INDIGO, "Mid Cap": ORANGE, "Small Cap": TEAL}
# Benchmark for every scheme = Nifty 50 (the app's own "LargeCap" series, same data the other pages use)
CAT_BENCHMARK = {"Large Cap": "LargeCap", "Mid Cap": "LargeCap", "Small Cap": "LargeCap"}


def ratio_table(full: pd.DataFrame, nav_col: str, asset_dfs: dict, rf_pct: float) -> pd.DataFrame:
    """One row per scheme, over the dates in `full` (already cut to the picked period).

    Sharpe ratio      = (average daily return − daily risk-free rate) × N ÷ (daily volatility × √N)
    Information ratio = average daily (fund − benchmark) return × N ÷ (std of that difference × √N)
    N = number of returns per year actually in the data (≈ 250 for daily prices), so a benchmark sheet
    with gaps can't inflate the ratios.
    Only dates on which the fund AND its category benchmark both have a price are used, for both ratios,
    so every number in a row covers exactly the same days."""
    bench = {cat: drop_glitches(asset_dfs[name].set_index("Date")["Close"])[0]
             for cat, name in CAT_BENCHMARK.items() if asset_dfs.get(name) is not None}
    rows = []
    for (cat, fund), g in full.dropna(subset=[nav_col]).groupby(["Group", "Fund"]):
        nav = g.set_index("Date")[nav_col].sort_index()
        b = bench.get(cat)
        if b is None:
            continue
        common = nav.index.intersection(b.index)
        if len(common) < 60:
            continue
        nav, bb = nav.loc[common], b.loc[common]
        r, br = nav.pct_change().dropna(), bb.pct_change().dropna()
        years = (common[-1] - common[0]).days / 365.25
        if years <= 0:
            continue
        ppy = len(r) / years                     # returns per year actually observed (≈ 250 for daily data)
        rf_p = (1 + rf_pct / 100) ** (1 / ppy) - 1
        vol, b_vol = r.std() * np.sqrt(ppy), br.std() * np.sqrt(ppy)
        active = r - br
        te = active.std() * np.sqrt(ppy)
        cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
        b_cagr = (bb.iloc[-1] / bb.iloc[0]) ** (1 / years) - 1
        rows.append({
            "Category": cat, "Fund": fund, "Benchmark": ASSET_SHORT.get(CAT_BENCHMARK[cat], CAT_BENCHMARK[cat]),
            "From": common[0].date(), "To": common[-1].date(), "Years": years,
            "CAGR %": cagr * 100, "Volatility %": vol * 100,
            "Sharpe ratio": (r.mean() - rf_p) * ppy / vol if vol else np.nan,
            "Benchmark CAGR %": b_cagr * 100,
            "Benchmark Sharpe": (br.mean() - rf_p) * ppy / b_vol if b_vol else np.nan,
            "Excess return %": active.mean() * ppy * 100,
            "Tracking error %": te * 100,
            "Information ratio": active.mean() * ppy / te if te else np.nan,
        })
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["Sharpe rank"] = t.groupby("Category")["Sharpe ratio"].rank(ascending=False, method="min").astype(int)
    t["IR rank"] = t.groupby("Category")["Information ratio"].rank(ascending=False, method="min").astype(int)
    t["Category"] = pd.Categorical(t["Category"], CATEGORIES, ordered=True)
    return t.sort_values(["Category", "Sharpe ratio"], ascending=[True, False]).reset_index(drop=True)


def _ratio_chart(t: pd.DataFrame, key: str):
    """Two horizontal bar charts side by side: Sharpe ratio and Information ratio, best at the top."""
    t = t.sort_values("Sharpe ratio")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("Sharpe ratio (higher = more return per unit of risk)",
                                        "Information ratio (higher = beats its index more consistently)"))
    colors = [CAT_COLORS.get(str(c), VIOLET) for c in t["Category"]]
    for col, (metric, fmt) in enumerate((("Sharpe ratio", ".2f"), ("Information ratio", ".2f")), start=1):
        fig.add_trace(go.Bar(x=t[metric], y=t["Fund"], orientation="h", marker_color=colors,
                             text=[f"{v:{fmt}}" for v in t[metric]], textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}<br>" + metric + ": %{x:.2f}<extra></extra>", showlegend=False),
                      row=1, col=col)
        fig.add_vline(x=0, line_color=MUTED, line_width=1, row=1, col=col)
    bench = t.drop_duplicates("Category")
    for _, b in bench.iterrows():                       # each index's own Sharpe, as a dashed reference line
        if pd.notna(b["Benchmark Sharpe"]) and t["Category"].nunique() == 1:
            fig.add_vline(x=b["Benchmark Sharpe"], line_dash="dash", line_color=CAT_COLORS.get(str(b["Category"])),
                          annotation_text=f"{b['Benchmark']} {b['Benchmark Sharpe']:.2f}",
                          annotation_position="bottom right", row=1, col=1)
    fig.update_layout(**PLOT_LAYOUT, height=max(320, 34 * len(t) + 120), margin=dict(l=10, r=40, t=60, b=10))
    fig.update_annotations(font_size=12)
    chart(fig, key=key)


def _ratio_table_view(t: pd.DataFrame, show_category: bool, key: str):
    cols = (["Category"] if show_category else []) + [
        "Fund", "Sharpe ratio", "Sharpe rank", "Information ratio", "IR rank", "CAGR %", "Benchmark CAGR %",
        "Excess return %", "Volatility %", "Tracking error %", "Benchmark", "From", "To"]
    view = t[cols].copy()
    view["Category"] = view["Category"].astype(str) if show_category else None
    if not show_category:
        view = view.drop(columns="Category")

    def tone(v):
        if pd.isna(v):
            return ""
        return f"color:{GOOD_TEXT};font-weight:700" if v > 0 else f"color:{BAD_TEXT};font-weight:700" if v < 0 else ""

    def best(col):                      # highlight the best fund of each category in this column
        out = []
        for cat_val, v in zip(t["Category"], t[col]):
            top = t.loc[t["Category"] == cat_val, col].max()
            out.append("background-color:#E8F7EE;font-weight:800" if pd.notna(v) and v == top else "")
        return out

    num = {c: "{:.2f}" for c in ["Sharpe ratio", "Information ratio", "CAGR %", "Benchmark CAGR %",
                                 "Excess return %", "Volatility %", "Tracking error %"]}
    sty = view.style.format(num, na_rep="—")
    apply_fn = sty.map if hasattr(sty, "map") else sty.applymap
    sty = apply_fn(tone, subset=["Sharpe ratio", "Information ratio", "Excess return %"])
    sty = sty.apply(lambda _s: best("Sharpe ratio"), subset=["Sharpe ratio"])
    sty = sty.apply(lambda _s: best("Information ratio"), subset=["Information ratio"])
    st.dataframe(sty, hide_index=True, width="stretch", key=key,
                 column_config={"From": st.column_config.DateColumn(format="DD-MM-YYYY"),
                                "To": st.column_config.DateColumn(format="DD-MM-YYYY"),
                                "Sharpe rank": st.column_config.NumberColumn("Rank (Sharpe)", help="Rank inside its category"),
                                "IR rank": st.column_config.NumberColumn("Rank (IR)", help="Rank inside its category")})


def render_ratios(full: pd.DataFrame, asset_dfs: dict, d_from, d_to):
    section("Sharpe ratio & Information ratio",
            f"Every scheme, {d_from:%d %b %Y} → {d_to:%d %b %Y} (change the dates above)", "🎯", VIOLET)
    with panel():
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        plan = c1.segmented_control("Plan", ["Regular", "Direct"], default="Regular", key="mf_ratio_plan")
        rf = c2.number_input("Risk-free rate (% a year)", min_value=0.0, max_value=15.0, value=0.0, step=0.25,
                             key="mf_rf", help="Used for the Sharpe ratio. 0% by default (same as the Main Dashboard); type e.g. 6.5 for the T-bill rate.")
    plan = plan or "Regular"
    st.session_state["_mf_plan_rf"] = (plan, rf)
    t = ratio_table(full, "NAV_Regular" if plan == "Regular" else "NAV_Direct", asset_dfs, rf)
    if t.empty:
        note("Not enough data in this date range to calculate the ratios. Pick a longer range.", AMBER, "⏳")
        return

    tabs = st.tabs(["🗂️ All", "🏢 Large Cap", "🏬 Mid Cap", "🏪 Small Cap"])
    for tab, cat in zip(tabs, ["All"] + CATEGORIES):
        with tab:
            sub = t if cat == "All" else t[t["Category"] == cat]
            if sub.empty:
                note(f"No {cat} scheme has data in this date range.", AMBER, "⏳")
                continue
            bs, bi = sub.loc[sub["Sharpe ratio"].idxmax()], sub.loc[sub["Information ratio"].idxmax()]
            cards = [
                dict(label="Best Sharpe ratio", value=f"{bs['Sharpe ratio']:.2f}", delta=bs["Fund"][:38],
                     tone="good", icon="🏆", accent=EMERALD),
                dict(label="Best Information ratio", value=f"{bi['Information ratio']:.2f}", delta=bi["Fund"][:38],
                     tone="good", icon="🎯", accent=VIOLET),
                dict(label="Average Sharpe", value=f"{sub['Sharpe ratio'].mean():.2f}",
                     delta=f"{len(sub)} schemes", icon="📊", accent=INDIGO),
            ]
            if cat != "All":
                cards.append(dict(label=f"{sub['Benchmark'].iloc[0]} Sharpe (index)",
                                  value=f"{sub['Benchmark Sharpe'].iloc[0]:.2f}",
                                  delta=f"{(sub['Sharpe ratio'] > sub['Benchmark Sharpe'].iloc[0]).sum()} of "
                                        f"{len(sub)} schemes beat it", icon="📍", accent=TEAL))
            kpi_row(cards)
            with panel():
                _ratio_chart(sub, key=f"mf_ratio_chart_{cat}")
                if cat == "All":
                    st.caption("Bar colours: 🔵 Large Cap · 🟠 Mid Cap · 🟢 Small Cap")
            with panel():
                _ratio_table_view(sub, show_category=(cat == "All"), key=f"mf_ratio_tbl_{cat}")
            st.download_button("⬇️  Download ratios CSV", sub.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"mf_sharpe_ir_{cat.replace(' ', '_').lower()}_{plan.lower()}.csv",
                               key=f"mf_ratio_dl_{cat}")

    st.caption(f"How it's calculated ({plan} plan, Growth option): daily NAV returns on the days both the scheme and "
               f"its index have a price, turned into yearly figures. **Sharpe ratio** = (yearly average return − "
               f"risk-free rate {rf:.2f}%) ÷ yearly volatility. **Information ratio** = yearly average of (scheme return − index return) ÷ tracking "
               "error (how much the scheme's return wanders from the index). Benchmark for every scheme: Nifty 50 "
               "(the same Nifty 50 data used on the other pages). "
               "Green = best in its category. Rank is inside the category.")


# ---------------------------------------------------------
# Sharpe ratio & Information ratio every week (same method as the Main Dashboard's Sharpe view)
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


@st.cache_data(show_spinner="Calculating weekly Sharpe and Information ratios...", max_entries=16)
def weekly_ratios(full: pd.DataFrame, nav_col: str, bench: pd.Series, d_from, d_to, rf: float):
    """{fund: frame with Sharpe and IR columns at every week}, newest week first, plus Nifty 50's own Sharpe.
    Full history is used for each 1-year look-back; only weeks inside d_from…d_to are returned."""
    out = {}
    for fund, g in full.dropna(subset=[nav_col]).groupby("Fund"):
        nav = g.set_index("Date")[nav_col].sort_index()
        last = min(nav.index[-1], pd.Timestamp(d_to))
        weeks = week_dates_back(last, max(pd.Timestamp(d_from), nav.index[0]))
        if len(weeks) == 0:
            continue
        sh = sharpe_at(nav, weeks, rf)
        ir = ir_at(nav, bench, weeks)
        f = pd.concat([sh, ir], axis=1).dropna(subset=["Sharpe", "Information ratio"], how="all")
        if len(f):
            out[fund] = f.sort_index()
    weeks = week_dates_back(min(bench.index[-1], pd.Timestamp(d_to)), pd.Timestamp(d_from))
    nifty = sharpe_at(bench, weeks, rf).dropna(how="all").sort_index()
    return out, nifty


def _weekly_chart(main: str, frames: dict, metric: str, title: str, color: str, key: str,
                  extra_hover: tuple, others: dict):
    """Line for `main` with green fill above 0 / red below, dashed average, other lines in the legend."""
    m = frames[main].dropna(subset=[metric])
    xs, y = m.index.strftime("%Y-%m-%d"), m[metric].round(3)
    fig = go.Figure()
    for vals, fill in ((y.clip(lower=0), hex_rgba(GOOD, 0.22)), (y.clip(upper=0), hex_rgba(BAD, 0.25))):
        fig.add_trace(go.Scatter(x=xs, y=vals, mode="lines", line=dict(width=0), fill="tozeroy",
                                 fillcolor=fill, hoverinfo="skip", showlegend=False))
    c1, c2 = extra_hover
    fig.add_trace(go.Scatter(
        x=xs, y=y, mode="lines+markers", name=main, marker=dict(size=3), line=dict(color=color, width=1.8),
        customdata=np.stack([m.index.strftime("%d-%m-%Y"), m[c1].round(2), m[c2].round(2)], axis=1),
        hovertemplate="%{customdata[0]}<br>" + c1.split(" (")[0] + " %{customdata[1]:+.2f}% · "
                      + c2.split(" (")[0] + " %{customdata[2]:.2f}%<br><b>" + metric + " %{y:.2f}</b><extra></extra>"))
    for name, (series, col) in others.items():
        fig.add_trace(go.Scatter(x=series.index.strftime("%Y-%m-%d"), y=series.round(3), mode="lines", name=name,
                                 line=dict(color=col, width=1.3), visible="legendonly",
                                 hovertemplate="<b>%{y:.2f}</b><extra>" + name[:40] + "</extra>"))
    fig.add_hline(y=0, line_color=FAINT, line_dash="dot")
    fig.add_hline(y=y.mean(), line_color=VIOLET, line_dash="dash", annotation_text=f"average {y.mean():.2f}",
                  annotation_position="top left", annotation_font=dict(color=VIOLET, size=11))
    n_rows = 1 + (len(others) + 2) // 3                       # legend rows below the chart (3 names per row)
    fig.update_layout(**PLOT_LAYOUT, height=470 + 22 * n_rows, hovermode="x unified",
                      title=dict(text=title, x=0.01, xanchor="left", y=0.98, yanchor="top"),
                      yaxis=dict(title=metric), xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
                      legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, xanchor="left",
                                  entrywidth=0.33, entrywidthmode="fraction", font=dict(size=11)),
                      margin=dict(l=12, r=12, t=56, b=40 + 22 * n_rows), autosize=True)
    chart(fig, key=key)


def render_weekly(full: pd.DataFrame, asset_dfs: dict, d_from, d_to, plan: str, rf: float):
    section("Sharpe ratio & Information ratio every week",
            "Each point = the last 1 year up to that week · Information ratio vs Nifty 50", "📈", INDIGO)
    if asset_dfs.get("LargeCap") is None:
        note("Nifty 50 data isn't loaded, so the weekly ratios can't be calculated.", AMBER, "⏳")
        return
    bench = drop_glitches(asset_dfs["LargeCap"].set_index("Date")["Close"])[0]
    nav_col = "NAV_Regular" if plan == "Regular" else "NAV_Direct"
    frames, nifty = weekly_ratios(full, nav_col, bench, d_from, d_to, rf)
    if not frames:
        note("Not enough history: each weekly point needs 1 year of NAVs before it. Pick a longer date range.",
             AMBER, "⏳")
        return

    group_of = full.groupby("Fund")["Group"].first()
    rank = {c: i for i, c in enumerate(CATEGORIES)}
    funds = sorted(frames, key=lambda f: (rank.get(group_of.get(f), 9), f))
    with panel():
        c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
        cat = c1.selectbox("Category", ["All"] + CATEGORIES, key="mf_w_cat")
        pool = funds if cat == "All" else [f for f in funds if group_of.get(f) == cat]
        if not pool:
            note(f"No {cat} scheme has a full year of data in this range.", AMBER, "⏳")
            return
        if st.session_state.get("mf_w_fund") not in pool:
            st.session_state["mf_w_fund"] = pool[0]
        main = c2.selectbox("Scheme", pool, key="mf_w_fund",
                            format_func=lambda f: f"{group_of.get(f, '')} · {f}")
    color = CAT_COLORS.get(group_of.get(main), INDIGO)
    peers = [f for f in funds if f != main and group_of.get(f) == group_of.get(main)]   # same category only
    palette = [INDIGO, ORANGE, TEAL, VIOLET, EMERALD, "#DB2777", "#2563EB", "#C8901A", "#0891B2", "#65A30D"]

    sharpe_others = {f: (frames[f]["Sharpe"].dropna(), palette[i % len(palette)]) for i, f in enumerate(peers)}
    if len(nifty):
        sharpe_others["Nifty 50"] = (nifty["Sharpe"].dropna(), "#0F1B3D")
    ir_others = {f: (frames[f]["Information ratio"].dropna(), palette[i % len(palette)]) for i, f in enumerate(peers)}

    which = st.segmented_control("Chart", ["⚖️ Sharpe ratio every week", "🎯 Information ratio every week"],
                                 default="⚖️ Sharpe ratio every week", key="mf_w_view",
                                 label_visibility="collapsed") or "⚖️ Sharpe ratio every week"
    if which.startswith("⚖️"):
        with panel():
            _weekly_chart(main, frames, "Sharpe", f"{main}: Sharpe ratio every week (risk-free {rf:.2f}%)", color,
                          "mf_w_sharpe", ("Rolling 1Y Return (%)", "Annualised σ (%)"), sharpe_others)
        st.caption("Sharpe = (1-year return − risk-free rate) ÷ yearly volatility (52 weekly log returns × √52). "
                   "Green = above 0, red = below 0. Click a name in the legend to add its line (other schemes of "
                   "the category and Nifty 50). Drag the slider under the chart to zoom.")
    else:
        with panel():
            _weekly_chart(main, frames, "Information ratio", f"{main}: Information ratio every week (vs Nifty 50)",
                          color, "mf_w_ir", ("Excess 1Y Return (%)", "Tracking error (%)"), ir_others)
        st.caption("Information ratio = (scheme 1-year return − Nifty 50 1-year return) ÷ tracking error "
                   "(52 weekly differences × √52). Above 0 = beat Nifty 50 over that year; the higher, the more "
                   "consistently. Click a name in the legend to compare other schemes.")

    m = frames[main].sort_index(ascending=False)
    table = pd.DataFrame({
        "Date": m.index.date,
        "1Y return (%)": m["Rolling 1Y Return (%)"], "Volatility (%)": m["Annualised σ (%)"], "Sharpe": m["Sharpe"],
        "Excess vs Nifty 50 (%)": m["Excess 1Y Return (%)"], "Tracking error (%)": m["Tracking error (%)"],
        "Information ratio": m["Information ratio"]})
    with st.expander(f"📋 Weekly table · {main}", expanded=False):
        st.dataframe(table.style.format({c: "{:.2f}" for c in table.columns if c != "Date"}, na_rep="—"),
                     hide_index=True, width="stretch", height=380,
                     column_config={"Date": st.column_config.DateColumn(format="DD-MM-YYYY")})
    st.download_button("⬇️  Download weekly Sharpe & IR (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"weekly_sharpe_ir_{re.sub(r'[^A-Za-z0-9]+', '_', main)}.csv", key="mf_w_dl")


# ---------------------------------------------------------
# Recovery: after a fall, how long until the NAV is back at its old high? (maths in src/mf_recovery.py)
# ---------------------------------------------------------
_FALL_C, _RISE_C = "#DC2626", "#16A34A"


def _when_back(r) -> str:
    return as_years_months(r["Recovery days"]) + ("" if r["Recovered"] else " (not back yet)")


def _recovery_bars(t: pd.DataFrame, key: str):
    """One bar per fund: time from the bottom of its biggest fall back to the old high. Fastest on top."""
    t = t.sort_values("Recovery days", ascending=False)
    fig = go.Figure(go.Bar(
        x=t["Recovery days"], y=t["Fund"], orientation="h",
        marker_color=[CAT_COLORS.get(c, VIOLET) for c in t["Category"]],
        marker_pattern_shape=["" if ok else "/" for ok in t["Recovered"]],
        text=[f"{_when_back(r)} · fell {r['Fall %']:.0f}%" for _, r in t.iterrows()],
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:,} days to recover<extra></extra>"))
    fig.update_layout(**PLOT_LAYOUT, height=max(300, 32 * len(t) + 80), showlegend=False,
                      xaxis=dict(title="Days from the bottom back to the old high"),
                      margin=dict(l=10, r=200, t=10, b=10))
    chart(fig, key=key)


def _fund_chart(nav: pd.Series, f: pd.DataFrame, key: str):
    """NAV line. Red zone = falling, green zone = climbing back to the old high."""
    nav = clean_nav(nav)
    fig = go.Figure(go.Scatter(x=nav.index, y=nav.values, mode="lines", line=dict(color=INDIGO, width=1.8),
                               hovertemplate="%{x|%d %b %Y}<br>NAV ₹%{y:,.2f}<extra></extra>"))
    for _, r in f.iterrows():
        end = r["Back date"] if r["Recovered"] else nav.index[-1]
        fig.add_vrect(x0=r["Top date"], x1=r["Bottom date"], fillcolor=hex_rgba(_FALL_C, 0.15), line_width=0)
        fig.add_vrect(x0=r["Bottom date"], x1=end, fillcolor=hex_rgba(_RISE_C, 0.12), line_width=0)
    fig.update_layout(**PLOT_LAYOUT, height=420, showlegend=False, yaxis=dict(title="NAV (₹)", tickprefix="₹"),
                      margin=dict(l=12, r=12, t=10, b=12))
    chart(fig, key=key)


def render_recovery(full: pd.DataFrame, plan: str):
    section("Recovery time", "When a fund falls, how long does it take to get back to its old high?", "⛰️", TEAL)
    note("<b>Top</b> = highest NAV before the fall · <b>Bottom</b> = lowest NAV in the fall · "
         "<b>Recovery time</b> = days from the Bottom until the NAV is back at the Top. "
         f"Uses the <b>{plan}</b> plan and the dates picked at the top of the page.", TEAL, "💡")
    nav_col = "NAV_Regular" if plan == "Regular" else "NAV_Direct"

    # ---- 1. every fund: its biggest fall ----
    t = biggest_falls(full, nav_col)
    if t.empty:
        note("Not enough NAV data in this date range.", AMBER, "⏳")
        return
    cat = st.segmented_control("Category", ["All"] + CATEGORIES, default="All", key="mf_rec_cat") or "All"
    sub = t if cat == "All" else t[t["Category"] == cat]
    st.markdown("##### Biggest fall of each fund")
    with panel():
        _recovery_bars(sub, key=f"mf_rec_bars_{cat}")
        st.caption("Shortest bar = recovered fastest. Striped = not back at its old high yet."
                   + (" 🔵 Large · 🟠 Mid · 🟢 Small Cap." if cat == "All" else ""))
    view = pd.DataFrame({
        "Category": sub["Category"], "Fund": sub["Fund"], "Fall": sub["Fall %"],
        "Bottom on": sub["Bottom date"].dt.date,
        "Back at top on": [d.date() if ok else None for d, ok in zip(sub["Back date"], sub["Recovered"])],
        "Recovery time": [_when_back(r) for _, r in sub.iterrows()],
    }).sort_values("Fall")
    with panel():
        st.dataframe(view.style.format({"Fall": "{:.1f}%"}, na_rep="not yet"), hide_index=True, width="stretch",
                     column_config={c: st.column_config.DateColumn(format="DD-MM-YYYY")
                                    for c in ("Bottom on", "Back at top on")})
        st.download_button("⬇️  Download (CSV)", view.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"mf_recovery_{plan.lower()}.csv", key="mf_rec_dl")

    # ---- 2. one fund: every fall of 10% or more ----
    st.markdown("##### One fund: every fall of 10% or more")
    funds = list(sub.sort_values(["Category", "Fund"])["Fund"])
    if st.session_state.get("mf_rec_fund") not in funds:
        st.session_state["mf_rec_fund"] = funds[0]
    fund = st.selectbox("Fund", funds, key="mf_rec_fund")
    nav = full[full["Fund"] == fund].set_index("Date")[nav_col]
    f = falls(nav, 10)
    with panel():
        _fund_chart(nav, f, key="mf_rec_fund_chart")
        st.caption("🟥 Red = falling from the top to the bottom. 🟩 Green = climbing back to the old high.")
    if f.empty:
        note(f"{fund} had no fall of 10% or more in these dates.", AMBER, "💡")
        return
    ev = f.iloc[::-1]
    with panel():
        st.dataframe(pd.DataFrame({
            "Top on": ev["Top date"].dt.date, "Bottom on": ev["Bottom date"].dt.date, "Fall": ev["Fall %"],
            "Back at top on": [d.date() if ok else None for d, ok in zip(ev["Back date"], ev["Recovered"])],
            "Recovery time": [_when_back(r) for _, r in ev.iterrows()],
        }).style.format({"Fall": "{:.1f}%"}, na_rep="not yet"), hide_index=True, width="stretch",
            column_config={c: st.column_config.DateColumn(format="DD-MM-YYYY")
                           for c in ("Top on", "Bottom on", "Back at top on")})


# ---------------------------------------------------------
# Page: ratios + data, no sidebar
# ---------------------------------------------------------
def _table(full: pd.DataFrame, col: str) -> pd.DataFrame:
    """Date down the side (oldest first), one column per fund (Large → Mid → Small Cap)."""
    wide = full.pivot_table(index="Date", columns="Fund", values=col, aggfunc="last")
    rank = {"Large Cap": 0, "Mid Cap": 1, "Small Cap": 2}
    group_of = full.groupby("Fund")["Group"].first()
    cols = sorted(wide.columns, key=lambda f: (rank.get(group_of.get(f), 9), f))
    wide = wide[cols].dropna(how="all").sort_index()
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

    # ---------------- date filter ----------------
    # Calendar allows: oldest fund start → 15-10-2026.
    # Default From = 01-01-2018 (DEFAULT_START, same as every other page).
    first_day = full["Date"].min().date()
    last_day = min(LAST_DATE, full["Date"].max().date())
    starts = pd.concat([full.dropna(subset=[c]).groupby("Fund")["Date"].min() for c in ("NAV_Regular", "NAV_Direct")])
    all_start = min(starts.max().date(), last_day)
    youngest = starts.idxmax()
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
                   f"**{first_day:%d %b %Y}** (oldest fund)."
                   + (f" Note: {youngest} only has data from **{all_start:%d %b %Y}**."
                      if all_start > DEFAULT_START else ""))
    if d_from > d_to:
        d_from, d_to = d_to, d_from
    full = full[(full["Date"] >= pd.Timestamp(d_from)) & (full["Date"] <= pd.Timestamp(d_to))]

    render_ratios(full, asset_dfs, d_from, d_to)
    plan, rf = st.session_state.get("_mf_plan_rf", ("Regular", 0.0))
    render_weekly(full, asset_dfs, d_from, d_to, plan, rf)
    render_recovery(full, plan)

    section("NAV data", "Daily NAV of every scheme for the dates above", "📋", INDIGO)
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