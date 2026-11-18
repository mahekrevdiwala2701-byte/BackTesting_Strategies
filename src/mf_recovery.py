"""Recovery time of a mutual fund: how long the NAV takes to climb from its BOTTOM back to its old TOP
(no Streamlit code here, so every number can be checked by hand or in Excel).

Step by step, for every day (NAV = Regular or Direct Growth NAV):

    1. Running top (peak)   = highest NAV from the first day up to today
                              Excel: C2 = MAX($B$2:B2)
    2. Fall from top (%)    = (NAV today ÷ running top − 1) × 100        (0% = at a new high, always ≤ 0)
                              Excel: D2 = (B2 / C2 − 1) * 100

A "fall" (drawdown episode) starts on the day after the NAV was last at its top, and ends on the first
day the NAV is back at or above that same top:

    Top date        = last day at the old high, before the fall started
    Bottom date     = day with the lowest NAV inside the fall
    Recovery date   = first day the NAV is ≥ the old top again (empty if it has not come back yet)

    Fall (%)                    = (Bottom NAV ÷ Top NAV − 1) × 100
    Days top → bottom           = Bottom date − Top date                (how long it kept falling)
    Days bottom → top (RECOVERY)= Recovery date − Bottom date           (the number asked for)
    Days underwater             = Recovery date − Top date              (whole fall, top → bottom → top)
    Gain needed from bottom (%) = (Top NAV ÷ Bottom NAV − 1) × 100      (a 50% fall needs a 100% gain)

All "days" are calendar days. Falls that have not recovered are counted up to the last NAV date and
marked "Not yet". Small dips (e.g. 1–2%) recover in days and would drown the picture, so tables
usually keep only falls of at least `min_fall` %.
"""
import numpy as np
import pandas as pd

from src.everyday_portfolio_page import drop_glitches


def clean_nav(nav: pd.Series) -> pd.Series:
    """NAV by date, sorted, one value per day, obvious data-entry spikes removed (same rule as the rest
    of the app: a jump of > 50% that later comes back)."""
    s = nav.dropna()
    s = s[s > 0]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return drop_glitches(s)[0]


def daily_steps(nav: pd.Series) -> pd.DataFrame:
    """The day-by-day working: Date, NAV, Running top, Fall from top (%)."""
    s = clean_nav(nav)
    top = s.cummax()
    return pd.DataFrame({"Date": s.index, "NAV": s.to_numpy(), "Running top": top.to_numpy(),
                         "Fall from top (%)": ((s / top - 1) * 100).to_numpy()})


def falls(nav: pd.Series, min_fall: float = 0.0) -> pd.DataFrame:
    """One row per fall (top → bottom → back to top), oldest first. Keeps falls of at least `min_fall` %.

    Columns: Top date, Top NAV, Bottom date, Bottom NAV, Recovery date, Fall (%), Gain needed (%),
             Days top → bottom, Days bottom → top, Days underwater, Recovered."""
    s = clean_nav(nav)
    cols = ["Top date", "Top NAV", "Bottom date", "Bottom NAV", "Recovery date", "Fall (%)", "Gain needed (%)",
            "Days top → bottom", "Days bottom → top", "Days underwater", "Recovered"]
    if len(s) < 2:
        return pd.DataFrame(columns=cols)
    top = s.cummax()
    below = (s < top).to_numpy()
    dates, vals = s.index, s.to_numpy(dtype=float)
    last = dates[-1]
    rows, i, n = [], 0, len(s)
    while i < n:
        if not below[i]:
            i += 1
            continue
        start = i                                   # first day below the old top
        while i < n and below[i]:
            i += 1
        end = i - 1                                 # last day below the old top
        top_i = start - 1                           # last day AT the old top
        bot_i = start + int(np.argmin(vals[start:end + 1]))
        recovered = i < n
        rec_date = dates[i] if recovered else pd.NaT
        top_d, bot_d = dates[top_i], dates[bot_i]
        fall = (vals[bot_i] / vals[top_i] - 1) * 100
        rows.append({
            "Top date": top_d, "Top NAV": vals[top_i],
            "Bottom date": bot_d, "Bottom NAV": vals[bot_i],
            "Recovery date": rec_date,
            "Fall (%)": fall,
            "Gain needed (%)": (vals[top_i] / vals[bot_i] - 1) * 100,
            "Days top → bottom": int((bot_d - top_d).days),
            "Days bottom → top": int(((rec_date if recovered else last) - bot_d).days),
            "Days underwater": int(((rec_date if recovered else last) - top_d).days),
            "Recovered": bool(recovered),
        })
    out = pd.DataFrame(rows, columns=cols)
    return out[out["Fall (%)"] <= -abs(min_fall)].reset_index(drop=True)


def fund_summary(full: pd.DataFrame, nav_col: str, min_fall: float = 10.0) -> pd.DataFrame:
    """One row per fund (full = long table with Date, Fund, Group and the NAV column).

    Biggest fall   : the deepest fall in the dates given, and how long it took from its bottom back to the top
    Longest recovery (falls ≥ min_fall): the slowest bottom → top among the falls at least min_fall % deep
    Average recovery: average bottom → top of the RECOVERED falls ≥ min_fall
    Now            : how far the latest NAV is below its top."""
    rows = []
    for (grp, fund), g in full.dropna(subset=[nav_col]).groupby(["Group", "Fund"]):
        nav = g.set_index("Date")[nav_col]
        f = falls(nav, 0.0)
        if f.empty:
            continue
        big = f.loc[f["Fall (%)"].idxmin()]
        sig = f[f["Fall (%)"] <= -abs(min_fall)]
        rec = sig[sig["Recovered"]]
        slow = sig.loc[sig["Days bottom → top"].idxmax()] if len(sig) else None
        steps = daily_steps(nav)
        rows.append({
            "Category": grp, "Fund": fund,
            "From": steps["Date"].iloc[0], "To": steps["Date"].iloc[-1],
            "Biggest fall (%)": big["Fall (%)"],
            "Top date": big["Top date"], "Bottom date": big["Bottom date"], "Recovery date": big["Recovery date"],
            "Days bottom → top": big["Days bottom → top"],
            "Recovered": big["Recovered"],
            "Days top → bottom": big["Days top → bottom"],
            "Days underwater": big["Days underwater"],
            f"Falls ≥ {min_fall:g}%": int(len(sig)),
            "Longest bottom → top (days)": slow["Days bottom → top"] if slow is not None else np.nan,
            "Longest: bottom date": slow["Bottom date"] if slow is not None else pd.NaT,
            "Longest recovered": bool(slow["Recovered"]) if slow is not None else None,
            "Average bottom → top (days)": float(rec["Days bottom → top"].mean()) if len(rec) else np.nan,
            "Now below top (%)": float(steps["Fall from top (%)"].iloc[-1]),
        })
    return pd.DataFrame(rows)


def as_years_months(days) -> str:
    """1,234 days → '3 y 4 m'. Blank for missing."""
    if days is None or pd.isna(days):
        return "—"
    months = int(round(float(days) / 30.4375))
    y, m = divmod(months, 12)
    if y and m:
        return f"{y} y {m} m"
    if y:
        return f"{y} y"
    return f"{m} m" if m else f"{int(days)} d"


# ---------------------------------------------------------
# Well-known market crashes: compare every fund on the SAME fall
# ---------------------------------------------------------
CRASHES = {                      # name -> (bottom must fall between these two dates)
    "2008 global crash": ("2008-01-01", "2009-12-31"),
    "2011 slowdown": ("2011-01-01", "2012-06-30"),
    "2015–16 correction": ("2015-03-01", "2016-06-30"),
    "2018–19 mid & small cap fall": ("2018-01-01", "2019-12-31"),
    "COVID crash 2020": ("2020-02-01", "2020-06-30"),
    "2024–25 correction": ("2024-09-01", "2025-06-30"),
}


def fall_in_window(f: pd.DataFrame, start, end):
    """The deepest fall whose BOTTOM date lies between start and end (None if there is none)."""
    if f is None or f.empty:
        return None
    w = f[(f["Bottom date"] >= pd.Timestamp(start)) & (f["Bottom date"] <= pd.Timestamp(end))]
    return None if w.empty else w.loc[w["Fall (%)"].idxmin()]