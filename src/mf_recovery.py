"""Recovery time of a mutual fund: after a fall, how long until the NAV is back at its old high?

    Top       = the highest NAV before the fall
    Bottom    = the lowest NAV during the fall
    Back      = the first day the NAV is at or above the old Top again
    Fall %    = Bottom ÷ Top − 1
    Recovery  = days from Bottom to Back   (if not back yet: days from Bottom to the last NAV)
"""
import pandas as pd

from src.everyday_portfolio_page import drop_glitches


def clean_nav(nav: pd.Series) -> pd.Series:
    """NAV by date, sorted, one value per day, data-entry spikes removed."""
    s = nav.dropna()
    s = s[s > 0]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return drop_glitches(s)[0]


def falls(nav: pd.Series, min_fall: float = 10.0) -> pd.DataFrame:
    """One row per fall of at least `min_fall` %, oldest first.
    Columns: Top date, Top NAV, Bottom date, Bottom NAV, Back date, Fall %, Recovery days, Recovered."""
    s = clean_nav(nav)
    cols = ["Top date", "Top NAV", "Bottom date", "Bottom NAV", "Back date", "Fall %", "Recovery days", "Recovered"]
    if len(s) < 2:
        return pd.DataFrame(columns=cols)

    below = (s < s.cummax()).to_numpy()
    dates, vals = s.index, s.to_numpy(dtype=float)
    rows, i, n = [], 0, len(s)
    while i < n:
        if not below[i]:
            i += 1
            continue
        start = i
        while i < n and below[i]:
            i += 1
        top_i = start - 1
        bot_i = start + int(vals[start:i].argmin())
        recovered = i < n
        back = dates[i] if recovered else pd.NaT
        fall = (vals[bot_i] / vals[top_i] - 1) * 100
        if fall <= -abs(min_fall):
            rows.append({"Top date": dates[top_i], "Top NAV": vals[top_i],
                         "Bottom date": dates[bot_i], "Bottom NAV": vals[bot_i],
                         "Back date": back, "Fall %": fall,
                         "Recovery days": int(((back if recovered else dates[-1]) - dates[bot_i]).days),
                         "Recovered": recovered})
    return pd.DataFrame(rows, columns=cols)


def biggest_falls(full: pd.DataFrame, nav_col: str) -> pd.DataFrame:
    """One row per fund: its deepest fall in the data given, and how long it took to recover."""
    rows = []
    for (grp, fund), g in full.dropna(subset=[nav_col]).groupby(["Group", "Fund"]):
        f = falls(g.set_index("Date")[nav_col], 0.0)
        if len(f):
            rows.append({"Category": grp, "Fund": fund, **f.loc[f["Fall %"].idxmin()].to_dict()})
    return pd.DataFrame(rows)


def as_years_months(days) -> str:
    """1,234 days → '3 y 4 m'."""
    if days is None or pd.isna(days):
        return "—"
    y, m = divmod(int(round(float(days) / 30.4375)), 12)
    if y and m:
        return f"{y} y {m} m"
    if y:
        return f"{y} y"
    return f"{m} m" if m else f"{int(days)} d"