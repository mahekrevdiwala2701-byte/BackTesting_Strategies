"""Sidebar 'Time period' control: quick presets or a calendar date range (up to today, IST)."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.ui import sidebar_label

MONTH_PRESETS = {
    "1 Month": 1, "3 Months": 3, "6 Months": 6, "1 Year": 12,
    "3 Years": 36, "5 Years": 60, "All Data": None, "Custom Range": "custom",
}

# Earliest date offered by the Custom Range calendar.
CUSTOM_RANGE_START = pd.Timestamp("2010-1-1").date()
IST = ZoneInfo("Asia/Kolkata")


def today_ist():
    """Today's date in India, so the calendar rolls over at IST midnight even on a UTC server."""
    return datetime.now(IST).date()


def period_selector(asset_dfs, default_preset="1 Year", key="main"):
    """Draws the period controls in the sidebar and returns (start_date, end_date).
    The end date never goes past today, even if the sheet has future-dated rows.
    Custom Range = pick a From date and a To date on a calendar."""
    today = today_ist()
    overall_min = min(df['Date'].min() for df in asset_dfs.values()).date()
    data_max = max(df['Date'].max() for df in asset_dfs.values()).date()
    overall_max = min(data_max, today)                   # ignore any future dates in the sheet

    sidebar_label("📅 Time period")
    options = list(MONTH_PRESETS.keys())
    preset = st.sidebar.selectbox("Period", options, index=options.index(default_preset),
                                  key=f"{key}_period")
    preset_val = MONTH_PRESETS[preset]

    if preset_val == "custom":
        range_min = CUSTOM_RANGE_START
        range_max = today                                # calendar ends today, moves forward daily

        from_date = st.sidebar.date_input(
            "📍 From date", value=range_min, min_value=range_min, max_value=range_max,
            format="DD-MM-YYYY", key=f"{key}_from_date")
        to_date = st.sidebar.date_input(
            "🏁 To date", value=range_max, min_value=range_min, max_value=range_max,
            format="DD-MM-YYYY", key=f"{key}_to_date")

        if from_date > to_date:
            st.sidebar.error("'From date' must be before 'To date'. Showing them swapped.")
            from_date, to_date = to_date, from_date

        start_date, end_date = from_date, to_date
        st.sidebar.caption(f"Showing **{start_date:%d %b %Y}** to **{end_date:%d %b %Y}**")
        if end_date > overall_max:
            st.sidebar.caption(f"Latest data available: {overall_max:%d %b %Y}")
        return start_date, end_date

    end_date = today
    if preset_val is None:
        start_date = overall_min
    else:
        start_date = max(overall_min, (pd.Timestamp(today) - pd.DateOffset(months=preset_val)).date())

    st.sidebar.caption(f"Showing **{start_date:%d %b %Y}** to **{end_date:%d %b %Y}**")
    if overall_max < today:
        st.sidebar.caption(f"Latest data available: {overall_max:%d %b %Y}")
    return start_date, end_date