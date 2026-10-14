"""Sidebar 'Select Time Period' control — month-wise (no day picking)."""
import pandas as pd
import streamlit as st

MONTH_PRESETS = {
    "1 Month": 1, "3 Months": 3, "6 Months": 6, "1 Year": 12,
    "3 Years": 36, "5 Years": 60, "All Data": None, "Custom Range": "custom",
}

# Custom Range starts here: the first date of the SmallCap data (all 4 assets have prices from then on).
CUSTOM_RANGE_START = pd.Timestamp("2015-11-23")


def period_selector(asset_dfs, default_preset="1 Year", key="main"):
    """Draws the period controls in the sidebar and returns (start_date, end_date).
    Custom Range = pick a From month and a To month."""
    overall_min = min(df['Date'].min() for df in asset_dfs.values()).date()
    overall_max = max(df['Date'].max() for df in asset_dfs.values()).date()

    st.sidebar.markdown("### 📅 Select Time Period")
    options = list(MONTH_PRESETS.keys())
    preset = st.sidebar.selectbox("Period", options, index=options.index(default_preset),
                                  key=f"{key}_period")
    preset_val = MONTH_PRESETS[preset]

    if preset_val == "custom":
        # Month list starts at Nov 2015 (SmallCap data start) instead of the oldest date of any asset
        range_min = max(overall_min, CUSTOM_RANGE_START.date())
        months = list(pd.period_range(pd.Timestamp(range_min).to_period('M'),
                                      pd.Timestamp(overall_max).to_period('M'), freq='M'))
        labels = [m.strftime('%b %Y') for m in months]           # e.g. "Nov 2015"
        from_lbl = st.sidebar.selectbox("📍 From month", labels, index=0,
                                        key=f"{key}_from_month")
        to_lbl = st.sidebar.selectbox("🏁 To month", labels, index=len(labels) - 1,
                                      key=f"{key}_to_month")
        from_m, to_m = months[labels.index(from_lbl)], months[labels.index(to_lbl)]
        if from_m > to_m:
            st.sidebar.error("'From month' must be before 'To month'. Showing them swapped.")
            from_m, to_m = to_m, from_m
        # From month → last day of To month; Nov 2015 starts on 23 Nov 2015 (first SmallCap date)
        start_date = max(range_min, from_m.start_time.date())
        end_date = min(overall_max, to_m.end_time.date())
        st.sidebar.caption(f"Showing **{start_date:%d %b %Y}** to **{end_date:%d %b %Y}**")
        return start_date, end_date

    if preset_val is None:
        start_date, end_date = overall_min, overall_max
    else:
        end_date = overall_max
        start_date = max(overall_min, (pd.Timestamp(overall_max) - pd.DateOffset(months=preset_val)).date())

    st.sidebar.caption(f"Showing **{start_date:%b %Y}** to **{end_date:%b %Y}**")
    return start_date, end_date