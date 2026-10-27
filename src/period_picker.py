"""'Select Time Period' logic (ported from src/period_picker.py, without the sidebar widgets).

All periods end on LAST_DATE (15-10-2026) at the latest.
"""
from datetime import date, datetime
from enum import Enum
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")

# Custom Range calendar starts here (the Streamlit date pickers did not allow earlier dates).
CUSTOM_RANGE_START = date(2010, 1, 1)

# Every time period stops on this date (no data after it is used).
LAST_DATE = date(2026, 10, 15)


class PeriodPreset(str, Enum):
    one_month = "1M"
    three_months = "3M"
    six_months = "6M"
    one_year = "1Y"
    three_years = "3Y"
    five_years = "5Y"
    all_data = "ALL"
    custom = "CUSTOM"


PRESET_MONTHS = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "3Y": 36, "5Y": 60, "ALL": None}
PRESET_LABELS = {"1M": "1 Month", "3M": "3 Months", "6M": "6 Months", "1Y": "1 Year",
                 "3Y": "3 Years", "5Y": "5 Years", "ALL": "All Data", "CUSTOM": "Custom Range"}


def real_today_ist() -> date:
    """Today's actual date in India."""
    return datetime.now(IST).date()


def today_ist() -> date:
    """The last date any period can reach: today (IST), but never later than LAST_DATE.
    Used everywhere as 'today', so presets count back from 15-10-2026."""
    return min(real_today_ist(), LAST_DATE)


def resolve_period(asset_dfs, preset: PeriodPreset | str = "1Y", start: date | None = None,
                   end: date | None = None):
    """Turn a preset (or a custom start/end) into (start_date, end_date, notes).

    * Presets count back from the end date (15-10-2026); "ALL" starts at the earliest date of any asset.
    * Giving `start` and/or `end` switches to a custom range (CUSTOM_RANGE_START .. 15-10-2026).
    * The end date never goes past 15-10-2026.
    """
    preset = PeriodPreset(preset).value
    last = today_ist()
    overall_min = min(df['Date'].min() for df in asset_dfs.values()).date()
    data_max = max(df['Date'].max() for df in asset_dfs.values()).date()
    overall_max = min(data_max, last)
    notes = []

    if preset == "CUSTOM" or start is not None or end is not None:
        from_date = start or CUSTOM_RANGE_START
        to_date = end or last
        if from_date > to_date:
            notes.append("'start' was after 'end', so they were swapped.")
            from_date, to_date = to_date, from_date
        if from_date < CUSTOM_RANGE_START:
            notes.append(f"Custom ranges start no earlier than {CUSTOM_RANGE_START:%d %b %Y}.")
            from_date = CUSTOM_RANGE_START
        if to_date > last:
            notes.append(f"End date moved back to {last:%d %b %Y} (the last allowed date).")
            to_date = last
        from_date = min(from_date, to_date)
        start_date, end_date = from_date, to_date
    else:
        months = PRESET_MONTHS[preset]
        end_date = last
        if months is None:
            start_date = overall_min
        else:
            start_date = max(overall_min, (pd.Timestamp(last) - pd.DateOffset(months=months)).date())

    if overall_max < end_date:
        notes.append(f"Latest data available: {overall_max:%d %b %Y}")
    return start_date, end_date, notes