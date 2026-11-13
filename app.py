"""Multi-Asset Lab: entry point.

Run with:  streamlit run app.py

Loads the 4 assets once (Google Sheets, falling back to the local CSV files) and shows
whichever page is selected in the sidebar.
"""
import streamlit as st

st.set_page_config(
    page_title="Multi-Asset Lab (INR ₹)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui import apply_custom_css, hero, note, BAD, AMBER  # noqa: E402  (after set_page_config)
import time  # noqa: E402
from src.asset_data import load_all_assets, backup_assets, short_reason, data_updated_at, ASSET_SHORT  # noqa: E402
from src.home_page import render_home_page  # noqa: E402
from src.investment_page import render_investment_page  # noqa: E402
from src.monte_carlo_page import render_monte_carlo_page  # noqa: E402
from src.everyday_portfolio_page import render_everyday_portfolio_page  # noqa: E402

apply_custom_css()
st.session_state["_panel_n"] = 0      # panel() numbering starts fresh every run

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
asset_dfs, fetch_errors = load_all_assets(force_refresh=st.session_state.pop("_force_sheet_refresh", False))

# "Live data · updated 12 min ago" under the sidebar settings
_t = data_updated_at()
if _t and not backup_assets(fetch_errors):
    _m = int((time.time() - _t) // 60)
    st.session_state["_data_updated_text"] = ("🟢 Live Google Sheets · updated "
                                              + ("just now" if _m < 1 else f"{_m} min ago" if _m < 120
                                                 else f"{_m // 60} h ago"))
else:
    st.session_state["_data_updated_text"] = None

if not asset_dfs:
    hero("Multi-Asset Lab", "No data could be loaded.", page="home", eyebrow="Data error")
    note("Unable to load a dataset for any asset. Check that the Google Sheets are shared as "
         "<b>Anyone with the link can view</b>, or place the CSV files next to <b>app.py</b>.", BAD, "⛔")
    for name, err in fetch_errors.items():
        st.write(f"**{name}:** {err}")
    st.stop()

# Never use old backup data silently: say which assets are not coming from the live Google Sheets
_backup = backup_assets(fetch_errors)
if _backup:
    _names = ", ".join(ASSET_SHORT.get(n, n) for n in _backup)
    _why = sorted({short_reason(fetch_errors[n]) for n in _backup})
    note(f"<b>Not live data for {_names}.</b> The Google Sheet could not be loaded ({'; '.join(_why)}), so the "
         f"old backup CSV files are shown and their prices can be wrong. Retried every 2 minutes, or click "
         f"<b>🔄 Refresh live sheet data</b> in the sidebar. Details: sidebar → Data load issues.", AMBER, "⚠️")

# ---------------------------------------------------------
# Router
# ---------------------------------------------------------
PAGES = {
    "home": render_home_page,              # 🏠 Main Dashboard (landing page)
    "invest": render_investment_page,
    "montecarlo": render_monte_carlo_page,
    "portfolio": render_everyday_portfolio_page,
}

view = st.session_state.get("view", "home")
if view not in PAGES:
    view = "home"
st.session_state["view"] = view

PAGES[view](asset_dfs, fetch_errors)