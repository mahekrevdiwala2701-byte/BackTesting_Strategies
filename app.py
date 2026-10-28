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

from src.ui import apply_custom_css, hero, note, BAD  # noqa: E402  (after set_page_config)
from src.asset_data import load_all_assets  # noqa: E402
from src.home_page import render_home_page  # noqa: E402
from src.dashboard_page import render_dashboard_page  # noqa: E402
from src.investment_page import render_investment_page  # noqa: E402
from src.monte_carlo_page import render_monte_carlo_page  # noqa: E402
from src.everyday_portfolio_page import render_everyday_portfolio_page  # noqa: E402

apply_custom_css()
st.session_state["_panel_n"] = 0      # panel() numbering starts fresh every run

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
asset_dfs, fetch_errors = load_all_assets()

if not asset_dfs:
    hero("Multi-Asset Lab", "No data could be loaded.", page="dashboard", eyebrow="Data error")
    note("Unable to load a dataset for any asset. Check that the Google Sheets are shared as "
         "<b>Anyone with the link can view</b>, or place the CSV files next to <b>app.py</b>.", BAD, "⛔")
    for name, err in fetch_errors.items():
        st.write(f"**{name}:** {err}")
    st.stop()

# ---------------------------------------------------------
# Router
# ---------------------------------------------------------
PAGES = {
    "home": render_home_page,              # 🏠 Main Dashboard (landing page)
    "dashboard": render_dashboard_page,    # 📊 Market Overview (the old main dashboard)
    "invest": render_investment_page,
    "montecarlo": render_monte_carlo_page,
    "portfolio": render_everyday_portfolio_page,
}

view = st.session_state.get("view", "home")
if view not in PAGES:
    view = "home"
st.session_state["view"] = view

PAGES[view](asset_dfs, fetch_errors)