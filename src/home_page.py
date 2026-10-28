"""🏠 Main Dashboard: the landing page (blank for now). 

Only the sidebar (navigation + refresh) and the page title are shown.
Add your own content below the title later.
"""
from src.ui import hero, sidebar_nav, sidebar_footer


def render_home_page(asset_dfs, fetch_errors):
    # ---------------- Sidebar ----------------
    sidebar_nav("home")
    sidebar_footer(fetch_errors, key="home")

    # ---------------- Title ----------------
    hero("Main Dashboard", page="home", eyebrow="Home")

    # ---------------- Your content goes here ----------------