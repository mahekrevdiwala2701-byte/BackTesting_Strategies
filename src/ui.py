"""Design system for the whole app.

Everything visual lives here so every page looks the same:
  * colour tokens (dark canvas, one vivid colour per asset / strategy)
  * the global CSS, including all animations and transitions
  * HTML building blocks: hero, section headers, KPI cards (with sparklines), chips, ticker tape,
    step flow, story card, 100-dot outcome grid, glossary, panels
  * one Plotly chart theme + a `chart()` helper that applies it
  * Indian number formatting (₹1,00,000 / ₹3.45 L / ₹1.20 Cr)
"""
import html

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# =========================================================
# Colour tokens
# =========================================================
BG = "#0A0F1E"          # page background
SURFACE = "#121A2E"     # cards / panels
SURFACE_2 = "#182240"   # raised elements, hover, tooltips
BORDER = "#232D4A"      # hairlines
TEXT = "#E6EAF5"        # main text
INK = "#FFFFFF"         # strong lines on charts (median etc.)
MUTED = "#8A94B3"       # secondary text
FAINT = "#5B6584"       # reference lines

NIGHT = "#0D1328"       # ticker tape
VIOLET = "#8B5CF6"
INDIGO = "#6366F1"
BLUE = "#3B82F6"
CYAN = "#22D3EE"
TEAL = "#14B8A6"
EMERALD = "#10B981"
AMBER = "#F59E0B"
ORANGE = "#FB923C"
PINK = "#EC4899"
ROSE = "#F43F5E"

GOOD = "#22C55E"        # gains
BAD = "#F43F5E"         # losses
GOOD_TEXT = "#4ADE80"
BAD_TEXT = "#FB7185"

# Hero gradients: every page has its own colour identity
GRADIENTS = {
    "dashboard": ("#6D28D9", "#2563EB", "#0891B2"),
    "invest": ("#047857", "#0D9488", "#2563EB"),
    "montecarlo": ("#C2410C", "#DB2777", "#7C3AED"),
    "portfolio": ("#1D4ED8", "#0891B2", "#059669"),
}

# One colour per strategy (same on the Investment and Monte Carlo pages)
STRAT_COLORS = {"bh": CYAN, "periodic": "#A78BFA", "diff": ORANGE}

# Navigation: key -> (icon, label, accent colour)
PAGES = {
    "dashboard": ("📊", "Market Dashboard", VIOLET),
    "invest": ("💰", "Calculated Investment", EMERALD),
    "montecarlo": ("🎲", "Monte Carlo Simulation", ORANGE),
    "portfolio": ("💼", "Everyday Portfolio", CYAN),
}


# =========================================================
# Number formatting (Indian digit grouping)
# =========================================================
def _group_indian(n: int) -> str:
    s = str(abs(int(n)))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def inr(v, dp: int = 0, sign: bool = False) -> str:
    """₹ with Indian grouping: inr(123456.7) -> '₹1,23,457', inr(-50, sign=True) -> '−₹50'."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v != v:  # NaN
        return "—"
    neg = v < 0
    text = f"{abs(v):.{dp}f}"
    whole, _, frac = text.partition(".")
    s = f"₹{_group_indian(int(whole))}" + (f".{frac}" if frac else "")
    if neg and float(text) != 0:
        return "−" + s
    return ("+" + s) if sign else s


def inr_short(v) -> str:
    """Compact rupees: ₹3.45 L, ₹1.20 Cr, ₹85,000."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if abs(v) >= 1e7:
        return f"₹{v / 1e7:,.2f} Cr"
    if abs(v) >= 1e5:
        return f"₹{v / 1e5:,.2f} L"
    return inr(v)


def pct(v, dp: int = 2, sign: bool = True) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    return f"{v:+.{dp}f}%" if sign else f"{v:.{dp}f}%"


def hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# =========================================================
# Global CSS
# =========================================================
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {{
  --bg:{BG}; --surface:{SURFACE}; --surface2:{SURFACE_2}; --border:{BORDER};
  --text:{TEXT}; --muted:{MUTED}; --faint:{FAINT}; --night:{NIGHT};
  --good:{GOOD}; --bad:{BAD}; --good-t:{GOOD_TEXT}; --bad-t:{BAD_TEXT};
  --display: 'Bricolage Grotesque', 'Plus Jakarta Sans', system-ui, sans-serif;
  --ease: cubic-bezier(.2,.75,.25,1);
}}

html, body, [class*="css"], .stApp, button, input, textarea, select {{
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
}}

/* ================= animations ================= */
@keyframes fadeUp   {{ from {{ opacity:0; transform: translateY(16px); }} to {{ opacity:1; transform:none; }} }}
@keyframes fadeIn   {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
@keyframes slideIn  {{ from {{ opacity:0; transform: translateX(-14px); }} to {{ opacity:1; transform:none; }} }}
@keyframes popIn    {{ 0% {{ opacity:0; transform: scale(.3); }} 70% {{ transform: scale(1.12); }} 100% {{ opacity:1; transform: scale(1); }} }}
@keyframes gradientShift {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
@keyframes floaty   {{ 0%,100% {{ transform: translate(0,0) scale(1); }} 50% {{ transform: translate(-18px,14px) scale(1.06); }} }}
@keyframes draw     {{ from {{ stroke-dashoffset: 1400; }} to {{ stroke-dashoffset: 0; }} }}
@keyframes marquee  {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
@keyframes shine    {{ from {{ left: -60%; }} to {{ left: 130%; }} }}
@keyframes growBar  {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
@keyframes glow     {{ 0%,100% {{ opacity:.55; }} 50% {{ opacity:1; }} }}

/* ---------- canvas: soft colour glows on near-black ---------- */
.stApp {{
  background:
    radial-gradient(900px 500px at 8% -10%, rgba(139,92,246,0.18), transparent 60%),
    radial-gradient(800px 480px at 100% 0%, rgba(34,211,238,0.11), transparent 60%),
    radial-gradient(700px 500px at 50% 110%, rgba(236,72,153,0.08), transparent 60%),
    var(--bg);
  background-attachment: fixed;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container, [data-testid="stMainBlockContainer"] {{
  padding-top: 2rem; padding-bottom: 4rem; max-width: 1480px;
}}
h1, h2, h3, h4, h5 {{ font-family: var(--display); letter-spacing: -0.015em; }}
[data-testid="stCaptionContainer"] {{ color: var(--muted); }}

/* every chart, table and tab panel glides in */
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{ animation: fadeUp .7s var(--ease) both; }}
.stTabs [role="tabpanel"], .stTabs [data-baseweb="tab-panel"] {{ animation: fadeIn .45s ease both; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
  background:
    radial-gradient(380px 280px at 0% 0%, rgba(139,92,246,0.22), transparent 70%),
    radial-gradient(360px 280px at 100% 100%, rgba(34,211,238,0.10), transparent 70%),
    linear-gradient(180deg, #10173A 0%, #0B1122 100%);
  border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top: .8rem; }}
section[data-testid="stSidebar"] hr {{ margin: 14px 0; border-color: var(--border); }}
.brand {{
  display:flex; align-items:center; gap:12px; padding: 6px 4px 18px 4px;
  border-bottom: 1px solid var(--border); margin-bottom: 12px; animation: fadeIn .6s ease both;
}}
.brand-logo {{
  width:44px; height:44px; border-radius:14px; display:grid; place-items:center; font-size:22px;
  background: linear-gradient(135deg, #8B5CF6, #3B82F6 45%, #22D3EE); background-size: 200% 200%;
  animation: gradientShift 6s ease infinite; box-shadow: 0 10px 26px rgba(99,102,241,0.40);
}}
.brand-name {{ font-family: var(--display); font-weight:800; font-size:1.12rem; color:#fff; line-height:1.1; }}
.brand-sub {{ font-size:0.7rem; color:var(--muted); letter-spacing:0.12em; margin-top:3px; }}
.side-label {{
  font-size:0.68rem; font-weight:800; letter-spacing:0.16em; text-transform:uppercase;
  color:var(--faint); margin: 16px 2px 8px 2px;
}}
section[data-testid="stSidebar"] .stButton > button {{
  justify-content:flex-start; text-align:left; padding: 0.6rem 0.9rem; min-height: 2.7rem;
  border-radius: 14px; font-weight:600; font-size:0.93rem; position: relative; overflow: hidden;
  transition: transform .2s var(--ease), background .2s ease, box-shadow .2s ease;
}}
section[data-testid="stSidebar"] .stButton > button > div {{ justify-content:flex-start; width:100%; }}
section[data-testid="stSidebar"] .stButton > button p {{ text-align:left; font-weight:600; }}
section[data-testid="stSidebar"] .stButton {{ margin-bottom: -6px; }}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{
  background: transparent; border: 1px solid transparent; color: #C3CAE0;
}}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {{
  background: rgba(139,92,246,0.12); border-color: rgba(139,92,246,0.32); color:#fff; transform: translateX(4px);
}}

/* ---------- buttons ---------- */
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] button, [data-testid="stBaseButton-primaryFormSubmit"] {{
  background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 50%, #3B82F6 100%) !important; background-size: 200% 200% !important;
  border: 0 !important; color: #fff !important; font-weight: 700 !important; position: relative; overflow: hidden;
  box-shadow: 0 8px 22px rgba(99,102,241,0.32);
  transition: transform .2s var(--ease), box-shadow .2s ease, filter .2s ease;
}}
.stButton > button[kind="primary"] p, [data-testid="stFormSubmitButton"] button p {{ color:#fff !important; }}
.stButton > button[kind="primary"]::after, [data-testid="stFormSubmitButton"] button::after {{
  content:""; position:absolute; top:0; left:-60%; width:40%; height:100%;
  background: linear-gradient(110deg, transparent, rgba(255,255,255,0.35), transparent);
}}
.stButton > button[kind="primary"]:hover::after, [data-testid="stFormSubmitButton"] button:hover::after {{
  animation: shine .8s ease;
}}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:hover {{
  transform: translateY(-2px); box-shadow: 0 14px 30px rgba(99,102,241,0.45); filter: brightness(1.08);
}}
[data-testid="stDownloadButton"] button {{
  border: 1px solid rgba(34,211,238,0.35); background: rgba(34,211,238,0.08); color: #A5F3FC;
  font-weight: 600; border-radius: 12px; transition: all .2s var(--ease);
}}
[data-testid="stDownloadButton"] button:hover {{ background: rgba(34,211,238,0.18); color:#fff; transform: translateY(-2px); }}

/* ---------- tabs as pills ---------- */
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {{
  gap: 6px; background: var(--surface); padding: 6px; border-radius: 16px;
  border: 1px solid var(--border); overflow-x: auto; width: fit-content; max-width: 100%;
}}
.stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"] {{
  height: auto; padding: 8px 16px; border-radius: 11px; color: var(--muted);
  font-weight: 600; background: transparent; white-space: nowrap; border: 0; margin: 0;
  transition: background .25s ease, color .25s ease, transform .2s var(--ease);
}}
.stTabs [data-testid="stTab"] p {{ font-weight: 600; font-size: 0.92rem; }}
.stTabs [data-baseweb="tab"]:hover, .stTabs [data-testid="stTab"]:hover {{
  color: var(--text); background: rgba(255,255,255,0.05); transform: translateY(-1px);
}}
.stTabs [role="tab"][aria-selected="true"] {{
  background: linear-gradient(135deg, #8B5CF6, #3B82F6) !important;
  color: #fff !important; box-shadow: 0 6px 18px rgba(99,102,241,0.32);
}}
.stTabs [role="tab"][aria-selected="true"] p {{ color: #fff !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"],
.stTabs .react-aria-SelectionIndicator {{ display: none !important; }}
.stTabs [data-baseweb="tab-panel"], .stTabs [role="tabpanel"] {{ padding-top: 1.1rem; }}

/* ---------- panels (st.container(border=True) via panel()) ---------- */
[data-testid="stVerticalBlockBorderWrapper"], [class*="st-key-panel_"] {{
  background: linear-gradient(180deg, rgba(24,34,64,0.60), rgba(18,26,46,0.60)) !important;
  border-radius: 20px !important; border-color: var(--border) !important;
  transition: box-shadow .3s ease, border-color .3s ease; animation: fadeUp .7s var(--ease) both;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover, [class*="st-key-panel_"]:hover {{
  border-color: rgba(139,92,246,0.35) !important; box-shadow: 0 16px 40px rgba(0,0,0,0.35);
}}
[data-testid="stForm"] {{
  background: linear-gradient(180deg, rgba(24,34,64,0.60), rgba(18,26,46,0.60));
  border: 1px solid var(--border) !important; border-radius: 20px !important; padding: 1.2rem 1.3rem;
}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
}}
[data-testid="stExpander"] details {{ border-radius: 14px; border-color: var(--border); background: var(--surface); }}
[data-testid="stAlert"] {{ border-radius: 14px; }}

/* ---------- radio as segmented pills ---------- */
[data-testid="stRadio"] div[role="radiogroup"] {{ gap: 8px; flex-wrap: wrap; }}
[data-testid="stRadio"] div[role="radiogroup"] > label,
[data-testid="stRadio"] [data-testid="stRadioOption"] {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 999px;
  padding: 7px 15px 7px 11px; margin: 0; transition: all .2s var(--ease);
}}
[data-testid="stRadio"] div[role="radiogroup"] > label:hover,
[data-testid="stRadio"] [data-testid="stRadioOption"]:hover {{ border-color: rgba(139,92,246,0.55); transform: translateY(-1px); }}
[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"] {{
  background: linear-gradient(135deg, rgba(139,92,246,0.22), rgba(59,130,246,0.18));
  border-color: rgba(139,92,246,0.85); box-shadow: 0 0 0 3px rgba(139,92,246,0.14);
}}
[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"] p {{ color: #fff; font-weight: 700; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label,
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] {{ padding: 4px 12px 4px 8px; }}

/* ---------- pills (multi-select toggles) ---------- */
[data-testid="stButtonGroup"] button {{ transition: all .2s var(--ease); border-radius: 999px !important; }}
[data-testid="stButtonGroup"] button:hover {{ transform: translateY(-1px); }}

/* ================= custom components ================= */

/* hero */
.hero {{
  position: relative; overflow: hidden; border-radius: 24px; padding: 30px 32px 26px 32px;
  margin-bottom: 1.3rem; color: #fff; isolation: isolate;
  background: linear-gradient(120deg, var(--g1), var(--g2), var(--g3), var(--g1));
  background-size: 260% 260%; animation: gradientShift 16s ease infinite, fadeUp .7s var(--ease) both;
  box-shadow: 0 18px 50px rgba(0,0,0,0.40), inset 0 1px 0 rgba(255,255,255,0.15);
}}
.hero .orb {{ position:absolute; border-radius:50%; z-index:-1; pointer-events:none;
  background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.30), rgba(255,255,255,0) 70%); }}
.hero .orb.o1 {{ width:320px; height:320px; right:-80px; top:-140px; animation: floaty 9s ease-in-out infinite; }}
.hero .orb.o2 {{ width:200px; height:200px; right:260px; bottom:-120px; opacity:.55; animation: floaty 12s ease-in-out infinite reverse; }}
.hero svg.hero-line {{ position:absolute; right:0; bottom:0; width:46%; height:68%; z-index:-1; opacity:.42; }}
.hero svg.hero-line path.l {{ stroke-dasharray: 1400; animation: draw 2.6s var(--ease) .2s both; }}
.hero-eyebrow {{
  display:inline-flex; align-items:center; gap:8px; font-size:0.72rem; font-weight:800;
  letter-spacing:0.16em; text-transform:uppercase; color: #fff;
  background: rgba(255,255,255,0.14); padding: 6px 12px; border-radius: 999px; margin-bottom: 14px;
  border: 1px solid rgba(255,255,255,0.22);
}}
.hero-title {{ font-family: var(--display); font-size: 2.45rem; font-weight: 800; line-height: 1.05;
  letter-spacing:-0.03em; margin:0; }}
.hero-sub {{ font-size: 1.02rem; color: rgba(255,255,255,0.86); margin-top: 10px; max-width: 780px; line-height:1.55; }}
.hero-chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top: 18px; }}
.hero-chip {{
  font-size:0.8rem; font-weight:700; padding:7px 13px; border-radius:999px;
  background: rgba(10,15,30,0.28); border: 1px solid rgba(255,255,255,0.22); color:#fff;
  backdrop-filter: blur(6px); animation: fadeUp .6s var(--ease) both;
}}
.hero-chip:nth-child(2) {{ animation-delay: .08s; }} .hero-chip:nth-child(3) {{ animation-delay: .16s; }}

/* ticker tape */
.ticker {{ position:relative; overflow:hidden; border-radius:14px; margin: 0 0 1.3rem 0;
  background: var(--night); border: 1px solid var(--border); animation: fadeIn .6s ease both; }}
.ticker::before, .ticker::after {{ content:""; position:absolute; top:0; bottom:0; width:60px; z-index:2; }}
.ticker::before {{ left:0; background: linear-gradient(90deg, var(--night), transparent); }}
.ticker::after {{ right:0; background: linear-gradient(-90deg, var(--night), transparent); }}
.ticker-track {{ display:flex; width:max-content; animation: marquee 32s linear infinite; }}
.ticker:hover .ticker-track {{ animation-play-state: paused; }}
.tk {{ display:flex; align-items:center; gap:10px; padding: 11px 26px; color:#C9CFE6; font-weight:600;
  font-size:0.9rem; white-space:nowrap; border-right: 1px solid rgba(255,255,255,0.06); }}
.tk-dot {{ width:9px; height:9px; border-radius:50%; animation: glow 2s ease-in-out infinite; }}
.tk-name {{ color:#fff; font-weight:800; letter-spacing:.02em; }}
.tk-up {{ color:{GOOD_TEXT}; font-weight:800; }} .tk-down {{ color:{BAD_TEXT}; font-weight:800; }}

/* section header */
.sec {{ display:flex; align-items:center; gap:13px; margin: 2rem 0 0.95rem 0; animation: slideIn .6s var(--ease) both; }}
.sec-icon {{
  width:40px; height:40px; flex: 0 0 40px; border-radius:12px; display:grid; place-items:center; font-size:19px;
  background: color-mix(in srgb, var(--c) 18%, transparent);
  border: 1px solid color-mix(in srgb, var(--c) 45%, transparent);
  box-shadow: 0 0 18px color-mix(in srgb, var(--c) 25%, transparent);
  transition: transform .3s var(--ease);
}}
.sec:hover .sec-icon {{ transform: rotate(-8deg) scale(1.08); }}
.sec-title {{ font-family: var(--display); font-size:1.32rem; font-weight:800; color:var(--text); line-height:1.15; letter-spacing:-0.02em; }}
.sec-sub {{ font-size:0.86rem; color:var(--muted); margin-top:3px; }}
.sec-bar {{ flex:1; height:2px; margin-left:10px; border-radius: 2px; transform-origin: left;
  background: linear-gradient(90deg, color-mix(in srgb, var(--c) 65%, transparent), transparent);
  animation: growBar 1s var(--ease) .15s both; }}

/* KPI card */
.kpi {{
  position:relative; overflow:hidden; border-radius:18px; padding:16px 18px 15px 18px;
  min-height: 126px; height:100%;
  background:
    radial-gradient(170px 110px at 100% 0%, color-mix(in srgb, var(--a) 22%, transparent), transparent 70%),
    linear-gradient(160deg, color-mix(in srgb, var(--a) 12%, {SURFACE}) 0%, {SURFACE} 62%);
  border: 1px solid color-mix(in srgb, var(--a) 28%, {BORDER});
  transition: transform .3s var(--ease), box-shadow .3s ease, border-color .3s ease;
  animation: fadeUp .65s var(--ease) both; margin-bottom: 4px;
}}
.kpi::before {{
  content:""; position:absolute; left:0; top:0; right:0; height:3px;
  background: linear-gradient(90deg, var(--a), color-mix(in srgb, var(--a) 15%, transparent));
}}
[data-testid="stColumn"]:nth-child(2) .kpi {{ animation-delay: .07s; }}
[data-testid="stColumn"]:nth-child(3) .kpi {{ animation-delay: .14s; }}
[data-testid="stColumn"]:nth-child(4) .kpi {{ animation-delay: .21s; }}
[data-testid="stColumn"]:nth-child(5) .kpi {{ animation-delay: .28s; }}
.kpi:hover {{
  transform: translateY(-4px); border-color: color-mix(in srgb, var(--a) 60%, {BORDER});
  box-shadow: 0 16px 34px color-mix(in srgb, var(--a) 22%, transparent);
}}
.kpi-top {{ display:flex; align-items:center; gap:9px; margin-bottom:10px; min-width:0; position:relative; z-index:1; }}
.kpi-icon {{
  width:30px; height:30px; flex: 0 0 30px; border-radius:9px; display:grid; place-items:center; font-size:15px;
  background: color-mix(in srgb, var(--a) 22%, transparent);
}}
.kpi-label {{ font-size:0.72rem; font-weight:800; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0; }}
.kpi-value {{
  font-family: var(--display); font-size:1.7rem; font-weight:800; color:#fff; letter-spacing:-0.03em;
  line-height:1.1; font-variant-numeric: tabular-nums; word-break: break-word; position:relative; z-index:1;
}}
.kpi-delta {{
  display:inline-flex; align-items:center; gap:5px; margin-top:9px; font-size:0.8rem; font-weight:700;
  padding:3px 10px; border-radius:999px; font-variant-numeric: tabular-nums; position:relative; z-index:1;
}}
.kpi-delta.good {{ color: var(--good-t); background: #10291F; border: 1px solid rgba(34,197,94,0.25); }}
.kpi-delta.bad {{ color: var(--bad-t); background: #2C1320; border: 1px solid rgba(244,63,94,0.25); }}
.kpi-delta.neutral {{ color: #AEB8D6; background: #1A2240; border: 1px solid rgba(138,148,179,0.20); font-weight:600; }}
.kpi-spark {{ position:absolute; right:10px; bottom:8px; width:44%; height:42px; z-index:0; opacity:.9; }}
.kpi-spark path.l {{ stroke-dasharray: 1400; animation: draw 1.8s var(--ease) .3s both; }}

/* chips, notes, explain */
.chip {{
  display:inline-flex; align-items:center; gap:7px; font-size:0.8rem; font-weight:600; color:var(--text);
  padding:5px 11px; border-radius:999px; margin: 0 6px 6px 0;
  background: color-mix(in srgb, var(--c) 14%, transparent); border: 1px solid color-mix(in srgb, var(--c) 40%, transparent);
}}
.chip-dot {{ width:8px; height:8px; border-radius:50%; background: var(--c); box-shadow: 0 0 8px var(--c); }}
.note {{
  display:flex; gap:10px; align-items:flex-start; padding: 11px 14px; border-radius: 13px; margin: 6px 0 12px 0;
  background: color-mix(in srgb, var(--c) 9%, transparent); border: 1px solid color-mix(in srgb, var(--c) 28%, transparent);
  color: #CDD4EA; font-size: 0.9rem; line-height:1.55; animation: fadeUp .5s var(--ease) both;
}}
.note b {{ color:#fff; }}
.explain {{
  border-radius:14px; padding: 14px 18px; margin: 4px 0 6px 0; font-size: 0.93rem; line-height:1.6; color:#CDD4EA;
  background: linear-gradient(110deg, color-mix(in srgb, var(--c) 14%, transparent), transparent 80%);
  border-left: 4px solid var(--c); animation: fadeUp .5s var(--ease) both;
}}
.wbadge {{ display:inline-block; font-weight:800; padding:4px 12px; border-radius:999px; font-size:0.85rem; }}
.wbadge.ok {{ color: var(--good-t); background: rgba(34,197,94,0.14); }}
.wbadge.off {{ color: #FCD34D; background: rgba(245,158,11,0.14); }}

/* step flow (how it works) */
.steps {{ display:grid; grid-template-columns: repeat(var(--n), 1fr); gap: 14px; margin: 6px 0 8px 0; }}
.step {{ position:relative; background: linear-gradient(165deg, color-mix(in srgb, var(--c) 12%, {SURFACE}), {SURFACE} 65%);
  border:1px solid color-mix(in srgb, var(--c) 28%, {BORDER}); border-radius:18px; padding:18px 18px 16px;
  animation: fadeUp .6s var(--ease) both; transition: transform .3s var(--ease), box-shadow .3s ease; }}
.step:hover {{ transform: translateY(-4px); box-shadow: 0 14px 30px color-mix(in srgb, var(--c) 20%, transparent); }}
.step:nth-child(2) {{ animation-delay: .1s; }} .step:nth-child(3) {{ animation-delay: .2s; }} .step:nth-child(4) {{ animation-delay: .3s; }}
.step-num {{ position:absolute; top:12px; right:16px; font-family: var(--display); font-weight:800; font-size:2.3rem;
  color: color-mix(in srgb, var(--c) 30%, transparent); line-height:1; }}
.step-icon {{ width:44px; height:44px; border-radius:13px; display:grid; place-items:center; font-size:21px; margin-bottom:10px;
  background: linear-gradient(135deg, var(--c), color-mix(in srgb, var(--c) 45%, {SURFACE}));
  box-shadow: 0 8px 20px color-mix(in srgb, var(--c) 35%, transparent); }}
.step-title {{ font-family: var(--display); font-weight:800; font-size:1.05rem; color:#fff; }}
.step-text {{ font-size:0.86rem; color:var(--muted); margin-top:4px; line-height:1.5; }}
.step:not(:last-child)::after {{ content:"→"; position:absolute; right:-14px; top:50%; transform: translate(50%,-50%);
  width:26px; height:26px; border-radius:50%; display:grid; place-items:center; background:{SURFACE_2};
  border:1px solid var(--border); color: var(--text); font-weight:800; z-index:2; font-size: 13px; }}

/* story card: the plain-language answer */
.story {{ position:relative; overflow:hidden; border-radius:22px; padding: 24px 26px; color:#fff;
  background: linear-gradient(125deg, var(--c1), var(--c2), var(--c1)); background-size: 220% 220%;
  animation: gradientShift 14s ease infinite, fadeUp .7s var(--ease) both;
  box-shadow: 0 18px 50px rgba(0,0,0,0.40); }}
.story-kicker {{ font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; opacity:.92; }}
.story-main {{ font-family: var(--display); font-size: 1.6rem; font-weight:800; line-height:1.3; margin-top:6px; letter-spacing:-.02em; }}
.story-main .big {{ font-size: 2.45rem; display:inline-block; padding: 0 4px; }}
.story-grid {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; margin-top:18px; }}
.story-cell {{ background: rgba(10,15,30,0.25); border:1px solid rgba(255,255,255,0.22); border-radius:15px; padding:12px 14px; backdrop-filter: blur(6px); }}
.story-cell .t {{ font-size:.76rem; font-weight:700; opacity:.9; text-transform: uppercase; letter-spacing:.08em; }}
.story-cell .v {{ font-family: var(--display); font-size:1.35rem; font-weight:800; margin-top:2px; }}
.story-cell .s {{ font-size:.8rem; opacity:.9; margin-top:2px; }}

/* 100-dot outcome grid */
.waffle-wrap {{ display:flex; gap:26px; align-items:center; flex-wrap:wrap; border:1px solid var(--border);
  background: linear-gradient(180deg, rgba(24,34,64,0.60), rgba(18,26,46,0.60));
  border-radius:20px; padding:20px 22px; animation: fadeUp .6s var(--ease) both; }}
.waffle {{ display:grid; grid-template-columns: repeat(20, 18px); gap:5px; }}
.wd {{ width:18px; height:18px; border-radius:6px; background: var(--c); animation: popIn .45s var(--ease) both;
  box-shadow: 0 0 10px color-mix(in srgb, var(--c) 45%, transparent); }}
.wlegend {{ flex:1; min-width: 260px; }}
.wl-row {{ display:flex; align-items:center; gap:10px; padding:7px 0; border-bottom:1px dashed var(--border); font-size:.92rem; color:#CDD4EA; }}
.wl-row:last-child {{ border-bottom:0; }}
.wl-sw {{ width:14px; height:14px; border-radius:5px; background: var(--c); flex: 0 0 14px; }}
.wl-n {{ margin-left:auto; font-family: var(--display); font-weight:800; font-size:1.15rem; color: #fff; }}

/* glossary */
.gloss {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap:10px; }}
.gl {{ background: var(--surface); border:1px solid var(--border); border-radius:14px; padding:12px 14px; font-size:.87rem; color:#C3CAE0; line-height:1.5; }}
.gl b {{ display:block; color: #fff; font-family: var(--display); font-size: .98rem; margin-bottom: 2px; }}

@media (max-width: 900px) {{
  .steps {{ grid-template-columns: 1fr 1fr; }} .step::after {{ display:none; }}
  .story-grid {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 640px) {{
  .hero {{ padding: 22px 18px; }} .hero-title {{ font-size: 1.7rem; }} .kpi-value {{ font-size: 1.35rem; }}
  .steps {{ grid-template-columns: 1fr; }} .waffle {{ grid-template-columns: repeat(10, 18px); }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition: none !important; }}
}}
</style>
"""


def apply_custom_css():
    """Inject the global stylesheet (call once per run, right after set_page_config)."""
    st.markdown(_CSS, unsafe_allow_html=True)


# =========================================================
# HTML building blocks
# =========================================================
def _e(s) -> str:
    return html.escape(str(s))


def _path(values, w, h, pad=2):
    """SVG path 'M x,y L ...' for a list of numbers scaled into a w x h box."""
    v = np.asarray([x for x in values if x == x], dtype=float)
    if len(v) < 2:
        return "", ""
    if len(v) > 160:                                   # keep the SVG small
        v = v[np.linspace(0, len(v) - 1, 160).astype(int)]
    lo, hi = v.min(), v.max()
    rng = (hi - lo) or 1.0
    xs = np.linspace(pad, w - pad, len(v))
    ys = h - pad - (v - lo) / rng * (h - 2 * pad)
    line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = line + f" L{xs[-1]:.1f},{h} L{xs[0]:.1f},{h} Z"
    return line, area


def sparkline_svg(values, color, cls="kpi-spark", w=200, h=46):
    line, area = _path(values, w, h)
    if not line:
        return ""
    gid = f"g{abs(hash((color, line[:40]))) % 10**8}"
    return (f'<svg class="{cls}" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity=".30"/><stop offset="1" stop-color="{color}" stop-opacity="0"/>'
            f'</linearGradient></defs><path d="{area}" fill="url(#{gid})"/>'
            f'<path class="l" d="{line}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round" '
            f'stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>')


def hero(title: str, subtitle: str = "", page: str = "dashboard", eyebrow: str = "", chips=None, line=None):
    g1, g2, g3 = GRADIENTS.get(page, GRADIENTS["dashboard"])
    icon = PAGES.get(page, ("📊",))[0]
    if line is None:                                   # gentle decorative wave
        t = np.linspace(0, 6.3, 80)
        line = np.sin(t) * 0.6 + np.sin(t * 2.3) * 0.25 + t * 0.35
    l_path, a_path = _path(line, 600, 200, pad=8)
    svg = (f'<svg class="hero-line" viewBox="0 0 600 200" preserveAspectRatio="none">'
           f'<path d="{a_path}" fill="rgba(255,255,255,0.08)"/>'
           f'<path class="l" d="{l_path}" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="2.5" '
           f'vector-effect="non-scaling-stroke"/></svg>') if l_path else ""
    chips_html = ""
    if chips:
        chips_html = '<div class="hero-chips">' + "".join(
            f'<span class="hero-chip">{_e(c)}</span>' for c in chips if c) + "</div>"
    eb = f'<div class="hero-eyebrow">{icon} {_e(eyebrow)}</div>' if eyebrow else ""
    sub = f'<div class="hero-sub">{_e(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="hero" style="--g1:{g1};--g2:{g2};--g3:{g3}"><div class="orb o1"></div><div class="orb o2"></div>'
        f'{svg}{eb}<div class="hero-title">{_e(title)}</div>{sub}{chips_html}</div>',
        unsafe_allow_html=True)


def ticker(items):
    """Scrolling ticker tape. items: list of (name, price_text, change_pct, colour)."""
    cells = ""
    for name, price, chg, color in items:
        cls = "tk-up" if chg >= 0 else "tk-down"
        arrow = "▲" if chg >= 0 else "▼"
        cells += (f'<div class="tk"><span class="tk-dot" style="background:{color};box-shadow:0 0 10px {color}"></span>'
                  f'<span class="tk-name">{_e(name)}</span><span>{_e(price)}</span>'
                  f'<span class="{cls}">{arrow} {abs(chg):.2f}%</span></div>')
    st.markdown(f'<div class="ticker"><div class="ticker-track">{cells * 4}</div></div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = "", icon: str = "◆", color: str = VIOLET):
    sub = f'<div class="sec-sub">{_e(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="sec" style="--c:{color}"><div class="sec-icon">{icon}</div>'
        f'<div><div class="sec-title">{_e(title)}</div>{sub}</div><div class="sec-bar"></div></div>',
        unsafe_allow_html=True)


def kpi(label: str, value: str, delta: str = None, tone: str = "neutral", icon: str = "", accent: str = VIOLET,
        spark=None):
    """A KPI card. tone: 'good' | 'bad' | 'neutral'. spark: optional list of numbers for a mini chart."""
    arrow = {"good": "▲ ", "bad": "▼ "}.get(tone, "")
    d = f'<div class="kpi-delta {tone}">{arrow}{_e(delta)}</div>' if delta else ""
    ic = f'<span class="kpi-icon">{icon}</span>' if icon else ""
    sp = sparkline_svg(spark, accent) if spark is not None else ""
    st.markdown(
        f'<div class="kpi" style="--a:{accent}">{sp}<div class="kpi-top">{ic}'
        f'<span class="kpi-label">{_e(label)}</span></div>'
        f'<div class="kpi-value">{_e(value)}</div>{d}</div>',
        unsafe_allow_html=True)


def kpi_row(cards):
    """Render a row of KPI cards. cards = list of dicts with kpi() keyword arguments."""
    for col, c in zip(st.columns(len(cards)), cards):
        with col:
            kpi(**c)


def tone_of(x, good_if_positive: bool = True) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "neutral"
    if x != x or x == 0:
        return "neutral"
    return "good" if (x > 0) == good_if_positive else "bad"


def render_metric_card(label: str, value: str, delta: str = None, is_positive: bool = True):
    """Backwards-compatible wrapper around kpi()."""
    kpi(label, value, delta, tone="good" if is_positive else "bad")


def chips(items):
    """items: list of (text, colour)."""
    st.markdown("".join(f'<span class="chip" style="--c:{c}"><span class="chip-dot"></span>{_e(t)}</span>'
                        for t, c in items), unsafe_allow_html=True)


def note(text_html: str, color: str = BLUE, icon: str = "ℹ️"):
    """Small coloured info strip. text_html may contain <b> tags (caller controls the content)."""
    st.markdown(f'<div class="note" style="--c:{color}"><span>{icon}</span><div>{text_html}</div></div>',
                unsafe_allow_html=True)


def explain(text: str, color: str = VIOLET):
    st.markdown(f'<div class="explain" style="--c:{color}">{_e(text)}</div>', unsafe_allow_html=True)


def steps(items):
    """Horizontal 'how it works' flow. items: list of (icon, title, text, colour)."""
    cells = "".join(
        f'<div class="step" style="--c:{c}"><div class="step-num">{i + 1}</div><div class="step-icon">{ic}</div>'
        f'<div class="step-title">{_e(t)}</div><div class="step-text">{_e(x)}</div></div>'
        for i, (ic, t, x, c) in enumerate(items))
    st.markdown(f'<div class="steps" style="--n:{len(items)}">{cells}</div>', unsafe_allow_html=True)


def story(kicker: str, main_html: str, cells, c1: str = "#C2410C", c2: str = "#7C3AED"):
    """Big gradient card with the plain-language answer. cells: list of (title, value, sub)."""
    grid = "".join(f'<div class="story-cell"><div class="t">{_e(t)}</div><div class="v">{_e(v)}</div>'
                   f'<div class="s">{_e(s)}</div></div>' for t, v, s in cells)
    st.markdown(f'<div class="story" style="--c1:{c1};--c2:{c2}"><div class="story-kicker">{_e(kicker)}</div>'
                f'<div class="story-main">{main_html}</div><div class="story-grid">{grid}</div></div>',
                unsafe_allow_html=True)


def waffle(buckets, caption: str = ""):
    """100-dot grid. buckets: list of (label, count, colour); counts must add up to 100."""
    dots, i = "", 0
    for _, n, c in buckets:
        for _ in range(int(n)):
            dots += f'<div class="wd" style="--c:{c};animation-delay:{i * 9}ms"></div>'
            i += 1
    legend = "".join(f'<div class="wl-row"><span class="wl-sw" style="--c:{c}"></span>{_e(lbl)}'
                     f'<span class="wl-n">{int(n)}</span></div>' for lbl, n, c in buckets)
    cap = f'<div style="font-size:.85rem;color:{MUTED};margin-bottom:6px">{_e(caption)}</div>' if caption else ""
    st.markdown(f'<div class="waffle-wrap"><div class="waffle">{dots}</div><div class="wlegend">{cap}{legend}</div></div>',
                unsafe_allow_html=True)


def glossary(items):
    """items: list of (term, plain explanation)."""
    st.markdown('<div class="gloss">' + "".join(f'<div class="gl"><b>{_e(t)}</b>{_e(x)}</div>' for t, x in items)
                + '</div>', unsafe_allow_html=True)


def weight_badge(total: float):
    ok = abs(total - 100) <= 0.01
    txt = f"Total {total:.1f}%" + ("  ✓" if ok else "  → scaled to 100%")
    st.markdown(f'<span class="wbadge {"ok" if ok else "off"}">{_e(txt)}</span>', unsafe_allow_html=True)


# =========================================================
# Sidebar: brand + navigation + footer
# =========================================================
def sidebar_nav(current: str):
    st.sidebar.markdown(
        '<div class="brand"><div class="brand-logo">📈</div><div>'
        '<div class="brand-name">Multi-Asset Lab</div>'
        '<div class="brand-sub">BACKTEST · SIMULATE · INR ₹</div></div></div>',
        unsafe_allow_html=True)
    st.sidebar.markdown('<div class="side-label">Navigate</div>', unsafe_allow_html=True)
    for key, (icon, label, _) in PAGES.items():
        if st.sidebar.button(f"{icon}   {label}", key=f"nav_{key}", width="stretch",
                             type="primary" if key == current else "secondary"):
            if key != current:
                st.session_state["view"] = key
                st.rerun()


def sidebar_label(text: str):
    st.sidebar.markdown(f'<div class="side-label">{_e(text)}</div>', unsafe_allow_html=True)


def sidebar_footer(fetch_errors, key: str = "main"):
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄  Refresh live sheet data", width="stretch", key=f"refresh_{key}"):
        st.cache_data.clear()
        st.rerun()
    if fetch_errors:
        with st.sidebar.expander(f"⚠️ Data load issues ({len(fetch_errors)})", expanded=False):
            for name, err in fetch_errors.items():
                st.write(f"**{name}:** {err}")


# =========================================================
# Plotly chart theme (dark)
# =========================================================
CHART_FONT = "Plus Jakarta Sans, system-ui, sans-serif"
GRID = "rgba(138,148,179,0.11)"

_template = go.layout.Template(pio.templates["plotly_dark"])
_template.layout.update(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=CHART_FONT, color="#C7CEE3", size=12),
    title=dict(font=dict(size=16, color=TEXT, family="Bricolage Grotesque, " + CHART_FONT), x=0.01, xanchor="left"),
    colorway=[VIOLET, CYAN, AMBER, PINK, EMERALD, BLUE, ORANGE, ROSE],
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickcolor=BORDER, showspikes=False),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickcolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#C7CEE3"), traceorder="normal"),
    hoverlabel=dict(bgcolor=SURFACE_2, bordercolor="#33406A", font=dict(color=TEXT, family=CHART_FONT)),
    margin=dict(l=12, r=12, t=56, b=12),
    transition=dict(duration=500, easing="cubic-in-out"),
)
pio.templates["lab"] = _template

PLOT_LAYOUT = dict(template="lab")


def panel():
    """A card around charts / inputs. Keys are numbered in page order so they stay stable between reruns
    (app.py resets the counter at the top of every run)."""
    n = st.session_state.get("_panel_n", 0) + 1
    st.session_state["_panel_n"] = n
    return st.container(border=True, key=f"panel_{n}")


def chart(fig, key=None, height=None):
    """Show a Plotly figure with the app theme (Streamlit's own theme would override ours)."""
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, theme=None, width="stretch", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})