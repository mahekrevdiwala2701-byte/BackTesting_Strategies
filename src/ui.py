"""Design system for the whole app: light, navy + indigo.

Soft light-grey page, white cards with gentle shadows, a deep navy → indigo banner at the top of every page,
indigo for buttons and selected items, teal as the second colour, green / red only for gains / losses.
Charts: one clear colour per asset, the portfolio as the navy line.

Everything visual lives here so every page looks the same:
  * colour tokens and the global CSS
  * HTML building blocks: top bar + page switcher, page banner, section headers, KPI cards (with sparklines),
    chips, ticker tape, step flow, story card, 100-dot outcome grid, glossary, panels
  * reveal() / action_bar(): Download CSV buttons and switches that show extra detail only when asked
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
BG = "#F3F5FA"          # page: soft light grey-blue
SURFACE = "#FFFFFF"     # cards and panels
SURFACE_2 = "#F1F4F9"   # inputs, hover, tracks
BORDER = "#E2E7F0"      # hairlines
TEXT = "#0F1B3D"        # main text: deep navy
INK = "#0F1B3D"         # strong lines on charts (median, typical case ...)
MUTED = "#64708A"       # secondary text
FAINT = "#A3AEC2"       # reference lines
SOFT = "#3A4766"        # body text on cards

NAVY = "#0B1B3F"        # banner
NAVY_2 = "#3730A3"      # banner, other end
NIGHT = "#FFFFFF"       # ticker tape
ACCENT = "#4F46E5"      # indigo: buttons, selected page / tab, links
ACCENT_DARK = "#4338CA"
ACCENT_SOFT = "#EEF0FF"
PORTFOLIO = "#0F1B3D"   # the total portfolio is the navy line on every chart

VIOLET = "#7C3AED"
INDIGO = "#4F46E5"
BLUE = "#2563EB"
CYAN = "#0891B2"
TEAL = "#0D9488"
EMERALD = "#059669"
ORANGE = "#EA580C"
AMBER = "#EA7A12"       # (name kept for older code: warm orange)
PINK = "#DB2777"
ROSE = "#E11D48"
LIME = "#65A30D"

GOOD = "#16A34A"        # gains
BAD = "#DC2626"         # losses
GOOD_TEXT = "#15803D"
BAD_TEXT = "#B91C1C"
GOOD_BG = "#E8F7EE"
BAD_BG = "#FDECEC"

# Banner colours (the same on every page, so the app feels like one product)
GRADIENTS = {k: (NAVY, NAVY_2) for k in ("home", "invest", "montecarlo", "portfolio")}

# One colour per strategy (same on the Investment and Monte Carlo pages)
STRAT_COLORS = {"bh": CYAN, "periodic": VIOLET, "diff": ORANGE}

# Navigation: key -> (icon, label, colour)
PAGES = {
    "home": ("🏠", "Main Dashboard", ACCENT),
    "invest": ("💰", "Calculated Investment", ACCENT),
    "montecarlo": ("🎲", "Monte Carlo", ACCENT),
    "portfolio": ("💼", "Everyday Portfolio", ACCENT),
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
# Global CSS (light, navy + indigo)
# =========================================================
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {{
  --bg:{BG}; --surface:{SURFACE}; --surface2:{SURFACE_2}; --border:{BORDER};
  --text:{TEXT}; --muted:{MUTED}; --faint:{FAINT}; --soft:{SOFT};
  --navy:{NAVY}; --navy2:{NAVY_2}; --accent:{ACCENT}; --accent-d:{ACCENT_DARK}; --accent-s:{ACCENT_SOFT};
  --teal:{TEAL};
  --good:{GOOD}; --bad:{BAD}; --good-t:{GOOD_TEXT}; --bad-t:{BAD_TEXT}; --good-bg:{GOOD_BG}; --bad-bg:{BAD_BG};
  --shadow: 0 1px 2px rgba(15,27,61,0.05), 0 6px 18px rgba(15,27,61,0.06);
  --shadow-lg: 0 2px 4px rgba(15,27,61,0.05), 0 14px 34px rgba(15,27,61,0.10);
  --font: 'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}

html, body, [class*="css"], .stApp, button, input, textarea, select {{ font-family: var(--font); }}
@keyframes marquee {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
@keyframes drawLine {{ from {{ stroke-dashoffset: 1600; }} to {{ stroke-dashoffset: 0; }} }}

/* ---------- canvas ---------- */
.stApp {{ background: var(--bg); color: var(--text); }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 1.1rem; padding-bottom: 4rem; max-width: 1400px; }}
h1, h2, h3, h4, h5 {{ font-family: var(--font); color: var(--text); letter-spacing: -0.02em; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--muted); }}
a {{ color: var(--accent); }}
*:focus-visible {{ outline: 2px solid var(--accent) !important; outline-offset: 2px; }}

/* ---------- top bar: brand + page switcher ---------- */
.st-key-topbar {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; box-shadow: var(--shadow);
  padding: 8px 10px; margin-bottom: 1.1rem; gap: 14px; }}
.brand {{ display:flex; align-items:center; gap:10px; white-space:nowrap; padding-left: 4px; }}
.brand-mark {{ width:34px; height:34px; display:grid; place-items:center; border-radius:10px; color:#fff; font-weight:800;
  font-size:17px; background: linear-gradient(135deg, var(--navy), var(--accent)); }}
.brand-name {{ font-weight:800; font-size:1.02rem; color: var(--text); letter-spacing:-0.02em; }}
.st-key-topbar [role="radiogroup"] {{ gap: 4px; flex-wrap: wrap; margin: 0; align-items: center; }}
/* keep the brand and the page buttons on one centre line */
.st-key-topbar > [data-testid="stElementContainer"] {{ margin: 0 !important; align-self: center; }}
.st-key-topbar [data-testid="stMarkdownContainer"] {{ margin: 0 !important; }}
.st-key-topbar [data-testid="stButtonGroup"] {{ margin: 0; padding: 0; }}
.st-key-topbar .brand {{ height: 38px; padding: 0 6px 0 4px; }}
.st-key-topbar button[data-variant="segmented_control"] {{
  background: transparent !important; border: 1px solid transparent !important; border-radius: 11px !important;
  box-shadow: none !important; padding: 0 14px !important; height: 38px; min-height: 0; transition: background .15s ease;
}}
.st-key-topbar button[data-variant="segmented_control"] p {{ color: var(--soft); font-weight: 600; font-size: .9rem; }}
.st-key-topbar button[data-variant="segmented_control"]:hover {{ background: var(--surface2) !important; }}
.st-key-topbar button[data-variant="segmented_control"][aria-checked="true"] {{
  background: var(--accent) !important; box-shadow: 0 4px 12px rgba(79,70,229,0.30) !important;
}}
.st-key-topbar button[data-variant="segmented_control"][aria-checked="true"] p {{ color: #fff; font-weight: 700; }}

/* ---------- sidebar: settings ---------- */
section[data-testid="stSidebar"] {{ background: var(--surface); border-right: 1px solid var(--border); }}
section[data-testid="stSidebar"]::before {{ content:""; position:absolute; left:0; right:0; top:0; height:4px;
  background: linear-gradient(90deg, var(--navy), var(--accent), var(--teal)); }}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top: .4rem; }}
section[data-testid="stSidebar"] hr {{ margin: 14px 0; border-color: var(--border); }}
.side-title {{ font-size:1.2rem; font-weight:800; color: var(--text); margin: 0 2px 2px 2px; }}
.side-hint {{ font-size:.82rem; color: var(--muted); margin: 0 2px 6px 2px; }}
.side-label {{ font-size:0.82rem; font-weight:700; color: var(--text); margin: 18px 2px 8px 2px;
  padding-top: 12px; border-top: 1px solid var(--border); }}

/* ---------- buttons ---------- */
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] button, [data-testid="stBaseButton-primaryFormSubmit"] {{
  background: linear-gradient(135deg, var(--accent), #6366F1) !important; border: 0 !important; color: #fff !important;
  font-weight: 700 !important; border-radius: 12px !important; box-shadow: 0 6px 16px rgba(79,70,229,0.28) !important;
  transition: filter .15s ease;
}}
.stButton > button[kind="primary"] p, [data-testid="stFormSubmitButton"] button p {{ color:#fff !important; font-weight:700; }}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:hover {{ filter: brightness(1.08); }}
.stButton > button[kind="secondary"], [data-testid="stBaseButton-secondary"] {{
  background: var(--surface); border: 1px solid var(--border); color: var(--text); border-radius: 12px;
}}
.stButton > button[kind="secondary"]:hover {{ border-color: var(--accent); color: var(--accent); }}
[data-testid="stDownloadButton"] button {{
  border: 1px solid #C7CBFB; background: var(--accent-s); color: var(--accent-d); font-weight: 700; border-radius: 12px;
}}
[data-testid="stDownloadButton"] button:hover {{ background: #E0E3FF; border-color: var(--accent); }}

/* ---------- st.tabs: underline ---------- */
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {{
  gap: 2px; border-bottom: 1px solid var(--border); overflow-x: auto; background: transparent;
}}
.stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"] {{
  height: auto; padding: 9px 14px; background: transparent; white-space: nowrap; border: 0;
  border-bottom: 2px solid transparent; margin: 0 0 -1px 0; border-radius: 0;
}}
.stTabs [data-testid="stTab"] p {{ font-weight: 600; font-size: 0.92rem; color: var(--muted); }}
.stTabs [role="tab"][aria-selected="true"] {{ border-bottom-color: var(--accent); }}
.stTabs [role="tab"][aria-selected="true"] p {{ color: var(--accent) !important; font-weight: 700; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"],
.stTabs .react-aria-SelectionIndicator {{ display: none !important; }}
.stTabs [data-baseweb="tab-panel"], .stTabs [role="tabpanel"] {{ padding-top: 1rem; }}

/* ---------- "show more" switches and action rows ---------- */
[class*="st-key-reveal_"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 7px 14px 5px 12px; width: fit-content !important; box-shadow: var(--shadow); }}
[class*="st-key-reveal_"]:hover {{ border-color: #A5B4FC; }}
[class*="st-key-reveal_"] p {{ font-weight: 700; color: var(--text); }}
[class*="st-key-actions_"] {{ gap: 10px; flex-wrap: wrap; margin: 4px 0 10px 0; }}

/* ---------- panels, forms, tables ---------- */
[data-testid="stVerticalBlockBorderWrapper"], [class*="st-key-panel_"] {{
  background: var(--surface) !important; border-radius: 16px !important; border-color: var(--border) !important;
  box-shadow: var(--shadow);
}}
[data-testid="stForm"] {{ background: var(--surface); border: 1px solid var(--border) !important; border-radius: 16px !important;
  padding: 1.1rem 1.2rem; box-shadow: var(--shadow); }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
  box-shadow: var(--shadow); }}
[data-testid="stExpander"] details {{ border-radius: 12px; border-color: var(--border); background: var(--surface); }}
[data-testid="stAlert"] {{ border-radius: 12px; }}

/* ---------- radio + pills ---------- */
[data-testid="stRadio"] div[role="radiogroup"] {{ gap: 6px; flex-wrap: wrap; }}
[data-testid="stRadio"] div[role="radiogroup"] > label, [data-testid="stRadio"] [data-testid="stRadioOption"] {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 999px; padding: 6px 14px 6px 10px; margin: 0;
}}
[data-testid="stRadio"] [data-testid="stRadioOption"]:hover {{ border-color: #A5B4FC; }}
[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"] {{ background: var(--accent-s); border-color: var(--accent); }}
[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"] p {{ color: var(--accent-d); font-weight: 700; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label,
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] {{ padding: 4px 11px 4px 8px; }}
[data-testid="stButtonGroup"] button {{ border-radius: 999px !important; }}

/* ================= custom components ================= */

/* page banner: deep navy -> indigo */
.hero {{ position: relative; overflow: hidden; border-radius: 20px; margin: 0 0 1.1rem 0; padding: 26px 28px 20px 28px;
  background: linear-gradient(120deg, var(--g1) 0%, var(--g2) 100%); isolation: isolate;
  box-shadow: 0 14px 34px rgba(30,27,94,0.22); }}
.hero::before {{ content:""; position:absolute; inset:0; z-index:-1;
  background: radial-gradient(520px 260px at 88% 115%, rgba(13,148,136,0.45), transparent 70%),
              radial-gradient(420px 240px at 70% -30%, rgba(129,140,248,0.35), transparent 70%); }}
.hero-top {{ display:flex; align-items:flex-end; gap: 28px; }}
.hero-text {{ flex: 1 1 auto; min-width: 0; }}
.hero-eyebrow {{ display:inline-block; font-size: .8rem; font-weight: 700; color: #C7D2FE; margin-bottom: 8px; }}
.hero-title {{ font-size: clamp(1.9rem, 3.6vw, 2.8rem); font-weight: 800; line-height: 1.06; letter-spacing: -0.035em;
  margin: 0; color: #fff; }}
.hero-sub {{ font-size: 1rem; color: rgba(255,255,255,0.82); margin-top: 10px; max-width: 66ch; line-height: 1.55; }}
.hero svg.hero-line {{ flex: 0 0 32%; height: 100px; }}
.hero svg.hero-line path.l {{ stroke-dasharray: 1600; animation: drawLine 1.6s ease-out .1s both; }}
.hero-chips {{ display:flex; flex-wrap:wrap; gap: 8px; margin-top: 16px; }}
.hero-chip {{ font-size: .82rem; font-weight: 600; color: #fff; padding: 6px 12px; border-radius: 999px;
  background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.20); }}

/* ticker tape */
.ticker {{ position:relative; overflow:hidden; margin: 0 0 1.1rem 0; border-radius: 14px; background: var(--surface);
  border: 1px solid var(--border); box-shadow: var(--shadow); }}
.ticker::before, .ticker::after {{ content:""; position:absolute; top:0; bottom:0; width:40px; z-index:2; }}
.ticker::before {{ left:0; background: linear-gradient(90deg, var(--surface), transparent); }}
.ticker::after {{ right:0; background: linear-gradient(-90deg, var(--surface), transparent); }}
.ticker-track {{ display:flex; width:max-content; animation: marquee 45s linear infinite; }}
.ticker:hover .ticker-track {{ animation-play-state: paused; }}
.tk {{ display:flex; align-items:center; gap:9px; padding: 10px 24px; color: var(--soft); font-size:0.9rem; white-space:nowrap;
  border-right: 1px solid var(--border); font-variant-numeric: tabular-nums; }}
.tk-dot {{ width:9px; height:9px; border-radius:50%; }}
.tk-name {{ color: var(--text); font-weight:700; }}
.tk-up {{ color:{GOOD_TEXT}; font-weight:700; }} .tk-down {{ color:{BAD_TEXT}; font-weight:700; }}

/* section header */
.sec {{ display:flex; align-items:center; gap:12px; margin: 1.8rem 0 0.9rem 0; }}
.sec-icon {{ width:36px; height:36px; flex: 0 0 36px; border-radius:11px; display:grid; place-items:center; font-size:17px;
  background: color-mix(in srgb, var(--c) 12%, white); border: 1px solid color-mix(in srgb, var(--c) 25%, white); }}
.sec-title {{ font-size: 1.18rem; font-weight: 800; color: var(--text); line-height: 1.2; letter-spacing: -0.02em; }}
.sec-sub {{ font-size: .86rem; color: var(--muted); margin-top: 2px; }}
.sec-bar {{ flex:1; height:2px; margin-left:12px; border-radius:2px;
  background: linear-gradient(90deg, color-mix(in srgb, var(--c) 45%, white), transparent); }}

/* KPI card */
.kpi {{ position:relative; overflow:hidden; border-radius:16px; padding:16px 18px 15px 18px; min-height: 126px; height:100%;
  margin-bottom: 4px; background: var(--surface); border: 1px solid var(--border); box-shadow: var(--shadow);
  transition: box-shadow .2s ease, transform .2s ease; }}
.kpi::before {{ content:""; position:absolute; left:0; right:0; top:0; height:4px;
  background: linear-gradient(90deg, var(--a), color-mix(in srgb, var(--a) 35%, white)); }}
.kpi:hover {{ box-shadow: var(--shadow-lg); transform: translateY(-2px); }}
.kpi-top {{ display:flex; align-items:center; gap:9px; margin-bottom:8px; min-width:0; position:relative; z-index:1; }}
.kpi-icon {{ width:30px; height:30px; flex: 0 0 30px; border-radius:50%; display:grid; place-items:center; font-size:14px;
  background: color-mix(in srgb, var(--a) 13%, white); }}
.kpi-label {{ font-size:0.86rem; font-weight:600; color: var(--muted); white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; min-width:0; }}
.kpi-value {{ font-size: clamp(1.35rem, 1.7vw, 2rem); font-weight: 800; color: var(--text); letter-spacing: -0.02em;
  line-height: 1.1; font-variant-numeric: tabular-nums; white-space: nowrap; position:relative; z-index:1; }}
.kpi-delta {{ display:inline-flex; align-items:center; gap:4px; margin-top:9px; font-size:0.8rem; font-weight:700;
  padding: 3px 9px; border-radius: 999px; font-variant-numeric: tabular-nums; position:relative; z-index:1; }}
.kpi-delta.good {{ color: var(--good-t); background: var(--good-bg); }}
.kpi-delta.bad {{ color: var(--bad-t); background: var(--bad-bg); }}
.kpi-delta.neutral {{ color: var(--soft); background: var(--surface2); font-weight: 600; }}
.kpi-spark {{ position:absolute; right:12px; bottom:10px; width:40%; height:40px; z-index:0; opacity:.9; }}

/* chips, notes, explain */
.chip {{ display:inline-flex; align-items:center; gap:7px; font-size:0.84rem; color: var(--soft); font-weight: 600;
  padding:5px 12px; border-radius:999px; margin: 0 6px 6px 0; background: var(--surface); border: 1px solid var(--border); }}
.chip-dot {{ width:8px; height:8px; border-radius:50%; background: var(--c); }}
.note {{ display:flex; gap:10px; align-items:flex-start; padding: 11px 15px; border-radius: 14px; margin: 6px 0 12px 0;
  background: color-mix(in srgb, var(--c) 7%, white); border: 1px solid color-mix(in srgb, var(--c) 24%, white);
  color: var(--soft); font-size: 0.9rem; line-height:1.55; }}
.note b {{ color: var(--text); }}
.explain {{ border-radius:14px; padding: 12px 16px; margin: 6px 0 10px 0; font-size: 0.94rem; line-height:1.6; color: var(--soft);
  background: color-mix(in srgb, var(--c) 7%, white); border-left: 4px solid var(--c); }}
.wbadge {{ display:inline-block; font-weight:700; padding:4px 12px; border-radius:999px; font-size:0.86rem; }}
.wbadge.ok {{ color: var(--good-t); background: var(--good-bg); }}
.wbadge.off {{ color: var(--accent-d); background: var(--accent-s); }}

/* step flow (how it works): a real sequence, so it is numbered */
.steps {{ display:grid; grid-template-columns: repeat(var(--n), 1fr); gap: 12px; margin: 6px 0 8px 0; }}
.step {{ position:relative; padding:18px 18px 16px; border-radius: 16px; background: var(--surface);
  border: 1px solid var(--border); box-shadow: var(--shadow); }}
.step-num {{ position:absolute; top:12px; right:16px; font-weight:800; font-size:2rem; line-height:1;
  color: color-mix(in srgb, var(--c) 30%, white); }}
.step-icon {{ width:42px; height:42px; border-radius:12px; display:grid; place-items:center; font-size:20px; margin-bottom:10px;
  background: color-mix(in srgb, var(--c) 13%, white); }}
.step-title {{ font-weight:800; font-size:1.02rem; color: var(--text); }}
.step-text {{ font-size:0.86rem; color: var(--muted); margin-top:4px; line-height:1.5; }}

/* story card: the plain-language answer, in the banner colours */
.story {{ position:relative; overflow:hidden; border-radius:20px; padding: 22px 24px; color:#fff; isolation:isolate;
  background: linear-gradient(120deg, {NAVY}, {NAVY_2}); box-shadow: 0 14px 34px rgba(30,27,94,0.22); }}
.story::before {{ content:""; position:absolute; inset:0; z-index:-1;
  background: radial-gradient(480px 240px at 90% 120%, color-mix(in srgb, var(--c1) 55%, transparent), transparent 70%); }}
.story-kicker {{ font-size:.92rem; font-weight:600; color: #C7D2FE; }}
.story-main {{ font-size: 1.4rem; font-weight:700; line-height:1.4; margin-top:6px; max-width: 62ch; }}
.story-main .big {{ font-size: 2.6rem; font-weight: 800; display:inline-block; padding: 0 4px; vertical-align: -3px; color: #fff; }}
.story-grid {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-top:18px; }}
.story-cell {{ padding:12px 15px; border-radius:14px; background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.18); }}
.story-cell .t {{ font-size:.84rem; color: #C7D2FE; }}
.story-cell .v {{ font-size:1.5rem; font-weight:800; margin-top:2px; }}
.story-cell .s {{ font-size:.82rem; color: rgba(255,255,255,0.78); margin-top:2px; }}

/* 100-dot outcome grid */
.waffle-wrap {{ display:flex; gap:28px; align-items:center; flex-wrap:wrap; border:1px solid var(--border);
  background: var(--surface); border-radius:16px; padding:20px 22px; box-shadow: var(--shadow); }}
.waffle {{ display:grid; grid-template-columns: repeat(20, 17px); gap:5px; }}
.wd {{ width:17px; height:17px; border-radius:5px; background: var(--c); }}
.wlegend {{ flex:1; min-width: 260px; }}
.wl-row {{ display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px solid var(--border); font-size:.92rem; color:var(--soft); }}
.wl-row:last-child {{ border-bottom:0; }}
.wl-sw {{ width:13px; height:13px; border-radius:4px; background: var(--c); flex: 0 0 13px; }}
.wl-n {{ margin-left:auto; font-weight:800; font-size:1.3rem; color: var(--text); }}

/* glossary */
.gloss {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:10px; }}
.gl {{ padding:13px 15px; font-size:.88rem; color: var(--soft); line-height:1.5; border-radius: 14px; background: var(--surface);
  border: 1px solid var(--border); box-shadow: var(--shadow); }}
.gl b {{ display:block; color: var(--text); font-size: .98rem; margin-bottom: 2px; }}

@media (max-width: 900px) {{
  .steps {{ grid-template-columns: 1fr 1fr; }}
  .story-grid {{ grid-template-columns: 1fr; }}
  .hero svg.hero-line {{ display:none; }}
}}
@media (max-width: 640px) {{
  .hero {{ padding: 20px 18px; }} .kpi-value {{ font-size: 1.6rem; }} .steps {{ grid-template-columns: 1fr; }}
  .waffle {{ grid-template-columns: repeat(10, 17px); }}
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


def hero(title: str, subtitle: str = "", page: str = "home", eyebrow: str = "", chips=None, line=None):
    """Page banner (navy → indigo): big title, one line of explanation, the portfolio's own line on
    the right (only when there is real data for it) and a row of read-outs underneath."""
    g1, g2 = GRADIENTS.get(page, (NAVY, NAVY_2))
    svg = ""
    if line is not None:
        l_path, a_path = _path(line, 600, 200, pad=8)
        if l_path:
            svg = (f'<svg class="hero-line" viewBox="0 0 600 200" preserveAspectRatio="none" aria-hidden="true">'
                   f'<path d="{a_path}" fill="rgba(255,255,255,0.14)"/>'
                   f'<path class="l" d="{l_path}" fill="none" stroke="#FFFFFF" stroke-width="2.6" '
                   f'vector-effect="non-scaling-stroke"/></svg>')
    chips_html = ""
    if chips:
        chips_html = '<div class="hero-chips">' + "".join(
            f'<span class="hero-chip">{_e(c)}</span>' for c in chips if c) + "</div>"
    eb = f'<div class="hero-eyebrow">{_e(eyebrow)}</div>' if eyebrow else ""
    sub = f'<div class="hero-sub">{_e(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="hero" style="--g1:{g1};--g2:{g2}"><div class="hero-top"><div class="hero-text">{eb}'
        f'<div class="hero-title">{_e(title)}</div>{sub}</div>{svg}</div>{chips_html}</div>',
        unsafe_allow_html=True)


def ticker(items):
    """Scrolling ticker tape. items: list of (name, price_text, change_pct, colour)."""
    cells = ""
    for name, price, chg, color in items:
        cls = "tk-up" if chg >= 0 else "tk-down"
        arrow = "▲" if chg >= 0 else "▼"
        cells += (f'<div class="tk"><span class="tk-dot" style="background:{color}"></span>'
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


def story(kicker: str, main_html: str, cells, c1: str = ACCENT, c2: str = VIOLET):
    """Highlighted card with the plain-language answer. cells: list of (title, value, sub)."""
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
# Top bar + settings sidebar
# =========================================================
def _go_to_page():
    """Page switcher callback: open the picked page (clicking the open page again keeps it open)."""
    pick = st.session_state.get("nav_page")
    if pick in PAGES:
        st.session_state["view"] = pick


def sidebar_nav(current: str):
    """Top bar (brand + page switcher) across the page, and the title of the settings sidebar.
    (Name kept from the old sidebar navigation so every page keeps calling it the same way.)"""
    st.session_state["nav_page"] = current                    # the switcher always shows the open page
    with st.container(key="topbar", horizontal=True, vertical_alignment="center"):
        st.markdown('<div class="brand"><div class="brand-mark">₹</div>'
                    '<div class="brand-name">Multi-Asset Lab</div></div>', unsafe_allow_html=True)
        st.segmented_control("Page", list(PAGES), format_func=lambda k: f"{PAGES[k][0]} {PAGES[k][1]}",
                             key="nav_page", on_change=_go_to_page, label_visibility="collapsed")
    st.sidebar.markdown(f'<div class="side-title">Settings</div>'
                        f'<div class="side-hint">For {_e(PAGES[current][1])}</div>', unsafe_allow_html=True)


def sidebar_label(text: str):
    st.sidebar.markdown(f'<div class="side-label">{_e(text)}</div>', unsafe_allow_html=True)


def sidebar_footer(fetch_errors, key: str = "main"):
    st.sidebar.markdown("---")
    updated = st.session_state.get("_data_updated_text")
    if updated:
        st.sidebar.caption(updated)
    if st.sidebar.button("🔄  Refresh live sheet data", width="stretch", key=f"refresh_{key}"):
        st.cache_data.clear()
        st.session_state["_force_sheet_refresh"] = True     # app.py downloads from Google right now
        st.rerun()
    if fetch_errors:
        with st.sidebar.expander(f"⚠️ Data load issues ({len(fetch_errors)})", expanded=False):
            for name, err in fetch_errors.items():
                st.write(f"**{name}:** {err}")



# =========================================================
# "Show more" switches: extra detail stays hidden (and is not even built) until you switch it on
# =========================================================
def reveal(label: str, key: str, help: str = None) -> bool:
    """A switch for extra detail (tables, extra charts). Returns True when it is on."""
    with st.container(key=f"reveal_{key}"):
        return st.toggle(label, key=key, help=help)


def action_bar(download=None, toggles=()):
    """One row with a Download CSV button and "show" switches.
    download: dict(label, data, file_name, key) or None.  toggles: list of (label, key).
    Returns {key: True/False} for the switches."""
    states = {}
    with st.container(key=f"actions_{(download or {}).get('key') or (toggles[0][1] if toggles else 'x')}",
                      horizontal=True, vertical_alignment="center"):
        if download:
            st.download_button(download['label'], download['data'], file_name=download['file_name'],
                               mime="text/csv", key=download['key'])
        for label, key in toggles:
            with st.container(key=f"reveal_{key}"):
                states[key] = st.toggle(label, key=key)
    return states


# =========================================================
# Keeping widget values
# =========================================================
def start_value(key: str, value, options=None):
    """Give a widget its starting value through st.session_state (instead of value= / index= / default=),
    so it can be kept without Streamlit warnings. With `options`, a saved value that is no longer
    one of the options is reset (lists are filtered)."""
    if key not in st.session_state:
        st.session_state[key] = value
    elif options is not None:
        cur = st.session_state[key]
        if isinstance(cur, list):
            st.session_state[key] = [v for v in cur if v in options]
        elif cur not in options:
            st.session_state[key] = value


# =========================================================
# Plotly chart theme (light)
# =========================================================
CHART_FONT = "Manrope, system-ui, sans-serif"
GRID = "#EDF0F6"

_template = go.layout.Template(pio.templates["plotly_white"])
_template.layout.update(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=CHART_FONT, color=SOFT, size=12),
    title=dict(font=dict(size=15, color=TEXT, family=CHART_FONT), x=0.01, xanchor="left"),
    colorway=[INDIGO, TEAL, ORANGE, PINK, BLUE, VIOLET, EMERALD, ROSE],
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickcolor=BORDER, showspikes=False),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickcolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SOFT), traceorder="normal"),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=TEXT, family=CHART_FONT)),
    margin=dict(l=12, r=12, t=52, b=12),
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