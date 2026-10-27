"""Design system for the whole app.

Everything visual lives here so every page looks the same:
  * colour tokens (one palette for the app, assets and strategies)
  * the global CSS (sidebar, buttons, tabs, tables, cards)
  * HTML building blocks: hero banner, section headers, KPI cards, chips
  * one Plotly chart theme + a `chart()` helper that applies it
  * Indian number formatting (₹1,00,000 / ₹3.45 L / ₹1.20 Cr)
"""
import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# =========================================================
# Colour tokens
# =========================================================
BG = "#0A0F1E"          # page background
SURFACE = "#121A2E"     # cards / panels
SURFACE_2 = "#182240"   # raised elements, hover
BORDER = "#232D4A"      # hairlines
TEXT = "#E6EAF5"        # main text
MUTED = "#8A94B3"       # secondary text
FAINT = "#5B6584"       # tertiary text, reference lines

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

# Page gradients for the hero banners (each page has its own colour identity)
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


# =========================================================
# Global CSS
# =========================================================
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

:root {{
  --bg:{BG}; --surface:{SURFACE}; --surface2:{SURFACE_2}; --border:{BORDER};
  --text:{TEXT}; --muted:{MUTED}; --faint:{FAINT};
  --good:{GOOD}; --bad:{BAD}; --good-t:{GOOD_TEXT}; --bad-t:{BAD_TEXT};
}}

html, body, [class*="css"], .stApp, button, input, textarea, select {{
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
}}

/* ---------- page background: soft colour glows ---------- */
.stApp {{
  background:
    radial-gradient(900px 500px at 8% -10%, rgba(139,92,246,0.16), transparent 60%),
    radial-gradient(800px 480px at 100% 0%, rgba(34,211,238,0.10), transparent 60%),
    radial-gradient(700px 500px at 50% 110%, rgba(236,72,153,0.07), transparent 60%),
    var(--bg);
  background-attachment: fixed;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container, [data-testid="stMainBlockContainer"] {{
  padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1480px;
}}

/* ---------- headings ---------- */
h1, h2, h3, h4, h5 {{ letter-spacing: -0.01em; }}
[data-testid="stCaptionContainer"] {{ color: var(--muted); }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0F1730 0%, #0B1122 100%);
  border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] .block-container,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top: 1rem; }}
.brand {{
  display:flex; align-items:center; gap:12px; padding: 6px 4px 18px 4px;
  border-bottom: 1px solid var(--border); margin-bottom: 14px;
}}
.brand-logo {{
  width:42px; height:42px; border-radius:12px; display:grid; place-items:center; font-size:22px;
  background: linear-gradient(135deg, #8B5CF6, #3B82F6 55%, #22D3EE);
  box-shadow: 0 8px 24px rgba(99,102,241,0.35);
}}
.brand-name {{ font-weight:800; font-size:1.02rem; color:var(--text); line-height:1.15; }}
.brand-sub {{ font-size:0.74rem; color:var(--muted); letter-spacing:0.04em; }}
.side-label {{
  font-size:0.7rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase;
  color:var(--faint); margin: 14px 2px 8px 2px;
}}
section[data-testid="stSidebar"] hr {{ margin: 14px 0; border-color: var(--border); }}

/* sidebar nav buttons */
section[data-testid="stSidebar"] .stButton > button {{
  justify-content:flex-start; text-align:left; padding: 0.55rem 0.85rem;
  border-radius: 12px; font-weight:600; font-size:0.92rem; min-height: 2.6rem;
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
  background: rgba(139,92,246,0.10); border-color: rgba(139,92,246,0.30); color: #fff;
}}

/* ---------- buttons ---------- */
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] button, [data-testid="stBaseButton-primaryFormSubmit"] {{
  background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 50%, #3B82F6 100%) !important;
  border: 0 !important; color: #fff !important; font-weight: 700 !important;
  box-shadow: 0 8px 22px rgba(99,102,241,0.30);
  transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover,
[data-testid="stFormSubmitButton"] button:hover {{
  transform: translateY(-1px); filter: brightness(1.08);
  box-shadow: 0 12px 28px rgba(99,102,241,0.42);
}}
[data-testid="stDownloadButton"] button {{
  border: 1px solid rgba(34,211,238,0.35); background: rgba(34,211,238,0.08); color: #A5F3FC;
  font-weight: 600; border-radius: 10px;
}}
[data-testid="stDownloadButton"] button:hover {{ background: rgba(34,211,238,0.16); color:#fff; }}

/* ---------- tabs as pills (works with old baseweb and new react-aria tabs) ---------- */
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {{
  gap: 6px; background: var(--surface); padding: 6px; border-radius: 14px;
  border: 1px solid var(--border); overflow-x: auto; width: fit-content; max-width: 100%;
}}
.stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"] {{
  height: auto; padding: 8px 16px; border-radius: 10px; color: var(--muted);
  font-weight: 600; background: transparent; white-space: nowrap; border: 0; margin: 0;
  transition: background .15s ease, color .15s ease;
}}
.stTabs [data-testid="stTab"] p {{ font-weight: 600; font-size: 0.92rem; }}
.stTabs [data-baseweb="tab"]:hover, .stTabs [data-testid="stTab"]:hover {{
  color: var(--text); background: rgba(255,255,255,0.05);
}}
.stTabs [role="tab"][aria-selected="true"] {{
  background: linear-gradient(135deg, #8B5CF6, #3B82F6) !important;
  color: #fff !important; box-shadow: 0 6px 18px rgba(99,102,241,0.32);
}}
.stTabs [role="tab"][aria-selected="true"] p {{ color: #fff !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"],
.stTabs .react-aria-SelectionIndicator {{ display: none !important; }}
.stTabs [data-baseweb="tab-panel"], .stTabs [role="tabpanel"] {{ padding-top: 1.1rem; }}

/* ---------- bordered containers = panels ---------- */
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stForm"] {{
  background: linear-gradient(180deg, rgba(24,34,64,0.55), rgba(18,26,46,0.55));
  border-radius: 18px !important;
}}
[data-testid="stForm"] {{ border: 1px solid var(--border) !important; padding: 1.2rem 1.3rem; }}

/* ---------- tables ---------- */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
}}

/* ---------- expander / alerts ---------- */
[data-testid="stExpander"] details {{ border-radius: 12px; border-color: var(--border); background: var(--surface); }}
[data-testid="stAlert"] {{ border-radius: 12px; }}

/* ---------- radio as segmented pills ---------- */
[data-testid="stRadio"] div[role="radiogroup"] {{ gap: 8px; flex-wrap: wrap; }}
[data-testid="stRadio"] div[role="radiogroup"] > label,
[data-testid="stRadio"] [data-testid="stRadioOption"] {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 999px;
  padding: 7px 15px 7px 11px; margin: 0; transition: all .15s ease;
}}
[data-testid="stRadio"] div[role="radiogroup"] > label:hover,
[data-testid="stRadio"] [data-testid="stRadioOption"]:hover {{ border-color: rgba(139,92,246,0.55); }}
[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"] {{
  background: linear-gradient(135deg, rgba(139,92,246,0.22), rgba(59,130,246,0.18));
  border-color: rgba(139,92,246,0.85); box-shadow: 0 0 0 3px rgba(139,92,246,0.14);
}}
[data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"] p {{ color: #fff; font-weight: 700; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label,
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] {{ padding: 4px 12px 4px 8px; }}

/* ================= custom components ================= */

/* hero banner */
.hero {{
  position: relative; overflow: hidden; border-radius: 22px; padding: 28px 30px 24px 30px;
  margin-bottom: 1.6rem; color: #fff;
  background: linear-gradient(125deg, var(--g1) 0%, var(--g2) 55%, var(--g3) 100%);
  box-shadow: 0 18px 50px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.15);
}}
.hero::before, .hero::after {{
  content:""; position:absolute; border-radius:50%; pointer-events:none;
  background: radial-gradient(circle, rgba(255,255,255,0.22), rgba(255,255,255,0) 70%);
}}
.hero::before {{ width:340px; height:340px; right:-90px; top:-150px; }}
.hero::after  {{ width:220px; height:220px; right:180px; bottom:-140px; opacity:.6; }}
.hero-eyebrow {{
  display:inline-flex; align-items:center; gap:8px; font-size:0.72rem; font-weight:700;
  letter-spacing:0.14em; text-transform:uppercase; color: rgba(255,255,255,0.85);
  background: rgba(255,255,255,0.14); padding: 5px 11px; border-radius: 999px; margin-bottom: 12px;
}}
.hero-title {{ font-size: 2.05rem; font-weight: 800; line-height: 1.15; letter-spacing:-0.02em; margin:0; }}
.hero-sub {{ font-size: 1rem; color: rgba(255,255,255,0.84); margin-top: 8px; max-width: 900px; line-height:1.5; }}
.hero-chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top: 16px; position:relative; z-index:1; }}
.hero-chip {{
  font-size:0.8rem; font-weight:600; padding:6px 12px; border-radius:999px;
  background: rgba(10,15,30,0.28); border: 1px solid rgba(255,255,255,0.22); color:#fff;
  backdrop-filter: blur(6px);
}}

/* section header */
.sec {{ display:flex; align-items:center; gap:12px; margin: 1.8rem 0 0.9rem 0; }}
.sec-icon {{
  width:38px; height:38px; flex: 0 0 38px; border-radius:11px; display:grid; place-items:center; font-size:18px;
  background: color-mix(in srgb, var(--c) 18%, transparent);
  border: 1px solid color-mix(in srgb, var(--c) 45%, transparent);
}}
.sec-title {{ font-size:1.2rem; font-weight:800; color:var(--text); line-height:1.2; }}
.sec-sub {{ font-size:0.86rem; color:var(--muted); margin-top:2px; }}
.sec-bar {{ flex:1; height:1px; margin-left:8px;
  background: linear-gradient(90deg, color-mix(in srgb, var(--c) 55%, transparent), transparent); }}

/* KPI card */
.kpi {{
  position:relative; overflow:hidden; border-radius:16px; padding:15px 17px 14px 17px;
  min-height: 122px; height:100%;
  background: linear-gradient(160deg, color-mix(in srgb, var(--a) 16%, {SURFACE}) 0%, {SURFACE} 62%);
  border: 1px solid color-mix(in srgb, var(--a) 30%, {BORDER});
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  margin-bottom: 4px;
}}
.kpi::before {{
  content:""; position:absolute; left:0; top:0; right:0; height:3px;
  background: linear-gradient(90deg, var(--a), color-mix(in srgb, var(--a) 20%, transparent));
}}
.kpi:hover {{
  transform: translateY(-2px); border-color: color-mix(in srgb, var(--a) 60%, {BORDER});
  box-shadow: 0 12px 30px color-mix(in srgb, var(--a) 18%, transparent);
}}
.kpi-top {{ display:flex; align-items:center; gap:9px; margin-bottom:10px; min-width:0; }}
.kpi-icon {{ flex: 0 0 28px; }}
.kpi-icon {{
  width:28px; height:28px; border-radius:8px; display:grid; place-items:center; font-size:14px;
  background: color-mix(in srgb, var(--a) 22%, transparent);
}}
.kpi-label {{ font-size:0.72rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; color:var(--muted);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0; }}
.kpi-value {{
  font-size:1.55rem; font-weight:800; color:#fff; letter-spacing:-0.02em; line-height:1.15;
  font-variant-numeric: tabular-nums; word-break: break-word;
}}
.kpi-delta {{
  display:inline-flex; align-items:center; gap:5px; margin-top:9px; font-size:0.8rem; font-weight:700;
  padding:3px 9px; border-radius:999px; font-variant-numeric: tabular-nums;
}}
.kpi-delta.good {{ color: var(--good-t); background: rgba(34,197,94,0.13); }}
.kpi-delta.bad {{ color: var(--bad-t); background: rgba(244,63,94,0.13); }}
.kpi-delta.neutral {{ color: #AEB8D6; background: rgba(138,148,179,0.13); font-weight:600; }}

/* chips & legend */
.chip {{
  display:inline-flex; align-items:center; gap:7px; font-size:0.8rem; font-weight:600; color:var(--text);
  padding:5px 11px; border-radius:999px; margin: 0 6px 6px 0;
  background: color-mix(in srgb, var(--c) 14%, transparent);
  border: 1px solid color-mix(in srgb, var(--c) 40%, transparent);
}}
.chip-dot {{ width:8px; height:8px; border-radius:50%; background: var(--c); box-shadow: 0 0 8px var(--c); }}

/* info strip */
.note {{
  display:flex; gap:10px; align-items:flex-start; padding: 11px 14px; border-radius: 12px; margin: 6px 0 12px 0;
  background: color-mix(in srgb, var(--c) 9%, transparent);
  border: 1px solid color-mix(in srgb, var(--c) 28%, transparent); color: #CDD4EA; font-size: 0.9rem; line-height:1.5;
}}
.note b {{ color:#fff; }}

/* strategy explainer card */
.explain {{
  border-radius:14px; padding: 14px 16px; margin: 4px 0 6px 0; font-size: 0.92rem; line-height:1.55; color:#CDD4EA;
  background: linear-gradient(135deg, color-mix(in srgb, var(--c) 12%, transparent), transparent 80%);
  border-left: 4px solid var(--c);
}}

/* weight total badge */
.wbadge {{ display:inline-block; font-weight:800; padding:4px 12px; border-radius:999px; font-size:0.85rem; }}
.wbadge.ok {{ color: var(--good-t); background: rgba(34,197,94,0.14); }}
.wbadge.off {{ color: #FCD34D; background: rgba(245,158,11,0.14); }}

@media (max-width: 640px) {{
  .hero {{ padding: 22px 18px; }}
  .hero-title {{ font-size: 1.5rem; }}
  .kpi-value {{ font-size: 1.3rem; }}
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


def hero(title: str, subtitle: str = "", page: str = "dashboard", eyebrow: str = "", chips=None):
    g1, g2, g3 = GRADIENTS.get(page, GRADIENTS["dashboard"])
    icon = PAGES.get(page, ("📊",))[0]
    chips_html = ""
    if chips:
        chips_html = '<div class="hero-chips">' + "".join(
            f'<span class="hero-chip">{_e(c)}</span>' for c in chips if c) + "</div>"
    eb = f'<div class="hero-eyebrow">{icon} {_e(eyebrow)}</div>' if eyebrow else ""
    sub = f'<div class="hero-sub">{_e(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="hero" style="--g1:{g1};--g2:{g2};--g3:{g3}">{eb}'
        f'<div class="hero-title">{_e(title)}</div>{sub}{chips_html}</div>',
        unsafe_allow_html=True)


def section(title: str, subtitle: str = "", icon: str = "◆", color: str = VIOLET):
    sub = f'<div class="sec-sub">{_e(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="sec" style="--c:{color}"><div class="sec-icon">{icon}</div>'
        f'<div><div class="sec-title">{_e(title)}</div>{sub}</div><div class="sec-bar"></div></div>',
        unsafe_allow_html=True)


def kpi(label: str, value: str, delta: str = None, tone: str = "neutral", icon: str = "", accent: str = VIOLET):
    """A coloured KPI card. tone: 'good' | 'bad' | 'neutral'."""
    arrow = {"good": "▲ ", "bad": "▼ "}.get(tone, "")
    d = f'<div class="kpi-delta {tone}">{arrow}{_e(delta)}</div>' if delta else ""
    ic = f'<span class="kpi-icon">{icon}</span>' if icon else ""
    st.markdown(
        f'<div class="kpi" style="--a:{accent}"><div class="kpi-top">{ic}'
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
# Plotly chart theme
# =========================================================
CHART_FONT = "Plus Jakarta Sans, system-ui, sans-serif"
GRID = "rgba(138,148,179,0.11)"

_template = go.layout.Template(pio.templates["plotly_dark"])
_template.layout.update(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=CHART_FONT, color="#C7CEE3", size=12),
    title=dict(font=dict(size=15, color=TEXT, family=CHART_FONT), x=0.01, xanchor="left"),
    colorway=[VIOLET, CYAN, AMBER, PINK, EMERALD, BLUE, ORANGE, ROSE],
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickcolor=BORDER, showspikes=False),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickcolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#C7CEE3"), traceorder="normal"),
    hoverlabel=dict(bgcolor=SURFACE_2, bordercolor="#33406A", font=dict(color=TEXT, family=CHART_FONT)),
    margin=dict(l=12, r=12, t=56, b=12),
)
pio.templates["lab"] = _template

PLOT_LAYOUT = dict(template="lab")


def chart(fig, key=None, height=None):
    """Show a Plotly figure with the app theme (Streamlit's own theme would override ours)."""
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, theme=None, width="stretch", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"