"""⛰️ MF Recovery page: when a mutual fund falls, how long does it take to get back to its old high?

Three parts, top to bottom:
  1. "How to read" strip: Top → Bottom → Back at the top
  2. ONE FUND: pick a fund and one of its falls → story cards + a big annotated NAV chart
     + a timeline of all its big falls (red = falling, green = climbing back)
  3. ALL FUNDS on the SAME crash (2008, COVID 2020, ...): who got back to the top fastest?

All the maths is in src/mf_recovery.py (plain pandas, every step explained there).
"""
import html
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import (panel, hero, note, section, kpi_row, chart, steps, sidebar_nav, PLOT_LAYOUT,
                    BAD, GOOD, AMBER, INDIGO, TEAL, ORANGE, VIOLET, EMERALD, FAINT, hex_rgba)
from src.mf_page import load_mf_history, MF_SHEET_LINK, CATEGORIES, CAT_COLORS
from src.mf_recovery import daily_steps, falls, as_years_months, CRASHES, fall_in_window

FALL_C, RISE_C, TOP_C = "#DC2626", "#16A34A", "#2563EB"


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%d %b %Y")


def _fall_label(r) -> str:
    back = as_years_months(r["Days bottom → top"]) + ("" if r["Recovered"] else " so far, not back yet")
    return f"{_d(r['Bottom date'])[3:]}: fell {abs(r['Fall (%)']):.0f}% · back to the top in {back}"


@st.cache_data(show_spinner=False, max_entries=64)
def _fund_data(full: pd.DataFrame, fund: str, nav_col: str):
    nav = full[full["Fund"] == fund].set_index("Date")[nav_col]
    return daily_steps(nav), falls(nav, 0.0)


# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------
def _story_chart(s: pd.DataFrame, r, fund: str):
    """NAV line with ONE fall drawn on it: red zone = falling, green zone = climbing back."""
    top_d, bot_d = r["Top date"], r["Bottom date"]
    end_d = r["Recovery date"] if r["Recovered"] else s["Date"].iloc[-1]
    pad = max(pd.Timedelta(days=120), (end_d - top_d) * 0.6)
    w = s[(s["Date"] >= top_d - pad) & (s["Date"] <= end_d + pad)]
    fig = go.Figure()
    fig.add_vrect(x0=top_d, x1=bot_d, fillcolor=hex_rgba(FALL_C, 0.10), line_width=0, layer="below",
                  annotation_text=f"<b>FALLING</b><br>{as_years_months(r['Days top → bottom'])}",
                  annotation_position="top", annotation_font=dict(color=FALL_C, size=13))
    fig.add_vrect(x0=bot_d, x1=end_d, fillcolor=hex_rgba(RISE_C, 0.10), line_width=0, layer="below",
                  annotation_text=(f"<b>CLIMBING BACK</b><br>{as_years_months(r['Days bottom → top'])}"
                                   + ("" if r["Recovered"] else " so far")),
                  annotation_position="top", annotation_font=dict(color=RISE_C, size=13))
    fig.add_trace(go.Scatter(x=w["Date"], y=w["NAV"], mode="lines", name="NAV", line=dict(color=INDIGO, width=2.4),
                             hovertemplate="%{x|%d %b %Y}<br>NAV ₹%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[top_d, end_d], y=[r["Top NAV"]] * 2, mode="lines", name="Old high to beat",
                             line=dict(color=TOP_C, width=1.6, dash="dash"), hoverinfo="skip"))
    pts = [(top_d, r["Top NAV"], TOP_C, "triangle-up", f"<b>TOP</b><br>₹{r['Top NAV']:,.2f}<br>{_d(top_d)}", "top center"),
           (bot_d, r["Bottom NAV"], FALL_C, "triangle-down",
            f"<b>BOTTOM</b> ({r['Fall (%)']:.0f}%)<br>₹{r['Bottom NAV']:,.2f}<br>{_d(bot_d)}", "bottom center")]
    if r["Recovered"]:
        pts.append((end_d, r["Top NAV"], RISE_C, "star", f"<b>BACK AT THE TOP</b><br>{_d(end_d)}", "top center"))
    for x, y, c, sym, txt, pos in pts:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", marker=dict(color=c, size=15, symbol=sym,
                                 line=dict(color="white", width=1.5)), text=[txt], textposition=pos,
                                 textfont=dict(color=c, size=12), showlegend=False, hoverinfo="skip"))
    lo, hi = float(w["NAV"].min()), float(w["NAV"].max())
    fig.update_layout(**PLOT_LAYOUT, height=520, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      yaxis=dict(title="NAV (₹)", tickprefix="₹", range=[lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.25]),
                      margin=dict(l=12, r=12, t=40, b=12))
    chart(fig, key="rec_story")


def _timeline(f: pd.DataFrame, last_day, key: str):
    """Every big fall of one fund as a bar: red part = falling, green part = climbing back."""
    f = f.sort_values("Top date")
    y = [f"{_d(t)[3:]}  ({x:.0f}%)" for t, x in zip(f["Top date"], f["Fall (%)"])]
    ms = 86_400_000
    end = [rd if ok else pd.Timestamp(last_day) for rd, ok in zip(f["Recovery date"], f["Recovered"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=y, x=(f["Bottom date"] - f["Top date"]).dt.days * ms, base=f["Top date"], orientation="h",
                         name="Falling (top → bottom)", marker_color=FALL_C,
                         customdata=np.stack([f["Days top → bottom"], f["Fall (%)"].round(1)], axis=1),
                         hovertemplate="Falling %{customdata[0]} days, %{customdata[1]}%<extra></extra>"))
    fig.add_trace(go.Bar(y=y, x=[(e - b).days * ms for e, b in zip(end, f["Bottom date"])], base=f["Bottom date"],
                         orientation="h", name="Climbing back (bottom → top)",
                         marker=dict(color=RISE_C, pattern_shape=["" if ok else "/" for ok in f["Recovered"]]),
                         text=[as_years_months(d) + ("" if ok else " (not yet)")
                               for d, ok in zip(f["Days bottom → top"], f["Recovered"])],
                         textposition="outside", cliponaxis=False,
                         customdata=f["Days bottom → top"],
                         hovertemplate="Climbing back %{customdata} days<extra></extra>"))
    fig.update_layout(**PLOT_LAYOUT, barmode="overlay", height=max(260, 46 * len(f) + 100),
                      xaxis=dict(type="date"), yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(l=12, r=80, t=40, b=12))
    chart(fig, key=key)


def _race(t: pd.DataFrame, key: str):
    """All funds, one crash: stacked bar of days falling (red) + days climbing back (green). Shortest on top."""
    t = t.sort_values("Days bottom → top", ascending=False)
    names = [f"{'🔵🟠🟢'[CATEGORIES.index(c)] if c in CATEGORIES else ''} {f}" for f, c in zip(t["Fund"], t["Category"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=t["Days top → bottom"], orientation="h", name="Falling", marker_color=FALL_C,
                         text=[f"{x:.0f}%" for x in t["Fall (%)"]], textposition="inside", insidetextanchor="middle",
                         hovertemplate="%{y}<br>Falling %{x} days<extra></extra>"))
    fig.add_trace(go.Bar(y=names, x=t["Days bottom → top"], orientation="h", name="Climbing back to the top",
                         marker=dict(color=RISE_C, pattern_shape=["" if ok else "/" for ok in t["Recovered"]]),
                         text=[as_years_months(d) + ("" if ok else " (not yet)")
                               for d, ok in zip(t["Days bottom → top"], t["Recovered"])],
                         textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}<br>Climbing back %{x} days<extra></extra>"))
    fig.update_layout(**PLOT_LAYOUT, barmode="stack", height=max(300, 36 * len(t) + 100),
                      xaxis=dict(title="Days"), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      margin=dict(l=12, r=110, t=40, b=12))
    chart(fig, key=key)


# ---------------------------------------------------------
# Page
# ---------------------------------------------------------
def render_mf_recovery_page(asset_dfs, fetch_errors):
    sidebar_nav("recovery")
    st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],"
                "[data-testid='stExpandSidebarButton']{display:none !important}</style>", unsafe_allow_html=True)
    full, source, err = load_mf_history()
    hero("MF Recovery", "When a mutual fund falls, how long does it take to get back to its old high?",
         page="recovery", eyebrow="Mutual funds · bottom → top")
    if full.empty:
        note(f"Couldn't load the <a href='{MF_SHEET_LINK}' target='_blank'>MF_NAV_History sheet</a> "
             f"({html.escape(str(err))}).", BAD, "⛔")
        st.stop()
    if source == "saved":
        note("Google Sheets couldn't be reached; showing the last copy downloaded.", AMBER, "⚠️")

    steps([
        ("🔺", "Top", "The fund's highest NAV before it started falling.", TOP_C),
        ("🔻", "Bottom", "The lowest NAV it reached during the fall.", FALL_C),
        ("⭐", "Back at the top", "The first day the NAV is at the old high again.", RISE_C),
        ("⏱️", "Recovery time", "Days from the Bottom to Back at the top. This is the number shown everywhere here.", VIOLET),
    ])

    with panel():
        c1, c2, c3 = st.columns([1, 2.2, 1], vertical_alignment="bottom")
        cat = c1.selectbox("Category", CATEGORIES, key="rec_cat")
        pool = sorted(full.loc[full["Group"] == cat, "Fund"].unique())
        if st.session_state.get("rec_fund") not in pool:
            st.session_state["rec_fund"] = pool[0]
        fund = c2.selectbox("Fund", pool, key="rec_fund")
        plan = c3.segmented_control("Plan", ["Regular", "Direct"], default="Regular", key="rec_plan") or "Regular"
    nav_col = "NAV_Regular" if plan == "Regular" else "NAV_Direct"

    # =========================== 1. ONE FUND ===========================
    s, f_all = _fund_data(full, fund, nav_col)
    big = f_all[f_all["Fall (%)"] <= -10].reset_index(drop=True)
    if s.empty or big.empty:
        note(f"{fund} has no fall of 10% or more in its {plan} NAV history.", AMBER, "💡")
        return
    section(f"{fund}", f"{plan} plan · NAV from {_d(s['Date'].iloc[0])} to {_d(s['Date'].iloc[-1])} · "
                       f"{len(big)} falls of 10% or more", "📉", CAT_COLORS.get(cat, INDIGO))
    order = big.sort_values("Fall (%)").index.tolist()                    # deepest first
    with panel():
        pick = st.selectbox("Pick a fall (deepest first)", order, key=f"rec_pick_{fund}_{plan}",
                            format_func=lambda i: _fall_label(big.loc[i]))
    r = big.loc[pick]
    kpi_row([
        dict(label="It fell", value=f"{r['Fall (%)']:.1f}%", delta=f"{_d(r['Top date'])} → {_d(r['Bottom date'])}",
             tone="bad", icon="🔻", accent=FALL_C),
        dict(label="Time falling", value=as_years_months(r["Days top → bottom"]),
             delta=f"{r['Days top → bottom']:,} days, top → bottom", icon="⬇️", accent=ORANGE),
        dict(label="Time to get back to the top", value=as_years_months(r["Days bottom → top"]) + ("" if r["Recovered"] else " +"),
             delta=(f"{r['Days bottom → top']:,} days, back on {_d(r['Recovery date'])}" if r["Recovered"]
                    else f"{r['Days bottom → top']:,} days so far, not back yet"),
             tone="good" if r["Recovered"] else "bad", icon="⏱️", accent=RISE_C),
        dict(label="Gain needed from the bottom", value=f"+{r['Gain needed (%)']:.0f}%",
             delta=f"₹{r['Bottom NAV']:,.2f} → ₹{r['Top NAV']:,.2f}", icon="📈", accent=VIOLET),
    ])
    note(f"<b>In short:</b> {fund} was at <b>₹{r['Top NAV']:,.2f}</b> on {_d(r['Top date'])}. It fell "
         f"<b>{abs(r['Fall (%)']):.1f}%</b> to <b>₹{r['Bottom NAV']:,.2f}</b> by {_d(r['Bottom date'])} "
         f"({r['Days top → bottom']:,} days). "
         + (f"Then it took <b>{as_years_months(r['Days bottom → top'])}</b> ({r['Days bottom → top']:,} days) to climb "
            f"back to ₹{r['Top NAV']:,.2f}, on <b>{_d(r['Recovery date'])}</b>." if r["Recovered"] else
            f"It has been climbing for <b>{as_years_months(r['Days bottom → top'])}</b> and is <b>not back</b> at "
            f"₹{r['Top NAV']:,.2f} yet."), TEAL, "📖")
    with panel():
        _story_chart(s, r, fund)

    st.markdown("#### Every fall of 10% or more")
    with panel():
        _timeline(big, s["Date"].iloc[-1], key="rec_timeline")
    st.caption("Each row is one fall. 🟥 Red = falling from the top to the bottom. 🟩 Green = climbing back to the top "
               "(the number at the end is the recovery time). Striped = still not back.")

    # =========================== 2. ALL FUNDS, SAME CRASH ===========================
    section("All funds, same crash", "Who got back to the top fastest?", "🏁", INDIGO)
    with panel():
        crash = st.segmented_control("Crash", list(CRASHES), default="COVID crash 2020", key="rec_crash") \
            or "COVID crash 2020"
    a, b = CRASHES[crash]
    rows, missing = [], []
    for (grp, fd), _g in full.groupby(["Group", "Fund"]):
        _, ff = _fund_data(full, fd, nav_col)
        x = fall_in_window(ff, a, b)
        if x is None or x["Fall (%)"] > -5:
            missing.append(fd)
            continue
        rows.append({"Category": grp, "Fund": fd, **x.to_dict()})
    t = pd.DataFrame(rows)
    if t.empty:
        note(f"No fund has a fall of 5% or more with its bottom in {crash} ({plan} plan).", AMBER, "💡")
        return
    done = t[t["Recovered"]]
    fast = done.loc[done["Days bottom → top"].idxmin()] if len(done) else None
    slow = t.loc[t["Days bottom → top"].idxmax()]
    kpi_row([
        dict(label="Fastest back to the top", value=as_years_months(fast["Days bottom → top"]) if fast is not None else "—",
             delta=fast["Fund"][:40] if fast is not None else "none yet", tone="good", icon="🚀", accent=EMERALD),
        dict(label="Slowest", value=as_years_months(slow["Days bottom → top"]) + ("" if slow["Recovered"] else " +"),
             delta=slow["Fund"][:40], tone="bad", icon="🐢", accent=BAD),
        dict(label="Average", value=as_years_months(done["Days bottom → top"].mean()) if len(done) else "—",
             delta=f"{len(done)} of {len(t)} funds back at the top", icon="📊", accent=INDIGO),
        dict(label="Average fall", value=f"{t['Fall (%)'].mean():.0f}%", delta=f"deepest {t['Fall (%)'].min():.0f}%",
             tone="bad", icon="📉", accent=ORANGE),
    ])
    with panel():
        _race(t, key=f"rec_race_{crash}")
    st.caption("Shortest bar at the top = got back fastest. 🟥 Red = falling (number inside = how much it fell). "
               "🟩 Green = climbing back to its old top (time written at the end). 🔵 Large · 🟠 Mid · 🟢 Small Cap."
               + (f" Not shown (no data or no fall in this crash): {', '.join(missing)}." if missing else ""))

    # =========================== 3. the numbers (hidden until asked) ===========================
    with st.expander("📋 See the numbers and download them"):
        view = pd.DataFrame({
            "Category": t["Category"], "Fund": t["Fund"],
            "Top date": t["Top date"].dt.date, "Top NAV": t["Top NAV"],
            "Bottom date": t["Bottom date"].dt.date, "Bottom NAV": t["Bottom NAV"],
            "Back at top": [d.date() if ok else None for d, ok in zip(t["Recovery date"], t["Recovered"])],
            "Fall %": t["Fall (%)"], "Days falling": t["Days top → bottom"],
            "Days bottom → top": t["Days bottom → top"],
            "Recovery time": [as_years_months(d) + ("" if ok else " (not yet)")
                              for d, ok in zip(t["Days bottom → top"], t["Recovered"])],
        }).sort_values("Days bottom → top")
        st.dataframe(view.style.format({"Top NAV": "₹{:,.2f}", "Bottom NAV": "₹{:,.2f}", "Fall %": "{:.1f}%"},
                                       na_rep="not yet"), hide_index=True, width="stretch",
                     column_config={c: st.column_config.DateColumn(format="DD-MM-YYYY")
                                    for c in ("Top date", "Bottom date", "Back at top")})
        safe = re.sub(r"[^A-Za-z0-9]+", "_", fund)
        c1, c2 = st.columns(2)
        c1.download_button(f"⬇️  {crash}: all funds (CSV)", view.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"recovery_{re.sub(r'[^A-Za-z0-9]+', '_', crash)}_{plan.lower()}.csv",
                           key="rec_dl_crash")
        c2.download_button(f"⬇️  {fund}: day-by-day working (CSV)", s.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"recovery_working_{safe}_{plan.lower()}.csv", key="rec_dl_steps")
        st.caption("Day-by-day CSV columns: Date · NAV · Running top (= highest NAV so far; Excel =MAX($B$2:B2)) · "
                   "Fall from top % (= NAV ÷ Running top − 1). Recovery time = first day the fall is back to 0% − "
                   "the day it was lowest. Calendar days.")