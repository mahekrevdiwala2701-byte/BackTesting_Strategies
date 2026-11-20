"""Excel export for the 💰 Calculated Investment page (no Streamlit code here).

Builds a workbook in which EVERY number is a live Excel formula that you can click and follow,
step by step, from the prices to the final result:

  All Steps            EVERYTHING on one sheet: inputs, result vs Buy & Hold, comparison, asset breakdown,
                       prices on every rebalance day, and every buy / rebalance worked out step by step
  Inputs               Amount, your weights (edit the blue cells → whole workbook recalculates)
  Prices               The daily closing price of each asset used by the calculation
  Weight Decisions     (Different Weights only) growth over the look-back → rank → new weight
  Steps - <Strategy>   Every buy / rebalance: units before, value before, target, units after, trade
  Daily - <Strategy>   Units, value of each asset, total, peak and drawdown for every day
  Steps / Daily - Buy & Hold   Same for Buy & Hold (used for the comparison)
  Comparison           Strategy vs Buy & Hold vs 100% in a single asset, plus a chart

Colour code: blue text = number typed in / taken from data, black = formula,
green = formula that links to another sheet, yellow fill = input you may change.
"""
import io
import math

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.series import SeriesLabel
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as L

# ---------------------------------------------------------
# Styles
# ---------------------------------------------------------
FONT = "Arial"
F_BASE = Font(name=FONT, size=10)
F_INPUT = Font(name=FONT, size=10, color="0000FF")
F_LINK = Font(name=FONT, size=10, color="008000")
F_BOLD = Font(name=FONT, size=10, bold=True)
F_HEAD = Font(name=FONT, size=10, bold=True, color="FFFFFF")
F_TITLE = Font(name=FONT, size=14, bold=True, color="1F3864")
F_SUB = Font(name=FONT, size=11, bold=True, color="1F3864")
F_NOTE = Font(name=FONT, size=9, italic=True, color="595959")

FILL_HEAD = PatternFill("solid", fgColor="1F3864")
FILL_BLOCK = PatternFill("solid", fgColor="D9E1F2")
FILL_TOTAL = PatternFill("solid", fgColor="F2F2F2")
FILL_INPUT = PatternFill("solid", fgColor="FFFF00")
FILL_GOOD = PatternFill("solid", fgColor="E2EFDA")

THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

MONEY = '"₹"#,##0.00;[Red]-"₹"#,##0.00;"-"'
PRICE = '"₹"#,##0.00'
UNITS = '#,##0.0000;[Red]-#,##0.0000;"-"'
PCT = '0.00%;[Red]-0.00%;"-"'
DATE = 'dd-mmm-yyyy'
NUM2 = '0.00'

ASSET_SHORT = {'Gold': 'GoldBEES', 'LargeCap': 'Nifty 50', 'MidCap': 'MidCap', 'SmallCap': 'SmallCap'}
STRAT_NAME = {"bh": "Buy & Hold", "periodic": "Same-Weight Rebalance", "diff": "Different Weights"}
STRAT_SHEET = {"bh": "Buy & Hold", "periodic": "Rebalance", "diff": "Diff Weights"}


def _q(sheet):
    """Sheet name ready for a formula: 'Steps - Rebalance'!"""
    return f"'{sheet}'!"


def _put(ws, row, col, value, fmt=None, font=F_BASE, fill=None, bold=False, border=True, align=None):
    c = ws.cell(row=row, column=col, value=value)
    c.font = Font(name=font.name, size=font.size, bold=bold or font.bold, italic=font.italic, color=font.color)
    if fmt:
        c.number_format = fmt
    if fill:
        c.fill = fill
    if border:
        c.border = BOX
    if align:
        c.alignment = Alignment(horizontal=align, vertical="center", wrap_text=True)
    return c


def _header(ws, row, col, labels, widths=None):
    for i, lab in enumerate(labels):
        c = _put(ws, row, col + i, lab, font=F_HEAD, fill=FILL_HEAD, align="center")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        if widths:
            ws.column_dimensions[L(col + i)].width = widths[i]
    lines = 2
    for i, lab in enumerate(labels):
        w = ws.column_dimensions[L(col + i)].width or 13
        lines = max(lines, math.ceil(len(str(lab or "")) * 1.15 / max(w - 1, 1)))
    ws.row_dimensions[row].height = 14 * lines + 2


def _title(ws, text, sub=None):
    ws["A1"] = text
    ws["A1"].font = F_TITLE
    if sub:
        ws["A2"] = sub
        ws["A2"].font = F_NOTE
    ws.sheet_view.showGridLines = False


def _note(ws, row, col, text):
    c = ws.cell(row=row, column=col, value=text)
    c.font = F_NOTE
    return c


def _link_font(formula):
    return F_LINK if "!" in str(formula) else F_BASE


# ---------------------------------------------------------
# Workbook builder
# ---------------------------------------------------------
class _Builder:
    def __init__(self, mode, prices, schedule, raw_weights, w0, amount, decisions,
                 freq_label, lookback_label, start_date, end_date):
        self.mode = mode
        self.prices = prices
        self.used = list(prices.columns)
        self.schedule = schedule
        self.raw_weights = raw_weights
        self.w0 = w0
        self.amount = amount
        self.decisions = decisions
        self.freq_label = freq_label
        self.lookback_label = lookback_label
        self.start_date, self.end_date = start_date, end_date
        self.wb = Workbook()
        self.n_days = len(prices)
        self.P0 = 5                                  # first data row on the Prices sheet
        self.PL = self.P0 + self.n_days - 1          # last data row on the Prices sheet
        self.metrics = {}                            # strategy key -> {metric: cell ref}

    # ---------- shared references ----------
    def price_ref(self, date, asset):
        r = self.P0 + self.prices.index.get_loc(date)
        return f"{_q('Prices')}${self.pcol[asset]}${r}"

    def weight_ref(self, asset):
        return self.inp_w[asset]

    # ---------- Inputs ----------
    def build_inputs(self):
        ws = self.wb.active
        ws.title = "Inputs"
        _title(ws, "Inputs", "Yellow cells can be changed: every other sheet recalculates from them.")
        ws.column_dimensions["A"].width = 34
        for c in "BCDE":
            ws.column_dimensions[c].width = 18

        _put(ws, 4, 1, "Total amount invested (₹)", bold=True)
        _put(ws, 4, 2, float(self.amount), MONEY, F_INPUT, FILL_INPUT)
        self.amt_ref = f"{_q('Inputs')}$B$4"
        _put(ws, 5, 1, "Strategy", bold=True)
        _put(ws, 5, 2, STRAT_NAME[self.mode])
        _put(ws, 6, 1, "Period selected", bold=True)
        _put(ws, 6, 2, pd.Timestamp(self.start_date).to_pydatetime(), DATE)
        _put(ws, 6, 3, pd.Timestamp(self.end_date).to_pydatetime(), DATE)
        _put(ws, 7, 1, "First trading day used", bold=True)
        _put(ws, 7, 2, f"=Prices!A{self.P0}", DATE, F_LINK)
        _put(ws, 8, 1, "Last trading day used", bold=True)
        _put(ws, 8, 2, f"=Prices!A{self.PL}", DATE, F_LINK)
        r = 9
        if self.mode != "bh":
            _put(ws, r, 1, "Rebalance every", bold=True)
            _put(ws, r, 2, self.freq_label)
            r += 1
        if self.mode == "diff":
            _put(ws, r, 1, "Compare growth over the previous", bold=True)
            _put(ws, r, 2, self.lookback_label)
            r += 1

        r += 1
        _put(ws, r, 1, "Step 1 · Normalise your weights", font=F_SUB, border=False)
        r += 1
        _header(ws, r, 1, ["Asset", "Your weight (typed)", "Weight used = typed ÷ total", "Money for this asset (₹)"])
        first = r + 1
        last = r + len(self.used)
        self.inp_w = {}
        for i, n in enumerate(self.used):
            rr = first + i
            _put(ws, rr, 1, ASSET_SHORT[n])
            _put(ws, rr, 2, self.raw_weights[n] / 100, PCT, F_INPUT, FILL_INPUT)
            _put(ws, rr, 3, f"=B{rr}/$B${last + 1}", PCT)
            _put(ws, rr, 4, f"=$B$4*C{rr}", MONEY)
            self.inp_w[n] = f"{_q('Inputs')}$C${rr}"
        _put(ws, last + 1, 1, "TOTAL", bold=True, fill=FILL_TOTAL)
        _put(ws, last + 1, 2, f"=SUM(B{first}:B{last})", PCT, bold=True, fill=FILL_TOTAL)
        _put(ws, last + 1, 3, f"=SUM(C{first}:C{last})", PCT, bold=True, fill=FILL_TOTAL)
        _put(ws, last + 1, 4, f"=SUM(D{first}:D{last})", MONEY, bold=True, fill=FILL_TOTAL)
        _note(ws, last + 2, 1, "If your typed weights don't add up to 100%, they are scaled so they do.")

        r = last + 4
        _put(ws, r, 1, "Colour code", font=F_SUB, border=False)
        legend = [("Blue text", "Number typed in, or taken from the price data", F_INPUT, None),
                  ("Black text", "Formula on the same sheet", F_BASE, None),
                  ("Green text", "Formula that pulls a value from another sheet", F_LINK, None),
                  ("Yellow fill", "Input you may change", F_BASE, FILL_INPUT)]
        for i, (a, b, f, fill) in enumerate(legend):
            _put(ws, r + 1 + i, 1, a, font=f, fill=fill)
            _put(ws, r + 1 + i, 2, b, border=False)
        if self.mode == "diff":
            _note(ws, r + 6, 1, "Note: dates of the rebalances and the look-back prices are fixed values from the "
                                "app; amount and weights are fully live.")

    # ---------- Prices ----------
    def build_prices(self):
        ws = self.wb.create_sheet("Prices")
        _title(ws, "Prices used (daily close, ₹)",
               "If an asset has no price on a day, its last known price is used (carried forward). "
               "Blank = the asset's data has not started yet.")
        cols = ["Date"] + [ASSET_SHORT[n] for n in self.used]
        _header(ws, 4, 1, cols, [14] + [14] * len(self.used))
        self.pcol = {n: L(2 + i) for i, n in enumerate(self.used)}
        for i, (dt, row) in enumerate(self.prices.iterrows()):
            r = self.P0 + i
            c = ws.cell(row=r, column=1, value=dt.to_pydatetime())
            c.number_format, c.font = DATE, F_INPUT
            for j, n in enumerate(self.used):
                v = row[n]
                c = ws.cell(row=r, column=2 + j, value=None if pd.isna(v) else float(v))
                c.number_format, c.font = PRICE, F_INPUT
        ws.freeze_panes = "B5"

    # ---------- Weight decisions (Different Weights) ----------
    def build_decisions(self):
        ws = self.wb.create_sheet("Weight Decisions")
        _title(ws, f"How the weights were decided (growth over the previous {self.lookback_label})",
               "Growth = Price now ÷ Price then − 1.  Rank 1 = highest growth.  The HIGHEST growth gets the "
               "SMALLEST of your weights, the LOWEST growth the BIGGEST.  No older price → keeps your weight.")
        heads = ["Rebalance #", "Date", "Asset", "Your weight", "Compare from", "Price then (₹)", "Price now (₹)",
                 "Growth", "Growth rank (1 = highest)", "Rank low→high", "Weight in pool", "New weight"]
        widths = [11, 13, 12, 11, 13, 13, 13, 11, 13, 11, 11, 12]
        _header(ws, 4, 1, heads, widths)
        self.new_w = {}                  # (k, asset) -> cell
        r = 5
        dec = self.decisions
        for k, (d, _) in enumerate(self.schedule):
            block = dec[dec['Rebalance #'] == k + 1]
            first, last = r, r + len(self.used) - 1
            for _, row in block.iterrows():
                n = row['Asset']
                fill = FILL_BLOCK if k % 2 == 0 else None
                _put(ws, r, 1, k + 1, fill=fill, align="center")
                _put(ws, r, 2, pd.Timestamp(d).to_pydatetime(), DATE, F_INPUT, fill)
                _put(ws, r, 3, ASSET_SHORT[n], fill=fill)
                _put(ws, r, 4, f"={self.weight_ref(n)}", PCT, F_LINK, fill)
                has = pd.notna(row['Price Then (₹)'])
                _put(ws, r, 5, row['Compare From'].to_pydatetime() if has else "no older data", DATE, F_INPUT, fill)
                pt = _put(ws, r, 6, float(row['Price Then (₹)']) if has else None, PRICE, F_INPUT, fill)
                if has:
                    pt.comment = Comment("Last price on or before the Compare-from date (full price history, "
                                         "can be before the selected period).", "Backtest")
                _put(ws, r, 7, f"={self.price_ref(d, n)}", PRICE, F_LINK, fill)
                _put(ws, r, 8, f'=IF(F{r}="","",G{r}/F{r}-1)', PCT, fill=fill)
                _put(ws, r, 9, f'=IF(H{r}="","",RANK(H{r},$H${first}:$H${last},0)+COUNTIF($H${first}:H{r},H{r})-1)',
                     "0", fill=fill, align="center")
                _put(ws, r, 10, f'=IF(H{r}="","",RANK(H{r},$H${first}:$H${last},1)+COUNTIF($H${first}:H{r},H{r})-1)',
                     "0", fill=fill, align="center")
                _put(ws, r, 11, f'=IF(H{r}="","",D{r})', PCT, fill=fill)
                if k == 0:
                    nw = f"=D{r}"
                else:
                    nw = f'=IF(H{r}="",D{r},LARGE($K${first}:$K${last},J{r}))'
                _put(ws, r, 12, nw, PCT, fill=FILL_GOOD, bold=True)
                self.new_w[(k, n)] = f"{_q('Weight Decisions')}$L${r}"
                self.wd_rows = getattr(self, "wd_rows", {})
                self.wd_rows[(k, n)] = r
                r += 1
            _put(ws, r, 3, "Total", bold=True, fill=FILL_TOTAL)
            _put(ws, r, 12, f"=SUM(L{first}:L{last})", PCT, bold=True, fill=FILL_TOTAL)
            if k == 0:
                _note(ws, r, 1, "First buy: your own weights are used.")
            r += 2
        ws.freeze_panes = "A5"

    # ---------- Rebalancing strategies ----------
    def build_rebalance_steps(self, key):
        name = f"Steps - {STRAT_SHEET[key]}"
        ws = self.wb.create_sheet(name)
        _title(ws, f"Step-by-step calculation · {STRAT_NAME[key]}",
               "Portfolio value = Σ (units held × price).   Target value = Portfolio value × target weight.   "
               "Units after = Target value ÷ price.   Trade = Units after − Units before.")
        heads = ["Step #", "Date", "Type", "Asset", "Price (₹)", "Units before", "Value before (₹)",
                 "Weight before", "Portfolio value (₹)", "Target weight", "Target value (₹)", "Units after",
                 "Units bought (+) / sold (−)", "Trade amount (₹)", "Action"]
        widths = [7, 13, 12, 12, 13, 13, 15, 11, 16, 11, 15, 13, 15, 15, 9]
        _header(ws, 4, 1, heads, widths)
        r = 5
        prev_after = {}
        units_cells = []                 # per step: {asset: units-after cell}
        for k, (d, w) in enumerate(self.schedule):
            first, last = r, r + len(self.used) - 1
            tot = last + 1
            fill = FILL_BLOCK if k % 2 == 0 else None
            for n in self.used:
                _put(ws, r, 1, k + 1, fill=fill, align="center")
                _put(ws, r, 2, pd.Timestamp(d).to_pydatetime(), DATE, F_INPUT, fill)
                _put(ws, r, 3, "First Buy" if k == 0 else "Rebalance", fill=fill)
                _put(ws, r, 4, ASSET_SHORT[n], fill=fill)
                _put(ws, r, 5, f"={self.price_ref(d, n)}", PRICE, F_LINK, fill)
                _put(ws, r, 6, f"=L{prev_after[n]}" if k else 0, UNITS, fill=fill)
                _put(ws, r, 7, f"=F{r}*E{r}", MONEY, fill=fill)
                _put(ws, r, 8, f"=IF($I{r}=0,0,G{r}/$I{r})" if k else 0, PCT, fill=fill)
                _put(ws, r, 9, f"={self.amt_ref}" if k == 0 else f"=$G${tot}", MONEY,
                     F_LINK if k == 0 else F_BASE, fill)
                tw = self.new_w[(k, n)] if key == "diff" else self.weight_ref(n)
                _put(ws, r, 10, f"={tw}", PCT, F_LINK, fill)
                _put(ws, r, 11, f"=I{r}*J{r}", MONEY, fill=fill)
                _put(ws, r, 12, f"=K{r}/E{r}", UNITS, fill=FILL_GOOD, bold=True)
                _put(ws, r, 13, f"=L{r}-F{r}", UNITS, fill=fill)
                _put(ws, r, 14, f"=M{r}*E{r}", MONEY, fill=fill)
                _put(ws, r, 15, f'=IF(ROUND(M{r},6)>0,"BUY",IF(ROUND(M{r},6)<0,"SELL","—"))', fill=fill, align="center")
                prev_after[n] = r
                r += 1
            _put(ws, tot, 4, "TOTAL", bold=True, fill=FILL_TOTAL)
            for col in (7, 8, 10, 11, 14):
                fmt = PCT if col in (8, 10) else MONEY
                _put(ws, tot, col, f"=SUM({L(col)}{first}:{L(col)}{last})", fmt, bold=True, fill=FILL_TOTAL)
            units_cells.append(dict(prev_after))
            r = tot + 2
        _note(ws, r - 1, 1, "First buy: there is nothing held yet, so the portfolio value is the amount invested. "
                            "On a rebalance the TOTAL trade amount is 0 (money from sells pays for the buys).")
        end_first = self._end_block(ws, r + 1, {n: f"L{prev_after[n]}" for n in self.used}, key)
        ws.freeze_panes = "E5"
        self.steps = getattr(self, "steps", {})
        self.steps[key] = dict(sheet=name, units=units_cells, end=end_first, last_row=r - 2,
                               trade_col="N", asset_col="D")
        return name

    def _end_block(self, ws, r, units_cell, key):
        """Final value of each asset on the last day. Returns {asset: row}."""
        last_date = self.prices.index[-1]
        _put(ws, r, 1, f"End value on the last day", font=F_SUB, border=False)
        r += 1
        _header(ws, r, 1, ["", "Date", "Type", "Asset", "Price (₹)", "Units held", "Value (₹)", "Weight"])
        first = r + 1
        last = first + len(self.used) - 1
        rows = {}
        for i, n in enumerate(self.used):
            rr = first + i
            _put(ws, rr, 2, last_date.to_pydatetime(), DATE, F_INPUT)
            _put(ws, rr, 3, "End Value")
            _put(ws, rr, 4, ASSET_SHORT[n])
            _put(ws, rr, 5, f"={self.price_ref(last_date, n)}", PRICE, F_LINK)
            _put(ws, rr, 6, f"={units_cell[n]}", UNITS)
            _put(ws, rr, 7, f"=F{rr}*E{rr}", MONEY, fill=FILL_GOOD, bold=True)
            _put(ws, rr, 8, f"=G{rr}/$G${last + 1}", PCT)
            rows[n] = rr
        _put(ws, last + 1, 4, "TOTAL", bold=True, fill=FILL_TOTAL)
        _put(ws, last + 1, 7, f"=SUM(G{first}:G{last})", MONEY, bold=True, fill=FILL_TOTAL)
        _put(ws, last + 1, 8, f"=SUM(H{first}:H{last})", PCT, bold=True, fill=FILL_TOTAL)
        return rows

    # ---------- Buy & Hold ----------
    def build_bh_steps(self):
        name = "Steps - Buy & Hold"
        ws = self.wb.create_sheet(name)
        _title(ws, "Step-by-step calculation · Buy & Hold",
               "Money for an asset = Amount × weight.   Units = Money ÷ price on its buy day.   "
               "Units never change after that.  Until an asset's data starts, its money waits as cash.")
        heads = ["Step #", "Buy date", "Type", "Asset", "Weight", "Money for this asset (₹)", "Buy price (₹)",
                 "Units bought", "Trade amount (₹)"]
        _header(ws, 4, 1, heads, [7, 13, 11, 12, 11, 17, 13, 13, 15])
        first = 5
        last = first + len(self.used) - 1
        bh = {}
        for i, n in enumerate(self.used):
            r = first + i
            dt = self.prices[n].first_valid_index()
            _put(ws, r, 1, 1, align="center")
            _put(ws, r, 2, dt.to_pydatetime(), DATE, F_INPUT)
            _put(ws, r, 3, "First Buy")
            _put(ws, r, 4, ASSET_SHORT[n])
            _put(ws, r, 5, f"={self.weight_ref(n)}", PCT, F_LINK)
            _put(ws, r, 6, f"={self.amt_ref}*E{r}", MONEY, F_LINK)
            _put(ws, r, 7, f"={self.price_ref(dt, n)}", PRICE, F_LINK)
            _put(ws, r, 8, f"=F{r}/G{r}", UNITS, fill=FILL_GOOD, bold=True)
            _put(ws, r, 9, f"=H{r}*G{r}", MONEY)
            bh[n] = r
        _put(ws, last + 1, 4, "TOTAL", bold=True, fill=FILL_TOTAL)
        for col, fmt in ((5, PCT), (6, MONEY), (9, MONEY)):
            _put(ws, last + 1, col, f"=SUM({L(col)}{first}:{L(col)}{last})", fmt, bold=True, fill=FILL_TOTAL)
        end = self._end_block(ws, last + 4, {n: f"H{bh[n]}" for n in self.used}, "bh")
        ws.freeze_panes = "E5"
        self.steps = getattr(self, "steps", {})
        self.steps["bh"] = dict(sheet=name, rows=bh, end=end, last_row=last, trade_col="I", asset_col="D")
        return name

    # ---------- Daily values ----------
    def build_daily(self, key):
        st = self.steps[key]
        name = f"Daily - {STRAT_SHEET[key]}"
        ws = self.wb.create_sheet(name)
        _title(ws, f"Daily value · {STRAT_NAME[key]}",
               "Value of an asset = units held × that day's price.  Peak = highest total so far.  "
               "Drawdown = Total ÷ Peak − 1.")
        nA = len(self.used)
        heads = (["Date"] + [f"Units · {ASSET_SHORT[n]}" for n in self.used]
                 + [f"Value · {ASSET_SHORT[n]} (₹)" for n in self.used]
                 + ["Total value (₹)", "Peak so far (₹)", "Drawdown", "Rebalance day?"])
        _header(ws, 4, 1, heads, [13] + [13] * nA + [15] * nA + [16, 16, 11, 11])
        cT, cP, cD, cR = L(2 + 2 * nA), L(3 + 2 * nA), L(4 + 2 * nA), L(5 + 2 * nA)
        sname = _q(st['sheet'])
        reb_dates = {pd.Timestamp(d): k for k, (d, _) in enumerate(self.schedule)} if key != "bh" else {}
        step_idx = -1
        for i, dt in enumerate(self.prices.index):
            r = self.P0 + i
            c = ws.cell(row=r, column=1, value=f"=Prices!A{r}")
            c.number_format, c.font = DATE, F_LINK
            if key != "bh" and dt in reb_dates:
                step_idx = reb_dates[dt]
            for j, n in enumerate(self.used):
                pc = f"Prices!{self.pcol[n]}{r}"
                uc, vc = ws.cell(row=r, column=2 + j), ws.cell(row=r, column=2 + nA + j)
                if key == "bh":
                    br = st['rows'][n]
                    bd = f"{sname}$B${br}"
                    uc.value = f"=IF(A{r}>={bd},{sname}$H${br},0)"
                    vc.value = f"=IF(A{r}>={bd},{L(2 + j)}{r}*{pc},{sname}$F${br})"
                else:
                    uc.value = f"={sname}$L${st['units'][step_idx][n]}"
                    vc.value = f"={L(2 + j)}{r}*{pc}"
                uc.number_format, uc.font = UNITS, F_LINK
                vc.number_format, vc.font = MONEY, F_LINK
            ws[f"{cT}{r}"] = f"=SUM({L(2 + nA)}{r}:{L(1 + 2 * nA)}{r})"
            ws[f"{cP}{r}"] = f"={cT}{r}" if i == 0 else f"=MAX({cP}{r - 1},{cT}{r})"
            ws[f"{cD}{r}"] = f"={cT}{r}/{cP}{r}-1"
            ws[f"{cR}{r}"] = ("Yes" if (key != "bh" and dt in reb_dates and reb_dates[dt] > 0) else
                              "First buy" if i == 0 else "")
            for col, fmt in ((cT, MONEY), (cP, MONEY), (cD, PCT)):
                ws[f"{col}{r}"].number_format = fmt
                ws[f"{col}{r}"].font = F_BOLD if col == cT else F_BASE
            ws[f"{cR}{r}"].font = F_BASE
        ws.freeze_panes = "B5"
        st.update(daily=name, total_col=cT, dd_col=cD)

    # ---------- Metrics block (used on Summary & Comparison) ----------
    def metric_formulas(self, key):
        st = self.steps[key]
        d = _q(st['daily'])
        T, D = st['total_col'], st['dd_col']
        a, b = self.P0, self.PL
        return {
            "Total invested (₹)": (f"={self.amt_ref}", MONEY),
            "Current value (₹)": (f"={d}{T}{b}", MONEY),
            "Profit / loss (₹)": (f"={d}{T}{b}-{self.amt_ref}", MONEY),
            "Total return": (f"={d}{T}{b}/{self.amt_ref}-1", PCT),
            "Years": (f"=(Prices!A{b}-Prices!A{a})/365.25", NUM2),
            "CAGR": (f"=({d}{T}{b}/{self.amt_ref})^(1/MAX((Prices!A{b}-Prices!A{a})/365.25,0.000001))-1", PCT),
            "Max drawdown": (f"=MIN({d}{D}{a}:{D}{b})", PCT),
            "Number of rebalances": ((len(self.schedule) - 1) if key != "bh" else 0, "0"),
        }

    # ---------- ALL STEPS (first sheet: everything on one page) ----------
    def build_all_steps(self, keys):
        ws = self.wb.create_sheet("All Steps", 0)
        self.all_ws = ws
        nA = len(self.used)
        _title(ws, f"Calculated Investment · {STRAT_NAME[self.mode]} · ALL STEPS",
               "Everything on this one sheet, top to bottom. Every number is a formula: click a cell to see "
               "where it comes from. Scroll down for every buy / rebalance.")
        ws.column_dimensions["A"].width = 46
        for i in range(2, 16):
            ws.column_dimensions[L(i)].width = 16
        ws.sheet_properties.tabColor = "1F3864"

        def section(r, text):
            c = _put(ws, r, 1, text, font=F_HEAD, fill=FILL_HEAD)
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=11)
            ws.row_dimensions[r].height = 20
            return r + 1

        # ===== 1. Inputs =====
        r = section(4, "1 · YOUR INPUTS")
        _put(ws, r, 1, "Amount invested (₹)", bold=True)
        _put(ws, r, 2, f"={self.amt_ref}", MONEY, F_LINK)
        _note(ws, r, 3, "← change it on the Inputs sheet (yellow cell)")
        r += 1
        _put(ws, r, 1, "Strategy", bold=True)
        _put(ws, r, 2, STRAT_NAME[self.mode])
        r += 1
        _put(ws, r, 1, "First / last trading day used", bold=True)
        _put(ws, r, 2, f"=Prices!A{self.P0}", DATE, F_LINK)
        _put(ws, r, 3, f"=Prices!A{self.PL}", DATE, F_LINK)
        r += 1
        if self.mode != "bh":
            _put(ws, r, 1, "Rebalance every", bold=True)
            _put(ws, r, 2, self.freq_label)
            r += 1
        if self.mode == "diff":
            _put(ws, r, 1, "Compare growth over the previous", bold=True)
            _put(ws, r, 2, self.lookback_label)
            r += 1
        r += 1
        _header(ws, r, 1, ["Asset", "Your weight (typed)", "Weight used", "Money for this asset (₹)"])
        for n in self.used:
            r += 1
            wr = self.inp_w[n].split("$")[-1]
            _put(ws, r, 1, ASSET_SHORT[n], bold=True)
            _put(ws, r, 2, f"=Inputs!B{wr}", PCT, F_LINK)
            _put(ws, r, 3, f"=Inputs!C{wr}", PCT, F_LINK)
            _put(ws, r, 4, f"=Inputs!D{wr}", MONEY, F_LINK)
        r += 3

        # ===== 2. Results =====
        r = section(r, "2 · RESULT" + (" (your strategy vs Buy & Hold)" if len(keys) == 2 else ""))
        heads = ["Metric"] + [STRAT_NAME[k] for k in keys]
        if len(keys) == 2:
            heads.append("Difference (strategy − B&H)")
        _header(ws, r, 1, heads)
        mets = {k: self.metric_formulas(k) for k in keys}
        res0 = r + 1
        for i, m in enumerate(mets[keys[0]]):
            rr = res0 + i
            _put(ws, rr, 1, m, bold=True)
            for j, k in enumerate(keys):
                f, fmt = mets[k][m]
                _put(ws, rr, 2 + j, f, fmt, _link_font(f),
                     FILL_GOOD if j == 0 and m == "Current value (₹)" else None)
            if len(keys) == 2 and m not in ("Total invested (₹)", "Years", "Number of rebalances"):
                _put(ws, rr, 4, f"=B{rr}-C{rr}", mets[keys[0]][m][1], bold=True)
        r = res0 + len(mets[keys[0]])
        if len(keys) == 2:
            cv = res0 + 1
            _put(ws, r, 1, "Verdict", bold=True)
            _put(ws, r, 2, f'=IF(B{cv}>=C{cv},"Strategy beat Buy & Hold by ₹"&TEXT(B{cv}-C{cv},"#,##0"),'
                           f'"Strategy was behind Buy & Hold by ₹"&TEXT(C{cv}-B{cv},"#,##0"))', border=False)
            r += 1
        _note(ws, r, 1, "Total return = Current value ÷ Invested − 1.   CAGR = (Current value ÷ Invested)^(1 ÷ Years) "
                        "− 1.   Max drawdown = worst fall from a previous peak (Daily sheet).")
        r += 3

        # ===== 3. Compare with 100% in one asset =====
        r = section(r, "3 · COMPARISON: your strategy vs Buy & Hold vs 100% in one asset")
        cws = self.wb["Comparison"]
        heads = [cws.cell(4, c).value for c in range(1, 9)]
        _header(ws, r, 1, heads)
        n_cmp = len(keys) + nA
        fmts = [None, MONEY, MONEY, MONEY, PCT, PCT, PCT, "0"]
        for i in range(n_cmp):
            r += 1
            for c in range(1, 9):
                _put(ws, r, c, f"=Comparison!{L(c)}{5 + i}", fmts[c - 1], F_LINK,
                     FILL_GOOD if i == 0 else None, bold=(c == 1))
        r += 3

        # ===== 4. Asset-wise breakdown =====
        r = section(r, f"4 · ASSET-WISE BREAKDOWN ({STRAT_NAME[self.mode]}) on the last day")
        st = self.steps[self.mode]
        sn = _q(st['sheet'])
        heads = ["Asset", "Starting weight", "Total bought (₹)", "Total sold (₹)", "Net invested (₹)",
                 "Units held", "Latest price (₹)", "Current value (₹)", "Current weight", "Profit / loss (₹)",
                 "Return"]
        _header(ws, r, 1, heads)
        first = r + 1
        last = first + nA - 1
        rng = f"{sn}${st['asset_col']}$5:${st['asset_col']}${st['last_row']}"
        trg = f"{sn}${st['trade_col']}$5:${st['trade_col']}${st['last_row']}"
        for i, n in enumerate(self.used):
            rr = first + i
            er = st['end'][n]
            _put(ws, rr, 1, ASSET_SHORT[n], bold=True)
            _put(ws, rr, 2, f"={self.weight_ref(n)}", PCT, F_LINK)
            _put(ws, rr, 3, f'=SUMIFS({trg},{rng},A{rr},{trg},">0")', MONEY, F_LINK)
            _put(ws, rr, 4, f'=-SUMIFS({trg},{rng},A{rr},{trg},"<0")', MONEY, F_LINK)
            _put(ws, rr, 5, f"=C{rr}-D{rr}", MONEY)
            _put(ws, rr, 6, f"={sn}F{er}", UNITS, F_LINK)
            _put(ws, rr, 7, f"={sn}E{er}", PRICE, F_LINK)
            _put(ws, rr, 8, f"={sn}G{er}", MONEY, F_LINK, FILL_GOOD, bold=True)
            _put(ws, rr, 9, f"=H{rr}/$H${last + 1}", PCT)
            _put(ws, rr, 10, f"=H{rr}+D{rr}-C{rr}", MONEY)
            _put(ws, rr, 11, f'=IF(C{rr}=0,"",(H{rr}+D{rr})/C{rr}-1)', PCT)
        t = last + 1
        _put(ws, t, 1, "TOTAL", bold=True, fill=FILL_TOTAL)
        for col, fmt in ((2, PCT), (3, MONEY), (4, MONEY), (5, MONEY), (8, MONEY), (9, PCT), (10, MONEY)):
            _put(ws, t, col, f"=SUM({L(col)}{first}:{L(col)}{last})", fmt, bold=True, fill=FILL_TOTAL)
        _put(ws, t, 11, f"=H{t}/E{t}-1", PCT, bold=True, fill=FILL_TOTAL)
        _note(ws, t + 1, 1, "Total bought / sold = all money put into / taken out of the asset on every buy and "
                            "rebalance.  Net invested = bought − sold.  Profit = Current value + sold − bought.")
        _note(ws, t + 2, 1, "Return per asset = (Current value + sold) ÷ bought − 1.  TOTAL return = Current value ÷ "
                            "amount invested − 1.")
        r = t + 5

        # ===== 5 & 6. Every buy / rebalance =====
        r = self.build_walkthrough(self.mode, ws, r, section)

        # ===== 7. Other sheets =====
        r += 1
        r = section(r, "7 · DETAIL SHEETS (click to open)")
        guide = [("Inputs", "Amount and weights (change the yellow cells → everything recalculates)"),
                 ("Prices", "Daily closing price of each asset used")]
        if self.mode == "diff":
            guide.append(("Weight Decisions", "Growth over the look-back, rank, and the weight each asset got"))
        for k in keys:
            guide.append((self.steps[k]['sheet'], f"Every buy / rebalance as a table · {STRAT_NAME[k]}"))
            guide.append((self.steps[k]['daily'], f"Value every day, peak and drawdown · {STRAT_NAME[k]}"))
        guide.append(("Comparison", "Comparison table and chart"))
        for i, (s, txt) in enumerate(guide):
            c = _put(ws, r + i, 1, s, font=Font(name=FONT, size=10, color="0563C1", underline="single"))
            c.hyperlink = f"#'{s}'!A1"
            _put(ws, r + i, 2, txt, border=False)
        ws.freeze_panes = "B4"

    # ---------- Comparison ----------
    def build_comparison(self, keys):
        ws = self.wb.create_sheet("Comparison")
        _title(ws, "Comparison", "Your strategy vs Buy & Hold vs putting 100% into a single asset "
                                 "(same amount, same period).")
        ws.column_dimensions["A"].width = 34
        for c in "BCDEFGHI":
            ws.column_dimensions[c].width = 16

        a, b = self.P0, self.PL
        heads = ["Option", "Invested (₹)", "Current value (₹)", "Profit / loss (₹)", "Total return", "CAGR",
                 "Max drawdown", "Rank (by value)"]
        _header(ws, 4, 1, heads)
        r = 5
        yrs = f"MAX((Prices!A{b}-Prices!A{a})/365.25,0.000001)"
        for k in keys:
            m = self.metric_formulas(k)
            _put(ws, r, 1, STRAT_NAME[k] + (" (your strategy)" if k == self.mode else ""), bold=True,
                 fill=FILL_GOOD if k == self.mode else None)
            for col, met in ((2, "Total invested (₹)"), (3, "Current value (₹)"), (4, "Profit / loss (₹)"),
                             (5, "Total return"), (6, "CAGR"), (7, "Max drawdown")):
                f, fmt = m[met]
                _put(ws, r, col, f, fmt, _link_font(f))
            r += 1
        single_first = r
        for n in self.used:
            first_dt = self.prices[n].first_valid_index()
            p0 = self.price_ref(first_dt, n)
            p1 = self.price_ref(self.prices.index[-1], n)
            col = self.pcol[n]
            _put(ws, r, 1, f"100% in {ASSET_SHORT[n]} (bought {first_dt:%d-%b-%Y})")
            _put(ws, r, 2, f"={self.amt_ref}", MONEY, F_LINK)
            _put(ws, r, 3, f"=B{r}/{p0}*{p1}", MONEY, F_LINK)
            _put(ws, r, 4, f"=C{r}-B{r}", MONEY)
            _put(ws, r, 5, f"=C{r}/B{r}-1", PCT)
            _put(ws, r, 6, f"=(C{r}/B{r})^(1/{yrs})-1", PCT, F_LINK)
            # max drawdown of a single asset = MIN over days of price/running-max - 1 → array formula avoided:
            # use helper column on this sheet further right
            _put(ws, r, 7, None, PCT)
            self._single_dd_helpers = getattr(self, "_single_dd_helpers", [])
            self._single_dd_helpers.append((r, n, first_dt))
            r += 1
        last = r - 1
        for rr in range(5, last + 1):
            _put(ws, rr, 8, f"=RANK(C{rr},$C$5:$C${last},0)", "0", align="center")
        _note(ws, r, 1, "Single-asset rows: Value = Amount ÷ first price × last price (no cash, no rebalancing).")

        # Helper columns for single-asset drawdowns (price ÷ running max − 1), far to the right
        hc = 12
        _put(ws, 3, hc, "Helper: single-asset drawdown (price ÷ peak − 1)", font=F_NOTE, border=False)
        for i, (row, n, first_dt) in enumerate(self._single_dd_helpers):
            pk, dd = L(hc + 2 * i), L(hc + 2 * i + 1)
            _header(ws, 4, hc + 2 * i, [f"Peak · {ASSET_SHORT[n]}", f"DD · {ASSET_SHORT[n]}"], [12, 10])
            start = self.P0 + self.prices.index.get_loc(first_dt)
            for rr in range(self.P0, self.PL + 1):
                pc = f"Prices!{self.pcol[n]}{rr}"
                if rr < start:
                    continue
                ws[f"{pk}{rr}"] = f"={pc}" if rr == start else f"=MAX({pk}{rr - 1},{pc})"
                ws[f"{dd}{rr}"] = f"={pc}/{pk}{rr}-1"
                ws[f"{pk}{rr}"].number_format, ws[f"{dd}{rr}"].number_format = PRICE, PCT
                ws[f"{pk}{rr}"].font = ws[f"{dd}{rr}"].font = F_NOTE
            ws.cell(row=row, column=7).value = f"=MIN({dd}{start}:{dd}{self.PL})"

        # Chart: total value over time for each strategy
        ch = LineChart()
        ch.title = "Portfolio value over time (₹)"
        ch.height, ch.width = 9, 24
        ch.roundedCorners = False
        ch.y_axis.title = "₹"
        ch.y_axis.number_format = '#,##0'
        ch.x_axis.number_format = 'mmm-yy'
        ch.x_axis.delete = False
        ch.y_axis.delete = False
        for k in keys:
            st = self.steps[k]
            dws = self.wb[st['daily']]
            col = dws[f"{st['total_col']}4"].column
            ref = Reference(dws, min_col=col, min_row=4, max_row=self.PL)
            ch.add_data(ref, titles_from_data=True)
        for s, k in zip(ch.series, keys):
            s.smooth = False
            s.graphicalProperties.line.width = 22000 if k == self.mode else 15000
        for s, k in zip(ch.series, keys):
            s.tx = SeriesLabel(v=STRAT_NAME[k])
        dws0 = self.wb[self.steps[keys[0]]['daily']]
        ch.set_categories(Reference(dws0, min_col=1, min_row=self.P0, max_row=self.PL))
        ch.x_axis.title = None
        ch.legend.position = "b"
        ch.legend.overlay = False
        ch.title.overlay = False
        ch.y_axis.title.overlay = False
        ws.add_chart(ch, f"A{r + 2}")


    # ---------- Every buy / rebalance explained step by step (written on the All Steps sheet) ----------
    def build_walkthrough(self, key, ws, r, section):
        """Section 5 = overview of prices on every buy / rebalance day.
        Section 6 = one block per event (first buy, each rebalance, end value): assets across, steps down.
        Calculated fresh from Prices / Inputs, then checked against the Steps sheet. Returns the next free row."""
        nA = len(self.used)
        cA = {n: 2 + i for i, n in enumerate(self.used)}       # asset columns
        cT = 2 + nA                                             # Total column
        cX = cT + 1                                             # "How" column
        st = self.steps[key]
        sq = _q(st['sheet'])
        last_day = self.prices.index[-1]

        if key == "bh":
            events = [("First Buy", {n: self.prices[n].first_valid_index() for n in self.used})]
        else:
            events = [("First Buy" if k == 0 else f"Rebalance {k}", {n: d for n in self.used})
                      for k, (d, _) in enumerate(self.schedule)]
        events.append(("End Value", {n: last_day for n in self.used}))
        n_ev = len(events)

        # ----- 5. Overview table (rows reserved, filled at the end) -----
        r = section(r, "5 · PRICES ON EVERY BUY / REBALANCE DAY (overview)")
        heads = (["Event"] + [f"Price · {ASSET_SHORT[n]} (₹)" for n in self.used]
                 + ["Portfolio value (₹)", "Change since previous"]
                 + [f"Price change · {ASSET_SHORT[n]}" for n in self.used])
        _header(ws, r, 1, heads)
        ov_rows = {i: r + 1 + i for i in range(n_ev)}
        r = r + n_ev + 4

        r = section(r, "6 · EVERY BUY / REBALANCE, STEP BY STEP")
        _note(ws, r, 1, "Read each block top to bottom: prices → what you held → what it was worth → target → "
                        "what to buy / sell → what you hold after → cash check.")
        r += 2

        prev_units = {n: None for n in self.used}   # row of 'units after' in the previous block
        prev_dates = None
        pv_refs = []
        for k, (label, dates) in enumerate(events):
            is_first, is_end = k == 0, label == "End Value"
            same_day = len({dates[n] for n in self.used}) == 1
            when = f"{dates[self.used[0]]:%d-%b-%Y}" if same_day else "each asset on its own first trading day"
            _put(ws, r, 1, f"{label} · {when}", font=F_HEAD, fill=PatternFill("solid", fgColor="2F5597"))
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=cX + 2)
            r += 1
            _header(ws, r, 1, ["Step"] + [ASSET_SHORT[n] for n in self.used] + ["Total", "How it is calculated"])
            r += 1
            R = {}
            step_no = [0]

            def row(tag, text, how, fmt, per_asset=None, total=None, bold=False, fill=None, font=F_BASE):
                nonlocal r
                step_no[0] += 1
                _put(ws, r, 1, f"{step_no[0]}. {text}", bold=bold, fill=fill)
                for n in self.used:
                    v = per_asset(n) if per_asset else None
                    _put(ws, r, cA[n], v, fmt, _link_font(v) if font is F_BASE else font, fill, bold=bold)
                _put(ws, r, cT, total, fmt, _link_font(total) if total else F_BASE, fill or FILL_TOTAL, bold=True)
                if how.startswith("="):                       # text, not a formula
                    how = "Formula: " + how[1:].strip()
                _put(ws, r, cX, how, font=F_NOTE, border=False)
                R[tag] = r
                r += 1

            def a(tag, n):
                return f"{L(cA[n])}{R[tag]}"

            def tot(tag):
                return f"${L(cT)}${R[tag]}"

            def set_sum(tag):
                ws.cell(row=R[tag], column=cT).value = \
                    f"=SUM({L(cA[self.used[0]])}{R[tag]}:{L(cA[self.used[-1]])}{R[tag]})"

            # --- A. Prices ---
            if not is_first:
                row("pdate", "Previous buy / rebalance date", "Date money was last moved", DATE,
                    lambda n: prev_dates[n].to_pydatetime(), font=F_INPUT)
                row("pprev", "Price on previous date (₹)", "From the Prices sheet", PRICE,
                    lambda n: f"={self.price_ref(prev_dates[n], n)}")
            row("date", "Date of this step", "Trading day used", DATE, lambda n: dates[n].to_pydatetime(),
                font=F_INPUT)
            row("price", "Price on this date (₹)", "From the Prices sheet", PRICE,
                lambda n: f"={self.price_ref(dates[n], n)}", bold=True)
            if not is_first:
                row("chg", "Price change since previous date", "= Price now ÷ Price on previous date − 1", PCT,
                    lambda n: f"={a('price', n)}/{a('pprev', n)}-1")

            # --- B. What you hold before ---
            row("ub", "Units held before this step", "First buy: nothing held yet. Later: units after the "
                "previous step", UNITS, lambda n: 0 if is_first else f"={L(cA[n])}{prev_units[n]}")
            if not is_first:
                row("vprev", "Value on previous date (₹)", "= Units held × Price on previous date", MONEY,
                    lambda n: f"={a('ub', n)}*{a('pprev', n)}")
                set_sum("vprev")
            row("vnow", "Value now, before trading (₹)", "= Units held × Price now", MONEY,
                lambda n: f"={a('ub', n)}*{a('price', n)}")
            set_sum("vnow")
            if not is_first:
                row("gain", "Gain / loss since previous date (₹)", "= Value now − Value on previous date", MONEY,
                    lambda n: f"={a('vnow', n)}-{a('vprev', n)}")
                set_sum("gain")
                row("wb", "Weight now (drifted with prices)", "= Value now ÷ Total value now", PCT,
                    lambda n: f"=IF({tot('vnow')}=0,0,{a('vnow', n)}/{tot('vnow')})")
                set_sum("wb")

            if is_end:
                row("ret", "Return since previous date", "= Value now ÷ Value on previous date − 1", PCT,
                    lambda n: f'=IF({a("vprev", n)}=0,"",{a("vnow", n)}/{a("vprev", n)}-1)')
                ws.cell(row=R["ret"], column=cT).value = f"={tot('vnow')}/{tot('vprev')}-1"
                pv_refs.append(tot('vnow'))
                r += 1
                break

            # --- C. Money to split ---
            row("pv", "Portfolio value to split (₹)", "First buy: the amount invested. Later: total value now",
                MONEY, None, f"={self.amt_ref}" if is_first else f"={tot('vnow')}", bold=True)
            pv_refs.append(tot('pv'))

            # --- D. (Different Weights) how the new weights were chosen ---
            if key == "diff":
                wd = _q('Weight Decisions')
                row("cf", f"Compare-from date ({self.lookback_label} back)", "From the Weight Decisions sheet", DATE,
                    lambda n: f'=IF({wd}F{self.wd_rows[(k, n)]}="","-",{wd}E{self.wd_rows[(k, n)]})')
                row("pthen", "Price on compare-from date (₹)", "Last price on or before that date", PRICE,
                    lambda n: f'=IF({wd}F{self.wd_rows[(k, n)]}="","-",{wd}F{self.wd_rows[(k, n)]})')
                row("gr", f"Growth over previous {self.lookback_label}", "= Price now ÷ Price then − 1", PCT,
                    lambda n: f"={wd}H{self.wd_rows[(k, n)]}")
                row("rk", "Growth rank (1 = highest)", "Highest growth gets the SMALLEST of your weights, lowest "
                    "growth the BIGGEST", "0", lambda n: f"={wd}I{self.wd_rows[(k, n)]}")
                row("yw", "Your weight", "From Inputs", PCT, lambda n: f"={self.weight_ref(n)}")
                set_sum("yw")

            # --- E. Target and trades ---
            tw_ref = (lambda n: f"={self.new_w[(k, n)]}") if key == "diff" else (lambda n: f"={self.weight_ref(n)}")
            row("tw", "Target weight", "Weight the asset should have after this step" +
                (" (first buy = your weights)" if key == "diff" and is_first else ""), PCT, tw_ref, bold=True)
            set_sum("tw")
            row("tv", "Target value (₹)", "= Portfolio value to split × Target weight", MONEY,
                lambda n: f"={tot('pv')}*{a('tw', n)}")
            set_sum("tv")
            row("tr", "Money to buy (+) / sell (−) (₹)", "= Target value − Value now", MONEY,
                lambda n: f"={a('tv', n)}-{a('vnow', n)}")
            set_sum("tr")
            row("tu", "Units to buy (+) / sell (−)", "= Money to buy or sell ÷ Price now", UNITS,
                lambda n: f"={a('tr', n)}/{a('price', n)}")
            row("act", "Action", "BUY if units go up, SELL if they go down", None,
                lambda n: f'=IF(ROUND({a("tu", n)},6)>0,"BUY",IF(ROUND({a("tu", n)},6)<0,"SELL","HOLD"))')
            row("ua", "Units held after this step", "= Units held before + Units bought / sold", UNITS,
                lambda n: f"={a('ub', n)}+{a('tu', n)}", bold=True, fill=FILL_GOOD)
            row("va", "Value after this step (₹)", "= Units after × Price now", MONEY,
                lambda n: f"={a('ua', n)}*{a('price', n)}")
            set_sum("va")
            row("wa", "Weight after this step", "= Value after ÷ Total value after", PCT,
                lambda n: f"={a('va', n)}/{tot('va')}")
            set_sum("wa")

            # --- F. Cash check ---
            rng = f"{L(cA[self.used[0]])}{R['tr']}:{L(cA[self.used[-1]])}{R['tr']}"
            row("sell", "Cash received from sells (₹)", "Sum of the negative amounts in step 'Money to buy / sell'",
                MONEY, None, f'=-SUMIF({rng},"<0")')
            row("buy", "Cash spent on buys (₹)", "Sum of the positive amounts", MONEY, None, f'=SUMIF({rng},">0")')
            row("net", "New cash needed (0 on every rebalance)",
                "First buy: the amount invested. Rebalance: sells pay exactly for buys", MONEY, None,
                f"={tot('buy')}-{tot('sell')}")
            if key == "bh":
                chk = lambda n: f'=IF(ABS({a("ua", n)}-{sq}H{st["rows"][n]})<0.000001,"✓ matches","✗ check")'
            else:
                chk = lambda n: f'=IF(ABS({a("ua", n)}-{sq}L{st["units"][k][n]})<0.000001,"✓ matches","✗ check")'
            row("chk", "Check: units after = Steps sheet", f"Same result as the '{st['sheet']}' sheet", None, chk)

            prev_units = {n: R["ua"] for n in self.used}
            prev_dates = dates
            r += 1

        # ----- fill the overview -----
        for k, (label, dates) in enumerate(events):
            orow = ov_rows[k]
            same = len({dates[n] for n in self.used}) == 1
            _put(ws, orow, 1, f"{k + 1}. {label} · {dates[self.used[0]]:%d-%b-%Y}" if same
                 else f"{k + 1}. {label} · each asset's own first day", bold=True)
            for i, n in enumerate(self.used):
                _put(ws, orow, cA[n], f"={self.price_ref(dates[n], n)}", PRICE, F_LINK)
                c = cT + 2 + i
                _put(ws, orow, c, None if k == 0 else f"={L(cA[n])}{orow}/{L(cA[n])}{orow - 1}-1", PCT)
            _put(ws, orow, cT, f"={pv_refs[k]}", MONEY, bold=True)
            _put(ws, orow, cT + 1, None if k == 0 else f"={L(cT)}{orow}/{L(cT)}{orow - 1}-1", PCT)
        return r

    # ---------- Build ----------
    def build(self):
        self.build_inputs()
        self.build_prices()
        keys = [self.mode] if self.mode == "bh" else [self.mode, "bh"]
        if self.mode == "diff":
            self.build_decisions()
        if self.mode != "bh":
            self.build_rebalance_steps(self.mode)
            self.build_daily(self.mode)
        self.build_bh_steps()
        self.build_daily("bh")
        self.build_comparison(keys)
        self.build_all_steps(keys)
        # sheet order: All Steps first, then the detail sheets
        order = ["All Steps", "Inputs", "Prices"]
        if self.mode == "diff":
            order.append("Weight Decisions")
        for k in keys:
            order += [self.steps[k]['sheet'], self.steps[k]['daily']]
        order.append("Comparison")
        self.wb._sheets = [self.wb[s] for s in order]
        self.wb.active = 0
        for w in self.wb.worksheets:
            w.sheet_view.tabSelected = (w.title == "All Steps")
        buf = io.BytesIO()
        self.wb.save(buf)
        return buf.getvalue()


def build_investment_workbook(mode, prices, schedule, raw_weights, w0, amount, decisions=None,
                              freq_label=None, lookback_label=None, start_date=None, end_date=None):
    """Return the .xlsx file (bytes) with the full step-by-step calculation.

    mode        : "bh" | "periodic" | "diff"
    prices      : the forward-filled price table the page used (Date index, one column per asset)
    schedule    : [(date, {asset: weight fraction})]  (for bh: [(first date, w0)])
    raw_weights : {asset: weight in % as typed by the user}  (only assets with weight > 0)
    w0          : {asset: normalised weight fraction}
    decisions   : DataFrame from build_diff_schedule() (Different Weights only)
    """
    return _Builder(mode, prices, schedule, raw_weights, w0, amount, decisions,
                    freq_label, lookback_label, start_date or prices.index[0],
                    end_date or prices.index[-1]).build()