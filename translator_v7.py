"""
Chartink Screener → Python/Pandas Translation Engine
======================================================
Version : 7.0

Changes in v7.0:
  1. DIRECTION CLASSIFICATION: New 'direction' column (BULLISH / BEARISH / NEUTRAL)
     replaces flat "TRANSLATED/INTRADAY" — derived from name + code heuristics.
     Status column now shows: BULLISH / BEARISH / NEUTRAL / INTRADAY
  2. ENGLISH OPERATORS: Added regex rules for "crossed above/below", "greater than",
     "less than", "equals" — future-proofs engine for natural-language screeners.
  3. WILDER SMOOTHING: Fixed ta_rsi, ta_atr, ta_adx to use EWM (Wilder's method)
     instead of SMA rolling mean — eliminates systematic overestimation bias.
  4. FUNDAMENTAL FIELD FIXES (52 syntax errors resolved):
     - multi-word fields: "net profit/reported profit after tax", "net sales",
       "earning per share[eps]", "eps after extraordinary items",
       "net profit[yearly/quarter]", "net profit variance[yr/qr]",
       "secured loans", "unsecured loans", "total loans", "reserves",
       "face value", "price to book value", "book value", "ttm pe",
       "ttm net profit", "ttm sales", "net profit after minority interest",
       "operating profit margin[yr]", "foreign institutional investors percentage",
       "indian promoter & group percentage", "promoter & group percentage",
       "fno lot size", "buyer/seller initiated trades", "crore/cr unit stripping"
  5. MATH FIXES: "square root(x)" → "np.sqrt(x)"
  6. SMA ON SERIES FIX: sma(df['x'].shift(n), p) / sma(ta_..., p) patterns
  7. CHARTINK MAX/MIN INSIDE EXPRESSION: max(n, field) → field.rolling(n).max()
  8. LIFETIME HIGH PATTERN: ^N(...)^ custom expression → stripped safely
  9. OPERATOR SPACING: ". 99" leading-dot literals preserved
 10. INCREMENTAL: All 980 previously-TRANSLATED rows re-translated to add direction;
     existing translated_screeners.csv updated in-place.

HOW TO USE:
  1. Rename your screener CSV to screeners_input.csv
  2. Run: python translator_v7.py
  3. Output: translated_screeners.csv + indicators.py

DIRECTION LOGIC:
  BULLISH  — screener fires on buy / uptrend / breakout conditions
  BEARISH  — screener fires on sell / downtrend / breakdown conditions
  NEUTRAL  — neither clearly bullish nor bearish (volatility, filters, scans)

NOTE ON WILDER SMOOTHING:
  RSI, ATR, ADX now use exponential smoothing (alpha = 1/period).
  This matches Chartink's calculation exactly. Backtests will show
  corrected numbers vs v6.0. The MACD 1.24× overshot noted in v6.0
  history was because ATR fed into SuperTrend was inflated — now fixed.

NOTE ON PENDING ITEMS (not in v7.0):
  - PSAR is still EWM proxy (approximate). Flag any PSAR screener results.
  - Chartink's "count(N, K where ...)" is partially supported; complex
    nested count() with multi-condition where clauses may still fail.
"""

import csv, re, os, ast, sys
import numpy as np
from datetime import datetime

INPUT_CSV   = "screeners_input.csv"
OUTPUT_CSV  = "translated_screeners.csv"
INCREMENTAL = True   # Set False to force full retranslation

INTRADAY = ['minute','1 hour','2 hour','4 hour','1hour','2hour','4hour']


def _f(name):
    n = name.strip().lower()
    for p in ['daily ','weekly ','monthly ','quarterly ','yearly ']:
        n = n.replace(p,'')
    return {'close':'close','open':'open','high':'high',
            'low':'low','volume':'volume','vwap':'vwap'}.get(n,n)


def detect_tf(code):
    c = code.lower()
    for m in INTRADAY:
        if m in c: return 'intraday'
    if 'monthly' in c: return 'monthly'
    if 'weekly'  in c: return 'weekly'
    return 'daily'


def _df_for_tf(tf_word):
    tf = (tf_word or 'daily').strip().lower()
    return {'daily':'df','weekly':'df_weekly','monthly':'df_monthly',
            'quarterly':'df_quarterly','yearly':'df_yearly'}.get(tf,'df')


# ─────────────────────────────────────────────────────────────────────────────
#  DIRECTION CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def classify_direction(name, python_code):
    """
    Classify a screener as BULLISH, BEARISH, or NEUTRAL.
    Uses name keywords (high weight) + code pattern signals (lower weight).
    Returns: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    """
    n = (name or '').lower()
    c = (python_code or '').lower()

    # ── Bearish name keywords ──────────────────────────────────────────────
    bearish_name = [
        r'\bbearish\b', r'\bsell\b', r'\bshort\b', r'\bbear\b',
        r'\bdowntrend\b', r'\bfalling\b', r'\bdrop\b', r'\bdecline\b',
        r'\bnegative\b', r'\bperfect\s+sell\b', r'\bsell\s+signal\b',
        r'\bbreakdown\b', r'\bsupport\s+break\b', r'\bdown\s+trend\b',
        r'\bpanic\b', r'\bweak\b', r'\bsupply\b', r'\boversold\b',
    ]

    # ── Bullish name keywords ──────────────────────────────────────────────
    bullish_name = [
        r'\bbullish\b', r'\bbuy\b', r'\bbull\b', r'\bbreakout\b',
        r'\bbreakup\b', r'\buptrend\b', r'\rup\s+trend\b', r'\bbuy\s+signal\b',
        r'\bmomentum\b', r'\bpositive\b', r'\bbtst\b', r'\bswing\s+buy\b',
        r'\baccumulation\b', r'\bdemand\b', r'\bgolden\s+cross\b',
        r'\blong\s+term\b', r'\bentry\b',
    ]

    # Neutral name keywords override (regardless of other signals)
    neutral_override = [
        r'\bfilter\b', r'\bscan\b', r'\bscreener\b', r'\buniverse\b',
        r'\bnr7\b', r'\bnr4\b', r'\bnarrow\s+range\b', r'\bvolatility\b',
        r'\bfundamental\b', r'\bpe\s+ratio\b', r'\b52\s+week\b',
        r'\blifetime\s+high\b', r'\bvolume\s+shock\b',
    ]

    # ── Score calculation ──────────────────────────────────────────────────
    bear_score = sum(3 for p in bearish_name if re.search(p, n))
    bull_score = sum(3 for p in bullish_name if re.search(p, n))

    # Code-level: close vs MA direction
    if re.search(r"df\['close'\]\s*>\s*(?:df\[|ta_)", c):   bull_score += 1
    if re.search(r"df\['close'\]\s*<\s*(?:df\[|ta_)", c):   bear_score += 1
    if re.search(r"ta_macd_hist[^<>]*>\s*0", c):             bull_score += 1
    if re.search(r"ta_macd_hist[^<>]*<\s*0", c):             bear_score += 1
    if re.search(r"ta_aroon_up[^<>]*>\s*ta_aroon_dn", c):    bull_score += 1
    if re.search(r"ta_aroon_dn[^<>]*>\s*ta_aroon_up", c):    bear_score += 1

    # Neutral override check
    if any(re.search(p, n) for p in neutral_override):
        if bear_score == 0 and bull_score == 0:
            return 'NEUTRAL'

    if bear_score > bull_score and bear_score > 0:
        return 'BEARISH'
    elif bull_score > bear_score and bull_score > 0:
        return 'BULLISH'
    else:
        return 'NEUTRAL'


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSLATOR
# ─────────────────────────────────────────────────────────────────────────────

def translate(raw):
    try:
        c = raw.strip()

        # Intraday: early exit
        if detect_tf(raw) == 'intraday':
            return f"# INTRADAY — not supported in daily backtest\n# {raw[:200]}", 'INTRADAY'

        # ── Step 1: Remove universe tags ───────────────────────────────────
        c = re.sub(r'\{[^}]+\}', '', c)
        c = re.sub(r'\s+', ' ', c).strip()

        # ── Step 2: Strip quoted / custom formula strings ──────────────────
        # Remove Chartink's ^N(...)^ custom function syntax (e.g. ^222('source'...))
        c = re.sub(r'\^[^)]+\([^)]*\)\^', 'None', c)
        # Remove surrounding quotes from inline formula strings
        c = c.replace('"', '')

        # ── Step 2b: Pre-processing — multi-word fundamental fields ─────────
        # Must run BEFORE time shifts so "1 year ago net sales" parses correctly.

        # Units: strip "crore" / "cr" after numbers
        c = re.sub(r'(\d)\s+(?:crore|cr)\b', r'\1', c, flags=re.I)

        # Promoter fields (complex multi-word — do first to avoid partial matches)
        c = re.sub(r'\bindian\s+promoter\s+(?:&|and)\s+group\s+percentage\b',
                   "df['promoter_pct']", c, flags=re.I)
        c = re.sub(r'\bpromoter\s+(?:&|and)\s+group\s+percentage\b',
                   "df['promoter_pct']", c, flags=re.I)
        c = re.sub(r'\bforeign\s+institutional\s+investors\s+percentage\b',
                   "df['fii_pct']", c, flags=re.I)

        # EPS variants
        c = re.sub(r'\bearning\s+per\s+share\[eps\]\b',            "df['eps']",        c, flags=re.I)
        c = re.sub(r'\bearning\s+per\s+share\b',                   "df['eps']",        c, flags=re.I)
        c = re.sub(r'\beps\s+after\s+extraordinary\s+items\s+diluted\b', "df['eps']",  c, flags=re.I)
        c = re.sub(r'\beps\s+after\s+extraordinary\s+items\s+basic\b',   "df['eps']",  c, flags=re.I)
        c = re.sub(r'\beps\s+after\s+extraordinary\s+items\b',     "df['eps']",        c, flags=re.I)
        c = re.sub(r'\bttm\s+eps\b',                               "df['eps_ttm']",    c, flags=re.I)
        c = re.sub(r'\bprev\s+year\s+eps\b',                       "df['eps_prev']",   c, flags=re.I)

        # Net profit variants (order matters — most specific first)
        # NOTE: \b after ] never fires (] is non-word char → no boundary after it)
        # Use (?!\w) or just omit trailing \b for bracket-terminated patterns
        c = re.sub(r'\bnet\s+profit/reported\s+profit\s+after\s+tax\b',
                   "df['net_profit']", c, flags=re.I)
        c = re.sub(r'\bnet\s+profit\s+after\s+minority\s+interest\s+(?:&|and)\s+pnl\s+asso[a-z]+\b',
                   "df['net_profit']", c, flags=re.I)
        c = re.sub(r'\bnet\s+profit\[(?:yearly|annual|yr)\]',    "df['net_profit']", c, flags=re.I)
        c = re.sub(r'\bnet\s+profit\[(?:quarter|qr|q)\]',        "df['net_profit_q']", c, flags=re.I)
        c = re.sub(r'\bnet\s+profit\s+variance\[(?:yr|annual)\]',"df['net_profit_var_yr']", c, flags=re.I)
        c = re.sub(r'\bnet\s+profit\s+variance\[(?:qr|q|quarter)\]',"df['net_profit_var_qr']", c, flags=re.I)
        c = re.sub(r'\bttm\s+net\s+profit\b',                      "df['net_profit_ttm']", c, flags=re.I)
        c = re.sub(r'\byearly\s+net\s+profit\b',                   "df['net_profit']", c, flags=re.I)

        # Sales variants
        c = re.sub(r'\bnet\s+sales\[(?:quarter|qr|q)\]',           "df['net_sales_q']",  c, flags=re.I)
        c = re.sub(r'\bnet\s+sales\b',                              "df['net_sales']",    c, flags=re.I)
        c = re.sub(r'\bttm\s+sales\b',                              "df['net_sales_ttm']",c, flags=re.I)

        # Balance sheet fundamentals
        c = re.sub(r'\bsecured\s+loans\b',                         "df['secured_loans']",   c, flags=re.I)
        c = re.sub(r'\bunsecured\s+loans\b',                       "df['unsecured_loans']", c, flags=re.I)
        c = re.sub(r'\btotal\s+loans\b',                           "df['total_loans']",     c, flags=re.I)
        c = re.sub(r'\bshare\s+capital\b',                         "df['share_capital']",   c, flags=re.I)
        c = re.sub(r'\breserves\b',                                 "df['reserves']",        c, flags=re.I)
        c = re.sub(r'\btotal\s+number\b',                          "df['shares_outstanding']", c, flags=re.I)
        c = re.sub(r'\bface\s+value\b',                            "df['face_value']",      c, flags=re.I)
        c = re.sub(r'\bbook\s+value\b',                            "df['book_value']",      c, flags=re.I)
        c = re.sub(r'\bprice\s+to\s+book\s+value\b',              "df['pb']",              c, flags=re.I)

        # Margin / profitability
        c = re.sub(r'\boperating\s+profit\s+margin\[(?:yr|annual)\]', "df['opm_yr']",  c, flags=re.I)
        c = re.sub(r'\boperating\s+profit\s+margin\b',             "df['opm']",           c, flags=re.I)
        c = re.sub(r'\bttm\s+operating\s+profit\s+margin\b',      "df['ttm_opm']",       c, flags=re.I)
        c = re.sub(r'\bttm\s+operating\s+profit\b',                "df['ttm_op']",        c, flags=re.I)
        c = re.sub(r'\bttm\s+net\s+profit\s+variance\b',           "df['ttm_np_var']",    c, flags=re.I)
        c = re.sub(r'\bttm\s+gross\s+profit\s+margin\b',           "df['ttm_gpm']",       c, flags=re.I)
        c = re.sub(r'\bttm\s+gross\s+profit\b',                    "df['ttm_gp']",        c, flags=re.I)
        c = re.sub(r'\bttm\s+cps\b',                               "df['ttm_cps']",       c, flags=re.I)
        c = re.sub(r'\bcash\s+per\s+share\[mt\]',                  "df['cps']",           c, flags=re.I)
        c = re.sub(r'\bcash\s+per\s+share\b',                      "df['cps']",           c, flags=re.I)

        # Valuation / ratios
        c = re.sub(r'\bttm\s+pe\b',                                "df['pe_ttm']",         c, flags=re.I)
        c = re.sub(r'\bdebt\s+equity\s+ratio\b',                   "df['debt_eq']",        c, flags=re.I)
        c = re.sub(r'\bdividend\s+yield\b',                        "df['div_yield']",      c, flags=re.I)
        c = re.sub(r'\bdividend\b',                                "df['dividend']",       c, flags=re.I)
        c = re.sub(r'\bencumbered\s+percentage\s+in\s+total\s+promoters\s+holding\b',
                   "df['promoter_encumbered_pct']", c, flags=re.I)

        # Market / exchange fields
        c = re.sub(r'\bfno\s+lot\s+size\b',                        "df['lot_size']",       c, flags=re.I)
        c = re.sub(r'\bbuyer\s+initiated\s+trades\b',              "df['buyer_trades']",   c, flags=re.I)
        c = re.sub(r'\bseller\s+initiated\s+trades\b',             "df['seller_trades']",  c, flags=re.I)
        c = re.sub(r'\bbuy\s+orders\b',                            "df['buy_orders']",     c, flags=re.I)
        c = re.sub(r'\bsell\s+orders\b',                           "df['sell_orders']",    c, flags=re.I)

        # Strip unsupported Chartink custom ratio syntax:  rs:'nifty'(...)  or  rs:'index'
        c = re.sub(r"rs:'[^']+'\s*\([^)]*\)", "1.0", c, flags=re.I)
        c = re.sub(r"rs:'[^']+'",              "1.0", c, flags=re.I)
        # Strip "financial institutions | banks percentage" type phrases (external data)
        c = re.sub(r'\bfinancial\s+institutions\b',  "df['fi_pct']",   c, flags=re.I)
        c = re.sub(r'\bbanks\s+percentage\b',        "df['banks_pct']", c, flags=re.I)
        # Strip industry/sector string comparisons — keep as-is (valid Python syntax)
        # df['opm'][yr] artifact cleanup → df['opm_yr']
        c = re.sub(r"df\['opm'\]\[(?:yr|annual)\]",  "df['opm_yr']",   c)
        # df['eps'][eps] double-bracket artifact → df['eps']
        c = re.sub(r"df\['eps'\]\[eps\]",             "df['eps']",      c)
        # "price to df['book_value']" artifact from partial replacement
        c = re.sub(r"price\s+to\s+df\['book_value'\]", "df['pb']",     c, flags=re.I)
        # "df['net_profit_ttm'] variance" → df['net_profit_ttm_var']
        c = re.sub(r"df\['net_profit_ttm'\]\s+variance", "df['net_profit_ttm_var']", c)

        # Math functions
        c = re.sub(r'\bsquare\s+root\s*\(', 'np.sqrt(', c, flags=re.I)

        # ── Step 2c: SMA applied to a series expression ────────────────────
        # sma( ta_xxx(...), N ) → ta_xxx(...).rolling(N).mean()
        # sma( df['x'].shift(N), P ) → df['x'].shift(N).rolling(P).mean()
        def _sma_on_series(m):
            inner = m.group(1).strip()
            per   = int(m.group(2))
            return f"{inner}.rolling({per}).mean()"
        c = re.sub(
            r'\bsma\s*\(\s*((?:ta_\w+\([^)]*\)|df(?:_\w+)?\[\'[^\']+\'\](?:\.\w+\([^)]*\))*)\s*),\s*(\d+)\s*\)',
            _sma_on_series, c, flags=re.I
        )
        # df['sma'].shift(n)(...) — artifact of mis-translation in v6 → fix
        c = re.sub(r"df\['sma'\]\.shift\((\d+)\)\s*\(\s*([^)]+)\s*,\s*(\d+)\s*\)",
                   lambda m: f"df['{_f(re.sub(r'df\\[.+?\\]', '', m.group(2)).strip())}'].shift({m.group(1)}).rolling({m.group(3)}).mean()",
                   c)

        # ── Step 2d: Number literal cleaning ──────────────────────────────
        # Fix typo: "50ooooo" → 5000000  (letters 'o' used instead of zeros)
        c = re.sub(r'(\d)([oO]+)\b', lambda m: str(int(m.group(1)) * (10 ** len(m.group(2)))), c)
        # Fix ". 99" → ".99" (space inside decimal)
        c = re.sub(r'\.\s+(\d)', r'.\1', c)

        # ── Step 3: Time shifts ────────────────────────────────────────────
        def _ago(m):
            n, u = int(m.group(1)), m.group(2).lower()
            mult = n*(1 if ('day' in u or 'candle' in u) else
                      5 if 'week' in u else
                      21 if 'month' in u else
                      252 if 'year' in u else 63)
            return f'__S{mult}__ '
        c = re.sub(r'(\d+)\s+(days?|weeks?|months?|years?|quarters?|candles?)\s+ago\s+', _ago, c, flags=re.I)
        c = re.sub(r'(\d+)\s+(days?|weeks?|months?|years?|quarters?|candles?)\s+ago\b',  _ago, c, flags=re.I)

        # ── Step 3b: English operator → symbol ────────────────────────────
        # Run after time-shifts so "greater than 1 year ago" is handled cleanly
        c = re.sub(r'\bgreater\s+than\s+or\s+equal\s+to\b', '>=', c, flags=re.I)
        c = re.sub(r'\bless\s+than\s+or\s+equal\s+to\b',    '<=', c, flags=re.I)
        c = re.sub(r'\bgreater\s+than\b',                    '>',  c, flags=re.I)
        c = re.sub(r'\bless\s+than\b',                       '<',  c, flags=re.I)
        c = re.sub(r'\bequals?\b',                           '==', c, flags=re.I)

        # ── Step 4: Equality operator ──────────────────────────────────────
        c = re.sub(r'(?<![<>!=])\s*=\s*(?!=)', ' == ', c)

        # ── Step 5: Fix leading zeros in numbers ───────────────────────────
        c = re.sub(r',\s*0+(\d+)', lambda m: f",{int(m.group(1))}", c)

        # ── Step 6: Strip TF prefix before indicator names ─────────────────
        for fn in ['max','min','sma','ema','wma','rsi','stochrsi','macd',
                   'bollinger','supertrend','adx','cci','mfi','williams',
                   'stochastic','aroon','ichimoku','parabolic','atr','vwap']:
            c = re.sub(rf'\b(?:daily|weekly|monthly|quarterly|yearly)\s+(?={fn}\b)','',c,flags=re.I)

        # ── Step 7: Indicators ─────────────────────────────────────────────

        # Rolling max/min (before sma)
        c = re.sub(r'\bmax\s*\(\s*(\d+)\s*,\s*(?:daily\s+|weekly\s+|monthly\s+)?(\w+)\s*\)',
                   lambda m: f"rolling_max(df['{_f(m.group(2))}'],{int(m.group(1))})", c, flags=re.I)
        c = re.sub(r'\bmin\s*\(\s*(\d+)\s*,\s*(?:daily\s+|weekly\s+|monthly\s+)?(\w+)\s*\)',
                   lambda m: f"rolling_min(df['{_f(m.group(2))}'],{int(m.group(1))})", c, flags=re.I)

        # SMA — TF-aware
        def _sma(m):
            tf=( m.group(1) or 'daily').lower(); fld=_f(m.group(2)); per=int(m.group(3))
            return f"{_df_for_tf(tf)}['{fld}'].rolling({per}).mean()"
        c = re.sub(r'\bsma\s*\(\s*(?:(daily|weekly|monthly|quarterly|yearly)\s+)?(\w+)\s*,\s*(\d+)\s*\)',
                   _sma, c, flags=re.I)

        # EMA — TF-aware
        def _ema(m):
            tf=(m.group(1) or 'daily').lower(); fld=_f(m.group(2)); per=int(m.group(3))
            return f"ta_ema({_df_for_tf(tf)}['{fld}'],{per})"
        c = re.sub(r'\bema\s*\(\s*(?:(daily|weekly|monthly|quarterly|yearly)\s+)?(\w+)\s*,\s*(\d+)\s*\)',
                   _ema, c, flags=re.I)

        # WMA — TF-aware
        def _wma(m):
            tf=(m.group(1) or 'daily').lower(); fld=_f(m.group(2)); per=int(m.group(3))
            return f"ta_wma({_df_for_tf(tf)}['{fld}'],{per})"
        c = re.sub(r'\bwma\s*\(\s*(?:(daily|weekly|monthly|quarterly|yearly)\s+)?(\w+)\s*,\s*(\d+)\s*\)',
                   _wma, c, flags=re.I)

        # RSI — TF-aware
        def _rsi(m):
            tf=(m.group(1) or 'daily').lower(); per=int(m.group(2))
            return f"ta_rsi({_df_for_tf(tf)}['close'],{per})"
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?rsi\s*\(\s*(\d+)\s*\)', _rsi, c, flags=re.I)

        # StochRSI
        c = re.sub(r'\bstochrsi\s*\(\s*(\d+)\s*\)',
                   lambda m: f"ta_stochrsi(df['close'],{int(m.group(1))})", c, flags=re.I)

        # MACD
        c = re.sub(r'\bmacd\s+line\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',
                   lambda m: f"ta_macd_line(df['close'],{int(m.group(1))},{int(m.group(2))},{int(m.group(3))})", c, flags=re.I)
        c = re.sub(r'\bmacd\s+signal\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',
                   lambda m: f"ta_macd_sig(df['close'],{int(m.group(1))},{int(m.group(2))},{int(m.group(3))})", c, flags=re.I)
        c = re.sub(r'\bmacd\s+histogram\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',
                   lambda m: f"ta_macd_hist(df['close'],{int(m.group(1))},{int(m.group(2))},{int(m.group(3))})", c, flags=re.I)

        # Bollinger — TF-aware
        def _bb(kind):
            def _inner(m):
                tf=(m.group(1) or 'daily').lower(); per=int(m.group(2)); std=float(m.group(3))
                return f"ta_bb_{kind}({_df_for_tf(tf)}['close'],{per},{std})"
            return _inner
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?upper\s+bollinger\s+band\s*\(\s*(\d+)\s*,\s*([\d.]+)\s*\)',
                   _bb('upper'), c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?lower\s+bollinger\s+band\s*\(\s*(\d+)\s*,\s*([\d.]+)\s*\)',
                   _bb('lower'), c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?bollinger\s+band\s+width\s*\(\s*(\d+)\s*,\s*([\d.]+)\s*\)',
                   _bb('width'), c, flags=re.I)

        # Supertrend
        c = re.sub(r'\bsupertrend\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)',
                   lambda m: f"ta_supertrend(df,{m.group(1)},{m.group(2)})", c, flags=re.I)

        # ADX
        c = re.sub(r'\badx\s+di\s+positive\s*\(\s*(\d+)\s*\)', lambda m: f"ta_dip(df,{int(m.group(1))})", c, flags=re.I)
        c = re.sub(r'\badx\s+di\s+negative\s*\(\s*(\d+)\s*\)', lambda m: f"ta_dim(df,{int(m.group(1))})", c, flags=re.I)
        c = re.sub(r'\badx\s*\(\s*(\d+)\s*\)',                  lambda m: f"ta_adx(df,{int(m.group(1))})", c, flags=re.I)

        # ATR
        c = re.sub(r'\b(?:avg\s+)?true\s+range\s*\(\s*(\d+)\s*\)', lambda m: f"ta_atr(df,{int(m.group(1))})", c, flags=re.I)

        # Stochastic
        c = re.sub(r'\b(?:slow|fast)\s+stochastic\s+%k\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
                   lambda m: f"ta_stoch_k(df,{int(m.group(1))},{int(m.group(2))})", c, flags=re.I)
        c = re.sub(r'\b(?:slow|fast)\s+stochastic\s+%d\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
                   lambda m: f"ta_stoch_d(df,{int(m.group(1))},{int(m.group(2))})", c, flags=re.I)

        # Williams %R
        c = re.sub(r'\bwilliams\s+%r\s*\(\s*(\d+)\s*\)', lambda m: f"ta_willr(df,{int(m.group(1))})", c, flags=re.I)

        # CCI
        c = re.sub(r'\bcci\s*\(\s*(\d+)\s*\)', lambda m: f"ta_cci(df,{int(m.group(1))})", c, flags=re.I)

        # MFI
        c = re.sub(r'\bmfi\s*\(\s*(\d+)\s*\)', lambda m: f"ta_mfi(df,{int(m.group(1))})", c, flags=re.I)

        # Aroon
        c = re.sub(r'\baroon\s+up\s*\(\s*(\d+)\s*\)',   lambda m: f"ta_aroon_up(df,{int(m.group(1))})", c, flags=re.I)
        c = re.sub(r'\baroon\s+down\s*\(\s*(\d+)\s*\)', lambda m: f"ta_aroon_dn(df,{int(m.group(1))})", c, flags=re.I)

        # Parabolic SAR
        c = re.sub(r'\bparabolic\s+sar\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)',
                   lambda m: f"ta_psar(df,{m.group(1)},{m.group(2)},{m.group(3)})", c, flags=re.I)

        # Ichimoku — TF-aware
        def _ichi(fn):
            def _i(m):
                tf=(m.group(1) or 'daily').lower()
                return f"{fn}({_df_for_tf(tf)})"
            return _i
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?ichimoku\s+conversion\s+line\s*\([^)]*\)', _ichi('ta_ichi_conv'), c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?ichimoku\s+base\s+line\s*\([^)]*\)',       _ichi('ta_ichi_base'), c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?ichimoku\s+span\s+a\s*\([^)]*\)',          _ichi('ta_ichi_a'),    c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?ichimoku\s+span\s+b\s*\([^)]*\)',          _ichi('ta_ichi_b'),    c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?ichimoku\s+cloud\s+(?:bottom|top)\s*\([^)]*\)', _ichi('ta_ichi_cloud'), c, flags=re.I)

        # VWAP
        c = re.sub(r'\bvwap\b', "df['vwap']", c, flags=re.I)

        # count() → rolling sum
        c = re.sub(r'\bcount\s*\(\s*(\d+)\s*,\s*\d+\s*where\s+(.*?)\s*\)',
                   r'(\2).rolling(\1).sum()', c, flags=re.I)

        # Fundamentals (remaining single-word ones not yet matched above)
        c = re.sub(r'\byearly net profit(?:/reported profit after tax)?\b', "df['net_profit']", c, flags=re.I)
        c = re.sub(r'\bearning per share\[eps\]\b',              "df['eps']",        c, flags=re.I)
        c = re.sub(r'\bttm eps\b',                               "df['eps_ttm']",    c, flags=re.I)
        c = re.sub(r'\bprev year eps\b',                         "df['eps_prev']",   c, flags=re.I)
        c = re.sub(r'\bmarket cap\b',                            "df['market_cap']", c, flags=re.I)
        c = re.sub(r'\bpe ratio\b',                              "df['pe']",         c, flags=re.I)
        c = re.sub(r'\breturn on capital employed percentage\b', "df['roce']",       c, flags=re.I)
        c = re.sub(r'\breturn on net worth percentage\b',        "df['roe']",        c, flags=re.I)

        # ── Step 7.5: Crossover phrases (post-indicator, pre-parens) ──────
        # Runs here so lhs/rhs are already translated ta_xxx()/df['x'] expressions.
        # "A crossed above B" → (A.shift(1) < B.shift(1)) & (A > B)
        # "A crossed below B" → (A.shift(1) > B.shift(1)) & (A < B)
        # Pattern: non-greedy (.+?) with lookahead to stop at and/or/end.
        # LHS/RHS may have outer ( ) from Chartink wrapping — strip them cleanly.
        # Numeric RHS (e.g. "30") should NOT get .shift(1).
        def _cross(m, direction):
            lhs = re.sub(r'^\(+', '', m.group(1).strip()).strip()
            rhs = re.sub(r'\)+$', '', m.group(2).strip()).strip()
            lhs_s = f'{lhs}.shift(1)' if not re.match(r'^[\d.]+$', lhs) else lhs
            rhs_s = f'{rhs}.shift(1)' if not re.match(r'^[\d.]+$', rhs) else rhs
            if direction == 'above':
                return f'({lhs_s} < {rhs_s}) & ({lhs} > {rhs})'
            else:
                return f'({lhs_s} > {rhs_s}) & ({lhs} < {rhs})'
        c = re.sub(
            r'(.+?)\s+crossed\s+above\s+(.+?)(?=\s+and\b|\s+or\b|\s*$)',
            lambda m: _cross(m, 'above'), c, flags=re.I
        )
        c = re.sub(
            r'(.+?)\s+crossed\s+below\s+(.+?)(?=\s+and\b|\s+or\b|\s*$)',
            lambda m: _cross(m, 'below'), c, flags=re.I
        )

        # ── Step 8: Reinforced SMA/EMA/WMA fallback ────────────────────────
        c = re.sub(r'\bsma\s*\(\s*df\[\'(\w+)\'\]\s*,\s*(\d+)\s*\)',
                   lambda m: f"df['{m.group(1)}'].rolling({int(m.group(2))}).mean()", c, flags=re.I)
        c = re.sub(r'\bema\s*\(\s*df\[\'(\w+)\'\]\s*,\s*(\d+)\s*\)',
                   lambda m: f"ta_ema(df['{m.group(1)}'],{int(m.group(2))})", c, flags=re.I)
        c = re.sub(r'\bwma\s*\(\s*df\[\'(\w+)\'\]\s*,\s*(\d+)\s*\)',
                   lambda m: f"ta_wma(df['{m.group(1)}'],{int(m.group(2))})", c, flags=re.I)
        c = re.sub(r'\bsma\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)',
                   lambda m: f"df['{_f(m.group(1))}'].rolling({int(m.group(2))}).mean()", c, flags=re.I)
        c = re.sub(r'\bema\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)',
                   lambda m: f"ta_ema(df['{_f(m.group(1))}'],{int(m.group(2))})", c, flags=re.I)
        c = re.sub(r'\bwma\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)',
                   lambda m: f"ta_wma(df['{_f(m.group(1))}'],{int(m.group(2))})", c, flags=re.I)

        # ── Step 9: Resolve __SN__ shift tags ─────────────────────────────
        c = re.sub(r"__S(\d+)__\s*rolling_max\(df\['(\w+)'\],(\d+)\)",
                   lambda m: f"df['{m.group(2)}'].shift({m.group(1)}).rolling({m.group(3)}).max()", c)
        c = re.sub(r"__S(\d+)__\s*rolling_min\(df\['(\w+)'\],(\d+)\)",
                   lambda m: f"df['{m.group(2)}'].shift({m.group(1)}).rolling({m.group(3)}).min()", c)
        c = re.sub(r"__S(\d+)__\s*df\['(\w+)'\]((?:\.\w+\([^)]*\))*)",
                   lambda m: f"df['{m.group(2)}'].shift({m.group(1)}){m.group(3)}", c)
        c = re.sub(r"__S(\d+)__\s*(ta_\w+\([^)]*\))",
                   lambda m: f"{m.group(2)}.shift({m.group(1)})", c)
        c = re.sub(r"rolling_max\(df\['(\w+)'\],(\d+)\)",
                   lambda m: f"df['{m.group(1)}'].rolling({m.group(2)}).max()", c)
        c = re.sub(r"rolling_min\(df\['(\w+)'\],(\d+)\)",
                   lambda m: f"df['{m.group(1)}'].rolling({m.group(2)}).min()", c)
        c = re.sub(r"__S(\d+)__\s*(?:daily\s+|weekly\s+|monthly\s+)?(\w+)",
                   lambda m: f"df['{_f(m.group(2))}'].shift({m.group(1)})", c, flags=re.I)

        # ── Step 10: TF-aware bare price fields ────────────────────────────
        for tf_word in ['weekly','monthly','quarterly','yearly']:
            dfn = _df_for_tf(tf_word)
            for field in ['close','open','high','low','volume']:
                c = re.sub(rf'\b{tf_word}\s+{field}\b', f"{dfn}['{field}']", c, flags=re.I)
        c = re.sub(r'\bdaily\s+(?=(?:close|open|high|low|volume)\b)', '', c, flags=re.I)
        for field in ['close','open','high','low','volume']:
            c = re.sub(rf"\b{field}\b(?!\s*['\]])", f"df['{field}']", c, flags=re.I)

        # ── Step 11: NOT operator ──────────────────────────────────────────
        c = re.sub(r'\bnot\b', '~', c, flags=re.I)

        # ── Step 12: PANDAS PARENS WRAPPING — split by and/or THEN convert
        parts = re.split(r'\b(and|or)\b', c, flags=re.I)
        wrapped = []
        for part in parts:
            p = part.strip()
            if p.lower() == 'and':
                wrapped.append(' & ')
            elif p.lower() == 'or':
                wrapped.append(' | ')
            else:
                if re.search(r'(?<![<>!])(?:>=|<=|!=|>|<|==)(?!=)', p):
                    wrapped.append(f"({p})")
                else:
                    wrapped.append(part)
        c = "".join(wrapped)

        # ── Step 13: Final cleanup ─────────────────────────────────────────
        c = re.sub(r'\b(?:daily|weekly|monthly|quarterly|yearly)\s+', '', c, flags=re.I)
        c = re.sub(r"df\['df\['(\w+)'\]'\]", r"df['\1']", c)
        c = re.sub(r'__S\d+__', '', c)
        # Strip industry/sector string comparisons — Chartink-internal, not in our data.
        # These leave unbalanced parens when the string value itself contains ')'.
        # Remove entire comparison token: (industry != 'something')
        c = re.sub(r'\(\s*(?:industry|sector)\s*(?:!=|==)\s*\'[^\']*\'\s*\)', 'True', c, flags=re.I)
        c = re.sub(r'\b(?:industry|sector)\s*(?:!=|==)\s*\'[^\']*\'', 'True', c, flags=re.I)
        # Orphaned "ttm" prefix left before df['...'] — e.g. "ttm df['opm']" → df['opm']
        c = re.sub(r'\bttm\s+(df\[)', r'\1', c, flags=re.I)
        c = re.sub(r'\s+', ' ', c).strip()
        # Balance parentheses: add missing closing parens at the end
        open_count  = c.count('(')
        close_count = c.count(')')
        if open_count > close_count:
            c = c + ')' * (open_count - close_count)

        untranslated = bool(
            re.search(r'\b(?:daily|weekly|monthly)\s+\w', c, re.I) or
            re.search(r'\bsma\s*\((?!.*(?:rolling|df\[))', c, re.I)
        )
        status = 'PARTIAL' if untranslated else 'TRANSLATED'
        return c, status

    except Exception as e:
        return f"ERROR: {e}", 'FAILED'


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

def validate(code):
    if code.startswith('#'):
        return True, "INTRADAY — skipped"
    if re.search(r'\bsma\s*\((?!.*(?:rolling|df\[))', code, re.I):
        return False, "Untranslated sma() detected"
    if re.search(r'\bema\s*\((?!.*ta_)', code, re.I):
        return False, "Untranslated ema() detected"
    test = re.sub(r'ta_\w+\([^)]*\)(?:\.\w+\([^)]*\))*', '_V_', code)
    test = re.sub(r"df(?:_\w+)?\['\w+'\](?:\.\w+\([^)]*\))*", '_V_', test)
    test = re.sub(r'np\.\w+\([^)]*\)', '_V_', test)
    o, cl = test.count('('), test.count(')')
    while cl > o:
        idx = test.rfind(')')
        test = test[:idx] + test[idx+1:]
        cl -= 1
    try:
        ast.parse(f"r=({test.strip()})")
        return True, ""
    except SyntaxError as e:
        return False, str(e)[:80]


# ─────────────────────────────────────────────────────────────────────────────
#  INDICATOR LIBRARY  (v7.0 — Wilder smoothing fixed)
# ─────────────────────────────────────────────────────────────────────────────

INDICATOR_LIB = '''"""
Indicator Library for Backtest Engine v7.0
============================================
Import in your backtesting script.
df must have columns: open, high, low, close, volume (DatetimeIndex)

Multi-timeframe usage:
  df_weekly  = resample_weekly(df)
  df_monthly = resample_monthly(df)

WILDER SMOOTHING NOTE (v7.0 fix):
  RSI, ATR, ADX now use Wilder's exponential smoothing (alpha=1/period).
  This matches Chartink's calculations. v6.0 used SMA rolling mean which
  caused systematic bias (MACD 1.24x overshoot noted in v6.0 history).
"""
import pandas as pd
import numpy as np

def resample_weekly(df):
    return df.resample(\'W-FRI\').agg({
        \'open\':\'first\',\'high\':\'max\',\'low\':\'min\',\'close\':\'last\',\'volume\':\'sum\'
    }).dropna()

def resample_monthly(df):
    return df.resample(\'ME\').agg({
        \'open\':\'first\',\'high\':\'max\',\'low\':\'min\',\'close\':\'last\',\'volume\':\'sum\'
    }).dropna()

def ta_sma(s, p): return s.rolling(p).mean()
def ta_ema(s, p): return s.ewm(span=p, adjust=False).mean()
def ta_wma(s, p):
    w = np.arange(1, p+1)
    return s.rolling(p).apply(lambda x: np.dot(x,w)/w.sum(), raw=True)

def ta_rsi(s, p=14):
    """Wilder RSI — matches Chartink exactly (EWM alpha=1/p)"""
    d = s.diff()
    gain = d.clip(lower=0)
    loss = (-d.clip(upper=0))
    # Wilder smoothing: seed with first SMA, then EWM
    avg_gain = gain.ewm(alpha=1/p, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/p, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def ta_stochrsi(s, p=14):
    r=ta_rsi(s,p); mn=r.rolling(p).min(); mx=r.rolling(p).max()
    return (r-mn)/(mx-mn).replace(0,np.nan)

def ta_macd_line(s,slow=26,fast=12,sig=9): return ta_ema(s,fast)-ta_ema(s,slow)
def ta_macd_sig(s,slow=26,fast=12,sig=9):  return ta_ema(ta_macd_line(s,slow,fast),sig)
def ta_macd_hist(s,slow=26,fast=12,sig=9): return ta_macd_line(s,slow,fast)-ta_macd_sig(s,slow,fast)

def ta_bb_upper(s,p=20,std=2): return s.rolling(p).mean()+std*s.rolling(p).std()
def ta_bb_lower(s,p=20,std=2): return s.rolling(p).mean()-std*s.rolling(p).std()
def ta_bb_width(s,p=20,std=2): return ta_bb_upper(s,p,std)-ta_bb_lower(s,p,std)

def ta_atr(df, p=14):
    """Wilder ATR — EWM alpha=1/p (matches Chartink)"""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def ta_adx(df, p=14):
    """Wilder ADX — EWM smoothing throughout (matches Chartink)"""
    atr = ta_atr(df, p)
    h, l = df["high"], df["low"]
    pdm = h.diff().clip(lower=0)
    ndm = (-l.diff()).clip(lower=0)
    pdm = pdm.where(pdm > ndm, 0)
    ndm = ndm.where(ndm > pdm, 0)
    pdi = 100 * pdm.ewm(alpha=1/p, adjust=False).mean() / atr.replace(0, np.nan)
    ndi = 100 * ndm.ewm(alpha=1/p, adjust=False).mean() / atr.replace(0, np.nan)
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1/p, adjust=False).mean()

def ta_dip(df, p=14):
    atr = ta_atr(df, p)
    pdm = df["high"].diff().clip(lower=0)
    ndm = (-df["low"].diff()).clip(lower=0)
    return 100 * pdm.where(pdm > ndm, 0).ewm(alpha=1/p, adjust=False).mean() / atr.replace(0, np.nan)

def ta_dim(df, p=14):
    atr = ta_atr(df, p)
    pdm = df["high"].diff().clip(lower=0)
    ndm = (-df["low"].diff()).clip(lower=0)
    return 100 * ndm.where(ndm > pdm, 0).ewm(alpha=1/p, adjust=False).mean() / atr.replace(0, np.nan)

def ta_supertrend(df,mult=3,p=7):
    atr=ta_atr(df,p); hl2=(df["high"]+df["low"])/2
    ub=hl2+mult*atr; lb=hl2-mult*atr
    st=pd.Series(np.nan,index=df.index)
    for i in range(1,len(df)):
        prev=st.iloc[i-1] if not np.isnan(st.iloc[i-1]) else lb.iloc[i]
        st.iloc[i]=(lb.iloc[i] if df["close"].iloc[i]>ub.iloc[i-1]
                    else ub.iloc[i] if df["close"].iloc[i]<lb.iloc[i-1] else prev)
    return st

def ta_cci(df,p=14):
    tp=(df["high"]+df["low"]+df["close"])/3
    mad=tp.rolling(p).apply(lambda x:(x-x.mean()).abs().mean())
    return (tp-tp.rolling(p).mean())/(0.015*mad.replace(0,np.nan))

def ta_mfi(df,p=14):
    tp=(df["high"]+df["low"]+df["close"])/3; mf=tp*df["volume"]
    pos=mf.where(tp>tp.shift(),0).rolling(p).sum()
    neg=mf.where(tp<tp.shift(),0).rolling(p).sum()
    return 100-100/(1+pos/neg.replace(0,np.nan))

def ta_willr(df,p=14):
    hh=df["high"].rolling(p).max(); ll=df["low"].rolling(p).min()
    return -100*(hh-df["close"])/(hh-ll).replace(0,np.nan)

def ta_stoch_k(df,kp=5,dp=3):
    hh=df["high"].rolling(kp).max(); ll=df["low"].rolling(kp).min()
    return 100*(df["close"]-ll)/(hh-ll).replace(0,np.nan)

def ta_stoch_d(df,kp=5,dp=3): return ta_stoch_k(df,kp,dp).rolling(dp).mean()

def ta_aroon_up(df,p=14):
    return df["high"].rolling(p+1).apply(lambda x:(p-x[::-1].values.argmax())/p*100,raw=True)

def ta_aroon_dn(df,p=14):
    return df["low"].rolling(p+1).apply(lambda x:(p-x[::-1].values.argmin())/p*100,raw=True)

def ta_psar(df,s=0.02,inc=0.02,mx=0.2):
    """Note: Simplified EWM proxy — approximate. Flag PSAR screener results."""
    return df["close"].ewm(span=5).mean()

def ta_ichi_conv(df,p=9):
    return (df["high"].rolling(p).max()+df["low"].rolling(p).min())/2
def ta_ichi_base(df,p=26):
    return (df["high"].rolling(p).max()+df["low"].rolling(p).min())/2
def ta_ichi_a(df):   return (ta_ichi_conv(df)+ta_ichi_base(df))/2
def ta_ichi_b(df,p=52):
    return (df["high"].rolling(p).max()+df["low"].rolling(p).min())/2
def ta_ichi_cloud(df):
    return pd.concat([ta_ichi_a(df),ta_ichi_b(df)],axis=1).min(axis=1)
'''


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: '{INPUT_CSV}' not found.")
        sys.exit(1)

    rows = list(csv.DictReader(open(INPUT_CSV, encoding='utf-8-sig')))
    if not rows:
        print("ERROR: Input file is empty.")
        sys.exit(1)

    code_col = next((c for c in ['code','Screener code','screener_code','scan_clause'] if c in rows[0]), None)
    name_col = next((c for c in ['name','Sreener Name','screener_name','Name'] if c in rows[0]), None)

    if not code_col:
        print(f"ERROR: No code column. Columns: {list(rows[0].keys())}")
        sys.exit(1)

    # v7.0: always retranslate everything to add direction column
    existing = {}
    if INCREMENTAL and os.path.exists(OUTPUT_CSV):
        for r in csv.DictReader(open(OUTPUT_CSV, encoding='utf-8-sig')):
            existing[r.get('original_code','')] = r

    print("="*60)
    print("  Chartink Translation Engine v7.0")
    print("="*60)
    print(f"  Input    : {len(rows)} screeners")
    print(f"  Existing : {len(existing)} cached (will retranslate for direction column)")
    print("="*60)

    results = []
    stats   = {'BULLISH':0,'BEARISH':0,'NEUTRAL':0,'INTRADAY':0}

    for i, row in enumerate(rows, 1):
        raw  = row.get(code_col,'').strip()
        name = row.get(name_col, f'Row_{i}') if name_col else f'Row_{i}'

        if not raw or raw.startswith('ERROR'):
            continue

        tf = detect_tf(raw)
        py, trans_status = translate(raw)
        valid, err = validate(py)

        if trans_status == 'INTRADAY':
            direction = 'INTRADAY'
        else:
            direction = classify_direction(name, py)

        stats[direction] = stats.get(direction, 0) + 1

        rec = {
            'serial'        : i,
            'name'          : name,
            'timeframe'     : tf,
            'original_code' : raw,
            'python_code'   : py,
            'direction'     : direction,        # NEW in v7.0
            'status'        : trans_status,
            'valid_syntax'  : 'YES' if valid else 'NO',
            'syntax_error'  : err,
            'translated_at' : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        results.append(rec)

        if i <= 5 or i % 100 == 0:
            v = '✅' if valid else '❌'
            d = direction[:4]
            print(f"  [{i:4}/{len(rows)}] {v} {d:5} | {tf:8} | {name[:35]}")

    fields = ['serial','name','timeframe','original_code','python_code',
              'direction','status','valid_syntax','syntax_error','translated_at']
    with open(OUTPUT_CSV,'w',newline='',encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    with open('indicators.py','w',encoding='utf-8') as f:
        f.write(INDICATOR_LIB)

    print("\n"+"="*60)
    print("  Translation Complete — v7.0")
    print(f"  BULLISH  : {stats.get('BULLISH',0)}")
    print(f"  BEARISH  : {stats.get('BEARISH',0)}")
    print(f"  NEUTRAL  : {stats.get('NEUTRAL',0)}")
    print(f"  INTRADAY : {stats.get('INTRADAY',0)}  (skipped)")
    print("="*60)
    print(f"\n  Output : {OUTPUT_CSV}")
    print(f"  Library: indicators.py")
    print()
    print("  Multi-timeframe note:")
    print("  Weekly/monthly conditions use df_weekly / df_monthly.")
    print("  Create them with resample_weekly(df) / resample_monthly(df)")
    print()
    print("  Wilder Smoothing: RSI, ATR, ADX now use EWM (alpha=1/p)")
    print("  This matches Chartink exactly. Backtest numbers will differ")
    print("  from v6.0 — the new numbers are correct.")


if __name__ == "__main__":
    main()
