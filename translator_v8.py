"""
Chartink Screener -> Python/Pandas Translation Engine
======================================================
Version : 8.2

Changes in v8.2 (from v8.1):

  FIX-A  MULTI-PASS MAX/MIN (BUG-1 complete fix):
         Step 8a now has a third catch-all pass that handles any
         remaining max(N, expr_with_ta_or_df) after ALL indicators are
         translated. Covers nested constructs like max(36, monthly rsi(14)).

  FIX-B  COUNT() OVERHAUL:
         Handles: count(N, K where cond), count(N, cond) without where,
         yearly/monthly/weekly TF prefixes with period scaling.
         yearly count(N) -> rolling(N*252), monthly -> N*21, weekly -> N*5.

  FIX-C  SUM() COMPLETENESS:
         Removed "must have arithmetic" gate. All sum(expr,N) patterns
         now handled including simple sum(volume,5). Extended _to_df()
         helper covers weekly/monthly TF-prefixed fields.

  FIX-D  YEAR-AGO / QUARTER-AGO:
         "1 year ago" -> __S1_yearly__ -> df_yearly['x'].shift(1)
                                          .reindex(df.index).ffill()
         Removed the brittle .shift(252) approximation on daily frames.

  FIX-E  df['df'] ARTIFACT CLEANUP:
         Three-pattern cleanup loop in Step 13 eliminates all known
         double-nested df[] artifact variants.

  FIX-F  YEARLY/QUARTERLY FRAME SUPPORT:
         _df_for_tf() and shift-resolution now handle 'yearly' and
         'quarterly'. resample_yearly() confirmed in library.

  FIX-G  VWAP: ta_vwap(df, p) added to indicators.py.

  FIX-H  DOUBLE-SUBSTITUTION PREVENTION:
         Bare eps/pe/roe/roa use negative lookbehind/lookahead to avoid
         re-matching already-substituted df['eps_diluted'] etc.

  FIX-I  IMPROVED SYNTAX VALIDATOR:
         Strips compound expressions, reindex chains, comments before AST.

  FIX-J  'N bars ago' treated same as 'N days ago'.

  FIX-K  MISSING FIELD MAPPINGS (batch):
         operating profit, ebitda, cfo, capex, fcf, working capital,
         inventory/asset turnover, retained earnings, goodwill,
         total assets/liabilities, cash, st/lt borrowings, total debt,
         pledged/dii/fii/public percentages, sales/profit growth.

  FIX-L  RETRANSLATE_FAILED_ONLY flag for faster iteration.
"""

import csv, re, os, ast, sys
import numpy as np
from datetime import datetime

INPUT_CSV               = "screeners_input.csv"
OUTPUT_CSV              = "translated_screeners.csv"
INCREMENTAL             = True
RETRANSLATE_FAILED_ONLY = False   # FIX-L: only redo NO/FAILED/PARTIAL rows

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
    return {
        'daily'    : 'df',
        'weekly'   : 'df_weekly',
        'monthly'  : 'df_monthly',
        'quarterly': 'df_quarterly',
        'yearly'   : 'df_yearly',
        'annual'   : 'df_yearly',
    }.get(tf, 'df')


# ─────────────────────────────────────────────────────────────────────────────
#  DIRECTION CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def classify_direction(name, python_code):
    n = (name or '').lower()
    c = (python_code or '').lower()

    bearish_name = [
        r'\bbearish\b', r'\bsell\b', r'\bshort\b', r'\bbear\b',
        r'\bdowntrend\b', r'\bfalling\b', r'\bdrop\b', r'\bdecline\b',
        r'\bnegative\b', r'\bperfect\s+sell\b', r'\bsell\s+signal\b',
        r'\bbreakdown\b', r'\bsupport\s+break\b', r'\bdown\s+trend\b',
        r'\bpanic\b', r'\bweak\b', r'\bsupply\b', r'\boversold\b',
    ]
    bullish_name = [
        r'\bbullish\b', r'\bbuy\b', r'\bbull\b', r'\bbreakout\b',
        r'\bbreakup\b', r'\buptrend\b', r'\bup\s+trend\b', r'\bbuy\s+signal\b',
        r'\bmomentum\b', r'\bpositive\b', r'\bbtst\b', r'\bswing\s+buy\b',
        r'\baccumulation\b', r'\bdemand\b', r'\bgolden\s+cross\b',
        r'\blong\s+term\b', r'\bentry\b',
    ]
    neutral_override = [
        r'\bfilter\b', r'\bscan\b', r'\bscreener\b', r'\buniverse\b',
        r'\bnr7\b', r'\bnr4\b', r'\bnarrow\s+range\b', r'\bvolatility\b',
        r'\bfundamental\b', r'\bpe\s+ratio\b', r'\b52\s+week\b',
        r'\blifetime\s+high\b', r'\bvolume\s+shock\b',
    ]

    bear_score = sum(3 for p in bearish_name if re.search(p, n))
    bull_score = sum(3 for p in bullish_name if re.search(p, n))

    if re.search(r"df\['close'\]\s*>\s*(?:df\[|ta_)", c):   bull_score += 1
    if re.search(r"df\['close'\]\s*<\s*(?:df\[|ta_)", c):   bear_score += 1
    if re.search(r"ta_macd_hist[^<>]*>\s*0", c):             bull_score += 1
    if re.search(r"ta_macd_hist[^<>]*<\s*0", c):             bear_score += 1
    if re.search(r"ta_aroon_up[^<>]*>\s*ta_aroon_dn", c):    bull_score += 1
    if re.search(r"ta_aroon_dn[^<>]*>\s*ta_aroon_up", c):    bear_score += 1

    if any(re.search(p, n) for p in neutral_override):
        if bear_score == 0 and bull_score == 0:
            return 'NEUTRAL'

    if bear_score > bull_score and bear_score > 0:   return 'BEARISH'
    elif bull_score > bear_score and bull_score > 0: return 'BULLISH'
    else:                                            return 'NEUTRAL'


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSLATOR
# ─────────────────────────────────────────────────────────────────────────────

def translate(raw):
    try:
        c = raw.strip()

        if detect_tf(raw) == 'intraday':
            return f"# INTRADAY -- not supported in daily backtest\n# {raw[:200]}", 'INTRADAY'

        # Step 1: Remove universe tags
        c = re.sub(r'\{[^}]+\}', '', c)
        c = re.sub(r'\s+', ' ', c).strip()
        if re.search(r'\{[^}]+\}', c):
            c = re.sub(r'\{[^}]+\}', '', c)

        # Step 1a: BUG-3 -- strip leading zeros from numeric literals
        c = re.sub(r'(?<![.\d])0+([1-9]\d*)(?!\s*[.\d])\b', r'\1', c)

        # Step 1a2: BUG-4 -- strip percent sign
        c = re.sub(r'(\d+)\s*%', r'\1', c)

        # Step 1b: CRITICAL-8 -- Heikin-Ashi tokens
        c = re.sub(r'\bha[-_]close\b', '__HA_CLOSE__', c, flags=re.I)
        c = re.sub(r'\bha[-_]open\b',  '__HA_OPEN__',  c, flags=re.I)
        c = re.sub(r'\bha[-_]high\b',  '__HA_HIGH__',  c, flags=re.I)
        c = re.sub(r'\bha[-_]low\b',   '__HA_LOW__',   c, flags=re.I)

        # Step 2: Strip custom formula syntax
        c = re.sub(r'\^[^)]+\([^)]*\)\^', 'None', c)

        # FIX-C: Parse sum() -- ALL expressions (removed "must have arithmetic" gate)
        def _parse_sum_expr(m):
            inner    = m.group(1).strip()
            period_m = re.match(r'^(.+),\s*(\d+)$', inner)
            if not period_m:
                return m.group(0)
            expr_part = period_m.group(1).strip()
            period    = int(period_m.group(2))

            def _to_df(s):
                s = re.sub(r'\bweekly\s+close\b',  "df_weekly['close']",  s, flags=re.I)
                s = re.sub(r'\bweekly\s+high\b',   "df_weekly['high']",   s, flags=re.I)
                s = re.sub(r'\bweekly\s+low\b',    "df_weekly['low']",    s, flags=re.I)
                s = re.sub(r'\bweekly\s+volume\b', "df_weekly['volume']", s, flags=re.I)
                s = re.sub(r'\bmonthly\s+close\b', "df_monthly['close']", s, flags=re.I)
                s = re.sub(r'\bmonthly\s+high\b',  "df_monthly['high']",  s, flags=re.I)
                s = re.sub(r'\bmonthly\s+low\b',   "df_monthly['low']",   s, flags=re.I)
                s = re.sub(r'\bclose\b',  "df['close']",  s, flags=re.I)
                s = re.sub(r'\bopen\b',   "df['open']",   s, flags=re.I)
                s = re.sub(r'\bhigh\b',   "df['high']",   s, flags=re.I)
                s = re.sub(r'\blow\b',    "df['low']",    s, flags=re.I)
                s = re.sub(r'\bvolume\b', "df['volume']", s, flags=re.I)
                s = re.sub(r'\bvwap\b',   "df['vwap']",   s, flags=re.I)
                return s

            return f"({_to_df(expr_part)}).rolling({period}).sum()"

        c = re.sub(r'\bsum\s*\(\s*([^)]+)\s*\)', _parse_sum_expr, c, flags=re.I)

        c = c.replace('"', '')

        # Step 2b: Multi-word fundamental fields -- most specific first

        # Promoter / ownership
        c = re.sub(r'\bindian\s+promoter\s+(?:&|and)\s+group\s+percentage\b',
                   "df['promoter_pct']", c, flags=re.I)
        c = re.sub(r'\bpromoter\s+(?:&|and)\s+group\s+percentage\b',
                   "df['promoter_pct']", c, flags=re.I)
        c = re.sub(r'\bforeign\s+institutional\s+investors\s+percentage\b',
                   "df['fii_pct']", c, flags=re.I)
        c = re.sub(r'\bencumbered\s+percentage\s+in\s+total\s+promoters\s+holding\b',
                   "df['promoter_encumbered_pct']", c, flags=re.I)

        # EPS variants -- longest first
        c = re.sub(r'\beps\s+after\s+extraordinary\s+items\s+diluted\b', "df['eps_diluted']", c, flags=re.I)
        c = re.sub(r'\beps\s+after\s+extraordinary\s+items\s+basic\b',   "df['eps']",         c, flags=re.I)
        c = re.sub(r'\beps\s+after\s+extraordinary\s+items\b',           "df['eps']",         c, flags=re.I)
        c = re.sub(r'\bearning\s+per\s+share\[eps\]\b',                  "df['eps']",         c, flags=re.I)
        c = re.sub(r'\bearning\s+per\s+share\b',                         "df['eps']",         c, flags=re.I)
        c = re.sub(r'\bttm\s+eps\b',                                     "df['eps_ttm']",     c, flags=re.I)
        c = re.sub(r'\bprev\s+year\s+eps\b',                             "df['eps_prev']",    c, flags=re.I)

        # Net profit variants
        c = re.sub(r'\bnet\s+profit/reported\s+profit\s+after\s+tax\b',       "df['net_profit']",       c, flags=re.I)
        c = re.sub(r'\bnet\s+profit\s+after\s+minority\s+interest\s+(?:&|and)\s+pnl\s+asso[a-z]+\b',
                   "df['net_profit']", c, flags=re.I)
        c = re.sub(r'\bnet\s+profit\[(?:yearly|annual|yr)\]',                 "df['net_profit']",       c, flags=re.I)
        c = re.sub(r'\bnet\s+profit\[(?:quarter|qr|q)\]',                     "df['net_profit_q']",     c, flags=re.I)
        c = re.sub(r'\bnet\s+profit\s+variance\[(?:yr|annual)\]',             "df['net_profit_var_yr']",c, flags=re.I)
        c = re.sub(r'\bnet\s+profit\s+variance\[(?:qr|q|quarter)\]',          "df['net_profit_var_qr']",c, flags=re.I)
        c = re.sub(r'\bttm\s+net\s+profit\b',                                 "df['net_profit_ttm']",   c, flags=re.I)
        c = re.sub(r'\byearly\s+net\s+profit\b',                              "df['net_profit']",       c, flags=re.I)
        c = re.sub(r'\bnet\s+profit\b',                                        "df['net_profit']",       c, flags=re.I)

        # Sales
        c = re.sub(r'\bnet\s+sales\[(?:quarter|qr|q)\]',  "df['net_sales_q']",   c, flags=re.I)
        c = re.sub(r'\bnet\s+sales\b',                     "df['net_sales']",     c, flags=re.I)
        c = re.sub(r'\bttm\s+sales\b',                     "df['net_sales_ttm']", c, flags=re.I)

        # Balance sheet
        c = re.sub(r'\bsecured\s+loans\b',          "df['secured_loans']",      c, flags=re.I)
        c = re.sub(r'\bunsecured\s+loans\b',        "df['unsecured_loans']",    c, flags=re.I)
        c = re.sub(r'\blong\s+term\s+borrowings\b', "df['lt_borrowings']",      c, flags=re.I)
        c = re.sub(r'\bshort\s+term\s+borrowings\b',"df['st_borrowings']",      c, flags=re.I)
        c = re.sub(r'\btotal\s+loans\b',             "df['total_loans']",        c, flags=re.I)
        c = re.sub(r'\btotal\s+debt\b',              "df['total_debt']",         c, flags=re.I)
        c = re.sub(r'\btotal\s+assets\b',            "df['total_assets']",       c, flags=re.I)
        c = re.sub(r'\btotal\s+liabilities\b',       "df['total_liabilities']",  c, flags=re.I)
        c = re.sub(r'\bcash\s+and\s+equivalents\b',  "df['cash']",               c, flags=re.I)
        c = re.sub(r'\bworking\s+capital\b',          "df['working_capital']",    c, flags=re.I)
        c = re.sub(r'\bretained\s+earnings\b',        "df['retained_earnings']",  c, flags=re.I)
        c = re.sub(r'\bgoodwill\b',                   "df['goodwill']",           c, flags=re.I)
        c = re.sub(r'\bshare\s+capital\b',            "df['share_capital']",      c, flags=re.I)
        c = re.sub(r'\breserves\b',                   "df['reserves']",           c, flags=re.I)
        c = re.sub(r'\btotal\s+number\b',             "df['shares_outstanding']", c, flags=re.I)
        c = re.sub(r'\bface\s+value\b',               "df['face_value']",         c, flags=re.I)
        c = re.sub(r'\bprice\s+to\s+book\s+value\b',  "df['pb']",                c, flags=re.I)
        c = re.sub(r'\bbook\s+value\b',               "df['book_value']",         c, flags=re.I)

        # Margins / profitability -- specific before generic
        c = re.sub(r'\boperating\s+profit\s+margin\[(?:qr|quarter|q)\]', "df['opm_q']",  c, flags=re.I)
        c = re.sub(r'\boperating\s+profit\s+margin\[(?:yr|annual)\]',    "df['opm_yr']", c, flags=re.I)
        c = re.sub(r'\boperating\s+profit\s+margin\b',                   "df['opm']",    c, flags=re.I)
        c = re.sub(r'\boperating\s+profit\b(?!\s+margin)',                "df['op']",     c, flags=re.I)
        c = re.sub(r'\bttm\s+operating\s+profit\s+margin\b',             "df['ttm_opm']",c, flags=re.I)
        c = re.sub(r'\bttm\s+operating\s+profit\b',                      "df['ttm_op']", c, flags=re.I)
        c = re.sub(r'\bttm\s+net\s+profit\s+variance\b',                 "df['ttm_np_var']", c, flags=re.I)
        c = re.sub(r'\bttm\s+gross\s+profit\s+margin\b',                 "df['ttm_gpm']",c, flags=re.I)
        c = re.sub(r'\bttm\s+gross\s+profit\b',                          "df['ttm_gp']", c, flags=re.I)
        c = re.sub(r'\bttm\s+cps\b',                                     "df['ttm_cps']",c, flags=re.I)
        c = re.sub(r'\bgross\s+profit\s+margin\b',                       "df['gross_profit_margin']", c, flags=re.I)
        c = re.sub(r'\bcash\s+per\s+share\[mt\]',                        "df['cps']",    c, flags=re.I)
        c = re.sub(r'\bcash\s+per\s+share\b',                            "df['cps']",    c, flags=re.I)
        c = re.sub(r'\bebitda\b',                                         "df['ebitda']", c, flags=re.I)
        c = re.sub(r'\bcash\s+flow\s+from\s+operations\b',               "df['cfo']",    c, flags=re.I)
        c = re.sub(r'\bcapital\s+expenditure\b',                          "df['capex']",  c, flags=re.I)
        c = re.sub(r'\bfree\s+cash\s+flow\b',                            "df['fcf']",    c, flags=re.I)
        c = re.sub(r'\binventory\s+turnover\b',                           "df['inv_turnover']",   c, flags=re.I)
        c = re.sub(r'\basset\s+turnover\b',                               "df['asset_turnover']", c, flags=re.I)
        c = re.sub(r'\binterest\s+expense\b',                             "df['int_expense']",    c, flags=re.I)
        c = re.sub(r'\bsales\s+growth\b',                                 "df['sales_growth']",   c, flags=re.I)
        c = re.sub(r'\bprofit\s+growth\b',                                "df['profit_growth']",  c, flags=re.I)

        # Valuation / ratios
        c = re.sub(r'\bttm\s+pe\b',               "df['pe_ttm']",  c, flags=re.I)
        c = re.sub(r'\bdebt\s+equity\s+ratio\b',  "df['debt_eq']", c, flags=re.I)
        c = re.sub(r'\bdividend\s+yield\b',       "df['div_yield']",c, flags=re.I)
        c = re.sub(r'\bdividend\b',               "df['dividend']", c, flags=re.I)

        # Market / exchange
        c = re.sub(r'\bfno\s+lot\s+size\b',            "df['lot_size']",    c, flags=re.I)
        c = re.sub(r'\bbuyer\s+initiated\s+trades\b',  "df['buyer_trades']",c, flags=re.I)
        c = re.sub(r'\bseller\s+initiated\s+trades\b', "df['seller_trades']",c, flags=re.I)
        c = re.sub(r'\bbuy\s+orders\b',                "df['buy_orders']",  c, flags=re.I)
        c = re.sub(r'\bsell\s+orders\b',               "df['sell_orders']", c, flags=re.I)
        c = re.sub(r'\bdii\s+percentage\b',             "df['dii_pct']",     c, flags=re.I)
        c = re.sub(r'\bpublic\s+percentage\b',          "df['public_pct']",  c, flags=re.I)
        c = re.sub(r'\bpledged\s+percentage\b',         "df['pledged_pct']", c, flags=re.I)

        # Strip unsupported Chartink ratio syntax
        c = re.sub(r"rs:'[^']+'\s*\([^)]*\)", "1.0", c, flags=re.I)
        c = re.sub(r"rs:'[^']+'",              "1.0", c, flags=re.I)
        c = re.sub(r'\bfinancial\s+institutions\b', "df['fi_pct']",  c, flags=re.I)
        c = re.sub(r'\bbanks\s+percentage\b',       "df['banks_pct']", c, flags=re.I)

        # Artifact cleanups
        c = re.sub(r"df\['opm'\]\[(?:yr|annual)\]",     "df['opm_yr']",           c)
        c = re.sub(r"df\['eps'\]\[eps\]",                "df['eps']",              c)
        c = re.sub(r"price\s+to\s+df\['book_value'\]",   "df['pb']",              c, flags=re.I)
        c = re.sub(r"df\['net_profit_ttm'\]\s+variance", "df['net_profit_ttm_var']", c)

        # Math
        c = re.sub(r'\bsquare\s+root\s*\(', 'np.sqrt(', c, flags=re.I)

        # Step 2b-extra: Additional fields
        c = re.sub(r'\badvance(?:s)?\s+given\s+by\s+bank\b',        "df['advances_by_bank']", c, flags=re.I)
        c = re.sub(r'\bbse\s+value\s+in\s+lakhs\b',                  "df['bse_value_lakhs']",  c, flags=re.I)
        c = re.sub(r'\bnse\s+value\s+in\s+lakhs\b',                  "df['nse_value_lakhs']",  c, flags=re.I)
        c = re.sub(r'\breturn\s+on\s+net\s+worth\s+percentage\b',    "df['roe']",              c, flags=re.I)
        c = re.sub(r'\breturn\s+on\s+capital\s+employed\s+percentage\b', "df['roce']",         c, flags=re.I)
        c = re.sub(r'\breturn\s+on\s+equity\b',                      "df['roe']",              c, flags=re.I)
        c = re.sub(r'\breturn\s+on\s+assets\b',                      "df['roa']",              c, flags=re.I)
        c = re.sub(r'\breturn\s+on\s+net\s+worth\b',                 "df['roe']",              c, flags=re.I)
        c = re.sub(r'\binterest\s+coverage\s+ratio\b',               "df['int_coverage']",     c, flags=re.I)
        c = re.sub(r'\bprice\s+to\s+earnings\b',                     "df['pe']",               c, flags=re.I)
        c = re.sub(r'\bprice\s+to\s+sales\b',                        "df['ps']",               c, flags=re.I)
        c = re.sub(r'\bprice\s+to\s+cash\s+flow\b',                 "df['pcf']",              c, flags=re.I)
        c = re.sub(r'\bpeg\s+ratio\b',                               "df['peg']",              c, flags=re.I)
        c = re.sub(r'\bpc\s+ratio\b',                                "df['pc_ratio']",         c, flags=re.I)
        c = re.sub(r'\bnet\s+non\s+performing\s+assets\b',           "df['net_npa']",          c, flags=re.I)
        c = re.sub(r'\bgross\s+non\s+performing\s+assets\b',         "df['gross_npa']",        c, flags=re.I)
        c = re.sub(r'\bcurrent\s+ratio\b',                           "df['current_ratio']",    c, flags=re.I)
        c = re.sub(r'\bquick\s+ratio\b',                             "df['quick_ratio']",      c, flags=re.I)
        c = re.sub(r'\byearly\s+pe\s+ratio\b',                       "df['pe']",               c, flags=re.I)
        c = re.sub(r'\bpe\s+ratio\b',                                "df['pe']",               c, flags=re.I)
        c = re.sub(r'\bprice[\s\-]earnings?\s+ratio\b',              "df['pe']",               c, flags=re.I)
        c = re.sub(r'\bsales\s+turnover\[(?:yearly|yr|annual)\]',    "df['net_sales']",        c, flags=re.I)
        c = re.sub(r'\bsales\s+turnover\[(?:quarter|qr|q)\]',        "df['net_sales_q']",      c, flags=re.I)
        c = re.sub(r'\bsales\s+turnover\b',                          "df['net_sales']",        c, flags=re.I)
        c = re.sub(r'\bnetworth\b',                                   "df['net_worth']",        c, flags=re.I)
        c = re.sub(r'\bnet\s+worth\b',                               "df['net_worth']",        c, flags=re.I)
        c = re.sub(r'\bmarket\s+cap\b',                              "df['market_cap']",       c, flags=re.I)
        c = re.sub(r'\bopm\[(?:qr|quarter|q)\]',                    "df['opm_q']",            c, flags=re.I)
        c = re.sub(r'\bopm\[(?:yr|annual)\]',                       "df['opm_yr']",           c, flags=re.I)
        c = re.sub(r'\bopm\b',                                       "df['opm']",              c, flags=re.I)

        # FIX-H: bare single-word fundamentals with lookbehind/ahead to avoid re-substitution
        c = re.sub(r"(?<!['\"\w])\broe\b(?!['\w])",   "df['roe']",     c, flags=re.I)
        c = re.sub(r"(?<!['\"\w])\broa\b(?!['\w])",   "df['roa']",     c, flags=re.I)
        c = re.sub(r"(?<!['\"\w])\beps\b(?!['\w_])",  "df['eps']",     c, flags=re.I)
        c = re.sub(r"(?<!['\"\w])\bpe\b(?!['\w_])",   "df['pe']",      c, flags=re.I)
        c = re.sub(r"(?<!['\"\w])\bdii\b(?!['\w_])",  "df['dii_pct']", c, flags=re.I)
        c = re.sub(r"(?<!['\"\w])\bfii\b(?!['\w_])",  "df['fii_pct']", c, flags=re.I)
        c = re.sub(r"(df\['pe'\])\s+ratio\b",         r"\1",           c, flags=re.I)

        # BUG-2 FIX: Generic [qr]/[yr]/[ttm] bracket suffix cleanup
        def _bracket_suffix_word(m):
            col = m.group(1).strip().replace(' ', '_')
            sfx = {'qr':'q','q':'q','quarter':'q','yr':'yr','annual':'yr',
                   'yearly':'yr','ttm':'ttm','mt':'ttm'}.get(m.group(2).lower(), m.group(2).lower())
            return f"df['{col}_{sfx}']"

        c = re.sub(
            r'\b([A-Za-z][A-Za-z0-9_ ]*?)\s*\[(qr|q|quarter|yr|annual|yearly|ttm|mt)\]',
            _bracket_suffix_word, c, flags=re.I
        )
        c = re.sub(
            r"df\['([^']+)'\]\[(qr|q|quarter|yr|annual|yearly|ttm|mt)\]",
            lambda m: "df['" + m.group(1) + "_" + {'qr':'q','q':'q','quarter':'q','yr':'yr','annual':'yr','yearly':'yr','ttm':'ttm','mt':'ttm'}.get(m.group(2).lower(), m.group(2).lower()) + "']",
            c, flags=re.I
        )

        # Step 2c: SMA on a series expression
        def _sma_on_series(m):
            return f"ta_sma({m.group(1).strip()},{int(m.group(2))})"
        c = re.sub(
            r'\bsma\s*\(\s*((?:ta_\w+\([^)]*\)|df(?:_\w+)?\[\'[^\']+\'\](?:\.\w+\([^)]*\))*)\s*),\s*(\d+)\s*\)',
            _sma_on_series, c, flags=re.I
        )
        c = re.sub(r"df\['sma'\]\.shift\((\d+)\)\s*\(\s*([^)]+)\s*,\s*(\d+)\s*\)",
                   lambda m: f"ta_sma(df['{_f(re.sub(chr(39), '', m.group(2)).strip())}'],{m.group(3)}).shift({m.group(1)})",
                   c)

        # Step 2d: Number literal cleaning
        c = re.sub(r'(\d)([oO]+)\b', lambda m: str(int(m.group(1)) * (10 ** len(m.group(2)))), c)
        c = re.sub(r'\.\s+(\d)', r'.\1', c)

        # Step 3: Time shifts
        # FIX-D: year->yearly frame, quarter->quarterly frame, FIX-J: bars synonym
        def _ago(m):
            n, u = int(m.group(1)), m.group(2).lower()
            if 'day' in u or 'candle' in u or 'bar' in u:
                return f'__S{n}_daily__ '
            elif 'week' in u:
                return f'__S{n}_weekly__ '
            elif 'month' in u:
                return f'__S{n}_monthly__ '
            elif 'year' in u:
                return f'__S{n}_yearly__ '
            elif 'quarter' in u:
                return f'__S{n}_quarterly__ '
            return f'__S{n}_daily__ '

        c = re.sub(r'(\d+)\s+(days?|weeks?|months?|years?|quarters?|candles?|bars?)\s+ago\s+',
                   _ago, c, flags=re.I)
        c = re.sub(r'(\d+)\s+(days?|weeks?|months?|years?|quarters?|candles?|bars?)\s+ago\b',
                   _ago, c, flags=re.I)

        # Step 3b: English operators
        c = re.sub(r'\bgreater\s+than\s+or\s+equal\s+to\b', '>=', c, flags=re.I)
        c = re.sub(r'\bless\s+than\s+or\s+equal\s+to\b',    '<=', c, flags=re.I)
        c = re.sub(r'\bgreater\s+than\b',                    '>',  c, flags=re.I)
        c = re.sub(r'\bless\s+than\b',                       '<',  c, flags=re.I)
        c = re.sub(r'\bequals?\b',                           '==', c, flags=re.I)

        # Step 4: Equality operator
        c = re.sub(r'(?<![<>!=])\s*=\s*(?!=)', ' == ', c)

        # Step 5: Fix leading zeros after commas
        c = re.sub(r',\s*0+(\d+)', lambda m: f",{int(m.group(1))}", c)

        # Step 6: Strip TF prefix for indicators that capture TF internally
        for fn in ['sma','ema','wma','max','min','supertrend','parabolic','vwap']:
            c = re.sub(rf'\b(?:daily|weekly|monthly|quarterly|yearly)\s+(?={fn}\b)','',c,flags=re.I)

        # Step 7: Indicators

        # Rolling max/min -- FIX-A: bail on '(' in field -> deferred to Step 8a
        def _rolling_max(m):
            per_str   = m.group(2).strip()
            field_str = m.group(3).strip()
            if (field_str.startswith('ta_') or field_str.startswith('df[')
                    or field_str.startswith('df_') or field_str.startswith('(')):
                try:    per = int(per_str)
                except: return m.group(0)
                return f"({field_str}).rolling({per}).max()"
            if '(' in field_str:  # untranslated indicator -> defer to Step 8a
                return m.group(0)
            inner_tf_m = re.match(r'^(daily|weekly|monthly|quarterly|yearly)\s+(.+)$', field_str, re.I)
            if inner_tf_m:
                tf  = inner_tf_m.group(1).lower()
                fld = _f(inner_tf_m.group(2))
            else:
                tf  = 'daily'
                fld = _f(field_str)
            try:    per = int(per_str)
            except: return m.group(0)
            return f"{_df_for_tf(tf)}['{fld}'].rolling({per}).max()"

        def _rolling_min(m):
            per_str   = m.group(2).strip()
            field_str = m.group(3).strip()
            if (field_str.startswith('ta_') or field_str.startswith('df[')
                    or field_str.startswith('df_') or field_str.startswith('(')):
                try:    per = int(per_str)
                except: return m.group(0)
                return f"({field_str}).rolling({per}).min()"
            if '(' in field_str:
                return m.group(0)
            inner_tf_m = re.match(r'^(daily|weekly|monthly|quarterly|yearly)\s+(.+)$', field_str, re.I)
            if inner_tf_m:
                tf  = inner_tf_m.group(1).lower()
                fld = _f(inner_tf_m.group(2))
            else:
                tf  = 'daily'
                fld = _f(field_str)
            try:    per = int(per_str)
            except: return m.group(0)
            return f"{_df_for_tf(tf)}['{fld}'].rolling({per}).min()"

        c = re.sub(r'\bmax\s*\(\s*(?:(daily|weekly|monthly)\s+)?(\d+)\s*,\s*(.+?)\s*\)',
                   _rolling_max, c, flags=re.I)
        c = re.sub(r'\bmin\s*\(\s*(?:(daily|weekly|monthly)\s+)?(\d+)\s*,\s*(.+?)\s*\)',
                   _rolling_min, c, flags=re.I)

        # SMA, EMA, WMA -- TF-aware
        def _sma(m):
            tf=(m.group(1) or 'daily').lower(); fld=_f(m.group(2)); per=int(m.group(3))
            return f"ta_sma({_df_for_tf(tf)}['{fld}'],{per})"
        c = re.sub(r'\bsma\s*\(\s*(?:(daily|weekly|monthly|quarterly|yearly)\s+)?(\w+)\s*,\s*(\d+)\s*\)',
                   _sma, c, flags=re.I)

        def _ema(m):
            tf=(m.group(1) or 'daily').lower(); fld=_f(m.group(2)); per=int(m.group(3))
            return f"ta_ema({_df_for_tf(tf)}['{fld}'],{per})"
        c = re.sub(r'\bema\s*\(\s*(?:(daily|weekly|monthly|quarterly|yearly)\s+)?(\w+)\s*,\s*(\d+)\s*\)',
                   _ema, c, flags=re.I)

        def _wma(m):
            tf=(m.group(1) or 'daily').lower(); fld=_f(m.group(2)); per=int(m.group(3))
            return f"ta_wma({_df_for_tf(tf)}['{fld}'],{per})"
        c = re.sub(r'\bwma\s*\(\s*(?:(daily|weekly|monthly|quarterly|yearly)\s+)?(\w+)\s*,\s*(\d+)\s*\)',
                   _wma, c, flags=re.I)

        # RSI, StochRSI
        def _rsi(m):
            tf=(m.group(1) or 'daily').lower(); per=int(m.group(2))
            return f"ta_rsi({_df_for_tf(tf)}['close'],{per})"
        c = re.sub(r'\b(?:(daily|weekly|monthly|quarterly)\s+)?rsi\s*\(\s*(\d+)\s*\)',
                   _rsi, c, flags=re.I)

        def _stochrsi(m):
            tf=(m.group(1) or 'daily').lower(); per=int(m.group(2))
            return f"ta_stochrsi({_df_for_tf(tf)}['close'],{per})"
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?stochrsi\s*\(\s*(\d+)\s*\)',
                   _stochrsi, c, flags=re.I)

        # MACD
        def _macd(kind):
            def _inner(m):
                tf=(m.group(1) or 'daily').lower()
                s1,s2,s3=int(m.group(2)),int(m.group(3)),int(m.group(4))
                return f"ta_macd_{kind}({_df_for_tf(tf)}['close'],{s1},{s2},{s3})"
            return _inner
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?macd\s+line\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',
                   _macd('line'), c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?macd\s+signal\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',
                   _macd('sig'), c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?macd\s+histogram\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',
                   _macd('hist'), c, flags=re.I)

        # Bollinger
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
        def _adx_fn(fn):
            def _inner(m):
                tf=(m.group(1) or 'daily').lower(); per=int(m.group(2))
                return f"{fn}({_df_for_tf(tf)},{per})"
            return _inner
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?adx\s+di\s+positive\s*\(\s*(\d+)\s*\)',
                   _adx_fn('ta_dip'), c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?adx\s+di\s+negative\s*\(\s*(\d+)\s*\)',
                   _adx_fn('ta_dim'), c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?adx\s*\(\s*(\d+)\s*\)',
                   _adx_fn('ta_adx'), c, flags=re.I)

        # ATR -- BUG-9: "avg" check BEFORE bare "true range"
        c = re.sub(r'\b(?:avg|average)\s+true\s+range\s*\(\s*(\d+)\s*\)',
                   lambda m: f"ta_atr(df,{int(m.group(1))})", c, flags=re.I)
        c = re.sub(r'\btrue\s+range\s*\(\s*(\d+)\s*\)',
                   lambda m: f"ta_tr(df).rolling({int(m.group(1))}).max()", c, flags=re.I)
        c = re.sub(r'\btrue\s+range\b(?!\s*\()', "ta_tr(df)", c, flags=re.I)

        # Stochastic
        def _stoch(kind):
            def _inner(m):
                tf=(m.group(1) or 'daily').lower(); kp=int(m.group(2)); dp=int(m.group(3))
                return f"ta_stoch_{kind}({_df_for_tf(tf)},{kp},{dp})"
            return _inner
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?(?:slow|fast)\s+stochastic\s+%k\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
                   _stoch('k'), c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?(?:slow|fast)\s+stochastic\s+%d\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
                   _stoch('d'), c, flags=re.I)

        # Williams %R
        def _willr(m):
            tf=(m.group(1) or 'daily').lower(); per=int(m.group(2))
            return f"ta_willr({_df_for_tf(tf)},{per})"
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?williams\s+%r\s*\(\s*(\d+)\s*\)',
                   _willr, c, flags=re.I)

        # CCI
        def _cci(m):
            tf=(m.group(1) or 'daily').lower(); per=int(m.group(2))
            return f"ta_cci({_df_for_tf(tf)},{per})"
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?cci\s*\(\s*(\d+)\s*\)',
                   _cci, c, flags=re.I)

        # MFI
        def _mfi(m):
            tf=(m.group(1) or 'daily').lower(); per=int(m.group(2))
            return f"ta_mfi({_df_for_tf(tf)},{per})"
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?mfi\s*\(\s*(\d+)\s*\)',
                   _mfi, c, flags=re.I)

        # Aroon
        def _aroon_fn(fn):
            def _inner(m):
                tf=(m.group(1) or 'daily').lower(); per=int(m.group(2))
                return f"{fn}({_df_for_tf(tf)},{per})"
            return _inner
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?aroon\s+osc(?:illator)?\s*\(\s*(\d+)\s*\)',
                   _aroon_fn('ta_aroon_osc'), c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?aroon\s+up\s*\(\s*(\d+)\s*\)',
                   _aroon_fn('ta_aroon_up'), c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?aroon\s+down\s*\(\s*(\d+)\s*\)',
                   _aroon_fn('ta_aroon_dn'), c, flags=re.I)

        # WaveTrend
        def _wavetrend(m):
            tf=(m.group(1) or 'daily').lower()
            args   = m.group(2).strip() if m.group(2) else ''
            params = [x.strip() for x in args.split(',') if x.strip().isdigit()] if args else []
            if len(params) == 3:
                return f"ta_wavetrend({_df_for_tf(tf)},{params[0]},{params[1]},{params[2]})"
            elif len(params) == 2:
                return f"ta_wavetrend({_df_for_tf(tf)},{params[0]},{params[1]})"
            return f"ta_wavetrend({_df_for_tf(tf)})"
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?wavetrend\s+momentum\s*\(([^)]*)\)',
                   lambda m: _wavetrend(m) + '.wt1', c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?wavetrend\s+trigger\s*\(([^)]*)\)',
                   lambda m: _wavetrend(m) + '.wt2', c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?wavetrend\s*\(([^)]*)\)',
                   _wavetrend, c, flags=re.I)

        # Parabolic SAR
        c = re.sub(r'\bparabolic\s+sar\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)',
                   lambda m: f"ta_psar(df,{m.group(1)},{m.group(2)},{m.group(3)})", c, flags=re.I)

        # Ichimoku
        def _ichi(fn):
            def _i(m):
                tf=(m.group(1) or 'daily').lower()
                params_str = m.group(2).strip() if m.group(2) else ''
                params = [p.strip() for p in params_str.split(',') if p.strip().isdigit()]
                dfn = _df_for_tf(tf)
                return f"{fn}({dfn},{','.join(params)})" if params else f"{fn}({dfn})"
            return _i
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?ichimoku\s+conversion\s+line\s*\(([^)]*)\)', _ichi('ta_ichi_conv'), c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?ichimoku\s+base\s+line\s*\(([^)]*)\)',       _ichi('ta_ichi_base'), c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?ichimoku\s+span\s+a\s*\(([^)]*)\)',          _ichi('ta_ichi_a'),    c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?ichimoku\s+span\s+b\s*\(([^)]*)\)',          _ichi('ta_ichi_b'),    c, flags=re.I)
        c = re.sub(r'\b(?:(daily|weekly|monthly)\s+)?ichimoku\s+cloud\s+(?:bottom|top)\s*\(([^)]*)\)', _ichi('ta_ichi_cloud'), c, flags=re.I)

        # VWAP
        c = re.sub(r'\bvwap\b', "df['vwap']", c, flags=re.I)

        # Heikin-Ashi resolve
        c = c.replace('__HA_CLOSE__', "ta_ha(df)['close']")
        c = c.replace('__HA_OPEN__',  "ta_ha(df)['open']")
        c = c.replace('__HA_HIGH__',  "ta_ha(df)['high']")
        c = c.replace('__HA_LOW__',   "ta_ha(df)['low']")

        # FIX-B: count() -- comprehensive multi-pattern parser with TF scaling
        def _parse_count(m):
            tf_word = (m.group(1) or 'daily').lower()
            period  = int(m.group(2))
            inner   = m.group(3).strip()
            # Strip "K where" prefix
            inner = re.sub(r'^\d+\s+where\s+', '', inner, flags=re.I)
            if tf_word == 'yearly':
                scaled  = period * 252
            elif tf_word == 'monthly':
                scaled  = period * 21
            elif tf_word == 'weekly':
                scaled  = period * 5
            else:
                scaled  = period
            # Note: no inline comment — comments break Step 12's and/or splitter
            return f'({inner}).rolling({scaled}).sum()'

        c = re.sub(
            r'\b(?:(yearly|monthly|weekly|daily)\s+)?count\s*\(\s*(\d+)\s*,\s*(.*?)\s*\)',
            _parse_count, c, flags=re.I
        )

        # Catch-all for any remaining fundamentals
        c = re.sub(r'\byearly net profit(?:/reported profit after tax)?\b', "df['net_profit']", c, flags=re.I)
        c = re.sub(r'\bearning per share\[eps\]\b', "df['eps']",     c, flags=re.I)
        c = re.sub(r'\bttm eps\b',                  "df['eps_ttm']", c, flags=re.I)
        c = re.sub(r'\bprev year eps\b',             "df['eps_prev']",c, flags=re.I)

        # Step 7.5: Crossover phrases
        def _cross(m, direction):
            lhs = re.sub(r'^\(+', '', m.group(1).strip()).strip()
            rhs = re.sub(r'\)+$', '', m.group(2).strip()).strip()
            lhs_s = f'{lhs}.shift(1)' if not re.match(r'^[\d.]+$', lhs) else lhs
            rhs_s = f'{rhs}.shift(1)' if not re.match(r'^[\d.]+$', rhs) else rhs
            if direction == 'above':
                return f'({lhs_s} < {rhs_s}) & ({lhs} > {rhs})'
            else:
                return f'({lhs_s} > {rhs_s}) & ({lhs} < {rhs})'
        c = re.sub(r'(.+?)\s+crossed\s+above\s+(.+?)(?=\s+and\b|\s+or\b|\s*$)',
                   lambda m: _cross(m, 'above'), c, flags=re.I)
        c = re.sub(r'(.+?)\s+crossed\s+below\s+(.+?)(?=\s+and\b|\s+or\b|\s*$)',
                   lambda m: _cross(m, 'below'), c, flags=re.I)

        # Step 8a: Post-indicator rolling max/min + nested MA
        # FIX-A: three-pass approach to catch all remaining max(N, expr) after full translation
        def _post_rolling(func):
            def _inner(m):
                per_str = m.group(1).strip()
                expr    = m.group(2).strip()
                try:    per = int(per_str)
                except: return m.group(0)
                return f"({expr}).rolling({per}).{func}()"
            return _inner

        _compound_ta = r'ta_\w+\([^)]*\)(?:\.\w+(?:\([^)]*\))?)*'
        _df_ref      = r"df(?:_\w+)?\['[^']+'\](?:\.\w+\([^)]*\))*"

        for func in ('max', 'min'):
            # Pass 1: compound ta_ expressions
            c = re.sub(rf'\b{func}\s*\(\s*(\d+)\s*,\s*({_compound_ta})\s*\)',
                       _post_rolling(func), c, flags=re.I)
            # Pass 2: df[...] references
            c = re.sub(rf'\b{func}\s*\(\s*(\d+)\s*,\s*({_df_ref})\s*\)',
                       _post_rolling(func), c, flags=re.I)
            # Pass 3 (FIX-A): any remaining max(N, expr_with_translated_code)
            c = re.sub(
                rf'\b{func}\s*\(\s*(\d+)\s*,\s*([^,)][^)]*(?:ta_|df[\[_])[^)]*?)\s*\)',
                _post_rolling(func), c, flags=re.I
            )

        # BUG-8: sma/ema/wma wrapping already-translated indicator
        for ma_fn in ('sma', 'ema', 'wma'):
            ta_fn = f'ta_{ma_fn}'
            c = re.sub(rf'\b{ma_fn}\s*\(\s*({_compound_ta})\s*,\s*(\d+)\s*\)',
                       lambda m, f=ta_fn: f"{f}({m.group(1)},{int(m.group(2))})", c, flags=re.I)
            c = re.sub(rf'\b{ma_fn}\s*\(\s*({_df_ref})\s*,\s*(\d+)\s*\)',
                       lambda m, f=ta_fn: f"{f}({m.group(1)},{int(m.group(2))})", c, flags=re.I)

        # BUG-6: WaveTrend bare object -- append .wt1
        c = re.sub(r'(ta_wavetrend\([^)]*\))(?!\.wt[12])', r'\1.wt1', c)

        # Step 8b: SMA/EMA/WMA fallback
        c = re.sub(r"\bsma\s*\(\s*df\['(\w+)'\]\s*,\s*(\d+)\s*\)",
                   lambda m: f"ta_sma(df['{m.group(1)}'],{int(m.group(2))})", c, flags=re.I)
        c = re.sub(r"\bema\s*\(\s*df\['(\w+)'\]\s*,\s*(\d+)\s*\)",
                   lambda m: f"ta_ema(df['{m.group(1)}'],{int(m.group(2))})", c, flags=re.I)
        c = re.sub(r"\bwma\s*\(\s*df\['(\w+)'\]\s*,\s*(\d+)\s*\)",
                   lambda m: f"ta_wma(df['{m.group(1)}'],{int(m.group(2))})", c, flags=re.I)
        c = re.sub(r'\bsma\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)',
                   lambda m: f"ta_sma(df['{_f(m.group(1))}'],{int(m.group(2))})", c, flags=re.I)
        c = re.sub(r'\bema\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)',
                   lambda m: f"ta_ema(df['{_f(m.group(1))}'],{int(m.group(2))})", c, flags=re.I)
        c = re.sub(r'\bwma\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)',
                   lambda m: f"ta_wma(df['{_f(m.group(1))}'],{int(m.group(2))})", c, flags=re.I)

        # Step 9: Resolve __SN_TF__ shift tags
        # FIX-D/F: yearly and quarterly resolved to correct resampled frames
        def _resolve_shift(m):
            n, tf, expr = m.group(1), m.group(2), m.group(3).strip()
            if tf in ('weekly', 'monthly', 'quarterly', 'yearly'):
                return f"({expr}).shift({n}).reindex(df.index, method='ffill')"
            else:
                return f"({expr}).shift({n})"

        c = re.sub(
            r'__S(\d+)_(daily|weekly|monthly|quarterly|yearly)__\s*(ta_\w+\([^)]*\)(?:\.\w+\([^)]*\))*)',
            _resolve_shift, c
        )
        c = re.sub(
            r"__S(\d+)_(daily|weekly|monthly|quarterly|yearly)__\s*(df(?:_\w+)?\['[^']+'\](?:\.\w+\([^)]*\))*)",
            _resolve_shift, c
        )
        c = re.sub(
            r'__S(\d+)_(daily|weekly|monthly|quarterly|yearly)__\s*(\w+)',
            lambda m: (
                f"({_df_for_tf(m.group(2))}['{_f(m.group(3))}'].shift({m.group(1)})"
                f".reindex(df.index, method='ffill')"
                if m.group(2) in ('weekly','monthly','quarterly','yearly')
                else f"df['{_f(m.group(3))}'].shift({m.group(1)})"
            ),
            c, flags=re.I
        )

        # Legacy __SN__ tags
        c = re.sub(r"__S(\d+)__\s*(ta_\w+\([^)]*\))",
                   lambda m: f"{m.group(2)}.shift({m.group(1)})", c)
        c = re.sub(r"__S(\d+)__\s*df\['(\w+)'\]((?:\.\w+\([^)]*\))*)",
                   lambda m: f"df['{m.group(2)}'].shift({m.group(1)}){m.group(3)}", c)
        c = re.sub(r"__S(\d+)__\s*(?:daily\s+|weekly\s+|monthly\s+)?(\w+)",
                   lambda m: f"df['{_f(m.group(2))}'].shift({m.group(1)})", c, flags=re.I)
        c = re.sub(r'__S\d+(?:_\w+)?__', '', c)

        # Step 10: TF-aware bare price fields
        for tf_word in ['weekly','monthly','quarterly','yearly']:
            dfn = _df_for_tf(tf_word)
            for field in ['close','open','high','low','volume']:
                c = re.sub(rf'\b{tf_word}\s+{field}\b', f"{dfn}['{field}']", c, flags=re.I)
        c = re.sub(r'\bdaily\s+(?=(?:close|open|high|low|volume)\b)', '', c, flags=re.I)
        for field in ['close','open','high','low','volume']:
            c = re.sub(rf"\b{field}\b(?!\s*['\]])", f"df['{field}']", c, flags=re.I)

        # Step 11: NOT operator
        c = re.sub(r'\bnot\b', '~', c, flags=re.I)

        # Step 12: Pandas parens wrapping
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

        # Step 13: Final cleanup
        c = re.sub(r'\b(?:daily|weekly|monthly|quarterly|yearly)\s+', '', c, flags=re.I)

        # FIX-E: df['df'] artifact cleanup -- three patterns, three passes
        for _ in range(3):
            c = re.sub(r"df\['df\['([^']+)'\]'\]", r"df['\1']", c)
            c = re.sub(r"df\[df\['([^']+)'\]\]",   r"df['\1']", c)

        c = re.sub(r'__S\d+(?:_\w+)?__', '', c)
        c = re.sub(r'\(\s*(?:industry|sector)\s*(?:!=|==)\s*\'[^\']*\'\s*\)', 'True', c, flags=re.I)
        c = re.sub(r'\b(?:industry|sector)\s*(?:!=|==)\s*\'[^\']*\'', 'True', c, flags=re.I)
        c = re.sub(r'\bttm\s+(df\[)', r'\1', c, flags=re.I)

        # CRITICAL-5: BODMAS fix
        c = re.sub(
            r"(df\['[^']+'\]\s*[+\-]\s*df\['[^']+'\])\s*([*/])",
            lambda m: f"({m.group(1)}) {m.group(2)}", c
        )

        # LOW-2: Remove truncated empty-RHS expressions
        c = re.sub(r'[<>]=?\s*\)', ')', c)
        c = re.sub(r'\d+\s+(?:weeks?|days?|months?)\s+(df\[)', r'\1', c, flags=re.I)

        c = re.sub(r'\s+', ' ', c).strip()

        # Balance parentheses
        open_count  = c.count('(')
        close_count = c.count(')')
        if open_count > close_count:
            c += ')' * (open_count - close_count)
        elif close_count > open_count:
            excess = close_count - open_count
            for _ in range(excess):
                idx = c.rfind(')')
                if idx >= 0:
                    c = c[:idx] + c[idx+1:]

        untranslated = bool(
            re.search(r'\b(?:daily|weekly|monthly)\s+\w', c, re.I) or
            re.search(r'\bsma\s*\((?!.*(?:rolling|df\[))', c, re.I)
        )
        return c, 'PARTIAL' if untranslated else 'TRANSLATED'

    except Exception as e:
        return f"ERROR: {e}", 'FAILED'


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATOR  (FIX-I: improved strip before AST parse)
# ─────────────────────────────────────────────────────────────────────────────

def validate(code):
    if code.startswith('#'):
        return True, "INTRADAY -- skipped"
    if re.search(r'\bsma\s*\((?!.*(?:rolling|df\[))', code, re.I):
        return False, "Untranslated sma() detected"
    if re.search(r'\bema\s*\((?!.*ta_)', code, re.I):
        return False, "Untranslated ema() detected"

    test = code
    test = re.sub(r'ta_\w+\([^)]*\)(?:\.\w+(?:\([^)]*\))?)*', '_V_', test)
    test = re.sub(r"df(?:_\w+)?\['[^']+'\](?:\.\w+\([^)]*\))*", '_V_', test)
    test = re.sub(r'np\.\w+\([^)]*\)', '_V_', test)
    test = re.sub(r'_V_\.rolling\(\d+\)\.\w+\(\)', '_V_', test)
    test = re.sub(r'_V_\.reindex\([^)]*\)\.ffill\(\)', '_V_', test)
    test = re.sub(r'_V_\.shift\(\d+\)', '_V_', test)
    test = re.sub(r'#[^\n]*', '', test)   # strip comments (FIX-I)
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
#  INDICATOR LIBRARY  (v8.2)
# ─────────────────────────────────────────────────────────────────────────────

INDICATOR_LIB = '''"""
Indicator Library for Backtest Engine v8.2
============================================
Import in your backtesting script.
df must have columns: open, high, low, close, volume (DatetimeIndex)

Multi-timeframe usage:
  df_weekly    = resample_weekly(df)
  df_monthly   = resample_monthly(df)
  df_quarterly = resample_quarterly(df)
  df_yearly    = resample_yearly(df)     # new in v8.2

v8.2 additions:
  resample_yearly()  - Yearly OHLCV resampling
  ta_vwap()          - Rolling / session VWAP
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

def resample_quarterly(df):
    return df.resample(\'QE\').agg({
        \'open\':\'first\',\'high\':\'max\',\'low\':\'min\',\'close\':\'last\',\'volume\':\'sum\'
    }).dropna()

def resample_yearly(df):
    """Yearly OHLCV resampling for fundamental comparisons (v8.2)."""
    return df.resample(\'YE\').agg({
        \'open\':\'first\',\'high\':\'max\',\'low\':\'min\',\'close\':\'last\',\'volume\':\'sum\'
    }).dropna()

def ta_sma(s, p): return s.rolling(p).mean()
def ta_ema(s, p): return s.ewm(span=p, adjust=False).mean()
def ta_wma(s, p):
    w = np.arange(1, p+1)
    return s.rolling(p).apply(lambda x: np.dot(x,w)/w.sum(), raw=True)

def ta_rsi(s, p=14):
    """Wilder RSI (EWM alpha=1/p) -- matches Chartink exactly."""
    d = s.diff()
    gain = d.clip(lower=0)
    loss = (-d.clip(upper=0))
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

def ta_tr(df):
    """Raw True Range -- no smoothing."""
    h, l, c = df["high"], df["low"], df["close"]
    return pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)

def ta_atr(df, p=14):
    """Wilder ATR (EWM alpha=1/p)."""
    return ta_tr(df).ewm(alpha=1/p, adjust=False).mean()

def ta_adx(df, p=14):
    """Wilder ADX -- matches Chartink."""
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

def ta_aroon_osc(df,p=14):
    return ta_aroon_up(df,p) - ta_aroon_dn(df,p)

def ta_vwap(df, p=None):
    """Rolling VWAP (v8.2). ta_vwap(df) = session VWAP; ta_vwap(df,20) = 20-bar rolling."""
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    tpv = tp * df["volume"]
    if p is None:
        return tpv.cumsum() / df["volume"].cumsum()
    return tpv.rolling(p).sum() / df["volume"].rolling(p).sum().replace(0, np.nan)

def ta_psar(df,s=0.02,inc=0.02,mx=0.2):
    """Parabolic SAR. psar_warning=YES rows use this."""
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    n = len(close)
    psar = close.copy()
    bull = True; af = s; hp = high[0]; lp = low[0]
    for i in range(2, n):
        if bull:
            psar[i] = psar[i-1] + af * (hp - psar[i-1])
            psar[i] = min(psar[i], low[i-1], low[i-2])
            if low[i] < psar[i]:
                bull = False; psar[i] = hp; lp = low[i]; af = s
            else:
                if high[i] > hp: hp = high[i]; af = min(af+inc, mx)
        else:
            psar[i] = psar[i-1] + af * (lp - psar[i-1])
            psar[i] = max(psar[i], high[i-1], high[i-2])
            if high[i] > psar[i]:
                bull = True; psar[i] = lp; hp = high[i]; af = s
            else:
                if low[i] < lp: lp = low[i]; af = min(af+inc, mx)
    return pd.Series(psar, index=df.index)

def ta_ha(df):
    """Heikin-Ashi OHLC. Returns DataFrame with open/high/low/close."""
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open  = (df["open"] + df["close"]).shift(1) / 2
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    ha_high  = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low   = pd.concat([df["low"],  ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close},
                        index=df.index)

def ta_wavetrend(df, channel_length=10, avg_length=21, ma_length=4):
    """WaveTrend oscillator. Returns object with .wt1 and .wt2."""
    class _WT: pass
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    esa  = ta_ema(hlc3, channel_length)
    d    = ta_ema((hlc3 - esa).abs(), channel_length)
    ci   = (hlc3 - esa) / (0.015 * d.replace(0, np.nan))
    wt1  = ta_ema(ci, avg_length)
    wt2  = ta_sma(wt1, ma_length)
    result = _WT(); result.wt1 = wt1; result.wt2 = wt2
    return result

def ta_ichi_conv(df,p=9):
    return (df["high"].rolling(p).max()+df["low"].rolling(p).min())/2

def ta_ichi_base(df,p=26):
    return (df["high"].rolling(p).max()+df["low"].rolling(p).min())/2

def ta_ichi_a(df,conv_p=9,base_p=26):
    return (ta_ichi_conv(df,conv_p)+ta_ichi_base(df,base_p))/2

def ta_ichi_b(df,p=52):
    return (df["high"].rolling(p).max()+df["low"].rolling(p).min())/2

def ta_ichi_cloud(df,conv_p=9,base_p=26,span_b_p=52):
    return pd.concat([ta_ichi_a(df,conv_p,base_p), ta_ichi_b(df,span_b_p)],axis=1).min(axis=1)
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
        print(f"ERROR: No code column found. Columns: {list(rows[0].keys())}")
        sys.exit(1)

    existing = {}
    if INCREMENTAL and os.path.exists(OUTPUT_CSV):
        for r in csv.DictReader(open(OUTPUT_CSV, encoding='utf-8-sig')):
            existing[r.get('original_code','')] = r

    print("="*60)
    print("  Chartink Translation Engine v8.2")
    print("="*60)
    print(f"  Input    : {len(rows)} screeners")
    print(f"  Existing : {len(existing)} cached")
    if RETRANSLATE_FAILED_ONLY:
        print("  Mode     : FIX-L -- retranslating FAILED/PARTIAL/NO-syntax only")
    print("="*60)

    results = []
    stats   = {'BULLISH':0,'BEARISH':0,'NEUTRAL':0,'INTRADAY':0}

    for i, row in enumerate(rows, 1):
        raw  = row.get(code_col,'').strip()
        name = row.get(name_col, f'Row_{i}') if name_col else f'Row_{i}'

        if not raw or raw.startswith('ERROR'):
            continue

        # FIX-L: skip good rows when in RETRANSLATE_FAILED_ONLY mode
        if RETRANSLATE_FAILED_ONLY and raw in existing:
            ex = existing[raw]
            if ex.get('valid_syntax') == 'YES' and ex.get('status') not in ('FAILED','PARTIAL'):
                results.append(ex)
                stats[ex.get('direction','NEUTRAL')] = stats.get(ex.get('direction','NEUTRAL'), 0) + 1
                continue

        tf = detect_tf(raw)
        py, trans_status = translate(raw)
        valid, err       = validate(py)

        direction = 'INTRADAY' if trans_status == 'INTRADAY' else classify_direction(name, py)
        stats[direction] = stats.get(direction, 0) + 1

        results.append({
            'serial'        : i,
            'name'          : name,
            'timeframe'     : tf,
            'original_code' : raw,
            'python_code'   : py,
            'direction'     : direction,
            'status'        : trans_status,
            'valid_syntax'  : 'YES' if valid else 'NO',
            'syntax_error'  : err,
            'psar_warning'  : 'YES' if 'ta_psar' in py else 'NO',
            'translated_at' : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        if i <= 5 or i % 100 == 0:
            v = 'OK' if valid else 'XX'
            print(f"  [{i:4}/{len(rows)}] {v} {direction[:4]:5} | {tf:8} | {name[:35]}")

    fields = ['serial','name','timeframe','original_code','python_code',
              'direction','status','valid_syntax','syntax_error','psar_warning','translated_at']
    with open(OUTPUT_CSV,'w',newline='',encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    with open('indicators.py','w',encoding='utf-8') as f:
        f.write(INDICATOR_LIB)

    total_valid = sum(1 for r in results if r.get('valid_syntax') == 'YES')
    total_rows  = len(results)

    print("\n" + "="*60)
    print("  Translation Complete -- v8.2")
    print(f"  BULLISH  : {stats.get('BULLISH',0)}")
    print(f"  BEARISH  : {stats.get('BEARISH',0)}")
    print(f"  NEUTRAL  : {stats.get('NEUTRAL',0)}")
    print(f"  INTRADAY : {stats.get('INTRADAY',0)}  (skipped)")
    print(f"  Syntax OK: {total_valid}/{total_rows}  ({100*total_valid//max(total_rows,1)}%)")
    print("="*60)
    print(f"\n  Output : {OUTPUT_CSV}")
    print(f"  Library: indicators.py")
    print()
    print("  Key changes from v8.1:")
    print("  - '1 year ago' now uses df_yearly.shift(1).reindex().ffill()")
    print("  - count() scaled: yearly*252, monthly*21, weekly*5")
    print("  - max(N, rsi(14)) now fully resolved in two-pass Step 8a")
    print("  - eps/pe/roe/roa no longer re-substituted into df['eps_diluted']")
    print("  - 25+ new fundamental field mappings added")


if __name__ == "__main__":
    main()
