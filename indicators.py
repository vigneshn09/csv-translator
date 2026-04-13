"""
Indicator Library for Backtest Engine v8.4
============================================
Import in your backtesting script.
df must have columns: open, high, low, close, volume (DatetimeIndex)

Multi-timeframe usage:
  df_weekly    = resample_weekly(df)
  df_monthly   = resample_monthly(df)
  df_quarterly = resample_quarterly(df)
  df_yearly    = resample_yearly(df)

v8.4 changelog (fixes over v8.3):
  [HARD BUG FIX] ta_aroon_up / ta_aroon_dn
      - lambda used x[::-1].values.argmax() but raw=True gives a numpy ndarray,
        not a pandas Series — so .values doesn't exist and crashes on every call.
      - Fix: replaced with np.argmax(x[::-1]) and np.argmin(x[::-1]).

  [HARD BUG FIX] ta_stochrsi
      - Was returning 0.0–1.0 range. Chartink StochRSI is 0–100.
      - Any screener with a condition like 'stochrsi > 20' would ALWAYS be False,
        silently producing zero signals — a silent killer bug.
      - Fix: result now multiplied by 100 to match Chartink scale.

  [SOFT BUG FIX] ta_bb_upper / ta_bb_lower / ta_bb_width — ddof=0
      - pandas rolling().std() defaults to ddof=1 (sample std).
      - Chartink uses ddof=0 (population std), which is standard for BBands.
      - At p=10 the difference is ~5.4%; at p=20 it is ~2.6%.
      - Fix: all BB functions now pass ddof=0 to match Chartink exactly.

  [SOFT BUG FIX] Added ta_ichi_cloud_top = max(span_a, span_b)
      - ta_ichi_cloud = min(A, B) = cloud bottom (Kumo support).
      - Many Chartink screeners check: close > max(span_a, span_b) — i.e. price
        above the full cloud. Without ta_ichi_cloud_top those screeners would
        incorrectly use the cloud bottom as the upper boundary.
      - Fix: ta_ichi_cloud_top added. ta_ichi_cloud (bottom) unchanged.
      - Also added to backtest1.py indicator namespace.

  [INFO — no change] ta_vwap cumsum
      - Each stock is processed independently so cumsum resets per stock. OK.
      - For daily bars, cumulative VWAP is the standard long-run anchor.
      - Rolling VWAP (ta_vwap(df, N)) is available for shorter windows.

  [INFO — no change] ta_supertrend loop
      - Pure Python loop runs ~0.02s per stock (~1 min total for 2700 stocks).
      - Acceptable for overnight batch. No change needed.

Zerodha Kite API alignment notes:
  - Kite historical_data(interval="day") returns: date, open, high, low, close, volume
  - data_fetcher.py normalises to DatetimeIndex, lowercase column names, float64 dtypes.
  - All functions here expect that exact format. No adjustments needed.
  - Volumes are float64 (cast in fetcher). All indicator math handles this correctly.
  - Kite timestamps are IST, stripped to date-only midnight UTC by data_fetcher.py.
    Indicators never touch the index so this is transparent.
"""
import pandas as pd
import numpy as np

# ─── RESAMPLERS ───────────────────────────────────────────────────────────────

def resample_weekly(df):
    return df.resample('W-FRI').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

def resample_monthly(df):
    return df.resample('ME').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

def resample_quarterly(df):
    return df.resample('QE').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

def resample_yearly(df):
    """Yearly OHLCV resampling for fundamental comparisons."""
    return df.resample('YE').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

# ─── BASIC MAs ────────────────────────────────────────────────────────────────

def ta_sma(s, p):
    return s.rolling(p).mean()

def ta_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def ta_wma(s, p):
    w = np.arange(1, p + 1)
    return s.rolling(p).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

# ─── MOMENTUM ─────────────────────────────────────────────────────────────────

def ta_rsi(s, p=14):
    """Wilder RSI (EWM alpha=1/p) — matches Chartink exactly."""
    d = s.diff()
    gain = d.clip(lower=0)
    loss = (-d.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / p, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / p, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def ta_stochrsi(s, p=14):
    """
    StochRSI on 0–100 scale — matches Chartink.
    v8.3 returned 0–1; v8.4 multiplies by 100.
    Screeners with 'stochrsi > 80' now fire correctly.
    """
    r = ta_rsi(s, p)
    mn = r.rolling(p).min()
    mx = r.rolling(p).max()
    raw = (r - mn) / (mx - mn).replace(0, np.nan)
    return raw * 100  # ← FIXED: was missing *100 in v8.3

def ta_macd_line(s, slow=26, fast=12, sig=9):
    return ta_ema(s, fast) - ta_ema(s, slow)

def ta_macd_sig(s, slow=26, fast=12, sig=9):
    return ta_ema(ta_macd_line(s, slow, fast), sig)

def ta_macd_hist(s, slow=26, fast=12, sig=9):
    return ta_macd_line(s, slow, fast) - ta_macd_sig(s, slow, fast, sig)

def ta_cci(df, p=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mad = tp.rolling(p).apply(lambda x: (x - x.mean()).abs().mean())
    return (tp - tp.rolling(p).mean()) / (0.015 * mad.replace(0, np.nan))

def ta_mfi(df, p=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(), 0).rolling(p).sum()
    neg = mf.where(tp < tp.shift(), 0).rolling(p).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))

def ta_willr(df, p=14):
    hh = df["high"].rolling(p).max()
    ll = df["low"].rolling(p).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)

def ta_stoch_k(df, kp=5, dp=3):
    hh = df["high"].rolling(kp).max()
    ll = df["low"].rolling(kp).min()
    return 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)

def ta_stoch_d(df, kp=5, dp=3):
    return ta_stoch_k(df, kp, dp).rolling(dp).mean()

def ta_aroon_up(df, p=14):
    """
    Aroon Up — fixed in v8.4.
    v8.3 used x[::-1].values.argmax() inside raw=True lambda, which crashes
    because raw=True passes a numpy ndarray (no .values attribute).
    Fix: np.argmax(x[::-1]) works on both ndarray and Series.
    """
    return df["high"].rolling(p + 1).apply(
        lambda x: (p - np.argmax(x[::-1])) / p * 100, raw=True
    )

def ta_aroon_dn(df, p=14):
    """
    Aroon Down — fixed in v8.4 (same raw=True crash as ta_aroon_up).
    """
    return df["low"].rolling(p + 1).apply(
        lambda x: (p - np.argmin(x[::-1])) / p * 100, raw=True
    )

def ta_aroon_osc(df, p=14):
    return ta_aroon_up(df, p) - ta_aroon_dn(df, p)

# ─── VOLATILITY ───────────────────────────────────────────────────────────────

def ta_bb_upper(s, p=20, std=2):
    """
    Bollinger Band upper — fixed in v8.4 to use ddof=0 (population std).
    Chartink uses population std. pandas default is ddof=1 (sample std).
    At p=20, sample std is ~2.6% wider than population std — causes false
    signals on tight band breakout screeners.
    """
    return s.rolling(p).mean() + std * s.rolling(p).std(ddof=0)

def ta_bb_lower(s, p=20, std=2):
    """Bollinger Band lower — fixed to ddof=0. See ta_bb_upper."""
    return s.rolling(p).mean() - std * s.rolling(p).std(ddof=0)

def ta_bb_width(s, p=20, std=2):
    return ta_bb_upper(s, p, std) - ta_bb_lower(s, p, std)

def ta_tr(df):
    """Raw True Range — no smoothing."""
    h, l, c = df["high"], df["low"], df["close"]
    return pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)

def ta_atr(df, p=14):
    """Wilder ATR (EWM alpha=1/p) — matches Chartink."""
    return ta_tr(df).ewm(alpha=1 / p, adjust=False).mean()

def ta_adx(df, p=14):
    """Wilder ADX — matches Chartink."""
    atr = ta_atr(df, p)
    h, l = df["high"], df["low"]
    pdm = h.diff().clip(lower=0)
    ndm = (-l.diff()).clip(lower=0)
    pdm = pdm.where(pdm > ndm, 0)
    ndm = ndm.where(ndm > pdm, 0)
    pdi = 100 * pdm.ewm(alpha=1 / p, adjust=False).mean() / atr.replace(0, np.nan)
    ndi = 100 * ndm.ewm(alpha=1 / p, adjust=False).mean() / atr.replace(0, np.nan)
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1 / p, adjust=False).mean()

def ta_dip(df, p=14):
    """DI+ (Directional Indicator Plus)."""
    atr = ta_atr(df, p)
    pdm = df["high"].diff().clip(lower=0)
    ndm = (-df["low"].diff()).clip(lower=0)
    return 100 * pdm.where(pdm > ndm, 0).ewm(alpha=1 / p, adjust=False).mean() / atr.replace(0, np.nan)

def ta_dim(df, p=14):
    """DI- (Directional Indicator Minus)."""
    atr = ta_atr(df, p)
    pdm = df["high"].diff().clip(lower=0)
    ndm = (-df["low"].diff()).clip(lower=0)
    return 100 * ndm.where(ndm > pdm, 0).ewm(alpha=1 / p, adjust=False).mean() / atr.replace(0, np.nan)

def ta_supertrend(df, mult=3, p=7):
    """
    Supertrend — logic is correct. Loop-based (~0.02s per 300-bar stock).
    Uses numpy arrays internally to avoid pandas iloc overhead.
    """
    atr = ta_atr(df, p)
    hl2 = (df["high"] + df["low"]) / 2
    ub  = (hl2 + mult * atr).values
    lb  = (hl2 - mult * atr).values
    close = df["close"].values
    n = len(close)
    st = np.full(n, np.nan)

    for i in range(1, n):
        prev = st[i - 1] if not np.isnan(st[i - 1]) else lb[i]
        if close[i] > ub[i - 1]:
            st[i] = lb[i]
        elif close[i] < lb[i - 1]:
            st[i] = ub[i]
        else:
            st[i] = prev

    return pd.Series(st, index=df.index)

# ─── OSCILLATORS ─────────────────────────────────────────────────────────────

def ta_vwap(df, p=None):
    """
    VWAP. ta_vwap(df) = cumulative VWAP from first bar (per-stock anchor).
    ta_vwap(df, N) = N-bar rolling VWAP.
    For daily data, cumulative VWAP is the correct long-run anchor.
    Each stock is processed independently so cumsum resets correctly per stock.
    """
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    tpv = tp * df["volume"]
    if p is None:
        return tpv.cumsum() / df["volume"].cumsum()
    return tpv.rolling(p).sum() / df["volume"].rolling(p).sum().replace(0, np.nan)

def ta_psar(df, s=0.02, inc=0.02, mx=0.2):
    """Parabolic SAR — correct implementation."""
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(close)
    psar  = close.copy()
    bull  = True; af = s; hp = high[0]; lp = low[0]

    for i in range(2, n):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
            psar[i] = min(psar[i], low[i - 1], low[i - 2])
            if low[i] < psar[i]:
                bull = False; psar[i] = hp; lp = low[i]; af = s
            else:
                if high[i] > hp:
                    hp = high[i]; af = min(af + inc, mx)
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
            psar[i] = max(psar[i], high[i - 1], high[i - 2])
            if high[i] > psar[i]:
                bull = True; psar[i] = lp; hp = high[i]; af = s
            else:
                if low[i] < lp:
                    lp = low[i]; af = min(af + inc, mx)

    return pd.Series(psar, index=df.index)

# ─── PATTERN / TRANSFORM ─────────────────────────────────────────────────────

def ta_ha(df):
    """
    Heikin-Ashi OHLC. Returns DataFrame with open/high/low/close.
    Uses .copy() to avoid pandas ChainedAssignment on the first-bar initialisation.
    """
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open  = (df["open"] + df["close"]).shift(1) / 2
    # Use .copy() + loc to avoid SettingWithCopyWarning in pandas 2.x / 3.x
    ha_open  = ha_open.copy()
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    ha_high  = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low   = pd.concat([df["low"],  ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame(
        {"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close},
        index=df.index
    )

def ta_wavetrend(df, channel_length=10, avg_length=21, ma_length=4):
    """WaveTrend oscillator. Returns object with .wt1 and .wt2 attributes."""
    class _WT:
        pass
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    esa  = ta_ema(hlc3, channel_length)
    d    = ta_ema((hlc3 - esa).abs(), channel_length)
    ci   = (hlc3 - esa) / (0.015 * d.replace(0, np.nan))
    wt1  = ta_ema(ci, avg_length)
    wt2  = ta_sma(wt1, ma_length)
    result = _WT()
    result.wt1 = wt1
    result.wt2 = wt2
    return result

# ─── ICHIMOKU ─────────────────────────────────────────────────────────────────

def ta_ichi_conv(df, p=9):
    """Tenkan-sen (Conversion Line)."""
    return (df["high"].rolling(p).max() + df["low"].rolling(p).min()) / 2

def ta_ichi_base(df, p=26):
    """Kijun-sen (Base Line)."""
    return (df["high"].rolling(p).max() + df["low"].rolling(p).min()) / 2

def ta_ichi_a(df, conv_p=9, base_p=26):
    """Senkou Span A (Leading Span A)."""
    return (ta_ichi_conv(df, conv_p) + ta_ichi_base(df, base_p)) / 2

def ta_ichi_b(df, p=52):
    """Senkou Span B (Leading Span B)."""
    return (df["high"].rolling(p).max() + df["low"].rolling(p).min()) / 2

def ta_ichi_cloud(df, conv_p=9, base_p=26, span_b_p=52):
    """
    Kumo cloud BOTTOM = min(Span A, Span B).
    Use for: close < ta_ichi_cloud(df)  → price below cloud (bearish)
    """
    return pd.concat([ta_ichi_a(df, conv_p, base_p), ta_ichi_b(df, span_b_p)], axis=1).min(axis=1)

def ta_ichi_cloud_top(df, conv_p=9, base_p=26, span_b_p=52):
    """
    Kumo cloud TOP = max(Span A, Span B).  ← NEW in v8.4
    Use for: close > ta_ichi_cloud_top(df)  → price above cloud (bullish)
    Many Chartink screeners check 'price above cloud' — this requires the top
    boundary, not the bottom. Without this function, those screeners were using
    the cloud bottom as the upper limit, creating false bullish signals.
    """
    return pd.concat([ta_ichi_a(df, conv_p, base_p), ta_ichi_b(df, span_b_p)], axis=1).max(axis=1)

# ─── ALIASES (for screener compatibility) ────────────────────────────────────
# Some screeners use these alternate names
ta_williams_r = ta_willr  # alias: ta_willr is the canonical name

def ta_bb_mid(s, p=20, std=2):
    """Bollinger Band midline (SMA). Alias used by some screeners."""
    return s.rolling(p).mean()
