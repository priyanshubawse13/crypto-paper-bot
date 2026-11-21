"""
Paper-trading engine for the PriyanshuTrend 4H strategy, futures-style: LONG and SHORT.

Same rules as user_data/strategies/PriyanshuTrend.py (PriyanshuTrend 1.5R + volume refinements):
  LONG   close > SMA300 (50-day avg) AND BTC close > BTC SMA300 AND close > highest high of the
         previous 120 candles (20-day breakout) [AND volume > VOL_MULT x 20-candle avg]
  SHORT  mirror: close < SMA300 AND BTC below its SMA300 AND close < lowest low of previous 120 candles
         -> enter at the next candle's open
  STOP   2.5 x ATR(14) from entry (clamped 2%-12%), fixed
  EXIT   take profit at 1.5 x the stop distance (checked on candle close),
         or close back across SMA300 (trend over)
  SIZE   1% of equity at risk per trade; margin <= 33% of equity at 2x leverage; max 3 open; min $5
  SAFETY 2 losing stop-outs within 24h -> 12h pause; daily realized loss >= 2% of start -> no new
         entries that UTC day; equity < 50% of start -> no new entries; no re-entry on a coin in the
         candle it exited
Not modelled: funding fees (perps), funding-rate crowding filter (no funding data on Binance.US).
Every closed candle is processed exactly once, in time order, so late or missed runs catch up.
Each trade records diagnostics: R multiple, best/worst move during the trade (MFE/MAE in R),
candles held, whether it reached +1R before exiting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import talib.abstract as ta

CANDLE = pd.Timedelta(hours=4)


@dataclass
class Variant:
    name: str
    vol_mult: float = 0.0          # 0 = no volume filter
    shorts: bool = True
    leverage: float = 2.0
    fee: float = 0.0005            # per side (futures taker); spot would be 0.001
    risk: float = 0.01
    r_mult: float = 1.5
    atr_stop: float = 2.5
    min_stop: float = 0.02
    max_stop: float = 0.12
    max_open: int = 3
    max_margin_frac: float = 0.33
    start_cap: float = 100.0
    daily_loss_cap: float = 0.02
    floor_frac: float = 0.5
    description: str = ""


def indicators(df: pd.DataFrame, btc: pd.DataFrame, vol_mult: float) -> pd.DataFrame:
    df = df.copy()
    df["sma300"] = ta.SMA(df, timeperiod=300)
    df["atr"] = ta.ATR(df, timeperiod=14)
    df["hh120"] = df["high"].rolling(120).max().shift(1)
    df["ll120"] = df["low"].rolling(120).min().shift(1)
    df["vol_avg"] = df["volume"].rolling(20).mean()
    b = btc[["date", "close"]].copy()
    bsma = ta.SMA(btc, timeperiod=300)
    b["btc_up"] = (b["close"] > bsma).astype(int)
    b["btc_down"] = (b["close"] < bsma).astype(int)
    df = df.merge(b[["date", "btc_up", "btc_down"]], on="date", how="left")
    df[["btc_up", "btc_down"]] = df[["btc_up", "btc_down"]].fillna(0)
    vol_ok = (df["volume"] > df["vol_avg"] * vol_mult) if vol_mult > 0 else (df["volume"] > 0)
    df["enter_long"] = ((df["close"] > df["sma300"]) & (df["btc_up"] == 1) & (df["close"] > df["hh120"]) & vol_ok).fillna(False)
    df["enter_short"] = ((df["close"] < df["sma300"]) & (df["btc_down"] == 1) & (df["close"] < df["ll120"]) & vol_ok).fillna(False)
    return df.set_index("date")


def new_state(v: Variant) -> dict:
    return {"variant": v.name, "cash": v.start_cap, "open": {}, "pending": {}, "closed": [],
            "sl_times": [], "pause_until": None, "last_t": None, "equity_curve": [], "started": None}


def equity(st: dict) -> float:
    """Cash plus margin locked in open trades (trades valued at entry, like Freqtrade's wallet)."""
    return st["cash"] + sum(t["margin"] for t in st["open"].values())


def mark_to_market(st: dict, prices: dict) -> float:
    eq = equity(st)
    for p, t in st["open"].items():
        px = prices.get(p, t["entry"])
        eq += t["side"] * t["qty"] * (px - t["entry"])
    return eq


def _ts(x) -> str:
    return pd.Timestamp(x).isoformat()


def _close_trade(st, v, pair, price, when, reason):
    t = st["open"].pop(pair)
    gross = t["side"] * t["qty"] * (price - t["entry"])
    exit_fee = t["qty"] * price * v.fee
    st["cash"] += t["margin"] + gross - exit_fee
    pnl = gross - exit_fee - t["entry_fee"]
    risk_usd = t["qty"] * t["entry"] * t["sp"]
    st["closed"].append({
        "pair": pair, "side": "long" if t["side"] == 1 else "short",
        "opened": t["opened"], "closed": _ts(when), "entry": t["entry"], "exit": round(price, 8),
        "stop": t["stop"], "target": t["target"], "margin": round(t["margin"], 4),
        "notional": round(t["qty"] * t["entry"], 4), "pnl": round(pnl, 4),
        "pnl_pct_margin": round(pnl / t["margin"] * 100, 2), "R": round(pnl / risk_usd, 3) if risk_usd else 0,
        "mfe_R": round(t["mfe_R"], 3), "mae_R": round(t["mae_R"], 3), "bars": t["bars"],
        "hit_1R": bool(t["mfe_R"] >= 1.0), "stop_pct": round(t["sp"] * 100, 2), "reason": reason,
    })
    if reason == "stop_loss" and pnl < 0:
        st["sl_times"].append(_ts(when))


def _fill(st, v, pair, price, when):
    p = st["pending"].pop(pair)
    margin, side, sp = p["margin"], p["side"], p["sp"]
    notional = margin * v.leverage
    entry_fee = notional * v.fee
    if margin + entry_fee > st["cash"] or notional < 5:
        return
    st["cash"] -= margin + entry_fee
    st["open"][pair] = {
        "side": side, "entry": round(price, 8), "qty": notional / price, "margin": margin,
        "entry_fee": entry_fee, "sp": sp,
        "stop": round(price * (1 - side * sp), 8), "target": round(price * (1 + side * v.r_mult * sp), 8),
        "opened": _ts(when), "signal": p["signal"], "mfe_R": 0.0, "mae_R": 0.0, "bars": 0,
    }


def step(st: dict, v: Variant, frames: dict, pairs: list, t: pd.Timestamp):
    """Process the CLOSED candle that opened at t for every pair."""
    candle_close = t + CANDLE
    for pair in pairs:
        df = frames[pair]
        if t not in df.index:
            continue
        row = df.loc[t]
        if pair in st["pending"]:
            _fill(st, v, pair, float(row["open"]), t)
        tr = st["open"].get(pair)
        if tr is None:
            continue
        s, e, sp = tr["side"], tr["entry"], tr["sp"]
        fav = (row["high"] - e) / e if s == 1 else (e - row["low"]) / e
        adv = (e - row["low"]) / e if s == 1 else (row["high"] - e) / e
        tr["mfe_R"] = max(tr["mfe_R"], fav / sp)
        tr["mae_R"] = max(tr["mae_R"], adv / sp)
        tr["bars"] += 1
        hit_stop = row["low"] <= tr["stop"] if s == 1 else row["high"] >= tr["stop"]
        hit_tp = row["close"] >= tr["target"] if s == 1 else row["close"] <= tr["target"]
        trend_over = row["close"] < row["sma300"] if s == 1 else row["close"] > row["sma300"]
        if hit_stop:
            price = min(tr["stop"], float(row["open"])) if s == 1 else max(tr["stop"], float(row["open"]))
            _close_trade(st, v, pair, price, candle_close, "stop_loss")
        elif hit_tp:
            _close_trade(st, v, pair, float(row["close"]), candle_close, "take_profit")
        elif trend_over:
            _close_trade(st, v, pair, float(row["close"]), candle_close, "trend_exit")

    recent = [x for x in st["sl_times"] if pd.Timestamp(x) > candle_close - 6 * CANDLE]
    st["sl_times"] = recent
    if len(recent) >= 2:
        st["pause_until"] = _ts(pd.Timestamp(max(recent)) + 3 * CANDLE)
        st["sl_times"] = []
    paused = st["pause_until"] and candle_close < pd.Timestamp(st["pause_until"])
    day = candle_close.normalize()
    day_pnl = sum(c["pnl"] for c in st["closed"] if pd.Timestamp(c["closed"]).normalize() == day)
    eq = equity(st)
    blocked = paused or day_pnl <= -v.daily_loss_cap * v.start_cap or eq < v.floor_frac * v.start_cap
    exited_now = {c["pair"] for c in st["closed"] if pd.Timestamp(c["closed"]) == candle_close}

    if not blocked:
        for pair in pairs:
            df = frames[pair]
            if t not in df.index or pair in st["open"] or pair in st["pending"] or pair in exited_now:
                continue
            if len(st["open"]) + len(st["pending"]) >= v.max_open:
                break
            row = df.loc[t]
            side = 1 if bool(row["enter_long"]) else (-1 if v.shorts and bool(row["enter_short"]) else 0)
            if side == 0 or math.isnan(row["atr"]):
                continue
            sp = float(np.clip(v.atr_stop * row["atr"] / row["close"], v.min_stop, v.max_stop))
            notional = eq * v.risk / sp
            reserved = sum(p["margin"] for p in st["pending"].values())
            margin = min(notional / v.leverage, eq * v.max_margin_frac, st["cash"] - reserved)
            if margin * v.leverage < 5:
                continue
            st["pending"][pair] = {"side": side, "sp": sp, "margin": margin, "signal": _ts(t)}

    st["last_t"] = _ts(t)
    st["equity_curve"].append([_ts(candle_close), round(equity(st), 4)])


def run(st: dict, v: Variant, raw: dict, btc: pd.DataFrame, pairs: list,
        now=None, backfill: bool = False) -> dict:
    """Advance the state through every closed candle not yet processed.
    raw frames may include the still-forming candle; it is only used to fill pending entries at its open
    and as the current price."""
    frames = {p: indicators(raw[p], btc, v.vol_mult) for p in pairs}
    now = now or pd.Timestamp.now(tz="UTC")
    all_t = sorted({t for f in frames.values() for t in f.index})
    closed_t = [t for t in all_t if t + CANDLE <= now]
    last = pd.Timestamp(st["last_t"]) if st["last_t"] else None
    if last is None and not backfill:
        closed_t = closed_t[-1:]
    if st.get("started") is None and closed_t:
        st["started"] = _ts(closed_t[0] + CANDLE)
    for t in closed_t:
        if last is None or t > last:
            step(st, v, frames, pairs, t)
    forming = [t for t in all_t if t + CANDLE > now]
    if forming and st["pending"]:
        ft = forming[0]
        for pair in list(st["pending"]):
            if ft in frames[pair].index:
                _fill(st, v, pair, float(frames[pair].loc[ft, "open"]), ft)
    st["last_prices"] = {p: float(frames[p]["close"].iloc[-1]) for p in pairs}
    return st
