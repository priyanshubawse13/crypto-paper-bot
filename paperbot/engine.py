"""
Paper-trading engine for the PriyanshuTrend 4H strategy (spot, longs only).

Same rules as user_data/strategies/PriyanshuTrend.py (PriyanshuTrendSpot + refinements):
  ENTRY  close > SMA300 (50-day avg) AND BTC close > BTC SMA300 AND close > highest high of
         the previous 120 candles (20-day breakout) [AND volume > VOL_MULT x 20-candle avg]
         -> buy at the next candle's open
  STOP   2.5 x ATR(14) below entry (clamped 2%-12%), fixed
  EXIT   take profit at 1.5 x the stop distance (checked on candle close)
         or close back below SMA300 (trend over)
  SIZE   1% of equity at risk per trade, max 33% of equity per trade, max 3 open, min $5
  SAFETY 2 losing stop-outs within 24h -> 12h pause; daily realized loss >= 2% of start
         -> no new entries that UTC day; equity < 50% of start -> no new entries; no
         re-entry on a coin in the same candle it exited
Every closed candle is processed exactly once, in time order, so late or missed runs catch up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import talib.abstract as ta

CANDLE = pd.Timedelta(hours=4)
FEE = 0.001  # Binance.US spot taker fee per side (backtest used the same)


@dataclass
class Variant:
    name: str
    vol_mult: float = 0.0          # 0 = no volume filter
    risk: float = 0.01
    r_mult: float = 1.5
    atr_stop: float = 2.5
    min_stop: float = 0.02
    max_stop: float = 0.12
    max_open: int = 3
    max_stake_frac: float = 0.33
    start_cap: float = 100.0
    daily_loss_cap: float = 0.02
    floor_frac: float = 0.5
    description: str = ""


def indicators(df: pd.DataFrame, btc: pd.DataFrame, vol_mult: float) -> pd.DataFrame:
    df = df.copy()
    df["sma300"] = ta.SMA(df, timeperiod=300)
    df["atr"] = ta.ATR(df, timeperiod=14)
    df["hh120"] = df["high"].rolling(120).max().shift(1)
    df["vol_avg"] = df["volume"].rolling(20).mean()
    b = btc[["date", "close"]].copy()
    b["btc_up"] = (b["close"] > ta.SMA(btc, timeperiod=300)).astype(int)
    df = df.merge(b[["date", "btc_up"]], on="date", how="left")
    df["btc_up"] = df["btc_up"].fillna(0)
    sig = (df["close"] > df["sma300"]) & (df["btc_up"] == 1) & (df["close"] > df["hh120"]) & (df["volume"] > 0)
    if vol_mult > 0:
        sig &= df["volume"] > df["vol_avg"] * vol_mult
    df["enter"] = sig.fillna(False)
    return df.set_index("date")


def new_state(v: Variant) -> dict:
    return {"variant": v.name, "cash": v.start_cap, "open": {}, "pending": {}, "closed": [],
            "sl_times": [], "pause_until": None, "last_t": None, "equity_curve": []}


def _equity(st: dict) -> float:
    return st["cash"] + sum(t["stake"] for t in st["open"].values())


def _ts(x) -> str:
    return pd.Timestamp(x).isoformat()


def _close_trade(st, pair, price, when, reason):
    t = st["open"].pop(pair)
    proceeds = t["qty"] * price * (1 - FEE)
    pnl = proceeds - t["stake"]
    st["cash"] += proceeds
    st["closed"].append({"pair": pair, "opened": t["opened"], "closed": _ts(when), "entry": t["entry"],
                         "exit": round(price, 8), "stake": round(t["stake"], 4), "pnl": round(pnl, 4),
                         "pnl_pct": round(pnl / t["stake"] * 100, 2), "reason": reason})
    if reason == "stop_loss" and pnl < 0:
        st["sl_times"].append(_ts(when))


def _fill(st, pair, price, when):
    p = st["pending"].pop(pair)
    stake = p["stake"]
    if stake > st["cash"] or stake < 5:
        return
    st["cash"] -= stake
    qty = stake * (1 - FEE) / price
    st["open"][pair] = {"entry": round(price, 8), "qty": qty, "stake": stake, "sp": p["sp"],
                        "stop": round(price * (1 - p["sp"]), 8),
                        "target": round(price * (1 + p["r_mult"] * p["sp"]), 8),
                        "opened": _ts(when), "signal": p["signal"]}


def step(st: dict, v: Variant, frames: dict[str, pd.DataFrame], pairs: list[str], t: pd.Timestamp):
    """Process the CLOSED candle that opened at t for every pair."""
    candle_close = t + CANDLE
    # 1) fills and trade management on candle t
    for pair in pairs:
        df = frames[pair]
        if t not in df.index:
            continue
        row = df.loc[t]
        if pair in st["pending"]:
            _fill(st, pair, float(row["open"]), t)
        tr = st["open"].get(pair)
        if tr is None:
            continue
        if row["low"] <= tr["stop"]:
            price = min(tr["stop"], float(row["open"]))
            _close_trade(st, pair, price, candle_close, "stop_loss")
        elif row["close"] >= tr["target"]:
            _close_trade(st, pair, float(row["close"]), candle_close, "take_profit")
        elif row["close"] < row["sma300"]:
            _close_trade(st, pair, float(row["close"]), candle_close, "trend_exit")

    # 2) protections
    recent = [s for s in st["sl_times"] if pd.Timestamp(s) > candle_close - 6 * CANDLE]
    st["sl_times"] = recent
    if len(recent) >= 2:
        st["pause_until"] = _ts(pd.Timestamp(max(recent)) + 3 * CANDLE)
        st["sl_times"] = []
    paused = st["pause_until"] and candle_close < pd.Timestamp(st["pause_until"])
    day = candle_close.normalize()
    day_pnl = sum(c["pnl"] for c in st["closed"] if pd.Timestamp(c["closed"]).normalize() == day)
    eq = _equity(st)
    blocked = paused or day_pnl <= -v.daily_loss_cap * v.start_cap or eq < v.floor_frac * v.start_cap
    exited_now = {c["pair"] for c in st["closed"] if pd.Timestamp(c["closed"]) == candle_close}

    # 3) new signals -> pending entries at next candle open
    if not blocked:
        for pair in pairs:
            df = frames[pair]
            if t not in df.index or pair in st["open"] or pair in st["pending"] or pair in exited_now:
                continue
            if len(st["open"]) + len(st["pending"]) >= v.max_open:
                break
            row = df.loc[t]
            if not bool(row["enter"]) or math.isnan(row["atr"]):
                continue
            sp = float(np.clip(v.atr_stop * row["atr"] / row["close"], v.min_stop, v.max_stop))
            stake = min(eq * v.risk / sp, eq * v.max_stake_frac, st["cash"] - sum(p["stake"] for p in st["pending"].values()))
            if stake < 5:
                continue
            st["pending"][pair] = {"sp": sp, "stake": stake, "r_mult": v.r_mult, "signal": _ts(t)}

    st["last_t"] = _ts(t)
    st["equity_curve"].append([_ts(candle_close), round(_equity(st), 4)])


def run(st: dict, v: Variant, raw: dict[str, pd.DataFrame], btc: pd.DataFrame, pairs: list[str],
        now: pd.Timestamp | None = None, backfill: bool = False) -> dict:
    """Advance the state through every closed candle not yet processed.
    raw frames may include the still-forming candle; it is only used to fill pending entries at its open."""
    frames = {p: indicators(raw[p], btc, v.vol_mult) for p in pairs}
    now = now or pd.Timestamp.now(tz="UTC")
    all_t = sorted({t for f in frames.values() for t in f.index})
    closed_t = [t for t in all_t if t + CANDLE <= now]
    last = pd.Timestamp(st["last_t"]) if st["last_t"] else None
    if last is None and not backfill:  # first live run: start from the latest closed candle only
        closed_t = closed_t[-1:]
    for t in closed_t:
        if last is None or t > last:
            step(st, v, frames, pairs, t)
    # fill pending entries at the open of the candle that is forming now
    forming = [t for t in all_t if t + CANDLE > now]
    if forming and st["pending"]:
        ft = forming[0]
        for pair in list(st["pending"]):
            df = frames[pair]
            if ft in df.index:
                _fill(st, pair, float(df.loc[ft, "open"]), ft)
                # mark so that processing candle ft later does not fill again
    st["last_prices"] = {p: float(frames[p]["close"].iloc[-1]) for p in pairs}
    return st
