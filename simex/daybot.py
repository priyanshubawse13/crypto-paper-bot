"""
Day-trading bot for the simulated exchange: intraday setups on 15m / 1h candles, longs and shorts.

Setups (each has a long and a mirrored short version; signals are read at the candle close, the order goes in
at the next candle's open):
  pullback  - trend pullback: 4H trend agrees, EMA20 > EMA50, price dips into EMA20 and closes back above it with a
              bullish engulfing or hammer candle, RSI 40-65. Stop under the low of the last two candles.
  sweep     - liquidity sweep / fake breakdown ("trap"): price takes out the last swing low (stops below support get
              hit) but closes back above it, on above-average volume, closing in the upper half of the candle.
              Stop under the sweep candle's low.
  breakout  - volume-confirmed breakout: close above the 20-candle high on > vol_mult x average volume with a strong
              body (>= 60% of the range, i.e. not a wick); weak-volume breakouts are skipped as likely fakes.
              Stop under the breakout candle.
  bounce    - support bounce: price tests the last swing low (support zone = +/- 0.3 ATR) and prints a bullish
              engulfing or hammer with RSI < 45. Stop under support.
Exits: exchange-side STOP_MARKET and TAKE_PROFIT_MARKET (target = r_mult x the stop distance), optional move of
the stop to break-even, and a time stop (day trade: closed after max_bars candles). 1% risk per trade; leverage up
to 20x with the liquidation guard; max 3 open; no new trades for the rest of the UTC day after a 3% loss day.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import talib as ta

H4 = 4 * 3600_000


@dataclass
class DayConfig:
    name: str
    tf: str = "1h"
    setups: tuple = ("pullback",)
    longs: bool = True
    shorts: bool = True
    htf_filter: bool = True
    r_mult: float = 1.5
    be_at: float = 0.0
    max_bars: int = 24
    risk: float = 0.01
    leverage: float = 20.0
    max_open: int = 3
    max_margin_frac: float = 0.33
    min_stop: float = 0.003
    max_stop: float = 0.03
    vol_mult: float = 1.5
    swing_k: int = 3
    daily_loss_cap: float = 0.03
    floor_frac: float = 0.5


def signals(df: pd.DataFrame, h4: pd.DataFrame, tf_ms: int, c: DayConfig) -> pd.DataFrame:
    """Vectorised, causal signals. df/h4 columns: t (open ms), open high low close volume.
    Output per candle: side (+1/-1/0), stop price, setup name."""
    d = df.copy()
    o, h, l, cl, v = (d[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
    d["ema20"], d["ema50"] = ta.EMA(cl, 20), ta.EMA(cl, 50)
    d["rsi"], d["atr"] = ta.RSI(cl, 14), ta.ATR(h, l, cl, 14)
    d["vavg"] = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    rng = np.where(h - l > 0, h - l, np.nan)
    body = np.abs(cl - o)
    pos = (cl - l) / rng                                   # where the close sits in the candle (0 low .. 1 high)
    o1, c1 = np.r_[np.nan, o[:-1]], np.r_[np.nan, cl[:-1]]
    bull_eng = (cl > o) & (c1 < o1) & (cl >= o1) & (o <= c1)
    bear_eng = (cl < o) & (c1 > o1) & (cl <= o1) & (o >= c1)
    lw, uw = np.minimum(o, cl) - l, h - np.maximum(o, cl)
    hammer = (lw >= 2 * body) & (uw <= np.maximum(body, 1e-12) * 1.0) & (pos >= 0.6)
    star = (uw >= 2 * body) & (lw <= np.maximum(body, 1e-12) * 1.0) & (pos <= 0.4)
    k = c.swing_k
    # swing levels: a pivot is only known k candles after it happened (shift k = no look-ahead)
    lo_s, hi_s = pd.Series(l), pd.Series(h)
    piv_lo = lo_s.where(lo_s == lo_s.rolling(2 * k + 1, center=True).min())
    piv_hi = hi_s.where(hi_s == hi_s.rolling(2 * k + 1, center=True).max())
    swing_lo = piv_lo.shift(k).ffill().shift(1).to_numpy()      # last confirmed swing low BEFORE this candle
    swing_hi = piv_hi.shift(k).ffill().shift(1).to_numpy()
    hh20 = hi_s.rolling(20).max().shift(1).to_numpy()
    ll20 = lo_s.rolling(20).min().shift(1).to_numpy()
    # 4H trend, using only 4H candles that closed by this candle's close
    h4c = h4[["t", "close"]].copy()
    h4c["e50"] = ta.EMA(h4c["close"].to_numpy(float), 50)
    h4c["up"] = (h4c["close"] > h4c["e50"]) & (h4c["e50"] > h4c["e50"].shift(6))
    h4c["dn"] = (h4c["close"] < h4c["e50"]) & (h4c["e50"] < h4c["e50"].shift(6))
    h4c["avail"] = h4c["t"] + H4
    m = pd.merge_asof(pd.DataFrame({"tc": d["t"] + tf_ms}), h4c[["avail", "up", "dn"]].rename(columns={"avail": "tc"}),
                      on="tc", direction="backward")
    up4, dn4 = m["up"].fillna(False).to_numpy(bool), m["dn"].fillna(False).to_numpy(bool)
    if not c.htf_filter:
        up4 = dn4 = np.ones(len(d), bool)
    e20, e50, rsi, atr, vavg = (d[x].to_numpy() for x in ("ema20", "ema50", "rsi", "atr", "vavg"))
    volok = v > c.vol_mult * vavg
    l1, h1 = np.r_[np.nan, l[:-1]], np.r_[np.nan, h[:-1]]
    S = {}
    S["pullback"] = (up4 & (e20 > e50) & (l <= e20 + 0.1 * atr) & (cl > e20) & (bull_eng | hammer) & (rsi > 40) & (rsi < 65),
                     np.fmin(l, l1) - 0.1 * atr,
                     dn4 & (e20 < e50) & (h >= e20 - 0.1 * atr) & (cl < e20) & (bear_eng | star) & (rsi < 60) & (rsi > 35),
                     np.fmax(h, h1) + 0.1 * atr)
    S["sweep"] = (up4 & (l < swing_lo) & (cl > swing_lo) & volok & (pos >= 0.5), l - 0.1 * atr,
                  dn4 & (h > swing_hi) & (cl < swing_hi) & volok & (pos <= 0.5), h + 0.1 * atr)
    S["breakout"] = (up4 & (cl > hh20) & volok & (body / rng >= 0.6), l - 0.1 * atr,
                     dn4 & (cl < ll20) & volok & (body / rng >= 0.6), h + 0.1 * atr)
    near_lo = (l <= swing_lo + 0.3 * atr) & (l >= swing_lo - 0.3 * atr)
    near_hi = (h >= swing_hi - 0.3 * atr) & (h <= swing_hi + 0.3 * atr)
    S["bounce"] = (up4 & near_lo & (bull_eng | hammer) & (rsi < 45), np.fmin(l, swing_lo) - 0.2 * atr,
                   dn4 & near_hi & (bear_eng | star) & (rsi > 55), np.fmax(h, swing_hi) + 0.2 * atr)
    side = np.zeros(len(d), int)
    stop = np.full(len(d), np.nan)
    name = np.array([""] * len(d), dtype=object)
    for sname in c.setups:                                   # first listed setup wins on a candle
        L, Ls, Sh, Ss = S[sname]
        L, Sh = np.nan_to_num(L).astype(bool), np.nan_to_num(Sh).astype(bool)
        take_l = c.longs & L & (side == 0)
        take_s = c.shorts & Sh & (side == 0) & ~take_l
        side[take_l], stop[take_l], name[take_l] = 1, Ls[take_l], sname
        side[take_s], stop[take_s], name[take_s] = -1, Ss[take_s], sname
    return pd.DataFrame({"t": d["t"].to_numpy(), "close": cl, "high": h, "low": l, "side": side, "stop": stop, "setup": name})


class DayBot:
    def __init__(self, ex, pairs, c: DayConfig, sig: dict[str, pd.DataFrame], tf_ms: int, start_cap: float):
        self.ex, self.pairs, self.c, self.tf = ex, pairs, c, tf_ms
        self.start_cap = start_cap
        self.sig = {p: (s["t"].to_numpy("int64"), s["side"].to_numpy(), s["stop"].to_numpy(), s["setup"].to_numpy(),
                        s["close"].to_numpy(), s["high"].to_numpy(), s["low"].to_numpy()) for p, s in sig.items()}
        self.meta: dict[str, dict] = {}
        self.seen = 0
        self.lev_used: list[float] = []
        self.skipped = {"min_notional": 0, "rejected": 0, "too_wide": 0}

    def _bar(self, p: str, t_open: int):
        T = self.sig[p][0]
        i = int(np.searchsorted(T, t_open))
        return i if i < len(T) and T[i] == t_open else None

    def _finish(self):
        for r in self.ex.closed_positions[self.seen:]:
            m = self.meta.pop(r["symbol"], {})
            ru = m.get("risk_usd")
            r["R"] = r["net"] / ru if ru else None
            r["R_gross"] = r["gross"] / ru if ru else None
            r["setup"], r["stop"], r["target"] = m.get("setup"), m.get("stop"), m.get("target")
        self.seen = len(self.ex.closed_positions)

    def on_close(self):
        ex, c = self.ex, self.c
        now = ex.now
        t_open = now - self.tf
        self._finish()
        # manage open trades: time stop, break-even
        for p in list(self.meta):
            if p not in ex.positions:
                continue
            m = self.meta[p]
            m["bars"] += 1
            i = self._bar(p, t_open)
            if i is None:
                continue
            _, _, _, _, cl, hi, lo = self.sig[p]
            side = m["side"]
            fav = (hi[i] - m["entry"]) if side == 1 else (m["entry"] - lo[i])
            m["mfe"] = max(m["mfe"], fav / m["dist"])
            sym = f"{p}/USDT:USDT"
            if m["bars"] >= c.max_bars:
                ex.cancel_all_orders(sym)
                ex.create_order(sym, "market", "sell" if side == 1 else "buy", ex.positions[p].qty,
                                params={"reduceOnly": True}, tag="time_exit")
                continue
            if c.be_at and not m["be"] and m["mfe"] >= c.be_at:
                tick = ex.market(sym)["precision"]["price"]
                be = m["entry"] * (1 + side * 0.0012)            # covers round-trip fees
                be = (math.ceil(be / tick) if side == 1 else math.floor(be / tick)) * tick
                okside = (be < cl[i]) if side == 1 else (be > cl[i])
                if okside:
                    ex.cancel_order(m["stop_id"])
                    so = ex.create_order(sym, "stop_market", "sell" if side == 1 else "buy", 0,
                                         params={"stopPrice": ex.price_to_precision(sym, be), "closePosition": True}, tag="breakeven")
                    if so["status"] == "open":
                        m["stop_id"], m["be"] = so["id"], True
        self._finish()
        # protections
        wallet = ex.wallet
        day0 = now // 86_400_000 * 86_400_000
        day_pnl = sum(r["net"] for r in ex.closed_positions if r["closed"] >= day0)
        if day_pnl <= -c.daily_loss_cap * self.start_cap or wallet < c.floor_frac * self.start_cap:
            return
        # entries
        open_now = set(ex.positions)
        if len(open_now) >= c.max_open:
            return
        for p in self.pairs:
            if len(open_now) >= c.max_open:
                break
            if p in open_now:
                continue
            i = self._bar(p, t_open)
            if i is None:
                continue
            T, SIDE, STOP, SETUP, CL, _, _ = self.sig[p]
            side = int(SIDE[i])
            if side == 0 or not np.isfinite(STOP[i]):
                continue
            sym = f"{p}/USDT:USDT"
            price = ex.fetch_ticker(sym)["last"]
            stop_px = STOP[i]
            sp = side * (price - stop_px) / price
            if sp <= 0:
                continue
            if sp < c.min_stop:                                  # too tight for noise: widen to the minimum
                stop_px, sp = price * (1 - side * c.min_stop), c.min_stop
            if sp > c.max_stop:
                self.skipped["too_wide"] += 1
                continue
            mmr = ex.specs[p].mmr
            lev = max(1.0, min(c.leverage, math.floor(1 / (1.5 * sp + mmr))))
            bal = ex.fetch_balance()["USDT"]
            notional = bal["total"] * c.risk / sp
            margin = min(notional / lev, bal["total"] * c.max_margin_frac, bal["free"] * 0.98)
            qty = ex.amount_to_precision(sym, margin * lev / price)
            if qty * price < ex.market(sym)["limits"]["cost"]["min"]:
                self.skipped["min_notional"] += 1
                continue
            ex.set_leverage(lev, sym)
            o = ex.create_order(sym, "market", "buy" if side == 1 else "sell", qty, tag="entry")
            if o["status"] != "closed":
                self.skipped["rejected"] += 1
                continue
            self.lev_used.append(lev)
            entry = o["average"]
            dist = abs(entry - stop_px)
            if side * (entry - stop_px) <= 0:                    # filled through the stop already
                ex.create_order(sym, "market", "sell" if side == 1 else "buy", qty, params={"reduceOnly": True}, tag="stop")
                self._finish()
                continue
            tick = ex.market(sym)["precision"]["price"]
            spx = ex.price_to_precision(sym, (math.floor(stop_px / tick) if side == 1 else math.ceil(stop_px / tick)) * tick)
            tpx = ex.price_to_precision(sym, entry + side * c.r_mult * dist)
            so = ex.create_order(sym, "stop_market", "sell" if side == 1 else "buy", 0, params={"stopPrice": spx, "closePosition": True}, tag="stop")
            to = ex.create_order(sym, "take_profit_market", "sell" if side == 1 else "buy", 0, params={"stopPrice": tpx, "closePosition": True}, tag="take_profit")
            if so["status"] != "open" or to["status"] != "open":
                ex.cancel_all_orders(sym)
                if p in ex.positions:
                    ex.create_order(sym, "market", "sell" if side == 1 else "buy", qty, params={"reduceOnly": True}, tag="order_error")
                self._finish()
                continue
            self.meta[p] = {"side": side, "entry": entry, "dist": dist, "stop": spx, "target": tpx, "stop_id": so["id"],
                            "risk_usd": qty * abs(entry - spx), "setup": SETUP[i], "bars": 0, "mfe": 0.0, "be": False}
            open_now.add(p)
