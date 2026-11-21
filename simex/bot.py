"""
The v3 trend bot, written against an exchange API (the simulator now; real Binance via ccxt later).

Every 4H close it:
  1. reads what happened since last time (stop-outs, liquidations) from the exchange
  2. exits open trades whose target was reached at the close (1.5R long / 1R short) or whose trend ended
     (close back across the 50-day average) with a reduce-only market order, and cancels their stop
  3. applies the protections (12h pause after 2 losing stops in 24h, 2% daily loss cap, 50% floor)
  4. opens new trades on signals: market order, then a reduce-only STOP_MARKET (closePosition) on the
     exchange at 2.5x ATR from the actual fill price, isolated margin, 2x leverage, 1% of wallet at risk
Signals use the same indicator code as the paper engine (paperbot.engine.indicators).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from paperbot.engine import Variant, indicators

H4 = 4 * 3600_000


def to_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df.drop(columns="t")


class TrendBot:
    def __init__(self, ex, pairs: list[str], v: Variant, ind: dict[str, pd.DataFrame], start_cap: float):
        """ind: per-pair indicator frames (index = candle open time). Only rows whose candle has CLOSED by
        ex.now are ever read (checked in _row)."""
        self.ex, self.pairs, self.v, self.ind = ex, pairs, v, ind
        self.start_cap = start_cap
        self.meta: dict[str, dict] = {}         # per open trade: target, stop order id, R risk
        self.losing_stops: list[int] = []
        self.pause_until = 0
        self.seen_closed = 0
        self.log: list[dict] = []
        self.skipped = {"min_notional": 0, "rejected": 0, "no_margin": 0}

    def _row(self, p: str, t_open: int):
        df = self.ind[p]
        ts = pd.Timestamp(t_open, unit="ms", tz="UTC")
        assert t_open + H4 <= self.ex.now, "look-ahead: candle not closed yet"
        return df.loc[ts] if ts in df.index else None

    def _note(self, kind, **kw):
        self.log.append({"t": self.ex.now, "kind": kind, **kw})

    def on_close(self):
        ex, v = self.ex, self.v
        now = ex.now
        t_open = now - H4                                     # the candle that just closed
        # 1. what happened on the exchange since last time
        for r in ex.closed_positions[self.seen_closed:]:
            m = self.meta.pop(r["symbol"], {})
            r["R"] = r["net"] / m["risk_usd"] if m.get("risk_usd") else None
            r["target"], r["stop"] = m.get("target"), m.get("stop")
            if r["reason"] in ("stop", "stop_market", "liquidation") and r["net"] < 0:
                self.losing_stops.append(r["closed"])
        self.seen_closed = len(ex.closed_positions)
        recent = [x for x in self.losing_stops if x > now - 6 * H4]
        self.losing_stops = recent
        if len(recent) >= 2:
            self.pause_until = max(recent) + 3 * H4
            self.losing_stops = []
        # 2. exits at the close
        positions = {ex.sym(p["symbol"]): p for p in ex.fetch_positions()}
        exited = set()
        for s, pos in positions.items():
            r = self._row(s, t_open)
            m = self.meta.get(s)
            if r is None or m is None:
                continue
            side = 1 if pos["side"] == "long" else -1
            hit_tp = r["close"] >= m["target"] if side == 1 else r["close"] <= m["target"]
            trend_over = r["close"] < r["sma300"] if side == 1 else r["close"] > r["sma300"]
            if hit_tp or trend_over:
                ex.cancel_all_orders(f"{s}/USDT:USDT")
                o = ex.create_order(f"{s}/USDT:USDT", "market", "sell" if side == 1 else "buy", pos["contracts"],
                                    params={"reduceOnly": True}, tag="take_profit" if hit_tp else "trend_exit")
                exited.add(s)
                if o["status"] != "closed":
                    self._note("exit_failed", symbol=s, status=o["status"])
        # attach R to positions closed just now by our own exits
        for r in ex.closed_positions[self.seen_closed:]:
            m = self.meta.pop(r["symbol"], {})
            r["R"] = r["net"] / m["risk_usd"] if m.get("risk_usd") else None
            r["target"], r["stop"] = m.get("target"), m.get("stop")
        self.seen_closed = len(ex.closed_positions)
        # 3. protections
        bal = ex.fetch_balance()["USDT"]
        wallet = bal["total"]
        day0 = now // 86_400_000 * 86_400_000
        day_pnl = sum(r["net"] for r in ex.closed_positions if r["closed"] >= day0)
        if now < self.pause_until or day_pnl <= -v.daily_loss_cap * self.start_cap or wallet < v.floor_frac * self.start_cap:
            return
        # 4. entries
        open_now = {ex.sym(p["symbol"]) for p in ex.fetch_positions()}
        for s in self.pairs:
            if len(open_now) >= v.max_open:
                break
            if s in open_now or s in exited:
                continue
            r = self._row(s, t_open)
            if r is None:
                continue
            side = 1 if bool(r["enter_long"]) else (-1 if v.shorts and bool(r["enter_short"]) else 0)
            if side == 0 or not np.isfinite(r["atr"]):
                continue
            atr_m = v.s_atr_stop if (side == -1 and v.s_atr_stop) else v.atr_stop
            sp = float(np.clip(atr_m * r["atr"] / r["close"], v.min_stop, v.max_stop))
            bal = ex.fetch_balance()["USDT"]
            notional = bal["total"] * v.risk * (v.s_risk if side == -1 else 1.0) / sp
            margin = min(notional / v.leverage, bal["total"] * v.max_margin_frac, bal["free"] * 0.98)
            price = ex.fetch_ticker(f"{s}/USDT:USDT")["last"]
            qty = ex.amount_to_precision(f"{s}/USDT:USDT", margin * v.leverage / price)
            if qty * price < ex.market(f"{s}/USDT:USDT")["limits"]["cost"]["min"]:
                self.skipped["min_notional"] += 1
                self._note("skip_min_size", symbol=s, notional=qty * price)
                continue
            sym = f"{s}/USDT:USDT"
            ex.set_leverage(v.leverage, sym)
            ex.set_margin_mode("isolated", sym)
            o = ex.create_order(sym, "market", "buy" if side == 1 else "sell", qty, tag="entry")
            if o["status"] != "closed":
                self.skipped["rejected"] += 1
                continue
            entry = o["average"]
            stop_raw = entry * (1 - side * sp)
            tick = ex.market(sym)["precision"]["price"]
            stop_px = (math.floor(stop_raw / tick) if side == 1 else math.ceil(stop_raw / tick)) * tick
            stop_px = ex.price_to_precision(sym, stop_px)
            so = ex.create_order(sym, "stop_market", "sell" if side == 1 else "buy", 0,
                                 params={"stopPrice": stop_px, "closePosition": True}, tag="stop")
            if so["status"] == "rejected":                    # would trigger at once: close immediately
                ex.create_order(sym, "market", "sell" if side == 1 else "buy", qty, params={"reduceOnly": True}, tag="stop")
                continue
            r_mult = (v.s_r_mult or v.r_mult) if side == -1 else v.r_mult
            self.meta[s] = {"target": entry * (1 + side * r_mult * sp), "stop": stop_px, "stop_id": so["id"],
                            "risk_usd": qty * abs(entry - stop_px)}
            open_now.add(s)
