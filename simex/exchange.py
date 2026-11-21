"""
SimBinanceUSDM: a replay of Binance USD(S)-M perpetual futures, driven by real 1-minute history.

What it copies from the real exchange
  * ccxt-style API (symbols like "SOL/USDT:USDT"): fetch_ohlcv, fetch_ticker, fetch_balance, fetch_positions,
    set_leverage, set_margin_mode, create_order, cancel_order, cancel_all_orders, fetch_open_orders,
    fetch_my_trades, amount_to_precision, price_to_precision, market
  * one-way mode, ISOLATED margin, leverage per symbol
  * order types: market, limit (GTC / post-only), STOP_MARKET, TAKE_PROFIT_MARKET (reduceOnly / closePosition),
    triggers on last price (CONTRACT_PRICE, Binance default) or MARK_PRICE
  * fees: maker 0.02%, taker 0.05% (VIP 0) + 18% GST on fees (charged to users in India)
  * tick size, lot step (measured from real trades) and 5 USDT minimum notional; orders that break them are rejected
  * funding every 8h with the real historical rates, paid from the position's isolated margin
  * liquidation on the real 1m MARK price using Binance's isolated-margin liquidation formula; the whole isolated
    margin is lost (insurance clearance fee takes what is left)
  * slippage: half the spread + size impact vs that minute's traded volume, always against you

How time works
  The exchange clock (self.now, ms) only moves when the caller calls advance(to_ms). Between bars nothing is
  known about the future. Market orders fill at the open of the next 1m bar (the order is sent at the start of
  that minute). advance() jumps straight over minutes where no order, stop or liquidation level is touched
  (vectorised scan), so years replay in minutes, and it stops exactly on the minute something happens.
  Inside one 1m bar the price path is assumed O -> nearer extreme -> other extreme -> C (standard OHLC path).
Not modelled: order-book queue position, partial fills of resting limit orders, ADL, exchange outages.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN = 60_000
H4 = 4 * 3600_000


@dataclass
class Fees:
    maker: float = 0.0002
    taker: float = 0.0005
    gst: float = 0.18                 # India: 18% GST on trading fees


@dataclass
class SymbolSpec:
    tick: float
    step: float
    min_notional: float = 5.0
    mmr: float = 0.01                 # maintenance margin rate, first bracket (conservative for alts)
    max_leverage: int = 20
    half_spread_bps: float = 1.0


class OrderRejected(Exception):
    pass


@dataclass
class Order:
    id: str
    symbol: str
    type: str                         # market | limit | stop_market | take_profit_market
    side: str                         # buy | sell
    amount: float
    price: float | None
    stop_price: float | None
    reduce_only: bool
    close_position: bool
    post_only: bool
    working_type: str
    created: int
    status: str = "open"
    filled: float = 0.0
    average: float | None = None
    fee: float = 0.0
    tag: str = ""


@dataclass
class Position:
    symbol: str
    side: int                         # 1 long, -1 short
    qty: float
    entry: float
    margin: float                     # isolated margin (wallet amount locked in this position)
    leverage: float
    opened: int
    funding_paid: float = 0.0
    fees_paid: float = 0.0
    max_qty: float = 0.0
    realized: float = 0.0
    tag: str = ""


@dataclass
class Bars:
    t: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    qv: np.ndarray
    mo: np.ndarray
    mh: np.ndarray
    ml: np.ndarray
    mc: np.ndarray
    fund_t: np.ndarray
    fund_r: np.ndarray


def make_bars(k1m: pd.DataFrame, mark1m: pd.DataFrame | None, funding: pd.DataFrame | None) -> Bars:
    k = k1m.sort_values("t").drop_duplicates("t").reset_index(drop=True)
    if mark1m is not None and len(mark1m):
        m = k[["t"]].merge(mark1m.drop_duplicates("t"), on="t", how="left")
        for col in "ohlc":                                   # missing mark minutes: fall back to last price
            m[col] = m[col].fillna(k[col])
    else:
        m = k
    ft = funding["t"].to_numpy("int64") if funding is not None and len(funding) else np.array([], dtype="int64")
    fr = funding["rate"].to_numpy(float) if funding is not None and len(funding) else np.array([], dtype=float)
    return Bars(k["t"].to_numpy("int64"), *(k[c].to_numpy(float) for c in "ohlc"),
                k["qv"].to_numpy(float) if "qv" in k else np.full(len(k), np.inf),
                *(m[c].to_numpy(float) for c in "ohlc"), ft, fr)


class SimBinanceUSDM:
    def __init__(self, bars: dict[str, Bars], specs: dict[str, SymbolSpec], start_ms: int, balance: float = 100.0,
                 fees: Fees | None = None, impact_bps: float = 10.0):
        self.bars, self.specs = bars, specs
        self.now = int(start_ms)
        self.wallet = float(balance)          # wallet balance (includes isolated margins locked in positions)
        self.fees = fees or Fees()
        self.impact_bps = impact_bps
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}
        self.leverage: dict[str, float] = {s: 1.0 for s in bars}
        self.trades: list[dict] = []          # every fill
        self.closed_positions: list[dict] = []  # one row per round trip (opened -> fully closed / liquidated)
        self.events: list[dict] = []          # funding, liquidations, rejections, triggers
        self._ids = itertools.count(1)
        self._h4 = {s: self._aggregate(b, H4) for s, b in bars.items()}
        self._cursor: dict[str, int] = {}
        self.totals = {"fees": 0.0, "gst": 0.0, "funding": 0.0, "slippage": 0.0, "liquidations": 0}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def sym(s: str) -> str:
        return s.split("/")[0]

    @staticmethod
    def _aggregate(b: Bars, ms: int) -> pd.DataFrame:
        df = pd.DataFrame({"t": b.t, "o": b.o, "h": b.h, "l": b.l, "c": b.c, "qv": b.qv})
        df["k"] = df["t"] // ms * ms
        g = df.groupby("k")
        out = pd.DataFrame({"o": g["o"].first(), "h": g["h"].max(), "l": g["l"].min(), "c": g["c"].last(),
                            "v": g["qv"].sum(), "last_t": g["t"].max()})
        return out.rename_axis("t").reset_index()

    def _idx(self, s: str, t: int) -> int:
        return int(np.searchsorted(self.bars[s].t, t, side="left"))

    def _round(self, x: float, step: float, mode: str = "floor") -> float:
        n = x / step
        n = math.floor(n + 1e-9) if mode == "floor" else (math.ceil(n - 1e-9) if mode == "ceil" else round(n))
        dec = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
        return round(n * step, dec + 2)

    def _fee(self, notional: float, maker: bool) -> tuple[float, float]:
        f = notional * (self.fees.maker if maker else self.fees.taker)
        return f, f * self.fees.gst

    def _log(self, kind: str, **kw):
        self.events.append({"t": self.now, "kind": kind, **kw})

    # ------------------------------------------------------------------ ccxt-style API
    def market(self, symbol: str) -> dict:
        sp = self.specs[self.sym(symbol)]
        return {"symbol": symbol, "precision": {"price": sp.tick, "amount": sp.step},
                "limits": {"amount": {"min": sp.step}, "cost": {"min": sp.min_notional},
                           "leverage": {"max": sp.max_leverage}}, "contractSize": 1, "linear": True}

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        return self._round(amount, self.specs[self.sym(symbol)].step, "floor")

    def price_to_precision(self, symbol: str, price: float) -> float:
        return self._round(price, self.specs[self.sym(symbol)].tick, "round")

    def milliseconds(self) -> int:
        return self.now

    def fetch_ohlcv(self, symbol: str, timeframe: str = "4h", since: int | None = None, limit: int = 500) -> list:
        """Closed candles plus the still-forming one, exactly what the real API returns at self.now."""
        assert timeframe == "4h", "only 4h is pre-aggregated"
        s = self.sym(symbol)
        df = self._h4[s]
        df = df[df["t"] < self.now]
        if len(df) and df["t"].iloc[-1] + H4 > self.now:      # forming candle: only minutes that have happened
            b, t0 = self.bars[s], int(df["t"].iloc[-1])
            i0, i1 = self._idx(s, t0), self._idx(s, self.now)
            if i1 > i0:
                last = [t0, b.o[i0], b.h[i0:i1].max(), b.l[i0:i1].min(), b.c[i1 - 1], b.qv[i0:i1].sum()]
            else:
                last = None
            df = df.iloc[:-1]
            rows = df[["t", "o", "h", "l", "c", "v"]].values.tolist()
            if last:
                rows.append(last)
        else:
            rows = df[["t", "o", "h", "l", "c", "v"]].values.tolist()
        if since is not None:
            rows = [r for r in rows if r[0] >= since]
        return rows[-limit:]

    def _last_bar(self, s: str) -> int:
        return max(self._idx(s, self.now) - 1, 0)

    def fetch_ticker(self, symbol: str) -> dict:
        s = self.sym(symbol)
        i, b = self._last_bar(s), self.bars[s]
        return {"symbol": symbol, "last": float(b.c[i]), "mark": float(b.mc[i]), "timestamp": int(b.t[i]) + MIN}

    def mark_price(self, s: str) -> float:
        return float(self.bars[s].mc[self._last_bar(s)])

    def unrealized(self, p: Position, mark: float | None = None) -> float:
        mark = self.mark_price(p.symbol) if mark is None else mark
        return p.side * p.qty * (mark - p.entry)

    def fetch_balance(self) -> dict:
        used = sum(p.margin for p in self.positions.values())
        upnl = sum(self.unrealized(p) for p in self.positions.values())
        return {"USDT": {"free": self.wallet - used, "used": used, "total": self.wallet},
                "info": {"walletBalance": self.wallet, "unrealizedProfit": upnl, "marginBalance": self.wallet + upnl}}

    def liq_price(self, p: Position) -> float:
        mmr = self.specs[p.symbol].mmr
        # Binance isolated, one-way, first bracket (maintenance amount 0):
        # LP = (WB - side*Q*EP) / (Q*MMR - side*Q)
        return (p.margin - p.side * p.qty * p.entry) / (p.qty * mmr - p.side * p.qty)

    def fetch_positions(self, symbols=None) -> list:
        out = []
        for s, p in self.positions.items():
            mark = self.mark_price(s)
            out.append({"symbol": f"{s}/USDT:USDT", "side": "long" if p.side == 1 else "short", "contracts": p.qty,
                        "entryPrice": p.entry, "markPrice": mark, "unrealizedPnl": self.unrealized(p, mark),
                        "leverage": p.leverage, "liquidationPrice": self.liq_price(p), "initialMargin": p.margin,
                        "marginMode": "isolated", "timestamp": p.opened})
        return out

    def set_leverage(self, leverage: float, symbol: str):
        s = self.sym(symbol)
        if leverage > self.specs[s].max_leverage:
            raise OrderRejected(f"leverage {leverage} > max {self.specs[s].max_leverage}")
        if s in self.positions:
            raise OrderRejected("cannot change leverage with an open position in this simulator")
        self.leverage[s] = float(leverage)

    def set_margin_mode(self, mode: str, symbol: str):
        if mode != "isolated":
            raise OrderRejected("simulator supports isolated margin only")

    def fetch_open_orders(self, symbol: str | None = None) -> list:
        return [vars(o) | {} for o in self.orders.values() if symbol is None or o.symbol == self.sym(symbol)]

    def fetch_my_trades(self, symbol: str | None = None, since: int | None = None) -> list:
        return [t for t in self.trades if (symbol is None or t["symbol"] == self.sym(symbol)) and (since is None or t["t"] >= since)]

    def cancel_order(self, oid: str, symbol: str | None = None):
        o = self.orders.pop(oid, None)
        if o:
            o.status = "canceled"
        return o

    def cancel_all_orders(self, symbol: str):
        s = self.sym(symbol)
        for oid in [k for k, o in self.orders.items() if o.symbol == s]:
            self.cancel_order(oid)

    def create_order(self, symbol: str, type: str, side: str, amount: float, price: float | None = None,
                     params: dict | None = None, tag: str = "") -> dict:
        params = params or {}
        s, sp = self.sym(symbol), self.specs[self.sym(symbol)]
        type = type.lower()
        close_pos = bool(params.get("closePosition"))
        reduce = bool(params.get("reduceOnly")) or close_pos
        stop = params.get("stopPrice")
        o = Order(str(next(self._ids)), s, type, side, float(amount or 0), price, stop, reduce, close_pos,
                  bool(params.get("postOnly")) or params.get("timeInForce") == "GTX",
                  params.get("workingType", "CONTRACT_PRICE"), self.now, tag=tag)
        # --- exchange filters (Binance rejects these) ---
        if not close_pos:
            if abs(o.amount / sp.step - round(o.amount / sp.step)) > 1e-6 or o.amount < sp.step:
                return self._reject(o, f"amount {o.amount} not a multiple of lot step {sp.step}")
        for px in (price, stop):
            if px is not None and abs(px / sp.tick - round(px / sp.tick)) > 1e-6:
                return self._reject(o, f"price {px} not a multiple of tick {sp.tick}")
        ref = price or stop or self.fetch_ticker(symbol)["last"]
        if not reduce and o.amount * ref < sp.min_notional:
            return self._reject(o, f"notional {o.amount * ref:.2f} < min {sp.min_notional}")
        pos = self.positions.get(s)
        if reduce and (pos is None or (pos.side == 1) == (side == "buy")):
            if type in ("market", "limit"):
                return self._reject(o, "reduce-only order would not reduce a position")
        if type in ("stop_market", "take_profit_market") and stop is None:
            return self._reject(o, "stopPrice required")
        if type == "market":
            self._fill_market(o)
        else:
            if type == "limit":
                last = self.fetch_ticker(symbol)["last"]
                crosses = (side == "buy" and price >= last) or (side == "sell" and price <= last)
                if crosses and o.post_only:
                    return self._reject(o, "post-only order would take liquidity (GTX)")
                if crosses:
                    o.type = "market"
                    self._fill_market(o)
                    return vars(o)
            if type in ("stop_market", "take_profit_market"):
                last = self.fetch_ticker(symbol)["last"]
                immediate = ((type == "stop_market" and ((side == "buy" and last >= stop) or (side == "sell" and last <= stop))) or
                             (type == "take_profit_market" and ((side == "buy" and last <= stop) or (side == "sell" and last >= stop))))
                if immediate:
                    return self._reject(o, "order would immediately trigger")
            self.orders[o.id] = o
        return vars(o)

    def _reject(self, o: Order, why: str) -> dict:
        o.status = "rejected"
        self._log("rejected", symbol=o.symbol, type=o.type, side=o.side, amount=o.amount, reason=why, tag=o.tag)
        return vars(o)

    # ------------------------------------------------------------------ fills
    def _slip(self, s: str, i: int, notional: float) -> float:
        sp, qv = self.specs[s], self.bars[s].qv[i]
        impact = self.impact_bps * math.sqrt(notional / qv) if qv > 0 else self.impact_bps
        return (sp.half_spread_bps + min(impact, 50.0)) / 1e4

    def _fill_market(self, o: Order, at_price: float | None = None, i: int | None = None, maker: bool = False):
        s = o.symbol
        if i is None:
            i = self._idx(s, self.now)                    # the minute starting now (order sent at its open)
            if i >= len(self.bars[s].t):
                return self._reject(o, "no market data")
        base = self.bars[s].o[i] if at_price is None else at_price
        pos = self.positions.get(s)
        qty = pos.qty if (o.close_position and pos) else o.amount
        if o.reduce_only:
            if pos is None:
                o.status = "canceled"
                return
            qty = min(qty, pos.qty)
        sign = 1 if o.side == "buy" else -1
        slip = 0.0 if maker else self._slip(s, i, qty * base)
        px = base * (1 + sign * slip)
        self.totals["slippage"] += qty * base * slip
        self._apply_fill(o, s, sign, qty, px, maker, int(self.bars[s].t[i]))

    def _apply_fill(self, o: Order, s: str, sign: int, qty: float, px: float, maker: bool, t: int):
        notional = qty * px
        fee, gst = self._fee(notional, maker)
        self.wallet -= fee + gst
        self.totals["fees"] += fee
        self.totals["gst"] += gst
        pos = self.positions.get(s)
        realized = 0.0
        if pos is None or pos.side == sign:                       # open / add
            lev = self.leverage[s]
            add_margin = notional / lev
            free = self.wallet - sum(p.margin for p in self.positions.values())
            if add_margin > free + 1e-9:
                self.wallet += fee + gst                         # undo fee: Binance rejects for insufficient margin
                self.totals["fees"] -= fee
                self.totals["gst"] -= gst
                return self._reject(o, f"insufficient margin: need {add_margin:.2f}, free {free:.2f}")
            if pos is None:
                pos = self.positions[s] = Position(s, sign, 0.0, px, 0.0, lev, t, tag=o.tag)
            pos.entry = (pos.entry * pos.qty + px * qty) / (pos.qty + qty)
            pos.qty += qty
            pos.max_qty = max(pos.max_qty, pos.qty)
            pos.margin += add_margin
            pos.fees_paid += fee + gst
        else:                                                     # reduce / close
            q = min(qty, pos.qty)
            realized = pos.side * q * (px - pos.entry)
            frac = q / pos.qty
            self.wallet += realized
            pos.margin -= pos.margin * frac
            pos.qty -= q
            pos.fees_paid += fee + gst
            pos.realized += realized
            if pos.qty <= 1e-12:
                self._record_close(pos, px, t, o.tag or o.type)
                del self.positions[s]
                for oid in [k for k, x in self.orders.items() if x.symbol == s and x.reduce_only]:
                    self.cancel_order(oid)                        # Binance cancels reduce-only/closePosition orders
        o.status, o.filled, o.average = "closed", qty, px
        o.fee = fee + gst
        self.orders.pop(o.id, None)
        self.trades.append({"t": t, "symbol": s, "side": o.side, "qty": qty, "price": px, "fee": fee, "gst": gst,
                            "maker": maker, "type": o.type, "reduce": o.reduce_only, "realized": realized,
                            "order": o.id, "tag": o.tag})

    # ------------------------------------------------------------------ time
    def _levels(self, s: str):
        """Prices whose touch would make something happen on this symbol: (level, use_mark, direction) where
        direction +1 = triggers when price goes >= level, -1 = when price goes <= level."""
        lv = []
        for o in self.orders.values():
            if o.symbol != s:
                continue
            mark = o.working_type == "MARK_PRICE"
            if o.type == "limit":
                lv.append((o.price, False, -1 if o.side == "buy" else 1))
            elif o.type == "stop_market":
                lv.append((o.stop_price, mark, 1 if o.side == "buy" else -1))
            elif o.type == "take_profit_market":
                lv.append((o.stop_price, mark, -1 if o.side == "buy" else 1))
        p = self.positions.get(s)
        if p is not None:
            lv.append((self.liq_price(p), True, -p.side))
        return lv

    def _first_hit(self, s: str, i0: int, i1: int) -> int:
        """Index of the first bar in [i0, i1) that touches any level, or i1."""
        b, first = self.bars[s], i1
        for level, mark, d in self._levels(s):
            hi, lo = (b.mh, b.ml) if mark else (b.h, b.l)
            seg = hi[i0:i1] >= level if d == 1 else lo[i0:i1] <= level
            k = np.argmax(seg) if seg.size else 0
            if seg.size and seg[k]:
                first = min(first, i0 + int(k))
        return first

    def _path(self, s: str, i: int, mark: bool):
        b = self.bars[s]
        o, h, l, c = (b.mo[i], b.mh[i], b.ml[i], b.mc[i]) if mark else (b.o[i], b.h[i], b.l[i], b.c[i])
        return [o, l, h, c] if c >= o else [o, h, l, c]

    def _process_bar(self, s: str, i: int):
        """Resolve everything that happens on symbol s during 1m bar i, walking the O-H-L-C path."""
        self.now = int(self.bars[s].t[i])
        for _ in range(6):                                        # a fill can create new levels; loop a few times
            hits = []
            for o in list(self.orders.values()):
                if o.symbol != s:
                    continue
                mark = o.working_type == "MARK_PRICE"
                path = self._path(s, i, mark)
                if o.type == "limit":
                    lvl, d = o.price, (-1 if o.side == "buy" else 1)
                elif o.type == "stop_market":
                    lvl, d = o.stop_price, (1 if o.side == "buy" else -1)
                elif o.type == "take_profit_market":
                    lvl, d = o.stop_price, (-1 if o.side == "buy" else 1)
                else:
                    continue
                seg = self._seg_hit(path, lvl, d, strict=(o.type == "limit"))
                if seg is not None:
                    hits.append((seg, o, lvl))
            p = self.positions.get(s)
            if p is not None:
                lp = self.liq_price(p)
                seg = self._seg_hit(self._path(s, i, True), lp, -p.side, strict=False)
                if seg is not None:
                    hits.append((seg, None, lp))
            if not hits:
                return
            seg, o, lvl = min(hits, key=lambda x: (x[0], x[1] is not None))     # earliest on the path; liquidation first on ties
            if o is None:
                self._liquidate(s, i, lvl)
                continue
            path = self._path(s, i, False)
            if o.type == "limit":
                self._log("limit_fill", symbol=s, side=o.side, price=lvl, tag=o.tag)
                sign = 1 if o.side == "buy" else -1
                self._apply_fill(o, s, sign, o.amount if not o.reduce_only else min(o.amount, self.positions[s].qty) if s in self.positions else 0,
                                 lvl, True, int(self.bars[s].t[i]))
            else:
                # trigger: fill at the trigger level, or at the open if the bar gapped through it
                gap_open = path[0]
                d = (1 if o.side == "buy" else -1) if o.type == "stop_market" else (-1 if o.side == "buy" else 1)
                px = max(lvl, gap_open) if (d == 1 and gap_open > lvl) else (min(lvl, gap_open) if (d == -1 and gap_open < lvl) else lvl)
                self._log("trigger", symbol=s, type=o.type, side=o.side, level=lvl, price=px, tag=o.tag)
                self.orders.pop(o.id, None)
                self._fill_market(o, at_price=px, i=i)

    @staticmethod
    def _seg_hit(path, level, d, strict):
        """Which leg of the path first reaches the level (0..2), or None."""
        for k in range(3):
            a, b = path[k], path[k + 1]
            lo, hi = min(a, b), max(a, b)
            if k == 0 and ((d == 1 and (a > level if strict else a >= level)) or (d == -1 and (a < level if strict else a <= level))):
                return 0
            if strict:
                if (d == 1 and hi > level) or (d == -1 and lo < level):
                    return k
            elif lo <= level <= hi:
                return k
        return None

    def _record_close(self, p: Position, px: float, t: int, reason: str):
        net = p.realized - p.fees_paid - p.funding_paid
        self.closed_positions.append({
            "symbol": p.symbol, "side": "long" if p.side == 1 else "short", "qty": p.max_qty, "entry": p.entry, "exit": px,
            "opened": p.opened, "closed": t, "gross": p.realized, "fees": p.fees_paid, "funding": -p.funding_paid,
            "net": net, "reason": reason, "entry_tag": p.tag})

    def _liquidate(self, s: str, i: int, lp: float):
        p = self.positions.pop(s)
        loss = -p.margin                                           # isolated margin is gone
        p.realized += loss                                         # net = -margin - fees - funding (wallet-consistent)
        self._record_close(p, lp, int(self.bars[s].t[i]), "liquidation")
        self.wallet += loss
        self.totals["liquidations"] += 1
        for oid in [k for k, x in self.orders.items() if x.symbol == s]:
            self.cancel_order(oid)
        self.trades.append({"t": int(self.bars[s].t[i]), "symbol": s, "side": "sell" if p.side == 1 else "buy", "qty": p.qty,
                            "price": lp, "fee": 0.0, "gst": 0.0, "maker": False, "type": "liquidation", "reduce": True,
                            "realized": loss, "order": "liq", "tag": "liquidation"})
        self._log("liquidation", symbol=s, price=lp, qty=p.qty, lost=p.margin)

    def _funding(self, s: str, t: int):
        p = self.positions.get(s)
        b = self.bars[s]
        k = np.searchsorted(b.fund_t, t)
        if p is None or k >= len(b.fund_t) or b.fund_t[k] != t:
            return
        i = max(self._idx(s, t) - 1, 0)
        mark = b.mc[i]
        pay = -p.side * p.qty * mark * b.fund_r[k]               # positive = received
        p.margin += pay
        self.wallet += pay
        p.funding_paid -= pay
        self.totals["funding"] += pay
        self._log("funding", symbol=s, rate=float(b.fund_r[k]), amount=pay)

    def advance(self, to_ms: int):
        """Move the clock to to_ms, resolving every fill, trigger, funding payment and liquidation on the way.
        Fast-forwards over minutes where nothing can happen."""
        to_ms = int(to_ms)
        while True:
            # next funding time among symbols with a position
            nxt_f = min((int(self.bars[s].fund_t[k]) for s in self.positions
                         for k in [np.searchsorted(self.bars[s].fund_t, self.now)] if k < len(self.bars[s].fund_t)),
                        default=None)
            stop_at = min(to_ms, nxt_f) if nxt_f is not None and nxt_f > self.now else to_ms
            # earliest bar that touches a level on any active symbol
            best = None
            for s in {o.symbol for o in self.orders.values()} | set(self.positions):
                i0, i1 = max(self._idx(s, self.now), self._cursor.get(s, 0)), self._idx(s, stop_at)
                k = self._first_hit(s, i0, i1) if i1 > i0 else i1
                if k < i1:
                    t = int(self.bars[s].t[k])
                    if best is None or t < best[0]:
                        best = (t, s, k)
            if best is not None:
                t, s, k = best
                self._process_bar(s, k)
                self._cursor[s] = k + 1                             # this symbol's minute is done
                self.now = t                                        # other symbols may still act in this minute
                continue
            self.now = stop_at
            if nxt_f is not None and stop_at == nxt_f:
                for s in list(self.positions):
                    self._funding(s, nxt_f)
                self.now = nxt_f + 1                                # funding handled; continue
                continue
            if self.now >= to_ms:
                self.now = to_ms
                return

    def equity(self) -> float:
        return self.wallet + sum(self.unrealized(p) for p in self.positions.values())
