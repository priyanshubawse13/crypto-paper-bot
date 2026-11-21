"""
Replay a day-trading configuration through the simulated Binance futures exchange.

    python -m simex.run_day --data data_fut --tf 1h --setups pullback,sweep --out sweep/x
Prints and saves summary.json: gross R (before costs) vs net R (after fees, GST, funding, slippage) per trade,
win rate, profit factor, drawdown, by setup / side / year and for 2022-Mar 2024 vs Apr 2024-now.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .daybot import DayBot, DayConfig, signals
from .exchange import TF_MS, Fees, SimBinanceUSDM, SymbolSpec, make_bars
from .run_sim import MMR, PAIRS12, SPREAD_BPS

H4 = 4 * 3600_000
SPLIT = int(pd.Timestamp("2024-04-01", tz="UTC").value // 10**6)


def frame(ex, ms, s):
    df = ex.candles(ms)[s].rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["t", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def stats(cp: pd.DataFrame) -> dict:
    if cp.empty:
        return {"n": 0}
    w, l = cp[cp.net > 0].net.sum(), -cp[cp.net <= 0].net.sum()
    return {"n": int(len(cp)), "win_rate": round(float((cp.net > 0).mean()), 3),
            "avg_R_gross": round(float(cp.R_gross.mean()), 3), "avg_R_net": round(float(cp.R.mean()), 3),
            "cost_R_per_trade": round(float((cp.R_gross - cp.R).mean()), 3),
            "net_usd": round(float(cp.net.sum()), 2), "pf": round(float(w / l), 2) if l > 0 else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_fut")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--pairs", default=",".join(PAIRS12))
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--setups", default="pullback")
    ap.add_argument("--no-htf", action="store_true")
    ap.add_argument("--longs-only", action="store_true")
    ap.add_argument("--shorts-only", action="store_true")
    ap.add_argument("--r", type=float, default=1.5)
    ap.add_argument("--be", type=float, default=0.0)
    ap.add_argument("--max-bars", type=int, default=None)
    ap.add_argument("--leverage", type=float, default=20.0)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--vol-mult", type=float, default=1.5)
    ap.add_argument("--balance", type=float, default=100.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="sweep/day")
    a = ap.parse_args(argv)
    t0 = time.time()
    D = Path(a.data)
    filters = json.loads((D / "filters.json").read_text())
    pairs = [p for p in a.pairs.split(",") if (D / f"{p}_1m.parquet").exists()]
    rd = lambda f: pd.read_parquet(f) if f.exists() else None
    bars, specs = {}, {}
    for s in pairs:
        bars[s] = make_bars(pd.read_parquet(D / f"{s}_1m.parquet"), rd(D / f"{s}_mark1m.parquet"), rd(D / f"{s}_funding.parquet"))
        f = filters.get(s, {})
        specs[s] = SymbolSpec(tick=f.get("tick", 1e-4), step=f.get("step", 1e-3), mmr=MMR.get(s, 0.01), max_leverage=75,
                              half_spread_bps=SPREAD_BPS.get(s, 2.0))
    ms = TF_MS[a.tf]
    start = int(pd.Timestamp(a.start, tz="UTC").value // 10**6)
    start = (start + ms - 1) // ms * ms
    end = int(pd.Timestamp(a.end, tz="UTC").value // 10**6) if a.end else int(max(b.t[-1] for b in bars.values())) + 60_000
    end = end // ms * ms
    ex = SimBinanceUSDM(bars, specs, start, balance=a.balance, fees=Fees())
    max_bars = a.max_bars or (24 if a.tf == "1h" else 96 if a.tf == "15m" else 6)       # ~one day
    c = DayConfig(name=a.name or f"{a.tf}-{a.setups}", tf=a.tf, setups=tuple(a.setups.split(",")), htf_filter=not a.no_htf,
                  longs=not a.shorts_only, shorts=not a.longs_only, r_mult=a.r, be_at=a.be, max_bars=max_bars,
                  leverage=a.leverage, risk=a.risk, vol_mult=a.vol_mult)
    sig = {p: signals(frame(ex, ms, p), frame(ex, H4, p), ms, c) for p in pairs}
    bot = DayBot(ex, pairs, c, sig, ms, a.balance)
    eq, last = [], time.time()
    for t in range(start, end + 1, ms):
        ex.advance(t)
        bot.on_close()
        if t % H4 == 0:
            eq.append([t, round(ex.equity(), 4)])
        if time.time() - last > 30:
            print(f"  {pd.Timestamp(t, unit='ms'):%Y-%m-%d} equity {ex.equity():.2f} trades {len(ex.closed_positions)}", flush=True)
            last = time.time()
    bot._finish()
    cp = pd.DataFrame(ex.closed_positions)
    if len(cp):
        cp = cp[cp.R.notna()]
    eqs = pd.Series([e[1] for e in eq]) if eq else pd.Series([a.balance])
    summary = {
        "name": c.name, "tf": a.tf, "setups": a.setups, "htf_filter": c.htf_filter, "r_mult": c.r_mult, "be_at": c.be_at,
        "max_bars": max_bars, "leverage": c.leverage, "risk": c.risk, "pairs": len(pairs),
        "final_equity": round(ex.equity(), 2), "max_drawdown_pct": round(float((1 - eqs / eqs.cummax()).max() * 100), 1),
        "all": stats(cp),
        "first_half": stats(cp[cp.closed < SPLIT]) if len(cp) else {}, "second_half": stats(cp[cp.closed >= SPLIT]) if len(cp) else {},
        "by_setup_side": {f"{k[0]}_{k[1]}": stats(g) for k, g in cp.groupby(["setup", "side"])} if len(cp) else {},
        "by_exit": cp.groupby("reason").R.agg(["size", "mean"]).round(3).to_dict("index") if len(cp) else {},
        "by_year": cp.groupby(pd.to_datetime(cp.closed, unit="ms").dt.year).net.sum().round(2).to_dict() if len(cp) else {},
        "fees": round(ex.totals["fees"], 2), "gst": round(ex.totals["gst"], 2), "funding": round(ex.totals["funding"], 2),
        "slippage": round(ex.totals["slippage"], 2), "liquidations": ex.totals["liquidations"], "skipped": bot.skipped,
        "leverage_used_avg": round(float(np.mean(bot.lev_used)), 1) if bot.lev_used else None,
        "runtime_s": round(time.time() - t0, 1),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("name", "final_equity", "max_drawdown_pct", "all", "first_half", "second_half", "liquidations", "runtime_s")}, default=str))
    return summary


if __name__ == "__main__":
    main()
