"""
Replay Binance futures history through the simulated exchange with the v3 bot.

    python -m simex.run_sim --data data_fut --start 2022-01-01 --out site/replay
    python -m simex.run_sim --data data_fut --regime research_data --variant v3_main

Writes <out>/replay.json (candles, trades, equity, events for the viewer), <out>/index.html (viewer) and prints a summary.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from paperbot.engine import Variant, indicators
from paperbot import regime as regime_mod
from .bot import TrendBot
from .exchange import Fees, SimBinanceUSDM, SymbolSpec, make_bars

H4 = 4 * 3600_000
PAIRS12 = "SOL XRP DOGE ADA LINK AVAX SUI LTC BNB NEAR XLM HBAR".split()
BASKET = ("SOL XRP DOGE ADA LINK AVAX SUI LTC BNB DOT ATOM NEAR UNI AAVE FIL APT ARB OP HBAR XLM TRX BCH SHIB PEPE ETC "
          "ALGO SEI TIA RENDER FET WIF BONK ENA ONDO POL ICP VET GRT CRV LDO JUP PENGU TRUMP").split()
MAIN_SHORT = dict(s_ema_rsi=True, s_rsi_lo=20, s_rsi_hi=50, s_squeeze=3.0, s_r_mult=1.0, leverage=5.0)
VARIANTS = {
    "v3_main": Variant("v3_main", **MAIN_SHORT, rot_no_alt_season_longs=True, rot_no_gold_shorts=True),
    "v3_no_rotation": Variant("v3_no_rotation", **MAIN_SHORT),
}
SPREAD_BPS = {"BTC": 0.5, "ETH": 0.5, "SOL": 1.0, "XRP": 1.0, "DOGE": 1.0, "BNB": 1.0}   # others 2.0
MMR = {"BTC": 0.004, "ETH": 0.005}                                                       # others 1% (conservative)


def load_us(d: Path, sym: str) -> pd.DataFrame | None:
    f = d / f"{sym}.json.gz"
    if not f.exists():
        return None
    df = pd.DataFrame(json.load(gzip.open(f, "rt")), columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    return df.astype({c: float for c in ["open", "high", "low", "close", "volume"]}).drop_duplicates("date")


def h4_frame(ex: SimBinanceUSDM, s: str) -> pd.DataFrame:
    df = ex._h4[s].rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_fut")
    ap.add_argument("--regime", default="research_data", help="dir with Binance.US 4H basket for the rotation regime")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--variant", default="v3_main")
    ap.add_argument("--pairs", default=",".join(PAIRS12))
    ap.add_argument("--balance", type=float, default=100.0)
    ap.add_argument("--out", default="site/replay")
    ap.add_argument("--leverage", type=float, default=None, help="override leverage (default 2x)")
    ap.add_argument("--risk", type=float, default=None, help="override risk per trade (default 0.01 = 1%%)")
    ap.add_argument("--max-margin-frac", type=float, default=None)
    ap.add_argument("--no-lev-guard", action="store_true", help="use the full leverage even if liquidation comes before the stop")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    D = Path(a.data)
    filters = json.loads((D / "filters.json").read_text())
    pairs = [p for p in a.pairs.split(",") if (D / f"{p}_1m.parquet").exists()]
    syms = sorted(set(pairs) | {"BTC"})
    rd = lambda f: pd.read_parquet(f) if f.exists() else None
    bars, specs = {}, {}
    for s in syms:
        bars[s] = make_bars(pd.read_parquet(D / f"{s}_1m.parquet"), rd(D / f"{s}_mark1m.parquet"), rd(D / f"{s}_funding.parquet"))
        f = filters.get(s, {})
        specs[s] = SymbolSpec(tick=f.get("tick", 1e-4), step=f.get("step", 1e-3), mmr=MMR.get(s, 0.01), max_leverage=125 if s == "BTC" else 75,
                              half_spread_bps=SPREAD_BPS.get(s, 2.0))
    start = int(pd.Timestamp(a.start, tz="UTC").value // 10**6)
    start = (start + H4 - 1) // H4 * H4
    end = int(pd.Timestamp(a.end, tz="UTC").value // 10**6) if a.end else int(max(b.t[-1] for b in bars.values())) + 60_000
    end = end // H4 * H4
    ex = SimBinanceUSDM(bars, specs, start, balance=a.balance, fees=Fees())
    print(f"data loaded in {time.time() - t0:.0f}s: {', '.join(f'{s}:{len(bars[s].t)}' for s in syms)}", flush=True)

    import dataclasses
    v = VARIANTS[a.variant]
    over = {k: x for k, x in (("leverage", a.leverage), ("risk", a.risk), ("max_margin_frac", a.max_margin_frac)) if x is not None}
    if over:
        v = dataclasses.replace(v, **over)
    btc4 = h4_frame(ex, "BTC")
    raw = {p: h4_frame(ex, p) for p in pairs}
    R = Path(a.regime)
    phase_note = "off"
    if (v.rot_no_alt_season_longs or v.rot_no_gold_shorts) and (R / "ETH.json.gz").exists():
        basket = {c: d for c in BASKET if (d := load_us(R, c)) is not None}
        reg = regime_mod.compute(load_us(R, "BTC"), load_us(R, "ETH"), load_us(R, "PAXG"), basket, 540)
        raw = regime_mod.attach(raw, reg)
        phase_note = "on (Binance.US basket)"
    ind = {p: indicators(raw[p], btc4, v) for p in pairs}
    bot = TrendBot(ex, pairs, v, ind, a.balance, lev_guard=not a.no_lev_guard)

    eq_curve, t_last_print = [], time.time()
    for t in range(start, end + 1, H4):
        ex.advance(t)
        bot.on_close()
        eq_curve.append([t, round(ex.wallet, 4), round(ex.equity(), 4)])
        if time.time() - t_last_print > 20:
            print(f"  {pd.Timestamp(t, unit='ms'):%Y-%m-%d}  equity {ex.equity():.2f}  trades {len(ex.closed_positions)}", flush=True)
            t_last_print = time.time()
    elapsed = time.time() - t0

    cp = pd.DataFrame(ex.closed_positions)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "variant": v.name, "start": a.start, "end": pd.Timestamp(end, unit="ms", tz="UTC").isoformat(), "rotation": phase_note,
        "start_balance": a.balance, "final_wallet": round(ex.wallet, 2), "final_equity": round(ex.equity(), 2),
        "trades": len(cp), "win_rate": round(float((cp.net > 0).mean()), 3) if len(cp) else None,
        "profit_factor": round(float(cp[cp.net > 0].net.sum() / -cp[cp.net <= 0].net.sum()), 2) if len(cp) and (cp.net <= 0).any() else None,
        "max_drawdown_pct": round(float((1 - pd.Series([e[2] for e in eq_curve]) / pd.Series([e[2] for e in eq_curve]).cummax()).max() * 100), 1),
        "fees": round(ex.totals["fees"], 2), "gst_on_fees": round(ex.totals["gst"], 2), "funding_net": round(ex.totals["funding"], 2),
        "slippage_cost": round(ex.totals["slippage"], 2), "liquidations": ex.totals["liquidations"],
        "rejected_orders": sum(1 for e in ex.events if e["kind"] == "rejected"), "skipped_min_size": bot.skipped["min_notional"],
        "by_reason": cp.groupby("reason").net.agg(["size", "sum"]).round(2).to_dict("index") if len(cp) else {},
        "by_side": cp.groupby("side").net.agg(["size", "sum"]).round(2).to_dict("index") if len(cp) else {},
        "by_year": cp.groupby(pd.to_datetime(cp.closed, unit="ms").dt.year).net.sum().round(2).to_dict() if len(cp) else {},
        "leverage_setting": v.leverage, "risk_per_trade": v.risk, "lev_guard": not a.no_lev_guard,
        "leverage_used_avg": round(float(np.mean(bot.lev_used)), 2) if bot.lev_used else None,
        "runtime_s": round(elapsed, 1),
    }
    print(json.dumps(summary, indent=1, default=str))
    if a.summary_only:
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        return summary
    # viewer payload: 4H candles per pair (compact), trades, equity, events
    candles, gaps = {}, {}
    for s_ in syms:
        t = bars[s_].t
        d = np.diff(t)
        big = np.where(d > 3600_000)[0]
        if len(big):
            gaps[s_] = [[pd.Timestamp(int(t[i]), unit="ms").isoformat(), round(float(d[i]) / 3600_000, 1)] for i in big[:20]]
    summary["data_gaps_over_1h"] = gaps
    if gaps:
        print("data gaps (start, hours):", json.dumps(gaps))
    for p in pairs:
        h = ex._h4[p].set_index("t")
        grid = np.arange(max(int(h.index[0]), start - 60 * H4), end + 1, H4)
        h = h.reindex(grid).rename_axis("t").reset_index()             # contiguous 4H grid; gaps become null
        nn = lambda x: [None if v != v else v for v in x.round(6).tolist()]
        candles[p] = {"t0": int(h.t.iloc[0]), "o": nn(h.o), "h": nn(h.h), "l": nn(h.l), "c": nn(h.c),
                      "sma": ind[p]["sma300"].reindex(pd.to_datetime(h.t, unit="ms", utc=True)).round(6).fillna(0).tolist()}
    payload = {"summary": summary, "candles": candles, "equity": eq_curve[::1],
               "trades": [{k: (round(x, 6) if isinstance(x, float) else x) for k, x in r.items()} for r in ex.closed_positions],
               "events": [e for e in ex.events if e["kind"] in ("liquidation", "rejected")][:2000] + bot.log[:2000],
               "funding": [e for e in ex.events if e["kind"] == "funding"][-5000:]}
    (out / "replay.json").write_text(json.dumps(payload, separators=(",", ":"), default=float))
    from .viewer import write_viewer
    write_viewer(out)
    return summary


if __name__ == "__main__":
    main()
