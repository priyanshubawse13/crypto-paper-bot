"""
Live paper-trading run: fetch Binance.US 4H candles, advance each variant's paper ledger through
every newly closed candle, save state, and rewrite DASHBOARD.md.

    python -m paperbot.run_live                       # live (Binance.US public API, no key needed)
    python -m paperbot.run_live --offline DIR --now "2026-10-01 01:00"   # replay from local data (testing)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .engine import CANDLE, Variant, indicators, new_state, run

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
PAIRS = ["SOL", "XRP", "DOGE", "ADA", "LINK", "AVAX", "SUI", "LTC"]
VARIANTS = [
    Variant("base", description="Current strategy (what runs on the Mac)"),
    Variant("vol13", vol_mult=1.3, description="+ breakout volume > 1.3x average"),
    Variant("vol15", vol_mult=1.5, description="+ breakout volume > 1.5x average"),
]
API = "https://api.binance.us/api/v3/klines?symbol={sym}USDT&interval=4h&limit=1000"


def fetch(sym: str) -> pd.DataFrame:
    last_err = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(API.format(sym=sym), headers={"User-Agent": "paperbot/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                rows = json.load(r)
            df = pd.DataFrame(rows).iloc[:, :6]
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
            return df.astype({c: float for c in ["open", "high", "low", "close", "volume"]})
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Binance.US fetch failed for {sym}: {last_err}")


def load_offline(d: str, sym: str, now: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_feather(Path(d) / f"{sym}_USDT-4h.feather")
    return df[df["date"] < now].reset_index(drop=True)


def dashboard(states: dict, raw: dict, btc: pd.DataFrame, now: pd.Timestamp) -> str:
    L = ["# Crypto paper-trading bot — live dashboard", "",
         f"**Paper money only.** Binance.US 4H candles, spot, longs only. Each version started with $100.  ",
         f"Last run: {now:%Y-%m-%d %H:%M} UTC ({now.tz_convert('America/Chicago'):%b %d %I:%M %p} Chicago)", "",
         "## Scoreboard", "", "| Version | Equity | Return | Trades | Win rate | Open now | Status |", "|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        st = states[v.name]
        lp = st.get("last_prices", {})
        unreal = sum(t["qty"] * lp.get(p, t["entry"]) * (1 - 0.001) - t["stake"] for p, t in st["open"].items())
        eq = st["cash"] + sum(t["stake"] for t in st["open"].values()) + unreal
        c = st["closed"]
        wr = f"{sum(x['pnl'] > 0 for x in c) / len(c):.0%}" if c else "-"
        paused = st.get("pause_until") and now < pd.Timestamp(st["pause_until"])
        status = f"paused until {pd.Timestamp(st['pause_until']):%b %d %H:%M} UTC" if paused else "active"
        L.append(f"| **{v.name}** — {v.description} | ${eq:.2f} | {eq - v.start_cap:+.2f}% | {len(c)} | {wr} | {len(st['open'])} | {status} |")
    L += ["", "## Open trades", ""]
    rows = []
    for v in VARIANTS:
        st = states[v.name]
        for p, t in st["open"].items():
            px = st.get("last_prices", {}).get(p, t["entry"])
            rows.append(f"| {v.name} | {p} | {pd.Timestamp(t['opened']):%b %d %H:%M} | {t['entry']:.5g} | {t['stop']:.5g} | {t['target']:.5g} | {px:.5g} | {(px / t['entry'] - 1) * 100:+.2f}% | ${t['stake']:.2f} |")
    L += (["| Version | Coin | Opened (UTC) | Entry | Stop | Target | Now | P/L | Size |", "|---|---|---|---|---|---|---|---|---|"] + rows) if rows else ["No open trades."]
    L += ["", "## Last 15 closed trades", ""]
    allc = sorted(((v.name, c) for v in VARIANTS for c in states[v.name]["closed"]), key=lambda x: x[1]["closed"], reverse=True)[:15]
    L += (["| Version | Coin | Opened | Closed | Entry | Exit | P/L | Reason |", "|---|---|---|---|---|---|---|---|"] +
          [f"| {n} | {c['pair']} | {pd.Timestamp(c['opened']):%b %d %H:%M} | {pd.Timestamp(c['closed']):%b %d %H:%M} | {c['entry']:.5g} | {c['exit']:.5g} | ${c['pnl']:+.2f} ({c['pnl_pct']:+.1f}%) | {c['reason']} |" for n, c in allc]) if allc else ["None yet."]
    # why no trade: market check on the last closed candle
    L += ["", "## Market check (last closed 4H candle)", "",
          "A buy needs all three: coin above its 50-day average, BTC above its 50-day average, and a close above the 20-day high.", "",
          "| Coin | Above 50-day avg | BTC above 50-day avg | Distance to 20-day high | Signal |", "|---|---|---|---|---|"]
    for p in PAIRS:
        f = indicators(raw[p], btc, 0.0)
        f = f[f.index + CANDLE <= now]
        if f.empty:
            continue
        r = f.iloc[-1]
        L.append(f"| {p} | {'yes' if r['close'] > r['sma300'] else 'no'} | {'yes' if r['btc_up'] == 1 else 'no'} | {(r['hh120'] / r['close'] - 1) * 100:+.1f}% | {'BUY' if r['enter'] else '-'} |")
    L += ["", "_Rules: entry on 20-day breakout in an uptrend; stop 2.5x ATR; take profit at 1.5x the stop distance; exit if price falls back below the 50-day average; 1% risk per trade, max 3 trades; pauses after 2 losing stops in 24h; no new trades after a 2% loss day or below $50._"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline")
    ap.add_argument("--now")
    a = ap.parse_args(argv)
    now = pd.Timestamp(a.now, tz="UTC") if a.now else pd.Timestamp.now(tz="UTC")
    get = (lambda s: load_offline(a.offline, s, now)) if a.offline else fetch
    raw = {p: get(p) for p in PAIRS}
    btc = get("BTC")
    STATE.mkdir(exist_ok=True)
    states = {}
    for v in VARIANTS:
        f = STATE / f"{v.name}.json"
        st = json.loads(f.read_text()) if f.exists() else new_state(v)
        st = run(st, v, raw, btc, PAIRS, now=now)
        st["equity_curve"] = st["equity_curve"][-2000:]
        f.write_text(json.dumps(st, indent=1))
        states[v.name] = st
        print(f"{v.name}: last candle {st['last_t']}, cash {st['cash']:.2f}, open {list(st['open'])}, pending {list(st['pending'])}, closed {len(st['closed'])}")
    (ROOT / "DASHBOARD.md").write_text(dashboard(states, raw, btc, now))
    return 0


if __name__ == "__main__":
    sys.exit(main())
