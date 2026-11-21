"""
Live paper-trading run: fetch Binance.US 4H candles, advance each variant's paper ledger through
every newly closed candle, save state, and write site/index.html (web dashboard) and DASHBOARD.md (short summary).

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

from . import dashboard, notify, regime
from .engine import CANDLE, Variant, mark_to_market, new_state, run

ROOT = Path(__file__).resolve().parent.parent
PAIRS8 = ["SOL", "XRP", "DOGE", "ADA", "LINK", "AVAX", "SUI", "LTC"]
PAIRS = PAIRS8 + ["BNB", "NEAR", "XLM", "HBAR"]          # 12 coins with real Binance.US volume and 2+ years of history
# basket of alts used only to measure the BTC <-> alt rotation (not traded)
BASKET = ("SOL XRP DOGE ADA LINK AVAX SUI LTC BNB DOT ATOM NEAR UNI AAVE FIL APT ARB OP HBAR XLM TRX BCH SHIB PEPE ETC "
          "ALGO SEI TIA RENDER FET WIF BONK ENA ONDO POL ICP VET GRT CRV LDO JUP PENGU TRUMP").split()
REGIME_WINDOW = 540                                       # 90 days of 4H candles
MAIN_SHORT = dict(s_ema_rsi=True, s_rsi_lo=20, s_rsi_hi=50, s_squeeze=3.0, s_r_mult=1.0, leverage=5.0)
ROTATION = dict(rot_no_alt_season_longs=True, rot_no_gold_shorts=True)
VARIANTS = [
    Variant("v3_main", **MAIN_SHORT, **ROTATION,
            description="NEW: 12 coins + market rotation (no alt longs late in alt season, no shorts while money hides in gold)"),
    Variant("v3_no_rotation", **MAIN_SHORT, description="Same 12 coins without the rotation rules (shows what rotation adds)"),
    Variant("v2_main", **MAIN_SHORT, description="Previous version: 8 coins, no rotation (comparison)"),
]
VARIANT_PAIRS = {"v3_main": PAIRS, "v3_no_rotation": PAIRS, "v2_main": PAIRS8}
SITE_URL = "https://priyanshubawse13.github.io/crypto-paper-bot/"
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
    gz = Path(d) / f"{sym}.json.gz"
    if gz.exists():
        import gzip
        df = pd.DataFrame(json.load(gzip.open(gz, "rt")), columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        df = df.astype({c: float for c in ["open", "high", "low", "close", "volume"]}).drop_duplicates("date")
    else:
        df = pd.read_feather(Path(d) / f"{sym}_USDT-4h.feather")
    return df[df["date"] < now].reset_index(drop=True)


def summary_md(states: dict, now: pd.Timestamp) -> str:
    L = ["# Crypto paper-trading bot", "",
         f"**Full dashboard: {SITE_URL}**", "",
         f"Last run: {now.tz_convert('America/Chicago'):%b %d %I:%M %p} Chicago. Paper money only, longs + shorts, $100 start each.", "",
         "| Version | Equity | Trades | Open |", "|---|---|---|---|"]
    for v in VARIANTS:
        st = states[v.name]
        eq = mark_to_market(st, st.get("last_prices", {}))
        L.append(f"| {v.name} | ${eq:.2f} ({(eq / v.start_cap - 1) * 100:+.2f}%) | {len(st['closed'])} | {len(st['open'])} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline")
    ap.add_argument("--now")
    ap.add_argument("--state-dir", default=str(ROOT / "state"))
    ap.add_argument("--site-dir", default=str(ROOT / "site"))
    ap.add_argument("--backfill", action="store_true", help="testing: replay all history on a fresh state")
    a = ap.parse_args(argv)
    now = pd.Timestamp(a.now, tz="UTC") if a.now else pd.Timestamp.now(tz="UTC")
    get = (lambda s: load_offline(a.offline, s, now)) if a.offline else fetch
    raw = {p: get(p) for p in PAIRS}
    btc = get("BTC")
    basket = {}
    for c in BASKET:
        try:
            basket[c] = raw[c] if c in raw else get(c)
        except Exception as e:  # noqa: BLE001  (a delisted/missing basket coin must not stop the run)
            print(f"basket: skip {c}: {e}")
    reg = regime.compute(btc, get("ETH"), get("PAXG"), basket, REGIME_WINDOW)
    reg = reg[reg["date"] + CANDLE <= now]
    raw = regime.attach(raw, reg)
    state_dir, site_dir = Path(a.state_dir), Path(a.site_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    site_dir.mkdir(parents=True, exist_ok=True)
    states = {}
    for v in VARIANTS:
        f = state_dir / f"{v.name}.json"
        st = json.loads(f.read_text()) if f.exists() else new_state(v)
        before = (dict(st["open"]), len(st["closed"]))
        st = run(st, v, raw, btc, VARIANT_PAIRS[v.name], now=now, backfill=a.backfill)
        if v is VARIANTS[0] and not a.backfill:
            notify.trade_changes(v.name, before, st)
        st["equity_curve"] = st["equity_curve"][-2000:]
        f.write_text(json.dumps(st, indent=1))
        states[v.name] = st
        print(f"{v.name}: last candle {st['last_t']}, cash {st['cash']:.2f}, open {list(st['open'])}, pending {list(st['pending'])}, closed {len(st['closed'])}")
    (site_dir / "index.html").write_text(dashboard.build(states, VARIANTS, raw, btc, PAIRS, now, reg))
    if not a.backfill:
        (ROOT / "DASHBOARD.md").write_text(summary_md(states, now))
    return 0


if __name__ == "__main__":
    sys.exit(main())
