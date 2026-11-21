"""Download Binance USD(S)-M futures history from data.binance.vision (public archive) into parquet:
  data_fut/{SYM}_1m.parquet       last-price 1m klines  (t ms, o h l c, v base qty, qv quote volume)
  data_fut/{SYM}_mark1m.parquet   mark-price 1m klines  (t, o h l c)
  data_fut/{SYM}_funding.parquet  funding events        (t ms, rate)
  data_fut/filters.json           tick size / lot step measured from real trades (aggTrades)
Monthly archives up to last month, daily archives for the current month. Already-downloaded months are skipped.
    python -m simex.fetch_data --start 2022-01 --symbols SOL,BTC
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/futures/um"
SYMBOLS = "SOL XRP DOGE ADA LINK AVAX SUI LTC BNB NEAR XLM HBAR BTC ETH".split()
OUT = Path("data_fut")


def _get(url: str) -> bytes | None:
    for a in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "simex"}), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(2 * (a + 1))
    raise RuntimeError(f"{url}: {err}")


def _csv(blob: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(blob))
    raw = z.read(z.namelist()[0])
    first = raw[:20].decode(errors="ignore")
    return pd.read_csv(io.BytesIO(raw), header=0 if first[:1].isalpha() else None)


def periods(start: str):
    now = pd.Timestamp.now(tz="UTC")
    months = pd.period_range(start, (now - pd.offsets.MonthBegin(1)).strftime("%Y-%m"), freq="M")
    days = pd.date_range(now.replace(day=1).normalize(), now.normalize() - pd.Timedelta(days=1), freq="D")
    return [str(m) for m in months], [d.strftime("%Y-%m-%d") for d in days]


def fetch_kind(sym: str, kind: str, months, days) -> pd.DataFrame:
    pair = f"{sym}USDT"
    urls = [f"{BASE}/monthly/{kind}/{pair}/1m/{pair}-1m-{m}.zip" for m in months] + \
           [f"{BASE}/daily/{kind}/{pair}/1m/{pair}-1m-{d}.zip" for d in days]
    with ThreadPoolExecutor(8) as ex:
        blobs = list(ex.map(_get, urls))
    frames = []
    for b in blobs:
        if b is None:
            continue
        df = _csv(b).iloc[:, :8]
        df.columns = ["t", "o", "h", "l", "c", "v", "ct", "qv"]
        frames.append(df.drop(columns="ct"))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).astype({"t": "int64"}).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    if kind == "markPriceKlines":
        df = df[["t", "o", "h", "l", "c"]]
    return df


def fetch_funding(sym: str, months) -> pd.DataFrame:
    pair = f"{sym}USDT"
    with ThreadPoolExecutor(8) as ex:
        blobs = list(ex.map(_get, [f"{BASE}/monthly/fundingRate/{pair}/{pair}-fundingRate-{m}.zip" for m in months]))
    frames = []
    for b in blobs:
        if b is not None:
            df = _csv(b)
            frames.append(pd.DataFrame({"t": df.iloc[:, 0].astype("int64"), "rate": df.iloc[:, -1].astype(float)}))
    return pd.concat(frames).drop_duplicates("t").sort_values("t").reset_index(drop=True) if frames else pd.DataFrame()


def measure_filters(sym: str) -> dict:
    """Tick size and lot step = the finest increments seen in a recent day of real trades."""
    pair = f"{sym}USDT"
    for back in range(2, 9):
        d = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
        b = _get(f"{BASE}/daily/aggTrades/{pair}/{pair}-aggTrades-{d}.zip")
        if b is None:
            continue
        z = zipfile.ZipFile(io.BytesIO(b))
        df = pd.read_csv(z.open(z.namelist()[0]), header=None, dtype=str, nrows=300000)
        if not df.iloc[0, 0].isdigit():
            df = df.iloc[1:]
        dec = lambda s: max((-Decimal(x).normalize().as_tuple().exponent for x in s.sample(min(len(s), 20000), random_state=1)), default=0)
        pd_, qd = max(dec(df.iloc[:, 1]), 0), max(dec(df.iloc[:, 2]), 0)
        return {"tick": float(Decimal(1).scaleb(-pd_)), "step": float(Decimal(1).scaleb(-qd)), "measured_on": d}
    return {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01")
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    months, days = periods(a.start)
    filters = json.loads((OUT / "filters.json").read_text()) if (OUT / "filters.json").exists() else {}
    for sym in a.symbols.split(","):
        t0 = time.time()
        for kind, name in (("klines", "1m"), ("markPriceKlines", "mark1m")):
            f = OUT / f"{sym}_{name}.parquet"
            old = pd.read_parquet(f) if f.exists() else pd.DataFrame()
            have = set(pd.to_datetime(old["t"], unit="ms").dt.strftime("%Y-%m")) if len(old) else set()
            need_m = [m for m in months if m not in have or m == months[-1]]
            new = fetch_kind(sym, kind, need_m, days)
            df = pd.concat([old, new]).drop_duplicates("t").sort_values("t").reset_index(drop=True) if len(new) else old
            if len(df):
                df.to_parquet(f, compression="zstd", index=False)
        fu = fetch_funding(sym, months)
        if len(fu):
            fu.to_parquet(OUT / f"{sym}_funding.parquet", index=False)
        if sym not in filters:
            filters[sym] = measure_filters(sym)
        n = len(pd.read_parquet(OUT / f"{sym}_1m.parquet")) if (OUT / f"{sym}_1m.parquet").exists() else 0
        print(f"{sym}: {n} 1m candles, filters {filters.get(sym)}, {time.time() - t0:.0f}s", flush=True)
    (OUT / "filters.json").write_text(json.dumps(filters, indent=1))


if __name__ == "__main__":
    main()
