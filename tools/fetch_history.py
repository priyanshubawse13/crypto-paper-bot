"""Research helper (run on GitHub Actions): download Binance.US 4H history for candidate coins into data/."""
import gzip, json, sys, time, urllib.request
from pathlib import Path
BASE = "https://api.binance.us/api/v3"
CANDIDATES = ("BTC ETH SOL XRP DOGE ADA LINK AVAX SUI LTC BNB DOT ATOM NEAR UNI AAVE FIL APT ARB OP INJ HBAR XLM TRX BCH "
              "SHIB PEPE ETC ALGO SEI TIA RENDER FET WIF BONK TAO ENA ONDO POL ICP VET GRT TON CRV LDO STX JUP PENGU TRUMP").split()
START_MS = 1640995200000  # 2022-01-01

def get(url):
    for a in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "paperbot"}), timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(2 * (a + 1))
    raise err

out = Path("data"); out.mkdir(exist_ok=True)
info = get(f"{BASE}/exchangeInfo")
live = {s["baseAsset"] for s in info["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"}
tick = {t["symbol"]: float(t["quoteVolume"]) for t in get(f"{BASE}/ticker/24hr") if t["symbol"].endswith("USDT")}
report = {}
for c in CANDIDATES:
    if c not in live:
        report[c] = "not on Binance.US"; continue
    rows, start = [], START_MS
    while True:
        k = get(f"{BASE}/klines?symbol={c}USDT&interval=4h&limit=1000&startTime={start}")
        if not k: break
        rows += [r[:6] for r in k]
        start = k[-1][0] + 1
        if len(k) < 1000: break
        time.sleep(0.2)
    with gzip.open(out / f"{c}.json.gz", "wt") as f:
        json.dump(rows, f)
    report[c] = {"candles": len(rows), "first": time.strftime("%Y-%m-%d", time.gmtime(rows[0][0] / 1000)) if rows else None,
                 "usd_vol_24h": round(tick.get(c + "USDT", 0))}
    print(c, report[c], flush=True)
json.dump(report, open(out / "report.json", "w"), indent=1)
