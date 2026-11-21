"""Market-rotation regime (BTC season -> ETH/large caps -> alt season -> risk-off/gold).

alt index  = cumulative average 4H excess return of a basket of alts vs BTC (rising = money rotating into alts)
ETH/BTC    = large-cap rotation
PAXG/BTC   = gold vs BTC (rising while BTC is in a downtrend = money hiding in gold)
Each is "rising" when above its own `window`-candle average. Only past data is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PHASES = {1: "BTC season", 2: "ETH / large caps", 3: "Alt season", 4: "Risk-off (gold > BTC)", 5: "Crypto down, gold not bid"}


def compute(btc: pd.DataFrame, eth: pd.DataFrame, paxg: pd.DataFrame, basket: dict, window: int = 540) -> pd.DataFrame:
    b = btc.set_index("date")["close"]
    idx = b.index
    lr = lambda s: np.log(s).diff()
    ex = pd.DataFrame({k: lr(v.set_index("date")["close"]).reindex(idx) for k, v in basket.items()}).sub(lr(b), axis=0)
    alt = ex.clip(-0.3, 0.3).mean(axis=1).fillna(0).cumsum()
    ethbtc = np.log(eth.set_index("date")["close"].reindex(idx).ffill() / b)
    gold = np.log(paxg.set_index("date")["close"].reindex(idx).ffill() / b)
    up = lambda s: s > s.rolling(window, min_periods=window // 2).mean()
    btc_up = b > b.rolling(300).mean()
    a, e, g = up(alt), up(ethbtc), up(gold)
    ph = pd.Series(0, index=idx)
    ph[btc_up & ~a & ~e] = 1
    ph[btc_up & ~a & e] = 2
    ph[btc_up & a] = 3
    ph[~btc_up & g] = 4
    ph[~btc_up & ~g] = 5
    ph[b.rolling(300).mean().isna()] = 0
    out = pd.DataFrame({"phase": ph, "alt_idx": alt, "ethbtc": ethbtc, "goldbtc": gold})
    return out.rename_axis("date").reset_index()


def attach(raw: dict, reg: pd.DataFrame) -> dict:
    return {p: df.drop(columns=["phase"], errors="ignore").merge(reg[["date", "phase"]], on="date", how="left") for p, df in raw.items()}
