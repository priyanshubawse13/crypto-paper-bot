# Crypto paper-trading bot — live dashboard

**Paper money only.** Binance.US 4H candles, spot, longs only. Each version started with $100.  
Last run: 2026-11-21 02:30 UTC (Nov 20 08:30 PM Chicago)

## Scoreboard

| Version | Equity | Return | Trades | Win rate | Open now | Status |
|---|---|---|---|---|---|---|
| **base** — Current strategy (what runs on the Mac) | $100.00 | +0.00% | 0 | - | 0 | active |
| **vol13** — + breakout volume > 1.3x average | $100.00 | +0.00% | 0 | - | 0 | active |
| **vol15** — + breakout volume > 1.5x average | $100.00 | +0.00% | 0 | - | 0 | active |

## Open trades

No open trades.

## Last 15 closed trades

None yet.

## Market check (last closed 4H candle)

A buy needs all three: coin above its 50-day average, BTC above its 50-day average, and a close above the 20-day high.

| Coin | Above 50-day avg | BTC above 50-day avg | Distance to 20-day high | Signal |
|---|---|---|---|---|
| SOL | no | no | +42.4% | - |
| XRP | no | no | +47.1% | - |
| DOGE | no | no | +43.8% | - |
| ADA | no | no | +40.5% | - |
| LINK | no | no | +39.6% | - |
| AVAX | no | no | +39.8% | - |
| SUI | no | no | +46.7% | - |
| LTC | no | no | +40.1% | - |

_Rules: entry on 20-day breakout in an uptrend; stop 2.5x ATR; take profit at 1.5x the stop distance; exit if price falls back below the 50-day average; 1% risk per trade, max 3 trades; pauses after 2 losing stops in 24h; no new trades after a 2% loss day or below $50._
