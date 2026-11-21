# Crypto paper-trading bot (PriyanshuTrend, 4H)

Runs every 4 hours on GitHub Actions, reads Binance.US public 4H candles (no API key, no account,
no real money) and paper-trades three versions of the strategy, each starting with $100.

**See [DASHBOARD.md](DASHBOARD.md)** for the scoreboard, open trades, closed trades and why it is or isn't buying.

| Version | Difference |
|---|---|
| base | current strategy |
| vol13 | breakout candle volume must be > 1.3x its 20-candle average |
| vol15 | breakout candle volume must be > 1.5x its 20-candle average |

Rules (all versions): buy a 20-day breakout when the coin and BTC are above their 50-day averages;
stop 2.5x ATR; take profit at 1.5x the stop distance; exit if price closes back below the 50-day average;
1% risk per trade, max 3 trades; 12h pause after 2 losing stops in 24h; no new trades after a 2% loss
day or below $50.

- Manual run: Actions tab -> paper-trade -> Run workflow.
- Every closed candle is processed once, in order, so late or missed runs catch up automatically.
- Ledgers live in `state/*.json`; delete one to restart that version from $100.
