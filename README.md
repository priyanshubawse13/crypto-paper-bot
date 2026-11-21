# Crypto paper-trading bot (PriyanshuTrend, 4H, long + short)

**Dashboard: https://priyanshubawse13.github.io/crypto-paper-bot/** (updates every 4 hours)

Runs on GitHub Actions every 4 hours, reads Binance.US public 4H candles (no API key, no account,
no real money) and paper-trades three versions of the strategy, each starting with $100.
Trades both directions, futures-style (max 2x leverage): long breakouts in uptrends, short breakdowns in downtrends.
In sideways markets it stays out on purpose.

| Version | Difference |
|---|---|
| ls_base | main strategy |
| ls_vol13 | breakout candle volume must be > 1.3x its 20-candle average |
| ls_vol15 | breakout candle volume must be > 1.5x its 20-candle average |

Rules (all versions):
- Long: coin and BTC above their 50-day averages, close above the 20-day high. Short: the mirror image.
- Stop 2.5x ATR (2-12%); take profit at 1.5x the stop distance; exit if price closes back across the 50-day average.
- 1% risk per trade, max 3 trades; 12h pause after 2 losing stops in 24h; no new trades after a 2% loss day or below $50.

Not modelled: funding fees (real futures pay/receive these every 8h).

The dashboard has a "Where it's lagging" section: win rate vs break-even, stop-outs that were up +1R first,
take-profits that kept running, fast stop-outs, long vs short, weakest coin.

- Manual run: Actions tab -> paper-trade -> Run workflow.
- Every closed candle is processed once, in order, so late or missed runs catch up automatically.
- Ledgers live in `state/*.json`; delete one to restart that version from $100.
