# Crypto paper-trading bot (PriyanshuTrend, 4H, long + short)

**Dashboard: https://priyanshubawse13.github.io/crypto-paper-bot/** (updates every 4 hours)

Runs on GitHub Actions every 4 hours, reads Binance.US public 4H candles (no API key, no account,
no real money) and paper-trades three versions of the strategy, each starting with $100.
Trades both directions, futures-style (max 2x leverage): long breakouts in uptrends, short breakdowns in downtrends.
In sideways markets it stays out on purpose.

| Version | What it is |
|---|---|
| v2_main | **the strategy**: longs unchanged; shorts with stricter rules (below) |
| v1_old | old version, shorts are the exact mirror of longs (comparison) |
| longs_only | no shorts, sits in cash in downtrends (comparison) |

Longs: coin and BTC above their 50-day averages, close above the 20-day high, target 1.5R.
Shorts (v2_main): coin and BTC below their 50-day averages, close below the 20-day low, EMA20 < EMA50,
RSI(14) between 20 and 50, price not more than 3 ATR below EMA20 (no shorting into an oversold squeeze), target 1R.
All: stop 2.5x ATR (2-12%); exit if price closes back across the 50-day average; 1% risk per trade, max 3 trades;
12h pause after 2 losing stops in 24h; no new trades after a 2% loss day or below $50.

Backtest Feb 2023 - Oct 2026 (same engine): v2_main +94%, worst drop 14%, shorts profitable every year;
v1_old +63%; longs_only +50%. Shorts research: research notes in the chat; 15m buy-zone filter and sell-the-bounce entries were tested and rejected.

Not modelled: funding fees (real futures pay/receive these every 8h).

The dashboard has a "Where it's lagging" section: win rate vs break-even, stop-outs that were up +1R first,
take-profits that kept running, fast stop-outs, long vs short, weakest coin.

- Manual run: Actions tab -> paper-trade -> Run workflow.
- Every closed candle is processed once, in order, so late or missed runs catch up automatically.
- Ledgers live in `state/*.json`; delete one to restart that version from $100.
