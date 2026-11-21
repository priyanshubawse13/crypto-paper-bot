# Crypto paper-trading bot (PriyanshuTrend, 4H, long + short)

**Dashboard: https://priyanshubawse13.github.io/crypto-paper-bot/** (updates every 4 hours)

Runs on GitHub Actions every 4 hours, reads Binance.US public 4H candles (no API key, no account,
no real money) and paper-trades three versions of the strategy, each starting with $100.
Trades both directions, futures-style (max 2x leverage): long breakouts in uptrends, short breakdowns in downtrends.
In sideways markets it stays out on purpose.

| Version | What it is |
|---|---|
| v3_main | **the strategy**: 12 coins, v2 long/short rules + market-rotation rules |
| v3_no_rotation | same 12 coins without rotation rules (shows what rotation adds) |
| v2_main | previous version: 8 coins, no rotation |

Coins: SOL XRP DOGE ADA LINK AVAX SUI LTC BNB NEAR XLM HBAR (enough Binance.US volume and 2+ years of history).

Longs: coin and BTC above their 50-day averages, close above the 20-day high, target 1.5R.
Shorts: coin and BTC below their 50-day averages, close below the 20-day low, EMA20 < EMA50, RSI(14) 20-50,
price not more than 3 ATR below EMA20, target 1R.
Market rotation (v3): phase from a 43-alt basket vs BTC, ETH/BTC and PAXG(gold)/BTC over 90 days:
BTC season / ETH season / alt season / risk-off (gold beats BTC) / crypto down without gold bid.
No alt longs in alt season (late stage, lost money in backtests); no shorts in gold risk-off (break-even in backtests).
All: 5x isolated leverage (auto-lowered if liquidation would sit inside 1.5x the stop), stop 2.5x ATR (2-12%); exit if price closes back across the 50-day average; 1% risk per trade, max 3 trades;
12h pause after 2 losing stops in 24h; no new trades after a 2% loss day or below $50.

Backtest Jan 2022 - Nov 2026 on Binance.US data: v3_main $100 -> $335, worst drop 12.5%, every year positive;
v3_no_rotation $278 (worst drop 23.5%); v2_main $199. Rejected after testing: per-coin strength vs BTC,
big-candle "news shock" pause, 15m buy-zone filter, sell-the-bounce shorts.

Telegram alerts (optional): add repo secrets TELEGRAM_TOKEN and TELEGRAM_CHAT_ID.

Not modelled: funding fees (real futures pay/receive these every 8h).

The dashboard has a "Where it's lagging" section: win rate vs break-even, stop-outs that were up +1R first,
take-profits that kept running, fast stop-outs, long vs short, weakest coin.

- Manual run: Actions tab -> paper-trade -> Run workflow.
- Every closed candle is processed once, in order, so late or missed runs catch up automatically.
- Ledgers live in `state/*.json`; delete one to restart that version from $100.
