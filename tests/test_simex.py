import numpy as np, pandas as pd, pytest
from simex.exchange import SimBinanceUSDM, SymbolSpec, Fees, make_bars, MIN

T0 = 1_704_067_200_000  # 2024-01-01


def bars(prices, highs=None, lows=None, qv=1e9, funding=None, flat=False):
    n = len(prices)
    o = np.array(prices, float)
    c = o.copy() if flat else np.r_[o[1:], o[-1]]
    h = np.maximum(o, c) if highs is None else np.array(highs, float)
    l = np.minimum(o, c) if lows is None else np.array(lows, float)
    k = pd.DataFrame({"t": T0 + np.arange(n) * MIN, "o": o, "h": h, "l": l, "c": c, "qv": qv})
    f = pd.DataFrame(funding, columns=["t", "rate"]) if funding else None
    return make_bars(k, None, f)


def ex(b, **kw):
    spec = SymbolSpec(tick=0.01, step=0.1, min_notional=5, mmr=0.01, half_spread_bps=0.0)
    return SimBinanceUSDM({"SOL": b}, {"SOL": spec}, T0, balance=100, fees=Fees(0.0002, 0.0005, 0.18), impact_bps=0, **kw)


def test_market_fill_fees_and_gst():
    e = ex(bars([100] * 10))
    e.set_leverage(2, "SOL/USDT:USDT")
    o = e.create_order("SOL/USDT:USDT", "market", "buy", 0.5)
    assert o["status"] == "closed" and o["average"] == 100
    fee = 50 * 0.0005
    assert e.wallet == pytest.approx(100 - fee * 1.18)
    p = e.fetch_positions()[0]
    assert p["initialMargin"] == pytest.approx(25) and p["side"] == "long"


def test_filters_reject():
    e = ex(bars([100] * 10))
    assert e.create_order("SOL/USDT:USDT", "market", "buy", 0.04)["status"] == "rejected"       # < step
    assert e.create_order("SOL/USDT:USDT", "market", "buy", 0.15)["status"] == "rejected"       # not multiple
    e2 = ex(bars([10] * 10))
    assert e2.create_order("SOL/USDT:USDT", "market", "buy", 0.4)["status"] == "rejected"      # $4 < $5 min
    assert e.create_order("SOL/USDT:USDT", "stop_market", "sell", 0.5, params={"stopPrice": 95.005, "reduceOnly": True})["status"] == "rejected"


def test_stop_triggers_and_gap():
    px = [100, 100, 99, 97, 96, 90, 90]
    e = ex(bars(px))
    e.set_leverage(2, "SOL/USDT:USDT")
    e.create_order("SOL/USDT:USDT", "market", "buy", 0.5)
    e.create_order("SOL/USDT:USDT", "stop_market", "sell", 0, params={"stopPrice": 96.5, "closePosition": True})
    e.advance(T0 + 7 * MIN)
    assert not e.positions
    fill = e.trades[-1]
    assert fill["price"] == pytest.approx(96.5) and fill["type"] == "stop_market"
    # gap: bar opens below the stop -> filled at the open
    e = ex(bars([100, 100, 95, 95], flat=True))
    e.set_leverage(2, "SOL/USDT:USDT")
    e.create_order("SOL/USDT:USDT", "market", "buy", 0.5)
    e.advance(T0 + MIN)
    e.create_order("SOL/USDT:USDT", "stop_market", "sell", 0, params={"stopPrice": 98, "closePosition": True})
    e.advance(T0 + 4 * MIN)
    assert e.trades[-1]["price"] == pytest.approx(95)


def test_liquidation_isolated():
    px = [100] * 5 + [95, 91, 90, 89, 88]
    e = ex(bars(px))
    e.set_leverage(10, "SOL/USDT:USDT")
    e.create_order("SOL/USDT:USDT", "market", "buy", 1.0)        # $100 notional, $10 margin
    lp = e.fetch_positions()[0]["liquidationPrice"]
    assert lp == pytest.approx((10 - 100) / (0.01 - 1))          # ~90.9
    w0 = e.wallet
    e.advance(T0 + 10 * MIN)
    assert not e.positions and e.totals["liquidations"] == 1
    assert e.wallet == pytest.approx(w0 - 10)


def test_funding_paid_by_long():
    ft = T0 + 3 * MIN
    e = ex(bars([100] * 10, funding=[(ft, 0.001)]))
    e.set_leverage(2, "SOL/USDT:USDT")
    e.create_order("SOL/USDT:USDT", "market", "buy", 0.5)
    w = e.wallet
    e.advance(T0 + 9 * MIN)
    assert e.wallet == pytest.approx(w - 50 * 0.001)
    assert e.totals["funding"] == pytest.approx(-0.05)


def test_short_take_profit_and_pnl():
    e = ex(bars([100, 100, 98, 96, 94, 94]))
    e.set_leverage(2, "SOL/USDT:USDT")
    e.create_order("SOL/USDT:USDT", "market", "sell", 0.5)
    e.create_order("SOL/USDT:USDT", "take_profit_market", "buy", 0, params={"stopPrice": 95, "closePosition": True})
    e.advance(T0 + 6 * MIN)
    assert not e.positions
    assert e.trades[-1]["realized"] == pytest.approx(0.5 * 5)


def test_limit_maker_needs_trade_through():
    e = ex(bars([100, 100, 99, 99, 98.5, 98.5], lows=[100, 99.5, 99, 98.99, 98.5, 98.5]))
    e.set_leverage(2, "SOL/USDT:USDT")
    e.create_order("SOL/USDT:USDT", "limit", "buy", 0.5, price=99.0)
    e.advance(T0 + 6 * MIN)
    f = e.trades[-1]
    assert f["maker"] and f["price"] == 99.0 and f["t"] == T0 + 3 * MIN   # low 99.0 at minute 2 is not enough


def test_fast_forward_and_ohlcv_no_future():
    n = 60 * 24 * 30
    e = ex(bars(list(100 + np.sin(np.arange(n) / 500))))
    e.advance(T0 + n * MIN - 1)
    assert e.now == T0 + n * MIN - 1
    e2 = ex(bars(list(np.arange(1, 500) * 1.0 + 100)))
    e2.advance(T0 + 300 * MIN)
    rows = e2.fetch_ohlcv("SOL/USDT:USDT", "4h")
    assert rows[-1][4] == 401 and max(r[2] for r in rows) <= 401      # last close = minute 299's close; nothing later


def test_closed_position_record_matches_wallet():
    ft = T0 + 3 * MIN
    e = ex(bars([100, 100, 100, 100, 105, 106, 106], funding=[(ft, 0.0005)]))
    e.set_leverage(2, "SOL/USDT:USDT")
    e.create_order("SOL/USDT:USDT", "market", "buy", 0.5)
    e.advance(T0 + 5 * MIN)
    e.create_order("SOL/USDT:USDT", "market", "sell", 0.5, params={"reduceOnly": True})
    r = e.closed_positions[-1]
    assert r["net"] == pytest.approx(e.wallet - 100)


def test_intraday_ohlcv():
    e = ex(bars(list(np.arange(300) * 1.0 + 100)))
    e.advance(T0 + 75 * MIN)
    rows = e.fetch_ohlcv("SOL/USDT:USDT", "15m")
    assert len(rows) == 5 and rows[-1][0] == T0 + 60 * MIN       # 5 closed, the next one has no minutes yet
    assert rows[-1][4] == 175                                     # close = close of minute 74
    e.advance(T0 + 80 * MIN)
    rows = e.fetch_ohlcv("SOL/USDT:USDT", "15m")
    assert len(rows) == 6 and rows[-1][4] == 180                  # forming candle: only minutes 75-79
    h1 = e.fetch_ohlcv("SOL/USDT:USDT", "1h")
    assert h1[0][2] == 160 and h1[0][4] == 160                    # first hour: high/close = minute 59 close


def test_stop_before_liquidation_in_same_minute():
    # one crash minute goes through the stop (110) and the liquidation level (~108.6) -> the stop must win
    e = ex(bars([113, 113, 112.4, 107, 107], lows=[113, 113, 104.6, 107, 107], flat=True))
    e.set_leverage(20, "SOL/USDT:USDT")
    e.create_order("SOL/USDT:USDT", "market", "buy", 0.3)
    e.create_order("SOL/USDT:USDT", "stop_market", "sell", 0, params={"stopPrice": 110.0, "closePosition": True})
    e.advance(T0 + 5 * MIN)
    assert e.totals["liquidations"] == 0 and e.trades[-1]["type"] == "stop_market"
