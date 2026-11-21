"""Builds site/index.html: a self-contained results dashboard (no external scripts or fonts)."""
from __future__ import annotations

import html
import math

import pandas as pd

from .engine import CANDLE, indicators, mark_to_market

COLORS = {"v3_main": "#2563eb", "v3_no_rotation": "#d97706", "v2_main": "#7c3aed"}
BREAKEVEN_WR = 1 / (1 + 1.5)  # 1.5R long target -> 40% win rate breaks even (shorts use 1R -> 50%)
BACKTEST = {"win_rate": 0.40, "avg_R": 0.10, "per_year": 120}


def esc(x) -> str:
    return html.escape(str(x))


def chi(ts) -> str:
    return pd.Timestamp(ts).tz_convert("America/Chicago").strftime("%b %d, %I:%M %p")


def fmt_money(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):,.2f}"


def stats(st: dict, start_cap: float) -> dict:
    c = pd.DataFrame(st["closed"])
    eq_mtm = mark_to_market(st, st.get("last_prices", {}))
    curve = [v for _, v in st["equity_curve"]] or [start_cap]
    peak, mdd = start_cap, 0.0
    for v in curve + [eq_mtm]:
        peak = max(peak, v)
        mdd = max(mdd, 1 - v / peak)
    s = {"equity": eq_mtm, "ret": (eq_mtm / start_cap - 1) * 100, "n": len(c), "mdd": mdd * 100,
         "open": len(st["open"])}
    if len(c):
        wins, losses = c[c.pnl > 0], c[c.pnl <= 0]
        s.update(win_rate=len(wins) / len(c), pf=(wins.pnl.sum() / -losses.pnl.sum()) if len(losses) and losses.pnl.sum() < 0 else float("inf"),
                 avg_R=c.R.mean(), avg_win_R=wins.R.mean() if len(wins) else 0, avg_loss_R=losses.R.mean() if len(losses) else 0,
                 net=c.pnl.sum())
    return s


def diagnostics(st: dict) -> list[tuple[str, str]]:
    """Plain-English findings: (level, text). level in good/warn/bad/info."""
    c = pd.DataFrame(st["closed"])
    out = []
    n = len(c)
    if n == 0:
        return [("info", "No closed trades yet. Diagnostics start after the first trades close.")]
    early = n < 20
    if early:
        out.append(("info", f"Only {n} closed trade{'s' if n != 1 else ''} so far. Below ~20 trades these numbers are mostly noise; treat them as early signals, not conclusions."))
    wr = (c.pnl > 0).mean()
    w, l = c[c.pnl > 0].R.mean(), c[c.pnl <= 0].R.mean()
    be = (-l / (w - l)) if (len(c[c.pnl > 0]) and len(c[c.pnl <= 0]) and w - l > 0) else BREAKEVEN_WR
    lvl = "good" if wr >= be + 0.05 else ("warn" if wr >= be - 0.03 else "bad")
    out.append((lvl, f"Win rate {wr:.0%} vs {be:.0%} needed to break even at the current average win ({w if w == w else 0:+.2f}R) "
                     f"and loss ({l if l == l else 0:+.2f}R). Backtest: about 40%."))
    losers = c[c.reason == "stop_loss"]
    if len(losers):
        gave_back = losers.hit_1R.mean()
        if gave_back >= 0.25:
            out.append(("warn", f"{gave_back:.0%} of stopped-out trades were up at least +1R before reversing to the stop. "
                                "A break-even stop after +1R would have saved them; worth backtesting."))
        else:
            out.append(("good", f"Only {gave_back:.0%} of stopped-out trades had been up +1R first, so few winners are turning into losers."))
        fast = (losers.bars <= 2).mean()
        if fast >= 0.4:
            out.append(("warn", f"{fast:.0%} of stop-outs happened within 2 candles (8h) of entry. Entries may be chasing breakouts; "
                                "a wider stop or waiting for a retest could help."))
    tp = c[c.reason == "take_profit"]
    if len(tp) >= 3:
        run_on = (tp.mfe_R >= 2.5).mean()
        if run_on >= 0.3:
            out.append(("warn", f"{run_on:.0%} of take-profit trades kept running past +2.5R. The take-profit target may be leaving money on the table."))
    slip = losers.R.mean() if len(losers) else None
    if slip is not None and slip < -1.15:
        out.append(("warn", f"Average stop-out costs {slip:.2f}R instead of about -1R: fees and gaps through the stop are eating into results."))
    for side in ("long", "short"):
        cs = c[c.side == side]
        if len(cs) >= 5:
            out.append(("good" if cs.pnl.sum() > 0 else "bad",
                        f"{side.title()}s: {len(cs)} trades, net {fmt_money(cs.pnl.sum())}, win rate {(cs.pnl > 0).mean():.0%}."))
    by_coin = c.groupby("pair").pnl.sum().sort_values()
    if len(by_coin) >= 3 and by_coin.iloc[0] < 0:
        out.append(("info", f"Weakest coin so far: {by_coin.index[0]} ({fmt_money(by_coin.iloc[0])}). Best: {by_coin.index[-1]} ({fmt_money(by_coin.iloc[-1])})."))
    te = c[c.reason == "trend_exit"]
    if len(te) >= 3 and te.pnl.mean() < 0:
        out.append(("info", f"Trend exits ({len(te)}) average {fmt_money(te.pnl.mean())}: trades are dying slowly before hitting stop or target."))
    return out


def svg_equity(states: dict, variants, w=860, h=260) -> str:
    series = {}
    for v in variants:
        pts = [(pd.Timestamp(t), val) for t, val in states[v.name]["equity_curve"]]
        if pts:
            series[v.name] = pts
    if not series or max(len(p) for p in series.values()) < 2:
        return '<p class="muted">The equity chart appears after a few runs.</p>'
    allp = [p for s in series.values() for p in s]
    t0, t1 = min(p[0] for p in allp), max(p[0] for p in allp)
    lo, hi = min(min(p[1] for p in allp), 100) * 0.995, max(max(p[1] for p in allp), 100) * 1.005
    span = (t1 - t0).total_seconds() or 1
    pad_l, pad_b, pad_t, pad_r = 46, 26, 10, 10

    def xy(t, val):
        x = pad_l + (t - t0).total_seconds() / span * (w - pad_l - pad_r)
        y = pad_t + (1 - (val - lo) / (hi - lo)) * (h - pad_t - pad_b)
        return f"{x:.1f},{y:.1f}"

    g = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Equity over time">']
    for k in range(5):
        val = lo + (hi - lo) * k / 4
        y = pad_t + (1 - k / 4) * (h - pad_t - pad_b)
        g.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" class="grid"/>'
                 f'<text x="{pad_l - 6}" y="{y + 4:.1f}" class="axis" text-anchor="end">${val:.0f}</text>')
    y100 = pad_t + (1 - (100 - lo) / (hi - lo)) * (h - pad_t - pad_b)
    g.append(f'<line x1="{pad_l}" y1="{y100:.1f}" x2="{w - pad_r}" y2="{y100:.1f}" class="base"/>')
    for i, frac in enumerate((0, 0.5, 1)):
        t = t0 + (t1 - t0) * frac
        x = pad_l + frac * (w - pad_l - pad_r)
        anchor = ["start", "middle", "end"][i]
        g.append(f'<text x="{x:.1f}" y="{h - 6}" class="axis" text-anchor="{anchor}">{t.tz_convert("America/Chicago"):%b %d}</text>')
    for name, pts in series.items():
        d = " ".join(xy(t, val) for t, val in pts)
        g.append(f'<polyline points="{d}" fill="none" stroke="{COLORS.get(name, "#333")}" stroke-width="2"/>')
    g.append("</svg>")
    legend = " ".join(f'<span class="key"><i style="background:{COLORS.get(v.name)}"></i>{esc(v.name)}</span>' for v in variants)
    return f'<div class="legend">{legend}</div>' + "".join(g)


def table(head, rows, empty="Nothing yet.") -> str:
    if not rows:
        return f'<p class="muted">{empty}</p>'
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def pl(x: float) -> str:
    cls = "pos" if x > 0 else ("neg" if x < 0 else "")
    return f'<span class="{cls}">{"+" if x > 0 else ""}{fmt_money(x) if not isinstance(x, str) else x}</span>'


def build(states: dict, variants, raw: dict, btc: pd.DataFrame, pairs: list, now: pd.Timestamp, reg=None) -> str:
    S = {v.name: stats(states[v.name], v.start_cap) for v in variants}
    # market regime
    mk_rows, ups, downs = [], 0, 0
    btc_up = None
    for p in pairs:
        f = indicators(raw[p], btc, variants[0])
        f = f[f.index + CANDLE <= now]
        if f.empty:
            continue
        r = f.iloc[-1]
        btc_up = bool(r["btc_up"])
        above = r["close"] > r["sma300"]
        ups += above
        downs += not above
        to_hi = (r["hh120"] / r["close"] - 1) * 100
        to_lo = (1 - r["ll120"] / r["close"]) * 100
        sig = "BUY" if r["enter_long"] else ("SELL (short)" if r["enter_short"] else "—")
        mk_rows.append([esc(p), "above" if above else "below", f"{r['close']:.5g}",
                        f"{to_hi:+.1f}%", f"-{to_lo:.1f}%", f"<b>{sig}</b>" if sig != "—" else sig])
    regime = ("Uptrend: bot looks for long breakouts" if btc_up and ups >= downs else
              "Downtrend: bot looks for short breakdowns" if (btc_up is False and downs >= ups) else
              "Mixed: coins and BTC disagree, fewer signals")
    if reg is not None and not reg.empty and int(reg["phase"].iloc[-1]) in (3, 4):
        regime = {3: "Alt season (late): v3 skips alt longs, still takes shorts if they set up",
                  4: "Downtrend, money hiding in gold: v3 waits (no new shorts) — the comparison versions still trade"}[int(reg["phase"].iloc[-1])]
    nxt = (now.floor("4h") + CANDLE + pd.Timedelta(minutes=7))

    cards = []
    for v in variants:
        s, st = S[v.name], states[v.name]
        paused = st.get("pause_until") and now < pd.Timestamp(st["pause_until"])
        status = f'<span class="tag warn">paused until {chi(st["pause_until"])}</span>' if paused else '<span class="tag ok">active</span>'
        wr = f"{s['win_rate']:.0%}" if s["n"] else "—"
        pf = (f"{s['pf']:.2f}" if s.get("pf") not in (None, float("inf")) else ("∞" if s["n"] else "—"))
        avgr = f"{s['avg_R']:+.2f}R" if s["n"] else "—"
        cards.append(f"""
        <div class="card" style="border-top:4px solid {COLORS.get(v.name)}">
          <div class="ch"><b>{esc(v.name)}</b>{status}</div>
          <div class="muted small">{esc(v.description)}</div>
          <div class="big {'pos' if s['ret'] > 0 else 'neg' if s['ret'] < 0 else ''}">{fmt_money(s['equity'])}</div>
          <div class="muted">{s['ret']:+.2f}% since start · includes open trades</div>
          <div class="grid4">
            <div><span>Trades</span><b>{s['n']}</b></div><div><span>Win rate</span><b>{wr}</b></div>
            <div><span>Profit factor</span><b>{pf}</b></div><div><span>Avg trade</span><b>{avgr}</b></div>
            <div><span>Open now</span><b>{s['open']}</b></div><div><span>Worst drop</span><b>{s['mdd']:.1f}%</b></div>
          </div>
        </div>""")

    open_rows = []
    for v in variants:
        st = states[v.name]
        for p, t in st["open"].items():
            px = st.get("last_prices", {}).get(p, t["entry"])
            upl = t["side"] * t["qty"] * (px - t["entry"])
            r_now = upl / (t["qty"] * t["entry"] * t["sp"])
            open_rows.append([esc(v.name), f"<b>{esc(p)}</b>", "LONG" if t["side"] == 1 else "SHORT", chi(t["opened"]),
                              f"{t['entry']:.5g}", f"{t['stop']:.5g}", f"{t['target']:.5g}", f"{px:.5g}",
                              pl(upl), f"{r_now:+.2f}R", fmt_money(t["margin"])])

    allc = sorted(((v.name, c) for v in variants for c in states[v.name]["closed"]), key=lambda x: x[1]["closed"], reverse=True)[:40]
    closed_rows = [[esc(n), f"<b>{esc(c['pair'])}</b>", c["side"].upper(), chi(c["opened"]), chi(c["closed"]),
                    f"{c['entry']:.5g}", f"{c['exit']:.5g}", pl(c["pnl"]), f"{c['R']:+.2f}R",
                    f"{c['mfe_R']:.1f}R / {c['mae_R']:.1f}R", esc(c["reason"].replace("_", " "))] for n, c in allc]

    breakdown = []
    for v in variants:
        c = pd.DataFrame(states[v.name]["closed"])
        if c.empty:
            continue
        rows = []
        for label, grp in [("Side", "side"), ("Exit", "reason"), ("Coin", "pair")]:
            for k, g in c.groupby(grp):
                rows.append([label, esc(str(k).replace("_", " ")), len(g), f"{(g.pnl > 0).mean():.0%}", pl(g.pnl.sum()), f"{g.R.mean():+.2f}R"])
        breakdown.append(f'<h3 style="color:{COLORS.get(v.name)}">{esc(v.name)}</h3>' + table(["Group", "Value", "Trades", "Win rate", "Net P/L", "Avg R"], rows))

    diag = []
    for v in variants:
        items = "".join(f'<li class="{lvl}">{esc(txt)}</li>' for lvl, txt in diagnostics(states[v.name]))
        diag.append(f'<div class="dg"><h3 style="color:{COLORS.get(v.name)}">{esc(v.name)}</h3><ul>{items}</ul></div>')

    started = min((pd.Timestamp(states[v.name]["started"]) for v in variants if states[v.name].get("started")), default=now)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto paper bot — results</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#111827;--mut:#6b7280;--line:#e5e7eb;--pos:#047857;--neg:#b91c1c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0b0f14;--card:#121821;--ink:#e5e7eb;--mut:#9ca3af;--line:#243041;--pos:#34d399;--neg:#f87171}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:20px 16px 60px}} h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:32px 0 10px}} h3{{font-size:15px;margin:16px 0 6px}}
.muted{{color:var(--mut)}} .small{{font-size:13px}} .pos{{color:var(--pos)}} .neg{{color:var(--neg)}}
.banner{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:14px 0;display:flex;gap:18px;flex-wrap:wrap}}
.banner div span{{display:block;color:var(--mut);font-size:12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
.ch{{display:flex;justify-content:space-between;align-items:center}} .big{{font-size:28px;font-weight:700;margin:8px 0 0}}
.grid4{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}} .grid4 span{{display:block;color:var(--mut);font-size:12px}}
.tag{{font-size:12px;padding:2px 8px;border-radius:999px}} .tag.ok{{background:#dcfce7;color:#166534}} .tag.warn{{background:#fef3c7;color:#92400e}}
.tw{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}} th{{color:var(--mut);font-weight:600}}
.chart{{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}} .grid{{stroke:var(--line)}} .base{{stroke:var(--mut);stroke-dasharray:4 4}} .axis{{fill:var(--mut);font-size:11px}}
.legend{{margin-bottom:6px}} .key{{margin-right:14px;font-size:13px}} .key i{{display:inline-block;width:12px;height:3px;margin-right:6px;vertical-align:middle}}
.dgs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}} .dg{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:6px 16px}}
.dg ul{{padding-left:18px}} .dg li{{margin:6px 0}} li.good::marker{{color:var(--pos)}} li.bad::marker{{color:var(--neg)}} li.warn::marker{{color:#d97706}} li.info::marker{{color:var(--mut)}}
.cd{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 14px;margin:8px 0}} .cd summary{{cursor:pointer}} .cd .chart{{border:0;margin-top:8px}} a{{color:#2563eb}}
footer{{margin-top:36px;color:var(--mut);font-size:13px}}
</style></head><body><main>
<h1>Crypto paper-trading bot</h1>
<div class="muted">Paper money only · Binance.US 4H candles · longs and shorts (futures-style, 2x max) · each version started with $100 · v3_main is the strategy, the other two are for comparison</div>
<div class="banner" style="justify-content:space-between"><div><span>Exchange replay</span><b><a href="replay/">Watch the bot trade 2022 → today on real 1-minute futures data, with fees, funding &amp; liquidation ↗</a></b></div></div>
<div class="banner">
  <div><span>Last update</span><b>{chi(now)}</b></div>
  <div><span>Next update</span><b>~{chi(nxt)}</b></div>
  <div><span>Running since</span><b>{chi(started)}</b></div>
  <div><span>Market</span><b>{esc(regime)}</b> <span class="small">{ups} of {ups + downs} coins above 50-day avg · BTC {'above' if btc_up else 'below'}</span></div>
</div>
{rotation_section(reg, now)}
<h2>Scoreboard</h2><div class="cards">{''.join(cards)}</div>
<h2>Equity over time</h2>{svg_equity(states, variants)}
<h2>Charts</h2>{charts_section(states, variants, raw, btc, pairs, now)}
<h2>Open trades</h2>{table(["Version","Coin","Side","Opened","Entry","Stop","Target","Now","P/L","R now","Margin"], open_rows, "No open trades right now.")}
<h2>Where it's lagging &amp; what to improve</h2>
<p class="muted small">Automatic checks against the backtest. R = profit measured in units of the amount risked (−1R = a normal stop-out, +1.5R = a normal win).</p>
<div class="dgs">{''.join(diag)}</div>
<h2>Closed trades (latest 40)</h2>
<p class="muted small">Best / worst = the furthest the trade went in your favour / against you before it closed, in R.</p>
{table(["Version","Coin","Side","Opened","Closed","Entry","Exit","P/L","R","Best / worst","Exit"], closed_rows, "No closed trades yet.")}
<h2>Breakdown</h2>{''.join(breakdown) or '<p class="muted">Appears after the first closed trade.</p>'}
<h2>Market check (last closed 4H candle)</h2>
<p class="muted small">Signals shown for v3_main (rotation rules included). Long: coin and BTC above their 50-day avg, close above the 20-day high.
Short: coin and BTC below their 50-day avg, close below the 20-day low, EMA20 below EMA50, RSI between 20 and 50, and price not more than 3 ATR below EMA20.</p>
{table(["Coin","vs 50-day avg","Price","To 20-day high","To 20-day low","Signal"], mk_rows)}
<footer>Rules (v3_main): longs and shorts below, plus market rotation (no alt longs in alt season, no shorts in gold risk-off). 20-day breakout/breakdown in the direction of the 50-day trend (coin and BTC); shorts also need EMA20 &lt; EMA50, RSI 20-50 and price not overstretched;
stop 2.5× ATR; take profit at 1.5× the stop distance for longs, 1× for shorts;
exit if price closes back across the 50-day average; 1% risk per trade, max 3 open; 12h pause after 2 losing stops in 24h; no new trades after a 2% loss day or below $50.
Not modelled: funding fees and the funding-rate filter of the real futures bot. Times in Chicago time.</footer>
</main></body></html>"""


def svg_candles(f: pd.DataFrame, pair: str, st: dict, n: int = 120, w: int = 1000, h: int = 340) -> str:
    """Candlestick chart of the last n closed 4H candles with averages, breakout levels and the main version's trades."""
    d = f.tail(n)
    if len(d) < 10:
        return ""
    pad_l, pad_r, pad_t, pad_b = 8, 62, 10, 22
    t0, t1 = d.index[0], d.index[-1]
    span = (t1 - t0).total_seconds() or 1
    trades_open = [t for p, t in st["open"].items() if p == pair]
    trades_closed = [c for c in st["closed"] if c["pair"] == pair and pd.Timestamp(c["closed"]) >= t0]
    levels = [x for t in trades_open for x in (t["stop"], t["target"], t["entry"])]
    cols = [c for c in ("sma300", "ema20", "hh120", "ll120") if c in d]
    lo = min(d["low"].min(), *(d[c].min() for c in cols), *levels) if levels else min(d["low"].min(), *(d[c].min() for c in cols))
    hi = max(d["high"].max(), *(d[c].max() for c in cols), *levels) if levels else max(d["high"].max(), *(d[c].max() for c in cols))
    lo, hi = lo * 0.995, hi * 1.005
    cw = (w - pad_l - pad_r) / len(d)
    X = lambda t: pad_l + (pd.Timestamp(t) - t0).total_seconds() / span * (w - pad_l - pad_r - cw) + cw / 2
    Y = lambda v: pad_t + (1 - (v - lo) / (hi - lo)) * (h - pad_t - pad_b)
    g = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(pair)} 4H chart">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        g.append(f'<line x1="{pad_l}" y1="{Y(v):.1f}" x2="{w - pad_r}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{w - pad_r + 4}" y="{Y(v) + 4:.1f}" class="axis">{v:.4g}</text>')
    for frac in (0, 0.5, 1):
        t = t0 + (t1 - t0) * frac
        g.append(f'<text x="{X(t):.1f}" y="{h - 6}" class="axis" text-anchor="middle">{t.tz_convert("America/Chicago"):%b %d}</text>')
    for t, r in d.iterrows():
        up = r["close"] >= r["open"]
        col = "var(--pos)" if up else "var(--neg)"
        x = X(t)
        yb, yt = Y(max(r["open"], r["close"])), Y(min(r["open"], r["close"]))
        g.append(f'<line x1="{x:.1f}" y1="{Y(r["high"]):.1f}" x2="{x:.1f}" y2="{Y(r["low"]):.1f}" stroke="{col}"/>'
                 f'<rect x="{x - cw * 0.35:.1f}" y="{yb:.1f}" width="{cw * 0.7:.1f}" height="{max(yt - yb, 0.8):.1f}" fill="{col}"/>')
    styles = {"sma300": ("#6b7280", ""), "ema20": ("#d97706", ""), "hh120": ("#2563eb", "3 3"), "ll120": ("#2563eb", "3 3")}
    for c in cols:
        s = d[c].dropna()
        pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in s.items())
        col, dash = styles[c]
        g.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.3" stroke-dasharray="{dash}"/>')
    for t in trades_open:
        xs = X(max(pd.Timestamp(t["opened"]), t0))
        for val, col, lab in ((t["target"], "var(--pos)", "target"), (t["entry"], "#2563eb", "entry"), (t["stop"], "var(--neg)", "stop")):
            g.append(f'<line x1="{xs:.1f}" y1="{Y(val):.1f}" x2="{w - pad_r}" y2="{Y(val):.1f}" stroke="{col}" stroke-width="1.5" stroke-dasharray="6 3"/>'
                     f'<text x="{w - pad_r - 4}" y="{Y(val) + (12 if lab == "target" and t["side"] == -1 or lab == "stop" and t["side"] == 1 else -4):.1f}" '
                     f'class="axis" text-anchor="end" style="fill:{col};font-weight:600">{lab} {val:.4g}</text>')
    for c in trades_closed + [dict(t, side="long" if t["side"] == 1 else "short") for t in trades_open]:
        ot = pd.Timestamp(c["opened"])
        if ot >= t0:
            x, y = X(ot), Y(c["entry"])
            tri = (f"{x:.1f},{y + 9:.1f} {x - 6:.1f},{y + 19:.1f} {x + 6:.1f},{y + 19:.1f}" if c["side"] == "long"
                   else f"{x:.1f},{y - 9:.1f} {x - 6:.1f},{y - 19:.1f} {x + 6:.1f},{y - 19:.1f}")
            g.append(f'<polygon points="{tri}" fill="#2563eb"><title>{c["side"]} entry {c["entry"]:.5g}</title></polygon>')
        if "exit" in c:
            x, y = X(min(pd.Timestamp(c["closed"]) - CANDLE, t1)), Y(c["exit"])
            col = "var(--pos)" if c["pnl"] > 0 else "var(--neg)"
            g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}" stroke="var(--card)" stroke-width="1.5">'
                     f'<title>exit {c["exit"]:.5g} ({c["reason"].replace("_", " ")}) {c["pnl"]:+.2f} USD</title></circle>')
    g.append("</svg>")
    return "".join(g)


def charts_section(states: dict, variants, raw: dict, btc: pd.DataFrame, pairs: list, now: pd.Timestamp) -> str:
    main = variants[0]
    st = states[main.name]
    order = sorted(pairs, key=lambda p: (p not in st["open"], p))
    out = []
    for p in order:
        f = indicators(raw[p], btc, main)
        f = f[f.index + CANDLE <= now]
        svg = svg_candles(f, p, st)
        if not svg:
            continue
        tv = f"https://www.tradingview.com/chart/?symbol=BINANCE:{p}USDT.P&interval=240"
        tag = ""
        if p in st["open"]:
            t = st["open"][p]
            tag = f' <span class="tag ok">{"LONG" if t["side"] == 1 else "SHORT"} open</span>'
        out.append(f'<details {"open" if p in st["open"] else ""} class="cd"><summary><b>{esc(p)}</b>{tag}'
                   f' <span class="muted small">last 20 days, 4H</span></summary>{svg}'
                   f'<div class="small"><a href="{tv}" target="_blank" rel="noopener">Open live {esc(p)} chart on TradingView ↗</a></div></details>')
    legend = ('<p class="muted small">Grey = 50-day average (trend) · orange = EMA20 · blue dotted = 20-day high / low (breakout levels) · '
              '▲ long entry, ▼ short entry · dots = exits (green profit, red loss) · dashed lines = open trade entry / stop / target. '
              f'Trades shown are {esc(main.name)}. Coins with an open trade are listed first and expanded.</p>')
    return legend + "".join(out)


PHASE_INFO = {
    1: ("BTC season", "#f59e0b", "BTC leads, alts lag. Historically the BEST time for alt breakout longs: the money has not rotated yet."),
    2: ("ETH / large caps", "#3b82f6", "Money moving from BTC into ETH and large caps. Alt longs work well; alt season often comes next."),
    3: ("Alt season", "#10b981", "Alts already outperforming BTC. Late stage: alt breakouts here lost money in the backtest, so v3 takes no alt longs."),
    4: ("Risk-off (gold > BTC)", "#6b7280", "BTC in a downtrend and gold beating BTC: money is hiding. Shorts here were break-even, so v3 does not short."),
    5: ("Crypto down, gold not bid", "#ef4444", "BTC downtrend without a flight to gold: active selling. Historically the best time for shorts."),
}


def rotation_section(reg, now) -> str:
    if reg is None or reg.empty:
        return ""
    r = reg.dropna(subset=["phase"])
    cur = int(r["phase"].iloc[-1])
    if cur == 0:
        return ""
    name, col, txt = PHASE_INFO[cur]
    d = r.set_index("date")["phase"].resample("1D").last().dropna().tail(120)
    w, h = 1000, 26
    bw = w / max(len(d), 1)
    bars = "".join(f'<rect x="{i * bw:.1f}" y="0" width="{bw + 0.5:.1f}" height="{h}" fill="{PHASE_INFO.get(int(v), ("", "#e5e7eb", ""))[1]}">'
                   f'<title>{t:%b %d}: {PHASE_INFO.get(int(v), ("n/a",))[0]}</title></rect>' for i, (t, v) in enumerate(d.items()))
    strip = f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:26px;border-radius:6px">{bars}</svg>'
    keys = " ".join(f'<span class="key"><i style="background:{c};height:10px"></i>{n}</span>' for n, c, _ in PHASE_INFO.values())
    return f"""<h2>Market rotation</h2>
<div class="card"><div class="ch"><b>Now: <span style="color:{col}">{esc(name)}</span></b><span class="muted small">90-day view of BTC vs alts, ETH/BTC and gold/BTC</span></div>
<p style="margin:6px 0 10px">{esc(txt)}</p>{strip}
<div class="muted small" style="display:flex;justify-content:space-between"><span>{d.index[0]:%b %d}</span><span>last 120 days</span><span>{d.index[-1]:%b %d}</span></div>
<div class="legend" style="margin-top:6px">{keys}</div></div>"""
