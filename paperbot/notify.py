"""Optional Telegram alerts. Active only if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID are set (GitHub repo secrets)."""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

SITE = "https://priyanshubawse13.github.io/crypto-paper-bot/"


def send(text: str) -> None:
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text, "disable_web_page_preview": "true"}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=15).read()
    except Exception as e:  # noqa: BLE001  (alerts must never break the bot)
        print("telegram failed:", e)


def trade_changes(name: str, before: tuple, st: dict) -> None:
    old_open, old_n = before
    for p, t in st["open"].items():
        if p not in old_open:
            side = "LONG" if t["side"] == 1 else "SHORT"
            send(f"PAPER {name}: opened {side} {p} at {t['entry']:.5g}\nstop {t['stop']:.5g} | target {t['target']:.5g}\n{SITE}")
    for c in st["closed"][old_n:]:
        send(f"PAPER {name}: closed {c['side'].upper()} {c['pair']} at {c['exit']:.5g} ({c['reason'].replace('_', ' ')})\n"
             f"P/L {c['pnl']:+.2f} USD ({c['R']:+.2f}R)\n{SITE}")
