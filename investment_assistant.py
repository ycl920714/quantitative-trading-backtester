"""
🤖 AI Investment Assistant
Platform: Trading 212 (UK)
Notification: Telegram Bot
Budget: £300
Author: Auto-generated for your use

DISCLAIMER: This is for educational purposes. 
Invest only what you can afford to lose.
"""

import os
import time
import json
import logging
import asyncio
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ─── Load config ────────────────────────────────────────────────────────────
load_dotenv()

T212_API_KEY    = os.getenv("T212_API_KEY")
T212_API_SECRET = os.getenv("T212_API_SECRET")
T212_MODE       = os.getenv("T212_MODE", "demo")   # "demo" or "live"
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TOTAL_BUDGET    = float(os.getenv("TOTAL_BUDGET", "300"))
MAX_PER_TRADE   = float(os.getenv("MAX_PER_TRADE", "30"))   # max £30 per trade
CONFIRM_TIMEOUT = int(os.getenv("CONFIRM_TIMEOUT", "300"))  # 5 min to confirm

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── Trading 212 API ──────────────────────────────────────────────────────────
BASE_URL = (
    "https://demo.trading212.com/api/v0"
    if T212_MODE == "demo"
    else "https://live.trading212.com/api/v0"
)

import base64

def t212_headers():
    creds = f"{T212_API_KEY}:{T212_API_SECRET}"
    encoded = base64.b64encode(creds.encode()).decode()
    return {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json"
    }

def get_account_cash():
    r = requests.get(f"{BASE_URL}/equity/account/cash", headers=t212_headers())
    r.raise_for_status()
    return r.json()

def get_portfolio():
    r = requests.get(f"{BASE_URL}/equity/portfolio", headers=t212_headers())
    r.raise_for_status()
    return r.json()

def get_instruments():
    r = requests.get(f"{BASE_URL}/equity/metadata/instruments", headers=t212_headers())
    r.raise_for_status()
    return r.json()

def place_market_order(ticker: str, value_gbp: float):
    """Place a fractional market order by GBP value."""
    payload = {
        "ticker": ticker,
        "value": value_gbp
    }
    r = requests.post(
        f"{BASE_URL}/equity/orders/market",
        headers=t212_headers(),
        json=payload
    )
    r.raise_for_status()
    return r.json()

# ─── Telegram ─────────────────────────────────────────────────────────────────
def tg_send(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    r = requests.post(url, json=payload)
    return r.json()

def tg_get_updates(offset=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {"timeout": 30}
    if offset:
        params["offset"] = offset
    r = requests.get(url, params=params)
    return r.json()

def wait_for_confirmation(trade_id: str, timeout_seconds: int = 300) -> bool:
    """
    Wait for user to reply YES or NO to a trade notification.
    Returns True if confirmed, False if rejected or timed out.
    """
    log.info(f"Waiting for confirmation of trade {trade_id}...")
    start = time.time()
    last_update_id = None

    # Get current update offset so we only read NEW messages
    updates = tg_get_updates()
    if updates.get("result"):
        last_update_id = updates["result"][-1]["update_id"] + 1

    while time.time() - start < timeout_seconds:
        updates = tg_get_updates(offset=last_update_id)
        for update in updates.get("result", []):
            last_update_id = update["update_id"] + 1
            msg = update.get("message", {})
            text = msg.get("text", "").strip().upper()
            chat_id = str(msg.get("chat", {}).get("id", ""))

            if chat_id == str(TELEGRAM_CHAT_ID):
                if text in ("YES", "Y", "確認", "是", "OK", "CONFIRM"):
                    log.info("Trade confirmed by user ✅")
                    return True
                elif text in ("NO", "N", "取消", "否", "CANCEL", "SKIP"):
                    log.info("Trade rejected by user ❌")
                    return False

        time.sleep(3)

    log.warning(f"Confirmation timeout for trade {trade_id}")
    return False

# ─── AI Strategy (using free Yahoo Finance data) ─────────────────────────────
def fetch_price_history(ticker_yahoo: str, period="3mo"):
    """
    Fetch price history from Yahoo Finance (free, no API key needed).
    ticker_yahoo: e.g. "VUSA.L", "AAPL", "SPY"
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker_yahoo)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        log.error(f"Failed to fetch {ticker_yahoo}: {e}")
        return None

def calculate_signals(hist):
    """
    Simple technical analysis signals:
    - 50-day and 20-day moving averages
    - RSI (Relative Strength Index)
    Returns: dict with signal and confidence
    """
    if hist is None or len(hist) < 30:
        return {"signal": "HOLD", "confidence": 0, "reason": "Insufficient data"}

    close = hist["Close"]

    # Moving averages
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    current = close.iloc[-1]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    # Recent momentum (5-day)
    momentum = (current - close.iloc[-6]) / close.iloc[-6] * 100

    signals = []
    score = 0

    # MA crossover
    if current > ma20 > ma50:
        signals.append("📈 Price above both MAs (bullish)")
        score += 2
    elif current < ma20 < ma50:
        signals.append("📉 Price below both MAs (bearish)")
        score -= 2
    else:
        signals.append("➡️ Mixed MA signals")

    # RSI
    if 40 < rsi < 60:
        signals.append(f"✅ RSI neutral ({rsi:.1f})")
        score += 1
    elif rsi < 30:
        signals.append(f"🔥 RSI oversold ({rsi:.1f}) - potential buy")
        score += 3
    elif rsi > 70:
        signals.append(f"⚠️ RSI overbought ({rsi:.1f}) - caution")
        score -= 2

    # Momentum
    if momentum > 2:
        signals.append(f"🚀 Strong 5-day momentum (+{momentum:.1f}%)")
        score += 1
    elif momentum < -2:
        signals.append(f"🔻 Negative 5-day momentum ({momentum:.1f}%)")
        score -= 1

    if score >= 3:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = min(abs(score) / 5 * 100, 95)

    return {
        "signal": signal,
        "confidence": round(confidence, 1),
        "score": score,
        "rsi": round(rsi, 1),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "current_price": round(current, 2),
        "momentum_5d": round(momentum, 2),
        "reasons": signals
    }

# ─── Watchlist ────────────────────────────────────────────────────────────────
# Format: (Trading212_ticker, Yahoo_Finance_ticker, name, allocation_%)
WATCHLIST = [
    # Low-risk ETFs (60% of budget)
    ("VUSA",     "VUSA.L",  "Vanguard S&P 500 ETF",       0.30),
    ("VWRL",     "VWRL.L",  "Vanguard All-World ETF",     0.20),
    ("ISF",      "ISF.L",   "iShares FTSE 100 ETF",       0.10),
    # Individual UK/US stocks (40% of budget)
    ("AAPL_US_EQ","AAPL",   "Apple Inc",                  0.15),
    ("MSFT_US_EQ","MSFT",   "Microsoft Corp",             0.15),
    ("RDSA_L_EQ", "SHEL.L", "Shell PLC",                  0.10),
]

# ─── Main Loop ────────────────────────────────────────────────────────────────
class InvestmentAssistant:
    def __init__(self):
        self.pending_trades = {}
        self.trade_counter = 0

    def run_analysis_cycle(self):
        """Analyse all watchlist items and propose trades."""
        log.info("═" * 50)
        log.info("🔍 Starting analysis cycle...")

        try:
            cash_data = get_account_cash()
            available_cash = cash_data.get("free", 0)
            total_value = cash_data.get("total", 0)
            log.info(f"💷 Account: free={available_cash:.2f}, total={total_value:.2f}")
        except Exception as e:
            log.error(f"Cannot fetch account info: {e}")
            return

        if available_cash < 5:
            log.info("⚠️ Less than £5 available – skipping cycle")
            tg_send("⚠️ <b>Investment Assistant</b>\nAvailable cash is below £5. Skipping this cycle.")
            return

        proposals = []

        for t212_ticker, yahoo_ticker, name, alloc in WATCHLIST:
            log.info(f"Analysing {name} ({yahoo_ticker})...")
            hist = fetch_price_history(yahoo_ticker)
            sig = calculate_signals(hist)

            if sig["signal"] == "BUY" and sig["confidence"] >= 50:
                trade_value = min(
                    TOTAL_BUDGET * alloc,
                    MAX_PER_TRADE,
                    available_cash * 0.5   # never use more than 50% of cash at once
                )
                if trade_value >= 1.0:
                    proposals.append({
                        "t212_ticker": t212_ticker,
                        "name": name,
                        "signal": sig,
                        "trade_value": round(trade_value, 2)
                    })

        if not proposals:
            msg = (
                "🤖 <b>Investment Assistant – Cycle Complete</b>\n"
                f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
                "📊 No strong BUY signals found this cycle.\n"
                "✅ Portfolio unchanged. Next check in 4 hours."
            )
            tg_send(msg)
            log.info("No buy signals this cycle.")
            return

        # Send proposals one by one and wait for confirmation
        for p in proposals:
            self.trade_counter += 1
            trade_id = f"T{self.trade_counter:04d}"
            sig = p["signal"]

            reasons_text = "\n".join(f"  • {r}" for r in sig["reasons"])
            msg = (
                f"🤖 <b>Investment Assistant – Trade Proposal</b>\n"
                f"🆔 Trade ID: <code>{trade_id}</code>\n"
                f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
                f"📦 <b>{p['name']}</b>\n"
                f"   Ticker: {p['t212_ticker']}\n"
                f"   Signal: {'🟢 BUY' if sig['signal']=='BUY' else '🔴 SELL'}\n"
                f"   Confidence: {sig['confidence']}%\n"
                f"   Current price: £{sig['current_price']}\n"
                f"   RSI: {sig['rsi']}\n"
                f"   5-day momentum: {sig['momentum_5d']}%\n\n"
                f"📋 <b>Signals:</b>\n{reasons_text}\n\n"
                f"💷 <b>Proposed investment: £{p['trade_value']}</b>\n\n"
                f"⚡ Reply <b>YES</b> to confirm or <b>NO</b> to skip\n"
                f"⏳ Auto-cancel in {CONFIRM_TIMEOUT//60} minutes"
            )
            tg_send(msg)
            log.info(f"Sent proposal {trade_id} for {p['name']}")

            confirmed = wait_for_confirmation(trade_id, CONFIRM_TIMEOUT)

            if confirmed:
                try:
                    order = place_market_order(p["t212_ticker"], p["trade_value"])
                    tg_send(
                        f"✅ <b>Order Placed!</b>\n"
                        f"🆔 {trade_id} – {p['name']}\n"
                        f"💷 £{p['trade_value']} invested\n"
                        f"📋 Order ID: {order.get('id', 'N/A')}\n"
                        f"Mode: <i>{'DEMO' if T212_MODE=='demo' else '🔴 LIVE'}</i>"
                    )
                    log.info(f"✅ Order placed for {p['name']}: {order}")
                except Exception as e:
                    tg_send(f"❌ <b>Order Failed</b>\n{p['name']}\nError: {e}")
                    log.error(f"Order failed for {p['name']}: {e}")
            else:
                tg_send(
                    f"⏭️ <b>Trade Skipped</b>\n"
                    f"🆔 {trade_id} – {p['name']} – No action taken."
                )

    def run_forever(self, interval_hours=4):
        """Run analysis every N hours."""
        log.info("🚀 Investment Assistant started")
        tg_send(
            "🤖 <b>Investment Assistant Online!</b>\n"
            f"💷 Budget: £{TOTAL_BUDGET}\n"
            f"🔄 Checking every {interval_hours} hours\n"
            f"📊 Watching {len(WATCHLIST)} instruments\n"
            f"Mode: <i>{'📋 DEMO (Paper Trading)' if T212_MODE=='demo' else '🔴 LIVE'}</i>\n\n"
            "Send /status to check portfolio\nSend /stop to pause"
        )

        while True:
            try:
                self.run_analysis_cycle()
            except Exception as e:
                log.error(f"Cycle error: {e}")
                tg_send(f"⚠️ <b>Error in analysis cycle:</b>\n{e}")

            next_run = datetime.now() + timedelta(hours=interval_hours)
            log.info(f"💤 Next run at {next_run.strftime('%H:%M')}")
            tg_send(f"😴 Next analysis at <b>{next_run.strftime('%d/%m %H:%M')}</b>")
            time.sleep(interval_hours * 3600)


if __name__ == "__main__":
    assistant = InvestmentAssistant()
    # Default: run every 4 hours
    assistant.run_forever(interval_hours=4)
