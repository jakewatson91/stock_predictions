import os
import json
import time
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
from gcs_utils import upload_to_gcs

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds/gcp-sa-key.json"
BUCKET = "kalshi-data-lake"

# ── FETCH TRADES ──────────────────────────────────────────────────────
def fetch_trades_batch(start_ts, end_ts=None, limit=1000, cursor=None):
    url = "https://api.elections.kalshi.com/trade-api/v2/markets/trades"
    params = {"limit": limit, "min_ts": start_ts, "max_ts": end_ts}
    if cursor:
        params["cursor"] = cursor

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    trades = data.get("trades", [])
    cursor = data.get("cursor")
    return trades, cursor

# ── FETCH MARKETS ─────────────────────────────────────────────────────
def fetch_markets_batch(min_close_ts=None, limit=1000, cursor=None):
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    params = {"limit": limit}
    if min_close_ts:
        params["min_close_ts"] = min_close_ts
    if cursor:
        params["cursor"] = cursor

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    markets = data.get("markets", [])
    cursor = data.get("cursor")
    return markets, cursor

# ── FETCH EVENTS ──────────────────────────────────────────────────────
def fetch_events_batch(limit=1000, cursor=None):
    url = "https://api.elections.kalshi.com/trade-api/v2/events"
    params = {"limit": limit}
    if cursor:
        params["cursor"] = cursor

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    events = data.get("events", [])
    cursor = data.get("cursor")
    return events, cursor

# ── STORE TRADES ──────────────────────────────────────────────────────
def store_trades(trades):
    for trade in trades:
        ts = datetime.fromisoformat(trade["created_time"].replace("Z", "+00:00"))
        trade["day_str"] = ts.strftime("%Y-%m-%d")

    df = pd.DataFrame(trades)
    for day, group in df.groupby("day_str"):
        filename = f"trades_{day}.parquet"
        group.drop(columns=["day_str"]).to_parquet(filename, index=False)
        upload_to_gcs(filename, f"trades/historical/01_2025/{filename}")
        os.remove(filename)

# ── STORE MARKETS ─────────────────────────────────────────────────────
def store_markets(markets):
    for market in markets:
        ts = datetime.fromisoformat(market["open_time"].replace("Z", "+00:00"))
        market["day_str"] = ts.strftime("%Y-%m-%d")

    df = pd.DataFrame(markets)
    for day, group in df.groupby("day_str"):
        filename = f"markets_{day}.parquet"
        group.drop(columns=["day_str"]).to_parquet(filename, index=False)
        upload_to_gcs(filename, f"markets/{filename}")
        os.remove(filename)

# ── MAIN LOOP EXAMPLE ─────────────────────────────────────────────────
if __name__ == "__main__":
    START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
    END_DATE = datetime(2025, 1, 2, tzinfo=timezone.utc)
    start_ts = int(START_DATE.timestamp())
    end_ts = int(END_DATE.timestamp())

    all_trades = []
    cursor = None
    while True:
        trades, cursor = fetch_trades_batch(start_ts, end_ts, cursor=cursor)
        all_trades.extend(trades)
        print(f"Fetched {len(trades)} trades. Cursor: {cursor}")
        if not cursor or not trades:
            break
        time.sleep(0.2)
    store_trades(all_trades)

    # Similar loop for markets
    all_markets = []
    cursor = None
    while True:
        markets, cursor = fetch_markets_batch(min_close_ts=start_ts, cursor=cursor)
        all_markets.extend(markets)
        print(f"Fetched {len(markets)} markets. Cursor: {cursor}")
        if not cursor or not markets:
            break
        time.sleep(0.2)
    store_markets(all_markets)