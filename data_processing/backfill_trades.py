import os
import json
import time
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from gcs_utils import upload_to_gcs
import pandas as pd
import numpy as np

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds/gcp-sa-key.json"
BUCKET = "kalshi-data-lake"

START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 2, 1, tzinfo=timezone.utc)

def fetch_and_store_trades(start_ts, end_ts=None):
    url = "https://api.elections.kalshi.com/trade-api/v2/markets/trades"
    cursor = None
    params = {"limit": 1000, "min_ts": start_ts, "max_ts" : end_ts}

    seen_ids = set()
    all_trades = []
    total_trades = 0
    i = 0
    while True:
        i += 1
        if cursor:
            params["cursor"] = cursor

        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        trades = data.get("trades", [])
        cursor = data.get("cursor")

        new_count = 0
        for trade in trades:
            tid = trade["trade_id"]
            if tid not in seen_ids:
                seen_ids.add(tid)
                ts = datetime.fromisoformat(trade["created_time"].replace("Z", "+00:00"))
                trade["day_str"] = ts.strftime("%Y-%m-%d")
                all_trades.append(trade)
                new_count += 1

        total_trades += new_count
        print(f"Retrieved {len(trades)} trades... New: {new_count} Total unique: {total_trades}")
        print("Final cursor:", cursor)

        if not cursor or not trades:
            break
        time.sleep(0.2)

        if i % 50 == 0:
            print(f"Step: {i}, Total trade count: {total_trades}")
            # break
    print("Total trades: ", total_trades)
    print("Final cursor:", cursor)

    # Dump all trades to Parquet grouped by hour
    df = pd.DataFrame(all_trades)
    for day, group in df.groupby("day_str"):
        filename = f"trades_{day}.parquet"
        group.drop(columns=["day_str"]).to_parquet(filename, index=False)
        upload_to_gcs(filename, f"trades/03_2025/{filename}")
        os.remove(filename)

def fetch_markets():
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    cursor = None
    seen_tickers = set()
    all_markets = []
    total_markets = 0
    i = 0

    while True:
        i += 1
        params = {"limit": 1000, "min_close_ts": start_ts}
        if cursor:
            params["cursor"] = cursor

        try:
            r = requests.get(url, params=params)
            if r.status_code == 429:
                print("Rate limit hit, sleeping 5 seconds...")
                time.sleep(5)
                continue

            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e}")
            break

        data = r.json()
        markets = data.get("markets", [])
        cursor = data.get("cursor")

        new_count = 0
        for market in markets:
            ticker = market["ticker"]
            if ticker not in seen_tickers:
                seen_tickers.add(ticker)
                ts = datetime.fromisoformat(market["open_time"].replace("Z", "+00:00"))
                market["day_str"] = ts.strftime("%Y-%m-%d")
                all_markets.append(market)
                new_count += 1

        total_markets += new_count
        print(f"Retrieved {len(markets)} markets... New: {new_count} Total unique: {total_markets}")
        print("Final cursor:", cursor)

        if not cursor or not markets:
            break

        time.sleep(0.2)
        # break

    # Save only unique markets
    df = pd.DataFrame(all_markets)
    filename = "01012025_to_04242025.parquet"
    df.to_parquet(filename, index=False)
    upload_to_gcs(filename, f"markets/{filename}")
    os.remove(filename)
    

def fetch_events():
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    cursor = None
    params = {"limit": 200}

    url = "https://api.elections.kalshi.com/trade-api/v2/events"
    r = requests.get(url)
    print("Status Code:", r.status_code)
    print("Response Text:", r.text)
    r.raise_for_status()

    data = r.json()

    all_events = []
    total_events = 0
    i = 0

    while True:
        i += 1
        if cursor:
            params["cursor"] = cursor
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        events = r.json().get("events", [])
        cursor = data.get("cursor")

        for event in events:
            all_events.append(event)

        print(f"Retrieved {len(events)} events...")
        total_events += len(events)

        if not cursor or not events:
            break
        time.sleep(0.2)

        if i % 50 == 0:
            print(f"Step: {i}, Total event count: {total_events}")
            # break
        print("Total events: ", total_events)
        print("Final cursor:", cursor)

        # Dump all events to Parquet
        df = pd.DataFrame(all_events)
        filename = f"all.parquet"
        df.to_parquet(filename, index=False)
        upload_to_gcs(filename, f"events/{filename}")
        os.remove(filename)        

if __name__ == "__main__":
    start_ts = int(START_DATE.timestamp())  # NOTE: Kalshi expects seconds, not ms
    end_ts = int(END_DATE.timestamp())  
    fetch_and_store_trades(end_ts)
    # fetch_markets()
    # fetch_events()