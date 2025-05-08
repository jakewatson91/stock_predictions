import requests
import json
import os
from datetime import datetime, timedelta, timezone

import requests
from datetime import datetime, timedelta, timezone

def fetch_trades():
    now = datetime.now(timezone.utc)
    min_ts = int((now - timedelta(hours=1)).timestamp() * 1000)

    url = "https://api.elections.kalshi.com/trade-api/v2/markets/trades"
    params = {
        "limit": 10,       # Fetch only 2 trades
    }

    r = requests.get(url, params=params)
    r.raise_for_status()  # Raise error if the request failed
    data = r.json()

    trades = data.get("trades", [])
    print(f"Fetched {len(trades)} trades:")
    for trade in trades:
        print(trade)

categories = set()
def fetch_events():
    url = "https://api.elections.kalshi.com/trade-api/v2/events"
    params = {
        "limit": 1,       # Fetch only 2 trades
    }
    r = requests.get(url, params=params)
    print("Status Code:", r.status_code)
    print("Response Text:", r.text)
    r.raise_for_status()

    data = r.json()
    events = data.get("events", [])
    print(f"Fetched {len(events)} events:")
    for event in events:  # Just print the first few
        if event['category'] not in categories:
            categories.add(event['category'])
        # print(event)


def fetch_markets():
    min_close_ts = int(datetime(2025, 3, 1, tzinfo=timezone.utc).timestamp())

    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    params = {
        "limit": 1,
        "min_close_ts": min_close_ts       # Fetch only 2 trades
    }

    r = requests.get(url, params=params)
    r.raise_for_status()  # Raise error if the request failed
    data = r.json()

    markets = data.get("markets", [])
    for market in markets:
        print(market)

# fetch_events()
# print(categories)
fetch_trades()
fetch_markets()