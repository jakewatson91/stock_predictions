import requests
import json
import os
from datetime import datetime, timedelta, timezone
from google.cloud import storage
from dotenv import load_dotenv

from gcs_utils import upload_to_gcs

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds/gcp-sa-key.json"

BUCKET_NAME = "kalshi-data-lake"

def fetch_trades():
    now = datetime.now(timezone.utc)
    min_ts = int((now - timedelta(hours=1)).timestamp() * 1000)
    cursor = None
    trades = []

    # while True:
    url = "https://api.elections.kalshi.com/trade-api/v2/markets/trades"
    params = {"limit": 10, "min_ts": min_ts}
    # if cursor:
    #     params["cursor"] = cursor
    r = requests.get(url, params=params)
    data = r.json()
    print(data)
    trades.extend(data.get("trades", []))
    print(trades)
    # cursor = data.get("cursor")
    # if not cursor:
    #     break

    # timestamp = now.strftime("%Y-%m-%dT%H")
    # filename = f"trades_{timestamp}.json"
    # with open(filename, "w") as f:
    #     json.dump(trades, f)
    # upload_to_gcs(filename, f"trades/{filename}")

def fetch_markets():
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    r = requests.get(url)
    markets = r.json().get("markets", [])
    print(markets[0].keys())
    for market in markets:
        market_json = {
            "ticker" : market['ticker'],
            "event_ticker" : market['event_ticker']   
        }

    # now = datetime.now(timezone.utc)
    # timestamp = now.strftime("%Y-%m-%dT%H")
    # filename = f"markets_{timestamp}.json"
    # with open(filename, "w") as f:
    #     json.dump(markets, f)
    # upload_to_gcs(filename, f"markets/{filename}")

if __name__ == "__main__":
    # fetch_markets()
    fetch_trades()