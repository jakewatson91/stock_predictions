import requests, time, json
from datetime import datetime, timedelta
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="kafka:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def fetch_trades():
    now = datetime.now(datetime.timezone.utc)
    min_ts = int((now - timedelta(hours=6)).timestamp() * 1000)
    url = "https://api.elections.kalshi.com/trade-api/v2/markets/trades"
    params = {"limit": 1000, "min_ts": min_ts}
    cursor = None

    while True:
        if cursor:
            params["cursor"] = cursor
        r = requests.get(url, params=params)
        data = r.json()
        for trade in data.get("trades", []):
            producer.send("kalshi.trades", trade)
        cursor = data.get("cursor")
        if not cursor: break
        time.sleep(0.2)

def fetch_markets():
    url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    r = requests.get(url)
    for market in r.json().get("markets", []):
        producer.send("kalshi.markets", market)

if __name__ == "__main__":
    fetch_markets()
    fetch_trades()
    producer.flush()