from kafka import KafkaConsumer
from google.cloud import storage
import os, json, datetime

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds/gcp-sa-key.json"

client = storage.Client()
bucket = client.bucket("kalshi-data-lake")

def upload_to_gcs(topic_name, message):
    now = datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H")
    folder = "trades" if topic_name == "kalshi.trades" else "markets"
    filename = f"{folder}/{now}.json"
    blob = bucket.blob(filename)
    blob.upload_from_string(json.dumps(message), content_type='application/json')
    print(f"Uploaded to {filename}")

consumer = KafkaConsumer(
    "kalshi.trades", "kalshi.markets",
    bootstrap_servers="kafka:9092",
    auto_offset_reset="earliest",
    group_id="gcs-loader",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

for msg in consumer:
    upload_to_gcs(msg.topic, msg.value)