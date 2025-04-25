# kafka_producer.py
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

data = {"message": "Hello from Jake's producer!"}

producer.send("kalshi-trades", data)
producer.flush()

print("✅ Message sent!")