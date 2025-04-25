FROM python:3.11

WORKDIR /apis
COPY . .

RUN pip install -r requirements.txt

CMD ["python", "kalshi_producer.py"]