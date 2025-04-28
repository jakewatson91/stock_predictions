import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp, col, min, max
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

import torch
import torch.nn as nn

from spark_read import read_parquet

PROJECT_ID='kalshi-456121'
BUCKET_NAME='kalshi-data-lake'
CLUSTER='kalshi-dataproc-cluster'
REGION='us-central1'

# Path to the GCS connector jar
from os.path import expanduser
gcs_connector_path = expanduser("/home/jwatson/gcs-connector-hadoop3-latest.jar") # change depending on where it's stored
# Start a local Spark session with the GCS connector
spark = SparkSession.builder \
    .appName("QueryGCSParquet") \
    .config("spark.jars", gcs_connector_path) \
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
    .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", "/home/jwatson/cs541/stock_predictions/creds/gcp-sa-key.json") \
    .config("spark.hadoop.fs.gs.project.id", "kalshi-456121") \
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .getOrCreate()

def read_parquet():
    # GCS bucket path to the Parquet file or folder
    gcs_path = f"gs://{BUCKET_NAME}/processed_data/"

    # Load the Parquet file from GCS
    # df = spark.read.parquet(f"{gcs_path}/*.parquet")
    # test file: part-00000-b3c7923c-db74-44a4-8ad7-37cd92c3a014-c000.snappy.parquet
    df = spark.read.parquet(f"{gcs_path}/part-00000-b3c7923c-db74-44a4-8ad7-37cd92c3a014-c000.snappy.parquet")


    print("Records: ", df.count())
    # df.printSchema()
    # df.show(1)
    return df

df = read_parquet()
df = df.drop("market_type") # only one type

date_cols = [
    "open_time",
    "close_time"
]
# purely categorical (string/date) features
categorical_cols = ["category"]

embedding_cols = [
    "market_title",
    "market_subtitle",
    "market_desc",
    "event_title",
    "event_subtitle",
]

# all the remaining numeric indicators
numeric_cols = [col for col in df.columns if col not in [categorical_cols, embedding_cols, date_cols]]

def cast_date(df, cols):
    for c in cols:
        df = df.withColumn(f"{c}_unix", unix_timestamp(c, "yyyy-MM-dd'T'HH:mm:ssX")).drop(c)
    return df

def cast_numeric(df, cols):
    # convert to double
    for c in cols:
        df = df.withColumn(c, col(c).cast("double"))
    return df

def cast_categorical(df, cols):
    # Create StringIndexer stages
    indexer = StringIndexer(inputCol="category", outputCol="category_index", handleInvalid="keep")
    indexer_model = indexer.fit(df)

    print("NUM CATEGORIES: ", indexer_model.labels)
    num_categories = indexer_model.labels

    df = indexer_model.transform(df)

    # Drop original string columns
    df = df.drop("category")

    categorical_df = df.select("category_index").toPandas()
    # print("DF INFO: ", categorical_df.info())

    category_input = torch.tensor(categorical_df.values.flatten(), dtype=torch.long)

    embedding_category = nn.Embedding(num_embeddings=num_categories, embedding_dim=4)
    category_embedded = embedding_category(category_input)
    print(category_embedded.shape)     # torch.Size([4, 4])
    
    return category_embedded

def embed_categorical(embedding_indices):
    embedding_category = nn.Embedding(num_embeddings=4, embedding_dim=4)
    
    # Get embeddings
    category_embedded = embedding_category(category_input)

    # Resulting shapes: [batch_size, embedding_dim]
    print(market_type_embedded.shape)  # torch.Size([4, 4])
    print(category_embedded.shape)     # torch.Size([4, 4])

    # Concatenate along feature dimension
    combined_embedding = torch.cat([market_type_embedded, category_embedded], dim=1)

    # Shape: [batch_size, total_embedding_dim]
    print(combined_embedding.shape)  # torch.Size([4, 8])

print(cast_categorical(df, categorical_cols))

df = cast_date(df, date_cols)
df = cast_numeric(df, numeric_cols)
df = cast_categorical(df, categorical_cols)

# Convert to Pandas DataFrame for PyTorch
df = df.toPandas()

